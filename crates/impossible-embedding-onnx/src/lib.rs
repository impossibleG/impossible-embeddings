//! Manifest-driven, in-process ONNX Runtime embedding engine.
//!
//! Models load only from local ONNX and tokenizer JSON files. This adapter never evaluates
//! repository code, Python, pickle, or other executable model content.

use impossible_embedding_core::{
    EmbeddingBatch, EmbeddingEngine, EmbeddingOutput, EngineFailure, ErrorCode, ExecutionControl,
    RequestedModel, ResolvedModelIdentity,
};
use ort::{
    session::{Session, SessionInputValue},
    value::Tensor,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::BTreeSet,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tokenizers::{Encoding, Tokenizer};

/// Stable runtime name used in model identities.
pub const ENGINE_NAME: &str = "onnx-runtime";

/// Semantic purpose of an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTask {
    /// A search query.
    Query,
    /// A document or passage.
    Document,
}

/// Behavior for tokenized inputs beyond the manifest limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// Reject the request.
    Reject,
    /// Retain the first `max_tokens` tokens.
    Truncate,
}

/// Per-request controls beyond the foundation trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedOptions {
    /// Task used to select a prefix.
    pub task: EmbeddingTask,
    /// Explicit over-length behavior.
    pub truncation: Truncation,
    /// Optional, manifest-approved Matryoshka dimension.
    pub dimensions: Option<usize>,
}
impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            task: EmbeddingTask::Document,
            truncation: Truncation::Reject,
            dimensions: None,
        }
    }
}

/// Pooling applied to token-level output.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Mask-weighted mean.
    Mean,
    /// First non-padding token.
    Cls,
    /// Last non-padding token.
    LastToken,
}

/// Names of graph inputs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InputNames {
    /// Token ids.
    pub input_ids: String,
    /// Attention mask.
    pub attention_mask: Option<String>,
    /// Optional segment ids.
    pub token_type_ids: Option<String>,
}

/// Optional task prefixes.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct Prefixes {
    /// Query prefix.
    pub query: Option<String>,
    /// Document prefix.
    pub document: Option<String>,
}

/// Auditable local model manifest.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    /// Supported schema version (currently 1).
    pub schema_version: u32,
    /// Canonical model id.
    pub canonical_id: String,
    /// Immutable revision.
    pub revision: String,
    /// ONNX filename relative to this manifest.
    pub model_file: PathBuf,
    /// Hugging Face tokenizer JSON filename relative to this manifest.
    pub tokenizer_file: PathBuf,
    /// Graph inputs.
    pub inputs: InputNames,
    /// Selected graph output.
    pub output: String,
    /// Pooling rule.
    pub pooling: Pooling,
    /// Limit including special tokens and prefix.
    pub max_tokens: usize,
    /// Token id used for request-time batch padding.
    #[serde(default)]
    pub pad_token_id: u32,
    /// Whether final vectors are L2-normalized.
    pub normalize: bool,
    /// Allowed Matryoshka dimensions; empty prohibits dimension truncation.
    #[serde(default)]
    pub matryoshka_dimensions: Vec<usize>,
    /// Task prefixes.
    #[serde(default)]
    pub prefixes: Prefixes,
}

#[derive(Debug)]
struct Diagnostic(String);
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for Diagnostic {}

struct LoadedModel {
    manifest: ModelManifest,
    tokenizer: Tokenizer,
    session: Session,
    identity: ResolvedModelIdentity,
}

/// Single-model ONNX engine with explicit lifecycle operations.
pub struct OnnxEmbeddingEngine {
    loaded: Mutex<Option<LoadedModel>>,
}
impl Default for OnnxEmbeddingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxEmbeddingEngine {
    /// Creates an unloaded engine.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            loaded: Mutex::new(None),
        }
    }

    /// Loads a local JSON manifest, replacing the current model only after validation succeeds.
    ///
    /// # Errors
    /// Returns stable invalid-manifest or unavailable-model errors with private diagnostics.
    pub fn load(&self, manifest_path: &Path) -> Result<ResolvedModelIdentity, EngineFailure> {
        let candidate = Self::load_candidate(manifest_path)?;
        let identity = candidate.identity.clone();
        *self.state()? = Some(candidate);
        Ok(identity)
    }

    fn load_candidate(manifest_path: &Path) -> Result<LoadedModel, EngineFailure> {
        if manifest_path.extension().and_then(|v| v.to_str()) != Some("json") {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        let manifest_bytes = fs::read(manifest_path)
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let manifest: ModelManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| EngineFailure::with_source(ErrorCode::InvalidRequest, e))?;
        validate_manifest(&manifest)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model_path = safe_artifact_path(root, &manifest.model_file, "onnx")?;
        let tokenizer_path = safe_artifact_path(root, &manifest.tokenizer_file, "json")?;
        let model_bytes = fs::read(&model_path)
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let tokenizer_bytes = fs::read(&tokenizer_path)
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| private_error(ErrorCode::ModelUnavailable, "tokenizer parse", e))?;
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(&model_path))
            .map_err(|e| private_error(ErrorCode::ModelUnavailable, "ONNX load", e))?;
        validate_graph_contract(&manifest, &session)?;
        let identity = ResolvedModelIdentity::new(
            manifest.canonical_id.clone(),
            manifest.revision.clone(),
            format!("{ENGINE_NAME}@{}", env!("CARGO_PKG_VERSION")),
            artifact_fingerprint(&manifest_bytes, &model_bytes, &tokenizer_bytes),
        )?;
        Ok(LoadedModel {
            manifest,
            tokenizer,
            session,
            identity,
        })
    }

    fn state(&self) -> Result<std::sync::MutexGuard<'_, Option<LoadedModel>>, EngineFailure> {
        self.loaded.lock().map_err(|_| {
            EngineFailure::with_source(
                ErrorCode::Internal,
                Diagnostic("model state lock is poisoned".into()),
            )
        })
    }

    /// Returns whether a model is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.lock().is_ok_and(|g| g.is_some())
    }

    /// Runs one private synthetic request to initialize runtime state.
    ///
    /// # Errors
    /// Returns the same stable errors as embedding.
    pub fn warm(&self) -> Result<(), EngineFailure> {
        let mut guard = self.state()?;
        let model = guard
            .as_mut()
            .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        let input = model
            .prefix_for(EmbeddingTask::Document)
            .unwrap_or("warmup")
            .to_owned();
        let encoding = model.encode(&input, Truncation::Truncate)?;
        drop(model.run(&[encoding], None)?);
        Ok(())
    }

    /// Unloads the current model; repeated calls are safe.
    ///
    /// # Errors
    /// Returns an internal failure only if lifecycle state was poisoned.
    pub fn unload(&self) -> Result<(), EngineFailure> {
        *self.state()? = None;
        Ok(())
    }

    /// Embeds with explicit task, truncation, and dimension controls.
    ///
    /// # Errors
    /// Returns stable privacy-safe request, model, control, or inference errors.
    pub fn embed_with_options(
        &self,
        requested: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        control: &ExecutionControl,
        options: EmbedOptions,
    ) -> Result<EmbeddingOutput, EngineFailure> {
        control.ensure_active()?;
        let mut guard = self.state()?;
        let model = guard
            .as_mut()
            .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        if requested.as_str() != model.manifest.canonical_id {
            return Err(EngineFailure::public(ErrorCode::ModelUnavailable));
        }
        validate_dimensions(&model.manifest, options.dimensions)?;
        let encodings = batch
            .inputs()
            .iter()
            .map(|text| {
                let prepared = apply_prefix(text, model.prefix_for(options.task));
                model.encode(&prepared, options.truncation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        control.ensure_active()?;
        let vectors = model.run(&encodings, options.dimensions)?;
        control.ensure_active()?;
        Ok(EmbeddingOutput {
            vectors,
            model: model.identity.clone(),
        })
    }
}

impl EmbeddingEngine for OnnxEmbeddingEngine {
    fn embed(
        &self,
        requested: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        control: &ExecutionControl,
    ) -> Result<EmbeddingOutput, EngineFailure> {
        self.embed_with_options(requested, batch, control, EmbedOptions::default())
    }
}

impl LoadedModel {
    fn prefix_for(&self, task: EmbeddingTask) -> Option<&str> {
        match task {
            EmbeddingTask::Query => self.manifest.prefixes.query.as_deref(),
            EmbeddingTask::Document => self.manifest.prefixes.document.as_deref(),
        }
        .filter(|p| !p.is_empty())
    }
    fn encode(&self, text: &str, truncation: Truncation) -> Result<Encoding, EngineFailure> {
        let mut encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| private_error(ErrorCode::InvalidRequest, "tokenization", e))?;
        if encoding.len() > self.manifest.max_tokens {
            match truncation {
                Truncation::Reject => return Err(EngineFailure::public(ErrorCode::InvalidRequest)),
                Truncation::Truncate => encoding.truncate(
                    self.manifest.max_tokens,
                    0,
                    tokenizers::TruncationDirection::Right,
                ),
            }
        }
        Ok(encoding)
    }

    fn run(
        &mut self,
        encodings: &[Encoding],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, EngineFailure> {
        let batch = encodings.len();
        let sequence = encodings
            .iter()
            .map(Encoding::len)
            .max()
            .unwrap_or(0)
            .max(1);
        let capacity = batch
            .checked_mul(sequence)
            .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?;
        let mut ids = vec![i64::from(self.manifest.pad_token_id); capacity];
        let mut masks = vec![0_i64; capacity];
        let mut types = vec![0_i64; capacity];
        for (row, encoding) in encodings.iter().enumerate() {
            for (column, id) in encoding.get_ids().iter().enumerate() {
                let i = row * sequence + column;
                ids[i] = i64::from(*id);
                masks[i] = i64::from(encoding.get_attention_mask()[column]);
                types[i] = i64::from(encoding.get_type_ids()[column]);
            }
        }
        let shape = [batch, sequence];
        let ids = Tensor::from_array((shape, ids)).map_err(inference_error)?;
        let mask_tensor = Tensor::from_array((shape, masks.clone())).map_err(inference_error)?;
        let types = Tensor::from_array((shape, types)).map_err(inference_error)?;
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(3);
        inputs.push((
            Cow::Borrowed(self.manifest.inputs.input_ids.as_str()),
            ids.into(),
        ));
        if let Some(name) = self.manifest.inputs.attention_mask.as_deref() {
            inputs.push((Cow::Borrowed(name), mask_tensor.into()));
        }
        if let Some(name) = self.manifest.inputs.token_type_ids.as_deref() {
            inputs.push((Cow::Borrowed(name), types.into()));
        }
        let outputs = self.session.run(inputs).map_err(inference_error)?;
        let output = outputs
            .get(self.manifest.output.as_str())
            .ok_or_else(|| EngineFailure::public(ErrorCode::InferenceFailed))?
            .try_extract_array::<f32>()
            .map_err(inference_error)?;
        let shape = output.shape();
        let mut vectors = match shape {
            [rows, hidden] if *rows == batch => (0..batch)
                .map(|r| (0..*hidden).map(|c| output[[r, c]]).collect())
                .collect(),
            [rows, tokens, hidden] if *rows == batch && *tokens == sequence => (0..batch)
                .map(|r| {
                    pool_tokens(
                        &output,
                        r,
                        *hidden,
                        &masks[r * sequence..(r + 1) * sequence],
                        self.manifest.pooling,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(EngineFailure::public(ErrorCode::InferenceFailed)),
        };
        for vector in &mut vectors {
            if !vector.iter().all(|v| v.is_finite()) {
                return Err(EngineFailure::public(ErrorCode::InferenceFailed));
            }
            // Matryoshka truncation precedes normalization by contract.
            if let Some(d) = dimensions {
                if d > vector.len() {
                    return Err(EngineFailure::public(ErrorCode::InferenceFailed));
                }
                vector.truncate(d);
            }
            if self.manifest.normalize {
                l2_normalize(vector)?;
            }
        }
        Ok(vectors)
    }
}

fn apply_prefix<'a>(input: &'a str, prefix: Option<&str>) -> Cow<'a, str> {
    match prefix {
        None => Cow::Borrowed(input),
        Some(p) if input.starts_with(p) => Cow::Borrowed(input),
        Some(p) => Cow::Owned(format!("{p}{input}")),
    }
}
fn validate_manifest(m: &ModelManifest) -> Result<(), EngineFailure> {
    let fields = m.schema_version == 1
        && !m.canonical_id.trim().is_empty()
        && !m.revision.trim().is_empty()
        && !m.inputs.input_ids.trim().is_empty()
        && m.inputs
            .attention_mask
            .as_ref()
            .is_none_or(|v| !v.trim().is_empty())
        && m.inputs
            .token_type_ids
            .as_ref()
            .is_none_or(|v| !v.trim().is_empty())
        && !m.output.trim().is_empty()
        && m.max_tokens > 0;
    let dimensions = m
        .matryoshka_dimensions
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if !fields || dimensions.len() != m.matryoshka_dimensions.len() || dimensions.contains(&0) {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    Ok(())
}
fn validate_graph_contract(
    manifest: &ModelManifest,
    session: &Session,
) -> Result<(), EngineFailure> {
    let available_inputs = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<BTreeSet<_>>();
    let required_inputs = [
        Some(manifest.inputs.input_ids.as_str()),
        manifest.inputs.attention_mask.as_deref(),
        manifest.inputs.token_type_ids.as_deref(),
    ];
    let has_inputs = required_inputs
        .into_iter()
        .flatten()
        .all(|name| available_inputs.contains(name));
    let has_output = session
        .outputs
        .iter()
        .any(|output| output.name == manifest.output);
    if !has_inputs || !has_output {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    Ok(())
}
fn validate_dimensions(m: &ModelManifest, requested: Option<usize>) -> Result<(), EngineFailure> {
    if requested.is_some_and(|d| !m.matryoshka_dimensions.contains(&d)) {
        Err(EngineFailure::public(ErrorCode::InvalidRequest))
    } else {
        Ok(())
    }
}
fn safe_artifact_path(
    root: &Path,
    relative: &Path,
    extension: &str,
) -> Result<PathBuf, EngineFailure> {
    use std::path::Component;
    if relative.is_absolute()
        || relative.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || relative.extension().and_then(|v| v.to_str()) != Some(extension)
    {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    let base = root
        .canonicalize()
        .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
    let path = root
        .join(relative)
        .canonicalize()
        .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
    if !path.starts_with(base) {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    Ok(path)
}
fn artifact_fingerprint(manifest: &[u8], model: &[u8], tokenizer: &[u8]) -> String {
    let mut hash = Sha256::new();
    for bytes in [manifest, model, tokenizer] {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("sha256:{:x}", hash.finalize())
}
fn pool_tokens(
    output: &ndarray::ArrayViewD<'_, f32>,
    row: usize,
    hidden: usize,
    mask: &[i64],
    pooling: Pooling,
) -> Result<Vec<f32>, EngineFailure> {
    let indices = mask
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v != 0).then_some(i))
        .collect::<Vec<_>>();
    if indices.is_empty() {
        return Err(EngineFailure::public(ErrorCode::InferenceFailed));
    }
    match pooling {
        Pooling::Cls => Ok((0..hidden).map(|d| output[[row, indices[0], d]]).collect()),
        Pooling::LastToken => {
            let token = indices[indices.len() - 1];
            Ok((0..hidden).map(|d| output[[row, token, d]]).collect())
        }
        Pooling::Mean => Ok((0..hidden)
            .map(|d| {
                let (sum, count) = indices.iter().fold((0.0_f32, 0.0_f32), |(sum, count), t| {
                    (sum + output[[row, *t, d]], count + 1.0)
                });
                sum / count
            })
            .collect()),
    }
}
fn l2_normalize(vector: &mut [f32]) -> Result<(), EngineFailure> {
    let n = vector.iter().map(|v| v * v).sum::<f32>();
    if !n.is_finite() || n <= f32::EPSILON {
        return Err(EngineFailure::public(ErrorCode::InferenceFailed));
    }
    let n = n.sqrt();
    for v in vector {
        *v /= n;
    }
    Ok(())
}
fn private_error(code: ErrorCode, operation: &str, error: impl fmt::Display) -> EngineFailure {
    EngineFailure::with_source(code, Diagnostic(format!("{operation} failed: {error}")))
}
fn inference_error(error: impl fmt::Display) -> EngineFailure {
    private_error(ErrorCode::InferenceFailed, "ONNX inference", error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_exactly_once() {
        assert_eq!(apply_prefix("hello", Some("query: ")), "query: hello");
        assert_eq!(
            apply_prefix("query: hello", Some("query: ")),
            "query: hello"
        );
    }
    #[test]
    fn normalization_rejects_zero_and_nan() {
        assert!(l2_normalize(&mut [0.0, 0.0]).is_err());
        assert!(l2_normalize(&mut [f32::NAN, 1.0]).is_err());
        let mut v = [3.0, 4.0];
        assert!(l2_normalize(&mut v).is_ok());
        assert!((v[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn pooling_obeys_non_contiguous_attention_masks() -> Result<(), EngineFailure> {
        let output = ndarray::Array::from_shape_vec(
            (1, 4, 2),
            vec![1.0, 10.0, 99.0, 99.0, 3.0, 30.0, 88.0, 88.0],
        )
        .map_err(|_| EngineFailure::public(ErrorCode::Internal))?
        .into_dyn();
        let view = output.view();
        assert_eq!(
            pool_tokens(&view, 0, 2, &[1, 0, 1, 0], Pooling::Mean)?,
            vec![2.0, 20.0]
        );
        assert_eq!(
            pool_tokens(&view, 0, 2, &[1, 0, 1, 0], Pooling::Cls)?,
            vec![1.0, 10.0]
        );
        assert_eq!(
            pool_tokens(&view, 0, 2, &[1, 0, 1, 0], Pooling::LastToken)?,
            vec![3.0, 30.0]
        );
        assert!(pool_tokens(&view, 0, 2, &[0, 0, 0, 0], Pooling::Mean).is_err());
        Ok(())
    }
}

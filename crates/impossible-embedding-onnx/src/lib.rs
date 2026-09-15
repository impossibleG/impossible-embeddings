//! Manifest-driven, in-process ONNX Runtime embedding engine.
//!
//! Models load only from local ONNX and tokenizer JSON files. This adapter never evaluates
//! repository code, Python, pickle, or other executable model content.

pub use impossible_embedding_core::{EmbedOptions, EmbeddingTask, Truncation};
use impossible_embedding_core::{
    EmbeddingBatch, EmbeddingEngine, EmbeddingOutput, EngineFailure, ErrorCode, ExecutionControl,
    RequestedModel, ResolvedModelIdentity,
};
use impossible_models::{Manifest, OnnxInputNames, Pooling, RuntimeMetadata, VerifiedModel};
use ort::{
    session::{Session, SessionInputValue},
    tensor::TensorElementType,
    value::Tensor,
    value::ValueType,
};
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

#[derive(Debug)]
struct Diagnostic(String);
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for Diagnostic {}

struct LoadedModel {
    manifest: Manifest,
    contract: OnnxContract,
    tokenizer: Tokenizer,
    session: Session,
    identity: ResolvedModelIdentity,
    _lease: VerifiedModel,
}

#[derive(Debug, Clone)]
struct OnnxContract {
    inputs: OnnxInputNames,
    output: String,
    pad_token_id: u32,
    normalize: bool,
}

/// Single-model ONNX engine with explicit lifecycle operations.
pub struct OnnxEmbeddingEngine {
    state: Mutex<EngineState>,
}

#[derive(Default)]
struct EngineState {
    epoch: u64,
    loaded: Option<LoadedModel>,
}

impl EngineState {
    fn reserve_transition(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1);
        self.epoch
    }

    const fn accepts(&self, epoch: u64) -> bool {
        self.epoch == epoch
    }
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
            state: Mutex::new(EngineState {
                epoch: 0,
                loaded: None,
            }),
        }
    }

    /// Loads a store-verified model, replacing the current model only after adapter validation.
    ///
    /// # Errors
    /// Returns stable invalid-manifest or unavailable-model errors with private diagnostics.
    pub fn load(&self, verified: &VerifiedModel) -> Result<ResolvedModelIdentity, EngineFailure> {
        let epoch = {
            let mut state = self.state()?;
            state.reserve_transition()
        };
        let candidate = Self::load_candidate(verified)?;
        let identity = candidate.identity.clone();
        let mut state = self.state()?;
        if !state.accepts(epoch) {
            return Err(EngineFailure::public(ErrorCode::ModelUnavailable));
        }
        state.loaded = Some(candidate);
        Ok(identity)
    }

    fn load_candidate(verified: &VerifiedModel) -> Result<LoadedModel, EngineFailure> {
        verified
            .revalidate_integrity()
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let manifest = verified.manifest().clone();
        let (model_file, tokenizer_file, contract) = onnx_contract(&manifest)?;
        let model_path = safe_artifact_path(verified, &model_file, "onnx")?;
        let tokenizer_path = safe_artifact_path(verified, &tokenizer_file, "json")?;
        let tokenizer_bytes = fs::read(&tokenizer_path)
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| private_error(ErrorCode::ModelUnavailable, "tokenizer parse", e))?;
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(&model_path))
            .map_err(|e| private_error(ErrorCode::ModelUnavailable, "ONNX load", e))?;
        validate_graph_contract(&contract, &session, manifest.dimensions.native)?;
        verified
            .revalidate_integrity()
            .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))?;
        let identity = ResolvedModelIdentity::new(
            manifest.canonical_id.clone(),
            manifest.revision.clone(),
            format!("{ENGINE_NAME}@{}", env!("CARGO_PKG_VERSION")),
            verified.artifact_fingerprint(),
            manifest
                .semantic_fingerprint()
                .map_err(|e| EngineFailure::with_source(ErrorCode::Internal, e))?,
        )?;
        Ok(LoadedModel {
            manifest,
            contract,
            tokenizer,
            session,
            identity,
            _lease: verified.clone(),
        })
    }

    fn state(&self) -> Result<std::sync::MutexGuard<'_, EngineState>, EngineFailure> {
        self.state.lock().map_err(|_| {
            EngineFailure::with_source(
                ErrorCode::Internal,
                Diagnostic("model state lock is poisoned".into()),
            )
        })
    }

    /// Returns whether a model is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.state.lock().is_ok_and(|state| state.loaded.is_some())
    }

    /// Runs one private synthetic request to initialize runtime state.
    ///
    /// # Errors
    /// Returns the same stable errors as embedding.
    pub fn warm(&self) -> Result<(), EngineFailure> {
        let mut guard = self.state()?;
        let model = guard
            .loaded
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
        let mut state = self.state()?;
        state.reserve_transition();
        state.loaded = None;
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
            .loaded
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
            EmbeddingTask::Query => Some(self.manifest.prefixes.query.as_str()),
            EmbeddingTask::Document => Some(self.manifest.prefixes.document.as_str()),
        }
        .filter(|p| !p.is_empty())
    }
    fn encode(&self, text: &str, truncation: Truncation) -> Result<Encoding, EngineFailure> {
        let mut encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| private_error(ErrorCode::InvalidRequest, "tokenization", e))?;
        let max_tokens = usize::try_from(self.manifest.tokenizer.max_tokens)
            .map_err(|error| EngineFailure::with_source(ErrorCode::Internal, error))?;
        if encoding.len() > max_tokens {
            match truncation {
                Truncation::Reject => return Err(EngineFailure::public(ErrorCode::InvalidRequest)),
                Truncation::Truncate => {
                    encoding.truncate(max_tokens, 0, tokenizers::TruncationDirection::Right);
                }
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
        let mut ids = vec![i64::from(self.contract.pad_token_id); capacity];
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
            Cow::Borrowed(self.contract.inputs.input_ids.as_str()),
            ids.into(),
        ));
        if let Some(name) = self.contract.inputs.attention_mask.as_deref() {
            inputs.push((Cow::Borrowed(name), mask_tensor.into()));
        }
        if let Some(name) = self.contract.inputs.token_type_ids.as_deref() {
            inputs.push((Cow::Borrowed(name), types.into()));
        }
        let outputs = self.session.run(inputs).map_err(inference_error)?;
        let output = outputs
            .get(self.contract.output.as_str())
            .ok_or_else(|| EngineFailure::public(ErrorCode::InferenceFailed))?
            .try_extract_array::<f32>()
            .map_err(inference_error)?;
        let shape = output.shape();
        let native = usize::try_from(self.manifest.dimensions.native)
            .map_err(|_| EngineFailure::public(ErrorCode::InferenceFailed))?;
        if !valid_runtime_output_shape(shape, batch, sequence, native) {
            return Err(EngineFailure::public(ErrorCode::InferenceFailed));
        }
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
            if self.contract.normalize {
                l2_normalize(vector)?;
            }
        }
        Ok(vectors)
    }
}

fn valid_runtime_output_shape(
    shape: &[usize],
    batch: usize,
    sequence: usize,
    native: usize,
) -> bool {
    matches!(shape, [rows, hidden] if *rows == batch && *hidden == native)
        || matches!(shape, [rows, tokens, hidden]
            if *rows == batch && *tokens == sequence && *hidden == native)
}

fn apply_prefix<'a>(input: &'a str, prefix: Option<&str>) -> Cow<'a, str> {
    match prefix {
        None => Cow::Borrowed(input),
        Some(p) if input.starts_with(p) => Cow::Borrowed(input),
        Some(p) => Cow::Owned(format!("{p}{input}")),
    }
}
fn onnx_contract(manifest: &Manifest) -> Result<(String, String, OnnxContract), EngineFailure> {
    let RuntimeMetadata::Onnx {
        model_file,
        tokenizer_file,
        inputs,
        output,
        pad_token_id,
        normalize,
    } = &manifest.runtime
    else {
        return Err(EngineFailure::public(ErrorCode::ModelUnavailable));
    };
    Ok((
        model_file.clone(),
        tokenizer_file.clone(),
        OnnxContract {
            inputs: inputs.clone(),
            output: output.clone(),
            pad_token_id: *pad_token_id,
            normalize: *normalize,
        },
    ))
}
fn validate_graph_contract(
    contract: &OnnxContract,
    session: &Session,
    native_dimensions: u32,
) -> Result<(), EngineFailure> {
    let available_inputs = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<BTreeSet<_>>();
    let required_inputs = [
        Some(contract.inputs.input_ids.as_str()),
        contract.inputs.attention_mask.as_deref(),
        contract.inputs.token_type_ids.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect::<BTreeSet<_>>();
    let has_exact_inputs = available_inputs.len() == session.inputs.len()
        && required_inputs.len() == session.inputs.len()
        && available_inputs == required_inputs;
    let valid_inputs = has_exact_inputs
        && session.inputs.iter().all(|input| {
            matches!(
                &input.input_type,
                ValueType::Tensor { ty, shape, .. }
                    if *ty == TensorElementType::Int64 && shape.len() == 2
            )
        });
    let output_count = session
        .outputs
        .iter()
        .filter(|output| output.name == contract.output)
        .count();
    let output = session
        .outputs
        .iter()
        .find(|output| output.name == contract.output);
    let valid_output = output_count == 1
        && output.is_some_and(|output| match &output.output_type {
            ValueType::Tensor { ty, shape, .. } if *ty == TensorElementType::Float32 => {
                matches!(shape.len(), 2 | 3)
                    && shape.last().is_some_and(|hidden| {
                        *hidden < 0 || u32::try_from(*hidden).ok() == Some(native_dimensions)
                    })
            }
            _ => false,
        });
    if !valid_inputs || !valid_output {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    Ok(())
}
fn validate_dimensions(m: &Manifest, requested: Option<usize>) -> Result<(), EngineFailure> {
    if requested.is_some_and(|dimension| {
        let requested = u32::try_from(dimension).unwrap_or(u32::MAX);
        requested != m.dimensions.native && !m.dimensions.matryoshka.contains(&requested)
    }) {
        Err(EngineFailure::public(ErrorCode::InvalidRequest))
    } else {
        Ok(())
    }
}
fn safe_artifact_path(
    verified: &VerifiedModel,
    relative: &str,
    extension: &str,
) -> Result<PathBuf, EngineFailure> {
    if Path::new(relative).extension().and_then(|v| v.to_str()) != Some(extension) {
        return Err(EngineFailure::public(ErrorCode::InvalidRequest));
    }
    verified
        .artifact_path(relative)
        .map_err(|e| EngineFailure::with_source(ErrorCode::ModelUnavailable, e))
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

    #[test]
    fn output_shapes_require_declared_native_width_for_2d_and_3d() {
        assert!(valid_runtime_output_shape(&[2, 384], 2, 7, 384));
        assert!(!valid_runtime_output_shape(&[2, 383], 2, 7, 384));
        assert!(valid_runtime_output_shape(&[2, 7, 384], 2, 7, 384));
        assert!(!valid_runtime_output_shape(&[2, 7, 383], 2, 7, 384));
        assert!(!valid_runtime_output_shape(&[2, 6, 384], 2, 7, 384));
    }

    #[test]
    fn native_dimension_is_always_an_explicit_valid_request() -> Result<(), EngineFailure> {
        let mut manifest = impossible_models::curated_manifests()
            .unwrap_or_default()
            .into_iter()
            .next()
            .ok_or_else(|| EngineFailure::public(ErrorCode::Internal))?;
        manifest.dimensions.matryoshka.clear();
        let native = usize::try_from(manifest.dimensions.native).unwrap_or_default();
        assert!(validate_dimensions(&manifest, Some(native)).is_ok());
        assert!(validate_dimensions(&manifest, Some(native.saturating_add(1))).is_err());
        Ok(())
    }

    #[test]
    fn lifecycle_epochs_reject_stale_concurrent_loads_and_post_unload_publish() {
        let mut state = EngineState::default();
        let first = state.reserve_transition();
        let second = state.reserve_transition();
        assert!(!state.accepts(first));
        assert!(state.accepts(second));
        state.reserve_transition();
        assert!(!state.accepts(second));
    }
}

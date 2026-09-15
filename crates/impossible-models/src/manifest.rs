use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use url::Url;

use crate::{Error, Result};

/// Current on-disk manifest schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// Immutable description of one model revision and all files required to load it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Schema version used to decode this document.
    pub schema_version: u32,
    /// Stable registry-qualified model identifier.
    pub canonical_id: String,
    /// Immutable upstream commit identifier.
    pub revision: String,
    /// License asserted by the upstream model repository.
    pub license: License,
    /// Whether output semantics have been independently validated by this project.
    pub semantic_verification: SemanticVerification,
    /// Tokenizer contract.
    pub tokenizer: TokenizerMetadata,
    /// Pooling algorithm applied to token representations.
    pub pooling: Pooling,
    /// Required input prefixes.
    pub prefixes: Prefixes,
    /// Output dimensions.
    pub dimensions: Dimensions,
    /// Tensor/runtime compatibility metadata.
    pub tensors: TensorMetadata,
    /// Runtime-specific loading contract, if this identity is directly executable.
    #[serde(default)]
    pub runtime: RuntimeMetadata,
    /// Immutable artifacts required by this manifest.
    pub artifacts: Vec<Artifact>,
}

/// SPDX license metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct License {
    /// SPDX identifier, such as `MIT` or `Apache-2.0`.
    pub spdx: String,
    /// Upstream HTTPS page from which the assertion was taken.
    pub source_url: String,
}

/// Honest semantic validation state, distinct from artifact integrity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum SemanticVerification {
    /// Golden vectors and runtime behavior were independently checked.
    Verified {
        /// Stable reference to the verification evidence.
        evidence: String,
    },
    /// Integrity is pinned, but embedding semantics are not yet certified.
    Unverified {
        /// Clear reason the model is not yet safe to advertise as loadable.
        reason: String,
    },
}

/// Tokenizer type and input bound.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerMetadata {
    /// Stable tokenizer family understood by an adapter.
    pub kind: String,
    /// Maximum tokenized sequence length.
    pub max_tokens: u32,
    /// Whether input normalization lowercases text.
    pub lowercase: bool,
}

/// Supported pooling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Use the classification token representation.
    Cls,
    /// Mean-pool non-padding token representations.
    Mean,
    /// Use the final non-padding token representation.
    LastToken,
}

/// Query/document prefix contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prefixes {
    /// Required prefix for query inputs.
    pub query: String,
    /// Required prefix for document inputs.
    pub document: String,
}

/// Native and optional truncatable embedding dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dimensions {
    /// Native output width.
    pub native: u32,
    /// Semantically verified truncation widths, if supported.
    pub matryoshka: Vec<u32>,
}

/// Serialized tensor format required by the future runtime adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMetadata {
    /// Serialization format, such as `safetensors`.
    pub format: String,
    /// Stored numeric type.
    pub dtype: String,
    /// Architecture identifier required by a compatible runtime.
    pub architecture: String,
}

/// Runtime-specific loading metadata kept in the canonical model manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeMetadata {
    /// Catalog metadata only; no in-process adapter contract is declared.
    #[default]
    CatalogOnly,
    /// ONNX Runtime graph and tokenizer contract.
    Onnx {
        /// ONNX artifact path from [`Manifest::artifacts`].
        model_file: String,
        /// Hugging Face tokenizer JSON artifact path from [`Manifest::artifacts`].
        tokenizer_file: String,
        /// ONNX graph input names.
        inputs: OnnxInputNames,
        /// Selected ONNX graph output name.
        output: String,
        /// Token id used for request-time batch padding.
        #[serde(default)]
        pad_token_id: u32,
        /// Whether final vectors are L2-normalized.
        normalize: bool,
    },
}

/// Names of inputs in an ONNX embedding graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnnxInputNames {
    /// Token ids input.
    pub input_ids: String,
    /// Optional attention-mask input.
    pub attention_mask: Option<String>,
    /// Optional segment/token-type input.
    pub token_type_ids: Option<String>,
}

/// One content-addressed file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// Safe relative location inside the installed identity.
    pub path: String,
    /// Immutable download URL.
    pub url: String,
    /// Lowercase SHA-256 digest.
    pub sha256: String,
    /// Exact byte length.
    pub size: u64,
}

impl Manifest {
    /// Parses and validates a versioned manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON decoding or any schema invariant fails.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Returns the canonical manifest JSON used for local identity records.
    ///
    /// # Errors
    ///
    /// Returns an error if validation or serialization fails.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Validates all identity, path, URL, size, hash, and tensor invariants.
    ///
    /// # Errors
    ///
    /// Returns an error describing the first invalid invariant.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::Invalid("unsupported schema_version".into()));
        }
        validate_id(&self.canonical_id)?;
        if self.revision.len() != 40
            || !self
                .revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(Error::Invalid(
                "revision must be a 40-character commit hash".into(),
            ));
        }
        if self.license.spdx.trim().is_empty()
            || Url::parse(&self.license.source_url).map_or(true, |url| url.scheme() != "https")
        {
            return Err(Error::Invalid(
                "license metadata must contain SPDX id and HTTPS source".into(),
            ));
        }
        if self.tokenizer.kind.trim().is_empty() || self.tokenizer.max_tokens == 0 {
            return Err(Error::Invalid("tokenizer metadata is incomplete".into()));
        }
        if self.dimensions.native == 0
            || self
                .dimensions
                .matryoshka
                .iter()
                .any(|value| *value == 0 || *value > self.dimensions.native)
        {
            return Err(Error::Invalid("invalid embedding dimensions".into()));
        }
        if self.tensors.format.trim().is_empty()
            || self.tensors.dtype.trim().is_empty()
            || self.tensors.architecture.trim().is_empty()
        {
            return Err(Error::Invalid("tensor metadata is incomplete".into()));
        }
        validate_runtime(&self.runtime)?;
        match &self.semantic_verification {
            SemanticVerification::Verified { evidence } if evidence.trim().is_empty() => {
                return Err(Error::Invalid(
                    "semantic verification evidence is empty".into(),
                ));
            }
            SemanticVerification::Unverified { reason } if reason.trim().is_empty() => {
                return Err(Error::Invalid(
                    "semantic verification reason is empty".into(),
                ));
            }
            _ => {}
        }
        if self.artifacts.is_empty() {
            return Err(Error::Invalid("manifest has no artifacts".into()));
        }
        let mut paths = HashSet::new();
        for artifact in &self.artifacts {
            validate_relative_path(&artifact.path)?;
            if !paths.insert(&artifact.path) {
                return Err(Error::Invalid("artifact paths must be unique".into()));
            }
            let url = Url::parse(&artifact.url)
                .map_err(|_| Error::Invalid("artifact URL is invalid".into()))?;
            let loopback_test_url = url.scheme() == "http"
                && url
                    .host_str()
                    .is_some_and(|host| matches!(host, "127.0.0.1" | "::1" | "localhost"));
            if (url.scheme() != "https" && !loopback_test_url)
                || url.host_str().is_none()
                || url.username() != ""
                || url.password().is_some()
            {
                return Err(Error::Invalid("artifact URL must use HTTPS (HTTP is allowed only for loopback tests) and contain no credentials".into()));
            }
            if artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || artifact.size == 0
            {
                return Err(Error::Invalid(
                    "artifact requires SHA-256 and non-zero size".into(),
                ));
            }
        }
        validate_runtime_artifacts(&self.runtime, &self.artifacts)?;
        Ok(())
    }

    /// Whether this exact model is safe for a runtime to advertise as loadable.
    #[must_use]
    pub const fn semantics_verified(&self) -> bool {
        matches!(
            self.semantic_verification,
            SemanticVerification::Verified { .. }
        )
    }
}

fn validate_runtime(runtime: &RuntimeMetadata) -> Result<()> {
    let RuntimeMetadata::Onnx {
        model_file,
        tokenizer_file,
        inputs,
        output,
        ..
    } = runtime
    else {
        return Ok(());
    };
    validate_relative_path(model_file)?;
    validate_relative_path(tokenizer_file)?;
    if model_file == tokenizer_file
        || inputs.input_ids.trim().is_empty()
        || output.trim().is_empty()
        || inputs
            .attention_mask
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        || inputs
            .token_type_ids
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
    {
        return Err(Error::Invalid("ONNX runtime metadata is incomplete".into()));
    }
    Ok(())
}

fn validate_runtime_artifacts(runtime: &RuntimeMetadata, artifacts: &[Artifact]) -> Result<()> {
    let RuntimeMetadata::Onnx {
        model_file,
        tokenizer_file,
        ..
    } = runtime
    else {
        return Ok(());
    };
    for required in [model_file, tokenizer_file] {
        if !artifacts.iter().any(|artifact| &artifact.path == required) {
            return Err(Error::Invalid(
                "ONNX runtime file is not declared as an artifact".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_relative_path(value: &str) -> Result<()> {
    let path = std::path::Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(Error::Invalid(format!("unsafe artifact path: {value}")));
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 200
        || value.starts_with('/')
        || value.ends_with('/')
        || value.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
    {
        return Err(Error::Invalid(
            "canonical_id contains unsafe characters".into(),
        ));
    }
    Ok(())
}

/// Returns the three manifests curated by this release.
///
/// # Errors
///
/// Returns an error if a committed manifest no longer satisfies the schema.
pub fn curated_manifests() -> Result<Vec<Manifest>> {
    [
        include_bytes!("../manifests/bge-small-en.json").as_slice(),
        include_bytes!("../manifests/multilingual-e5-small.json").as_slice(),
        include_bytes!("../manifests/nomic-embed-text-v1.5.json").as_slice(),
    ]
    .into_iter()
    .map(Manifest::from_json)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_catalog_is_exact_and_valid() -> Result<()> {
        let manifests = curated_manifests()?;
        assert_eq!(manifests.len(), 3);
        assert_eq!(
            manifests
                .iter()
                .map(|m| m.canonical_id.as_str())
                .collect::<Vec<_>>(),
            [
                "BAAI/bge-small-en-v1.5",
                "intfloat/multilingual-e5-small",
                "nomic-ai/nomic-embed-text-v1.5"
            ]
        );
        assert!(
            manifests
                .iter()
                .all(|manifest| !manifest.semantics_verified())
        );
        Ok(())
    }

    #[test]
    fn rejects_traversal_and_unknown_fields() {
        let mut json: serde_json::Value =
            serde_json::from_slice(include_bytes!("../manifests/bge-small-en.json"))
                .unwrap_or_default();
        json["artifacts"][0]["path"] = "../outside".into();
        assert!(Manifest::from_json(&serde_json::to_vec(&json).unwrap_or_default()).is_err());
        json["artifacts"][0]["path"] = "model.safetensors".into();
        json["surprise"] = true.into();
        assert!(Manifest::from_json(&serde_json::to_vec(&json).unwrap_or_default()).is_err());
    }
}

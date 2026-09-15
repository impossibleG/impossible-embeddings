//! Secure, transport-independent model catalog and artifact lifecycle.
//!
//! The crate never discovers arbitrary paths, executes model repository code, or downloads from
//! an origin that the caller has not explicitly allowlisted.

mod install;
mod manifest;
mod store;

pub use install::{CancelToken, InstallOptions, Installer};
pub use manifest::{
    Artifact, Dimensions, License, Manifest, Pooling, Prefixes, SemanticVerification,
    TensorMetadata, TokenizerMetadata, curated_manifests,
};
pub use store::{CacheLayout, DiscoveryRoot, ModelStatus, ModelStore};

use std::{fmt, io};

/// Errors returned by catalog and model lifecycle operations.
#[derive(Debug)]
pub enum Error {
    /// A manifest or caller-controlled value failed validation.
    Invalid(String),
    /// An operation was cancelled before promotion.
    Cancelled,
    /// Network access was requested while offline mode was enabled.
    Offline,
    /// A URL did not match an explicitly allowed HTTPS origin.
    OriginNotAllowed(String),
    /// A response exceeded an artifact's declared size or configured limit.
    SizeLimit {
        /// Declared or configured upper bound.
        expected: u64,
        /// Bytes observed or declared by the response.
        actual: u64,
    },
    /// An artifact's digest differs from its immutable manifest.
    HashMismatch {
        /// Immutable digest in the manifest.
        expected: String,
        /// Digest calculated while streaming.
        actual: String,
    },
    /// A filesystem operation failed.
    Io(io::Error),
    /// An HTTP operation failed.
    Http(reqwest::Error),
    /// Manifest JSON could not be decoded.
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid model data: {message}"),
            Self::Cancelled => formatter.write_str("operation cancelled"),
            Self::Offline => formatter.write_str("network access is disabled in offline mode"),
            Self::OriginNotAllowed(origin) => {
                write!(formatter, "download origin is not allowed: {origin}")
            }
            Self::SizeLimit { expected, actual } => write!(
                formatter,
                "artifact size limit exceeded: expected at most {expected} bytes, received {actual}"
            ),
            Self::HashMismatch { expected, actual } => write!(
                formatter,
                "artifact hash mismatch: expected {expected}, received {actual}"
            ),
            Self::Io(error) => write!(formatter, "filesystem error: {error}"),
            Self::Http(error) => write!(formatter, "HTTP error: {error}"),
            Self::Json(error) => write!(formatter, "manifest JSON error: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Result type for model lifecycle operations.
pub type Result<T> = std::result::Result<T, Error>;

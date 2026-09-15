//! Secure, transport-independent model catalog and artifact lifecycle.
//!
//! The crate never discovers arbitrary paths, executes model repository code, or downloads from
//! an origin that the caller has not explicitly allowlisted.

mod install;
mod manifest;
mod store;

pub use impossible_embedding_core::ModelVerificationStatus as ModelStatus;
pub use install::{CancelToken, CommitDecision, InstallCommitGate, InstallOptions, Installer};
pub use manifest::{
    Artifact, Dimensions, License, MAX_ARTIFACTS, MAX_TOTAL_ARTIFACT_BYTES, Manifest,
    OnnxInputNames, Pooling, Prefixes, RuntimeMetadata, SemanticVerification, TensorMetadata,
    TokenizerMetadata, curated_manifests,
};
pub use store::{
    CacheLayout, DiscoveryRoot, ModelStore, SemanticTrustRoot, TrustedSemanticEvidence,
    VerifiedModel,
};

use std::{fmt, io};

/// Sanitized category for an HTTP transport failure.
///
/// The originating request error is deliberately reduced to this closed category before it is
/// stored so signed URLs and other request metadata cannot escape through formatting or logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpFailureKind {
    /// A connection could not be established.
    Connect,
    /// A response body could not be read completely.
    Body,
    /// The operation exceeded a transport timeout.
    Timeout,
    /// An HTTP request could not be constructed or sent.
    Request,
    /// A response could not be decoded.
    Decode,
    /// A transport failure not covered by a more specific stable category.
    Other,
}

/// Errors returned by catalog and model lifecycle operations.
pub enum Error {
    /// A manifest or caller-controlled value failed validation.
    Invalid(String),
    /// An operation was cancelled before promotion.
    Cancelled,
    /// Network access was requested while offline mode was enabled.
    Offline,
    /// The exact model identity currently has one or more runtime leases.
    InUse,
    /// A bounded wait for another model lifecycle operation expired.
    Busy,
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
    Http(HttpFailureKind),
    /// Manifest JSON could not be decoded.
    Json(serde_json::Error),
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(_) => formatter
                .debug_tuple("Invalid")
                .field(&"[REDACTED]")
                .finish(),
            Self::Cancelled => formatter.write_str("Cancelled"),
            Self::Offline => formatter.write_str("Offline"),
            Self::InUse => formatter.write_str("InUse"),
            Self::Busy => formatter.write_str("Busy"),
            Self::OriginNotAllowed(_) => formatter
                .debug_tuple("OriginNotAllowed")
                .field(&"[REDACTED]")
                .finish(),
            Self::SizeLimit { expected, actual } => formatter
                .debug_struct("SizeLimit")
                .field("expected", expected)
                .field("actual", actual)
                .finish(),
            Self::HashMismatch { .. } => formatter
                .debug_struct("HashMismatch")
                .field("expected", &"[REDACTED]")
                .field("actual", &"[REDACTED]")
                .finish(),
            Self::Io(error) => formatter.debug_tuple("Io").field(&error.kind()).finish(),
            Self::Http(kind) => formatter.debug_tuple("Http").field(kind).finish(),
            Self::Json(_) => formatter.debug_tuple("Json").field(&"[REDACTED]").finish(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid model data: {message}"),
            Self::Cancelled => formatter.write_str("operation cancelled"),
            Self::Offline => formatter.write_str("network access is disabled in offline mode"),
            Self::InUse => formatter.write_str("model identity is currently in use"),
            Self::Busy => formatter.write_str("model identity is busy; retry later"),
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
            Self::Http(kind) => write!(formatter, "HTTP transport error: {kind}"),
            Self::Json(error) => write!(formatter, "manifest JSON error: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// Whether retrying the same operation later may succeed without changing its input.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::Busy | Self::InUse)
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<reqwest::Error> for Error {
    fn from(value: reqwest::Error) -> Self {
        let kind = if value.is_connect() {
            HttpFailureKind::Connect
        } else if value.is_body() {
            HttpFailureKind::Body
        } else if value.is_timeout() {
            HttpFailureKind::Timeout
        } else if value.is_request() {
            HttpFailureKind::Request
        } else if value.is_decode() {
            HttpFailureKind::Decode
        } else {
            HttpFailureKind::Other
        };
        Self::Http(kind)
    }
}

impl fmt::Display for HttpFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Connect => "connection failed",
            Self::Body => "response body failed",
            Self::Timeout => "request timed out",
            Self::Request | Self::Other => "request failed",
            Self::Decode => "response decoding failed",
        })
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

/// Result type for model lifecycle operations.
pub type Result<T> = std::result::Result<T, Error>;

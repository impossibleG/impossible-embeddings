//! Engine-neutral embedding domain contracts.

use std::{
    borrow::Cow,
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

/// A validated non-empty batch of text inputs. Individual inputs may be empty strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingBatch<'a> {
    inputs: Vec<Cow<'a, str>>,
}

/// Semantic purpose of an embedding input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTask {
    /// A search query.
    Query,
    /// A document or passage.
    Document,
}

/// Behavior for tokenized inputs beyond a model's declared limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truncation {
    /// Reject the request.
    Reject,
    /// Retain the leading tokens that fit the model limit.
    Truncate,
}

/// Transport-independent embedding request options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedOptions {
    /// Task used to select a model prefix.
    pub task: EmbeddingTask,
    /// Explicit over-length behavior.
    pub truncation: Truncation,
    /// Optional, manifest-approved output dimension.
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

/// Integrity and semantic readiness of an exact model identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVerificationStatus {
    /// No final installation exists.
    Missing,
    /// Files exist but fail validation or cannot safely be inspected.
    Invalid,
    /// Every byte matches, but semantic verification is pending.
    IntegrityVerified,
    /// Integrity and semantic behavior are both verified.
    Loadable,
}

impl<'a> EmbeddingBatch<'a> {
    /// Validates and constructs a batch.
    ///
    /// # Errors
    ///
    /// Returns a stable invalid-request failure when no inputs are provided.
    pub fn new(inputs: impl IntoIterator<Item = Cow<'a, str>>) -> Result<Self, EngineFailure> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        if inputs.is_empty() {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        Ok(Self { inputs })
    }

    /// Returns the ordered input texts.
    #[must_use]
    pub fn inputs(&self) -> &[Cow<'a, str>] {
        &self.inputs
    }
}

/// A user-supplied model alias or canonical id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestedModel(String);

impl RequestedModel {
    /// Validates a requested model name.
    ///
    /// # Errors
    ///
    /// Returns an invalid-request failure when the name is blank.
    pub fn new(value: impl Into<String>) -> Result<Self, EngineFailure> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        Ok(Self(value))
    }

    /// Returns the alias exactly as requested.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable identity of the exact model artifact and runtime used for inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelIdentity {
    /// Registry-qualified or locally assigned canonical model id.
    pub canonical_id: String,
    /// Immutable upstream revision or content-addressed local revision.
    pub revision: String,
    /// Runtime adapter and compatibility version.
    pub runtime: String,
    /// Content fingerprint covering all inference-affecting artifacts.
    pub artifact_fingerprint: String,
    /// Canonical fingerprint covering the complete inference semantics and artifact content.
    pub semantic_fingerprint: String,
}

impl ResolvedModelIdentity {
    /// Creates an identity only when every reproducibility field is present.
    ///
    /// # Errors
    ///
    /// Returns an internal failure when a resolver produces an incomplete identity.
    pub fn new(
        canonical_id: impl Into<String>,
        revision: impl Into<String>,
        runtime: impl Into<String>,
        artifact_fingerprint: impl Into<String>,
        semantic_fingerprint: impl Into<String>,
    ) -> Result<Self, EngineFailure> {
        let identity = Self {
            canonical_id: canonical_id.into(),
            revision: revision.into(),
            runtime: runtime.into(),
            artifact_fingerprint: artifact_fingerprint.into(),
            semantic_fingerprint: semantic_fingerprint.into(),
        };
        if [
            &identity.canonical_id,
            &identity.revision,
            &identity.runtime,
            &identity.artifact_fingerprint,
            &identity.semantic_fingerprint,
        ]
        .into_iter()
        .any(|value| value.trim().is_empty())
        {
            return Err(EngineFailure::public(ErrorCode::Internal));
        }
        Ok(identity)
    }
}

/// Output vectors in the same order as their inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutput {
    /// Dense vectors, ordered by request input.
    pub vectors: Vec<Vec<f32>>,
    /// Exact model identity used to produce the vectors.
    pub model: ResolvedModelIdentity,
}

/// Stable, privacy-safe codes exposed by every transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The request is structurally or semantically invalid.
    InvalidRequest,
    /// The selected model cannot currently serve requests.
    ModelUnavailable,
    /// The bounded admission queue has no capacity.
    QueueFull,
    /// The caller cancelled the request.
    Cancelled,
    /// The request deadline elapsed.
    DeadlineExceeded,
    /// The runtime failed during inference.
    InferenceFailed,
    /// An unexpected server failure occurred.
    Internal,
}

impl ErrorCode {
    /// Returns the stable wire representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::ModelUnavailable => "model_unavailable",
            Self::QueueFull => "queue_full",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::InferenceFailed => "inference_failed",
            Self::Internal => "internal",
        }
    }
}

/// Whether retrying a failed request can be useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryability {
    /// Retrying unchanged cannot succeed.
    Never,
    /// Retrying later may succeed.
    Retryable,
    /// The server cannot safely make a retry claim.
    Unknown,
}

/// Stable error data that is safe to serialize to clients and normal logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicError {
    /// Machine-readable stable code.
    pub code: ErrorCode,
    /// Retry guidance independent of transport status codes.
    pub retryability: Retryability,
    /// Privacy-reviewed message containing no engine diagnostics or local paths.
    pub message: &'static str,
}

impl PublicError {
    /// Returns the canonical privacy-reviewed representation for a stable code.
    #[must_use]
    pub const fn for_code(code: ErrorCode) -> Self {
        match code {
            ErrorCode::InvalidRequest => Self::new(
                code,
                Retryability::Never,
                "the embedding request is invalid",
            ),
            ErrorCode::ModelUnavailable => Self::new(
                code,
                Retryability::Retryable,
                "the requested model is unavailable",
            ),
            ErrorCode::QueueFull => {
                Self::new(code, Retryability::Retryable, "the request queue is full")
            }
            ErrorCode::Cancelled => {
                Self::new(code, Retryability::Never, "the request was cancelled")
            }
            ErrorCode::DeadlineExceeded => Self::new(
                code,
                Retryability::Never,
                "the request deadline was exceeded",
            ),
            ErrorCode::InferenceFailed => {
                Self::new(code, Retryability::Unknown, "embedding inference failed")
            }
            ErrorCode::Internal => Self::new(
                code,
                Retryability::Unknown,
                "an internal server error occurred",
            ),
        }
    }

    const fn new(code: ErrorCode, retryability: Retryability, message: &'static str) -> Self {
        Self {
            code,
            retryability,
            message,
        }
    }
}

/// An engine failure with strictly separated public and internal detail.
///
/// Transports must serialize only [`Self::public_error`]. The diagnostic source is intended for
/// access-controlled debug telemetry and may contain sensitive runtime information.
pub struct EngineFailure {
    public: PublicError,
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl EngineFailure {
    /// Creates a failure without internal diagnostics.
    #[must_use]
    pub const fn public(code: ErrorCode) -> Self {
        Self {
            public: PublicError::for_code(code),
            source: None,
        }
    }

    /// Attaches a private diagnostic source to a stable public failure.
    #[must_use]
    pub fn with_source(code: ErrorCode, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            public: PublicError::for_code(code),
            source: Some(Box::new(source)),
        }
    }

    /// Returns the only error representation transports may expose.
    #[must_use]
    pub const fn public_error(&self) -> &PublicError {
        &self.public
    }

    /// Returns private diagnostics for explicitly access-controlled telemetry.
    #[must_use]
    pub fn diagnostic_source(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.source.as_deref()
    }
}

impl fmt::Debug for EngineFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineFailure")
            .field("public", &self.public)
            .field("has_diagnostic_source", &self.source.is_some())
            .finish()
    }
}

impl fmt::Display for EngineFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.public.message)
    }
}

// Intentionally do not expose the diagnostic through `Error::source`. Generic error reporters
// commonly traverse that chain into ordinary logs and client responses. Callers that are allowed
// to handle private diagnostics must opt in through `diagnostic_source` instead.
impl Error for EngineFailure {}

/// Cloneable cancellation signal shared by admission, engine, and response layers.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// Requests cancellation. Calls are idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Admission and publication constraints for one request.
#[derive(Debug, Clone)]
pub struct ExecutionControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl ExecutionControl {
    /// Creates request controls with an optional absolute monotonic deadline.
    #[must_use]
    pub const fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
        }
    }

    /// Returns the shared cancellation token.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Checks whether queued work may start or completed work may be published.
    ///
    /// Cancellation wins when both cancellation and deadline are observed.
    ///
    /// # Errors
    ///
    /// Returns a stable cancelled or deadline-exceeded failure.
    pub fn ensure_active(&self) -> Result<(), EngineFailure> {
        if self.cancellation.is_cancelled() {
            return Err(EngineFailure::public(ErrorCode::Cancelled));
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(EngineFailure::public(ErrorCode::DeadlineExceeded));
        }
        Ok(())
    }
}

/// Engine boundary implemented by in-process and delegated runtimes.
pub trait EmbeddingEngine: Send + Sync {
    /// Produces one dense vector for each input, preserving order.
    ///
    /// Implementations must check `control` immediately before invoking native inference. Native
    /// runtimes are not required to support interruption after invocation begins.
    ///
    /// # Errors
    ///
    /// Returns a stable failure, optionally carrying a private diagnostic source.
    fn embed(
        &self,
        requested_model: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        control: &ExecutionControl,
    ) -> Result<EmbeddingOutput, EngineFailure>;
}

/// Executes an admitted request and prevents late native results from being published.
///
/// This wrapper checks control state before dispatch and after inference. If cancellation or a
/// deadline occurs while a non-interruptible runtime is executing, the runtime may finish, but its
/// result is discarded here.
///
/// # Errors
///
/// Returns cancellation/deadline failures around the engine's own stable failures.
pub fn execute_embedding(
    engine: &dyn EmbeddingEngine,
    requested_model: &RequestedModel,
    batch: &EmbeddingBatch<'_>,
    control: &ExecutionControl,
) -> Result<EmbeddingOutput, EngineFailure> {
    control.ensure_active()?;
    let result = engine.embed(requested_model, batch, control);
    control.ensure_active()?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_error_chain(mut error: &(dyn Error + 'static)) -> String {
        let mut messages = vec![error.to_string()];
        while let Some(source) = error.source() {
            messages.push(source.to_string());
            error = source;
        }
        messages.join(": ")
    }

    fn identity() -> Result<ResolvedModelIdentity, EngineFailure> {
        ResolvedModelIdentity::new(
            "registry.example/model",
            "revision-sha256",
            "test-runtime@1",
            "sha256:artifact",
            "sha256:semantics",
        )
    }

    struct CancelsDuringInference;

    struct CancelsThenFails;

    struct ExpiresThenFails;

    struct MustNotRun;

    impl EmbeddingEngine for MustNotRun {
        fn embed(
            &self,
            _requested_model: &RequestedModel,
            _batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            Err(EngineFailure::public(ErrorCode::Internal))
        }
    }

    impl EmbeddingEngine for CancelsDuringInference {
        fn embed(
            &self,
            _requested_model: &RequestedModel,
            _batch: &EmbeddingBatch<'_>,
            control: &ExecutionControl,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            control.ensure_active()?;
            control.cancellation().cancel();
            Ok(EmbeddingOutput {
                vectors: vec![vec![1.0]],
                model: identity()?,
            })
        }
    }

    impl EmbeddingEngine for CancelsThenFails {
        fn embed(
            &self,
            _requested_model: &RequestedModel,
            _batch: &EmbeddingBatch<'_>,
            control: &ExecutionControl,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            control.cancellation().cancel();
            Err(EngineFailure::public(ErrorCode::InferenceFailed))
        }
    }

    impl EmbeddingEngine for ExpiresThenFails {
        fn embed(
            &self,
            _requested_model: &RequestedModel,
            _batch: &EmbeddingBatch<'_>,
            control: &ExecutionControl,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            let Some(deadline) = control.deadline else {
                return Err(EngineFailure::public(ErrorCode::Internal));
            };
            while Instant::now() < deadline {
                std::hint::spin_loop();
            }
            Err(EngineFailure::public(ErrorCode::InferenceFailed))
        }
    }

    #[test]
    fn rejects_empty_batches_with_stable_public_error() -> Result<(), &'static str> {
        let result = EmbeddingBatch::new(Vec::<Cow<'_, str>>::new());
        let Some(error) = result.err() else {
            return Err("empty batches must fail");
        };
        assert_eq!(error.public_error().code, ErrorCode::InvalidRequest);
        assert_eq!(error.public_error().retryability, Retryability::Never);
        assert!(error.diagnostic_source().is_none());
        Ok(())
    }

    #[test]
    fn preserves_requested_alias_separately_from_resolved_identity() -> Result<(), EngineFailure> {
        let requested = RequestedModel::new("default")?;
        let resolved = identity()?;
        assert_eq!(requested.as_str(), "default");
        assert_eq!(resolved.canonical_id, "registry.example/model");
        assert_eq!(resolved.revision, "revision-sha256");
        Ok(())
    }

    #[test]
    fn rejects_queued_work_after_deadline() -> Result<(), &'static str> {
        let control = ExecutionControl::new(CancellationToken::default(), Some(Instant::now()));
        let Some(error) = control.ensure_active().err() else {
            return Err("expired work must fail");
        };
        assert_eq!(error.public_error().code, ErrorCode::DeadlineExceeded);
        Ok(())
    }

    #[test]
    fn cancelled_queued_work_never_reaches_engine() -> Result<(), EngineFailure> {
        let batch = EmbeddingBatch::new([Cow::Borrowed("input")])?;
        let requested = RequestedModel::new("default")?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let control = ExecutionControl::new(cancellation, None);
        let Some(error) = execute_embedding(&MustNotRun, &requested, &batch, &control).err() else {
            return Err(EngineFailure::public(ErrorCode::Internal));
        };
        assert_eq!(error.public_error().code, ErrorCode::Cancelled);
        Ok(())
    }

    #[test]
    fn discards_late_native_result_after_cancellation() -> Result<(), EngineFailure> {
        let batch = EmbeddingBatch::new([Cow::Borrowed("input")])?;
        let requested = RequestedModel::new("default")?;
        let control = ExecutionControl::new(CancellationToken::default(), None);
        let Some(error) =
            execute_embedding(&CancelsDuringInference, &requested, &batch, &control).err()
        else {
            return Err(EngineFailure::public(ErrorCode::Internal));
        };
        assert_eq!(error.public_error().code, ErrorCode::Cancelled);
        Ok(())
    }

    #[test]
    fn cancellation_takes_precedence_over_late_engine_failure() -> Result<(), EngineFailure> {
        let batch = EmbeddingBatch::new([Cow::Borrowed("input")])?;
        let requested = RequestedModel::new("default")?;
        let control = ExecutionControl::new(CancellationToken::default(), None);
        let Some(error) = execute_embedding(&CancelsThenFails, &requested, &batch, &control).err()
        else {
            return Err(EngineFailure::public(ErrorCode::Internal));
        };
        assert_eq!(error.public_error().code, ErrorCode::Cancelled);
        Ok(())
    }

    #[test]
    fn deadline_takes_precedence_over_late_engine_failure() -> Result<(), EngineFailure> {
        let batch = EmbeddingBatch::new([Cow::Borrowed("input")])?;
        let requested = RequestedModel::new("default")?;
        let control = ExecutionControl::new(
            CancellationToken::default(),
            Some(Instant::now() + std::time::Duration::from_millis(10)),
        );
        let Some(error) = execute_embedding(&ExpiresThenFails, &requested, &batch, &control).err()
        else {
            return Err(EngineFailure::public(ErrorCode::Internal));
        };
        assert_eq!(error.public_error().code, ErrorCode::DeadlineExceeded);
        Ok(())
    }

    #[test]
    fn public_error_codes_have_stable_wire_values() {
        assert_eq!(ErrorCode::InvalidRequest.as_str(), "invalid_request");
        assert_eq!(ErrorCode::ModelUnavailable.as_str(), "model_unavailable");
        assert_eq!(ErrorCode::QueueFull.as_str(), "queue_full");
        assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
        assert_eq!(ErrorCode::DeadlineExceeded.as_str(), "deadline_exceeded");
        assert_eq!(ErrorCode::InferenceFailed.as_str(), "inference_failed");
        assert_eq!(ErrorCode::Internal.as_str(), "internal");
        assert_eq!(
            PublicError::for_code(ErrorCode::QueueFull).retryability,
            Retryability::Retryable
        );
        assert_eq!(
            PublicError::for_code(ErrorCode::InferenceFailed).retryability,
            Retryability::Unknown
        );
    }

    #[test]
    fn ordinary_error_reporting_never_exposes_private_source() {
        #[derive(Debug)]
        struct SensitiveDiagnostic;
        impl fmt::Display for SensitiveDiagnostic {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("private local path")
            }
        }
        impl Error for SensitiveDiagnostic {}

        let failure = EngineFailure::with_source(ErrorCode::InferenceFailed, SensitiveDiagnostic);
        let debug = format!("{failure:?}");
        assert!(!debug.contains("private local path"));
        assert_eq!(failure.to_string(), "embedding inference failed");
        assert!(std::error::Error::source(&failure).is_none());

        let ordinary_chain = format_error_chain(&failure);
        assert_eq!(ordinary_chain, "embedding inference failed");
        assert!(!ordinary_chain.contains("private local path"));

        let diagnostic = failure
            .diagnostic_source()
            .map(ToString::to_string)
            .ok_or("private diagnostics must remain explicitly accessible")
            .unwrap_or_default();
        assert_eq!(diagnostic, "private local path");
        assert!(failure.diagnostic_source().is_some());
    }
}

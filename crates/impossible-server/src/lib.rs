//! Transport-independent application runtime for model lifecycle and bounded inference.

pub mod application;
pub mod runtime;

pub use application::{
    AppState, AppStateError, ApplicationError, ApplicationErrorKind, CatalogError, EmbedCommand,
    EmbeddingApplication, LoadStatus, ModelCatalog, ModelInfo, PreloadOutcome, RequestIdSource,
    RuntimeStatus, SemanticStatus, ShutdownTrigger, StartupReport,
};

pub use runtime::{
    ApplicationRuntime, BatchPolicy, LifecycleError, RuntimeEngine, RuntimeLease, RuntimeSnapshot,
};

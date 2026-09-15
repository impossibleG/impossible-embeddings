//! Transport-independent application runtime for model lifecycle and bounded inference.

pub mod runtime;

pub use runtime::{
    ApplicationRuntime, BatchPolicy, LifecycleError, RuntimeEngine, RuntimeLease, RuntimeSnapshot,
};

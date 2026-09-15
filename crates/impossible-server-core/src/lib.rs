//! Transport-neutral operational foundations for Impossible servers.

pub mod config;
pub mod doctor;
pub mod health;
pub mod metrics;
pub mod security;
pub mod shutdown;
pub mod telemetry;

pub use config::{ConfigError, ServerConfig};
pub use health::{
    HealthRegistry, LifecycleState, ModelFailureReason, ModelKey, ModelState, Readiness,
    ReadinessReason,
};
pub use shutdown::{ShutdownCoordinator, ShutdownState};

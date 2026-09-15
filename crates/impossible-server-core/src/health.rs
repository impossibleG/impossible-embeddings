//! Lock-safe liveness, readiness, and per-model state.

use impossible_embedding_core::ModelVerificationStatus;
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
};

/// Process lifecycle state exposed through health adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// Configuration and runtime initialization are incomplete.
    Starting,
    /// The process can accept work.
    Ready,
    /// The process is draining and rejects new work.
    Draining,
    /// The process has stopped.
    Stopped,
}

/// Stable opaque model identifier safe for aggregate health output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelKey(pub u64);

/// Model state used by schedulers and health adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// Model registration exists but initialization has not started.
    Unloaded,
    /// Model artifacts or runtime are being initialized.
    Loading,
    /// Model can accept inference requests.
    Ready,
    /// Model is unavailable with a stable reason code.
    Failed(&'static str),
}

impl From<ModelVerificationStatus> for ModelState {
    fn from(status: ModelVerificationStatus) -> Self {
        match status {
            ModelVerificationStatus::Missing => Self::Failed("model_missing"),
            ModelVerificationStatus::Invalid => Self::Failed("model_invalid"),
            ModelVerificationStatus::IntegrityVerified => {
                Self::Failed("semantic_verification_pending")
            }
            // Artifact readiness permits adapter loading; it does not prove an initialized runtime.
            ModelVerificationStatus::Loadable => Self::Loading,
        }
    }
}

/// A privacy-safe readiness snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Readiness {
    /// Current process lifecycle.
    pub state: LifecycleState,
    /// Stable, non-sensitive reason code when the service is not ready.
    pub reason_code: Option<&'static str>,
}

impl Readiness {
    /// Returns whether the process can accept new work.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self.state, LifecycleState::Ready)
    }
}

#[derive(Debug)]
struct HealthState {
    live: bool,
    lifecycle: LifecycleState,
    reason: Option<&'static str>,
    models: BTreeMap<ModelKey, ModelState>,
}

/// Shared health registry. Lock poisoning degrades health instead of panicking.
#[derive(Clone, Debug)]
pub struct HealthRegistry(Arc<RwLock<HealthState>>);

impl Default for HealthRegistry {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HealthState {
            live: true,
            lifecycle: LifecycleState::Starting,
            reason: Some("starting"),
            models: BTreeMap::new(),
        })))
    }
}

impl HealthRegistry {
    /// Returns whether the process event loop is alive.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.0.read().is_ok_and(|state| state.live)
    }

    /// Register or update an opaque model state.
    pub fn set_model(&self, key: ModelKey, model: ModelState) {
        if let Ok(mut state) = self.0.write() {
            state.models.insert(key, model);
        }
    }

    /// Register model artifact readiness from the canonical verification status.
    ///
    /// A loadable identity becomes [`ModelState::Loading`]; only the runtime adapter may promote
    /// it to [`ModelState::Ready`] after successful initialization and warmup.
    pub fn set_model_verification(&self, key: ModelKey, status: ModelVerificationStatus) {
        self.set_model(key, status.into());
    }

    /// Transition the process lifecycle. Stopped cannot transition back to service.
    #[must_use]
    pub fn transition(&self, next: LifecycleState, reason: Option<&'static str>) -> bool {
        let Ok(mut state) = self.0.write() else {
            return false;
        };
        if state.lifecycle == LifecycleState::Stopped {
            return next == LifecycleState::Stopped;
        }
        if state.lifecycle == LifecycleState::Draining
            && matches!(next, LifecycleState::Starting | LifecycleState::Ready)
        {
            return false;
        }
        state.lifecycle = next;
        state.reason = reason;
        if next == LifecycleState::Stopped {
            state.live = false;
        }
        true
    }

    /// Obtain a health snapshot without exposing model identity or host details.
    #[must_use]
    pub fn readiness(&self) -> Readiness {
        let Ok(state) = self.0.read() else {
            return Readiness {
                state: LifecycleState::Draining,
                reason_code: Some("health_state_unavailable"),
            };
        };
        Readiness {
            state: state.lifecycle,
            reason_code: state.reason,
        }
    }

    /// Return aggregate model counts without exposing identities.
    #[must_use]
    pub fn model_counts(&self) -> (usize, usize) {
        let Ok(state) = self.0.read() else {
            return (0, 0);
        };
        (
            state
                .models
                .values()
                .filter(|value| **value == ModelState::Ready)
                .count(),
            state.models.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_requires_process_and_model() {
        let health = HealthRegistry::default();
        assert!(!health.readiness().is_ready());
        health.set_model(ModelKey(1), ModelState::Ready);
        assert!(health.transition(LifecycleState::Ready, None));
        assert!(health.readiness().is_ready());
        assert!(health.transition(LifecycleState::Draining, Some("shutdown")));
        assert!(!health.readiness().is_ready());
        assert!(!health.transition(LifecycleState::Ready, None));
        assert!(health.transition(LifecycleState::Stopped, Some("stopped")));
        assert!(!health.is_live());
        assert!(!health.transition(LifecycleState::Starting, None));
    }
}

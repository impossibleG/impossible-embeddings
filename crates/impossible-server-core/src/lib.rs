//! Transport-neutral lifecycle and readiness primitives.

/// Process lifecycle state exposed through future health adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// Configuration and runtime initialization are incomplete.
    Starting,
    /// The process can accept work.
    Ready,
    /// The process is draining and rejects new work.
    Draining,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_ready_state_accepts_work() {
        assert!(
            Readiness {
                state: LifecycleState::Ready,
                reason_code: None
            }
            .is_ready()
        );
        assert!(
            !Readiness {
                state: LifecycleState::Draining,
                reason_code: Some("shutdown")
            }
            .is_ready()
        );
    }
}

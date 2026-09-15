//! Coordinated, idempotent graceful shutdown.

use std::{
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

/// Global shutdown state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownState {
    /// New work may be admitted.
    Running,
    /// No new work is accepted; active work is draining.
    Draining,
    /// All tracked work has completed.
    Stopped,
}

#[derive(Debug)]
struct State {
    shutdown: ShutdownState,
    active: usize,
}

/// Shared shutdown coordination independent of an async runtime.
#[derive(Clone, Debug)]
pub struct ShutdownCoordinator(Arc<(Mutex<State>, Condvar)>);

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self(Arc::new((
            Mutex::new(State {
                shutdown: ShutdownState::Running,
                active: 0,
            }),
            Condvar::new(),
        )))
    }
}

/// RAII permit tracking admitted work through cancellation and errors.
#[derive(Debug)]
pub struct WorkPermit(Option<ShutdownCoordinator>);

impl ShutdownCoordinator {
    /// Current state. Poisoned coordination state fails closed as stopped.
    #[must_use]
    pub fn state(&self) -> ShutdownState {
        self.0
            .0
            .lock()
            .map_or(ShutdownState::Stopped, |state| state.shutdown)
    }

    /// Admit work only while running.
    #[must_use]
    pub fn admit(&self) -> Option<WorkPermit> {
        let Ok(mut state) = self.0.0.lock() else {
            return None;
        };
        if state.shutdown != ShutdownState::Running {
            return None;
        }
        state.active = state.active.saturating_add(1);
        Some(WorkPermit(Some(self.clone())))
    }

    /// Begin draining. Repeated calls are harmless.
    pub fn begin(&self) {
        let Ok(mut state) = self.0.0.lock() else {
            return;
        };
        if state.shutdown == ShutdownState::Running {
            state.shutdown = ShutdownState::Draining;
        }
        if state.active == 0 {
            state.shutdown = ShutdownState::Stopped;
        }
        self.0.1.notify_all();
    }

    /// Wait until all tracked work completes or the deadline expires.
    #[must_use]
    pub fn wait(&self, timeout: Duration) -> bool {
        let started = Instant::now();
        let Ok(mut state) = self.0.0.lock() else {
            return false;
        };
        while state.shutdown != ShutdownState::Stopped {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return false;
            }
            let Ok((next, result)) = self.0.1.wait_timeout(state, remaining) else {
                return false;
            };
            state = next;
            if result.timed_out() && state.shutdown != ShutdownState::Stopped {
                return false;
            }
        }
        true
    }
}

impl Drop for WorkPermit {
    fn drop(&mut self) {
        let Some(coordinator) = self.0.take() else {
            return;
        };
        let Ok(mut state) = coordinator.0.0.lock() else {
            return;
        };
        state.active = state.active.saturating_sub(1);
        if state.active == 0 && state.shutdown == ShutdownState::Draining {
            state.shutdown = ShutdownState::Stopped;
        }
        coordinator.0.1.notify_all();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_rejects_new_work_and_waits_for_active_work() {
        let shutdown = ShutdownCoordinator::default();
        let permit = shutdown.admit().expect("admitted");
        shutdown.begin();
        assert_eq!(shutdown.state(), ShutdownState::Draining);
        assert!(shutdown.admit().is_none());
        assert!(!shutdown.wait(Duration::from_millis(1)));
        drop(permit);
        assert!(shutdown.wait(Duration::from_millis(10)));
        assert_eq!(shutdown.state(), ShutdownState::Stopped);
        shutdown.begin();
        assert_eq!(shutdown.state(), ShutdownState::Stopped);
    }

    #[test]
    fn shutdown_without_work_stops_immediately() {
        let shutdown = ShutdownCoordinator::default();
        shutdown.begin();
        assert!(shutdown.wait(Duration::ZERO));
    }
}

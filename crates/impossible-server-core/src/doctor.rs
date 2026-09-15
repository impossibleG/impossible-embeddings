//! Sanitized diagnostic report types.

use crate::health::{LifecycleState, Readiness, ReadinessReason};

/// Safe-by-default `doctor` report.
///
/// Hostnames, usernames, absolute paths, hardware identifiers, IP addresses,
/// environment values, and model names are intentionally absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoctorReport {
    /// Published application version.
    pub version: &'static str,
    /// Current lifecycle state.
    pub lifecycle: LifecycleState,
    /// Stable non-sensitive readiness reason.
    pub reason_code: Option<ReadinessReason>,
    /// Total configured model count.
    pub model_count: usize,
    /// Ready model count.
    pub ready_model_count: usize,
}

impl DoctorReport {
    /// Build a report from sanitized health data.
    #[must_use]
    pub const fn from_readiness(
        version: &'static str,
        readiness: &Readiness,
        ready_models: usize,
        total_models: usize,
    ) -> Self {
        Self {
            version,
            lifecycle: readiness.state,
            reason_code: readiness.reason_code,
            model_count: total_models,
            ready_model_count: ready_models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_contains_only_product_and_aggregate_state() {
        let readiness = Readiness {
            state: LifecycleState::Ready,
            reason_code: None,
        };
        let report = DoctorReport::from_readiness("0.1.0", &readiness, 1, 2);
        let debug = format!("{report:?}");
        assert!(!debug.contains("hostname"));
        assert!(!debug.contains("processor"));
        assert!(!debug.contains("sentinel-private-path"));
    }
}

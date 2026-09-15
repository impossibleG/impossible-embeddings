//! Prometheus metrics with compile-time bounded label vocabularies.

use std::{
    collections::BTreeMap,
    fmt::Write,
    sync::{Arc, Mutex},
    time::Duration,
};

/// Transport-independent operation class. No user-controlled route is accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Operation {
    /// Embedding inference.
    Embed,
    /// Model listing.
    ListModels,
    /// Model administration.
    AdminModel,
    /// Health probe.
    Health,
}

impl Operation {
    const ALL: [Self; 4] = [
        Self::Embed,
        Self::ListModels,
        Self::AdminModel,
        Self::Health,
    ];
    const fn label(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::ListModels => "list_models",
            Self::AdminModel => "admin_model",
            Self::Health => "health",
        }
    }
}

/// Stable request outcome class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    /// Successful request.
    Ok,
    /// Invalid client input.
    Invalid,
    /// Overloaded admission queue.
    Overloaded,
    /// Deadline exceeded or cancellation.
    Cancelled,
    /// Internal service failure.
    Failed,
}

impl Outcome {
    const ALL: [Self; 5] = [
        Self::Ok,
        Self::Invalid,
        Self::Overloaded,
        Self::Cancelled,
        Self::Failed,
    ];
    const fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Invalid => "invalid",
            Self::Overloaded => "overloaded",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    requests: BTreeMap<(Operation, Outcome), u64>,
    durations_micros: BTreeMap<Operation, u64>,
    queue_depth: u64,
    active: u64,
}

/// Cloneable registry whose cardinality cannot be expanded with request data.
#[derive(Clone, Debug, Default)]
pub struct Metrics(Arc<Mutex<Inner>>);

impl Metrics {
    /// Observe a completed operation.
    pub fn observe(&self, operation: Operation, outcome: Outcome, elapsed: Duration) {
        if let Ok(mut inner) = self.0.lock() {
            *inner.requests.entry((operation, outcome)).or_default() += 1;
            let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
            let total = inner.durations_micros.entry(operation).or_default();
            *total = total.saturating_add(micros);
        }
    }

    /// Set current admitted queue depth.
    pub fn set_queue_depth(&self, depth: usize) {
        if let Ok(mut inner) = self.0.lock() {
            inner.queue_depth = u64::try_from(depth).unwrap_or(u64::MAX);
        }
    }

    /// Set current active inference count.
    pub fn set_active(&self, active: usize) {
        if let Ok(mut inner) = self.0.lock() {
            inner.active = u64::try_from(active).unwrap_or(u64::MAX);
        }
    }

    /// Increment the active request gauge without a racy read/replace sequence.
    pub fn increment_active(&self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.active = inner.active.saturating_add(1);
        }
    }

    /// Decrement the active request gauge, saturating defensively at zero.
    pub fn decrement_active(&self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.active = inner.active.saturating_sub(1);
        }
    }

    /// Increment the aggregate pre-native waiting-request gauge.
    pub fn increment_queue_depth(&self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.queue_depth = inner.queue_depth.saturating_add(1);
        }
    }

    /// Decrement the aggregate pre-native waiting-request gauge.
    pub fn decrement_queue_depth(&self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.queue_depth = inner.queue_depth.saturating_sub(1);
        }
    }

    /// Render deterministic Prometheus text exposition.
    #[must_use]
    pub fn render(&self) -> String {
        let Ok(inner) = self.0.lock() else {
            return "# metrics unavailable\n".to_owned();
        };
        let mut output = String::from(
            "# HELP impossible_requests_total Completed requests.\n# TYPE impossible_requests_total counter\n",
        );
        for operation in Operation::ALL {
            for outcome in Outcome::ALL {
                let value = inner
                    .requests
                    .get(&(operation, outcome))
                    .copied()
                    .unwrap_or_default();
                let _ = writeln!(
                    output,
                    "impossible_requests_total{{operation=\"{}\",outcome=\"{}\"}} {value}",
                    operation.label(),
                    outcome.label()
                );
            }
        }
        output.push_str("# HELP impossible_operation_duration_seconds_total Aggregate operation time.\n# TYPE impossible_operation_duration_seconds_total counter\n");
        for operation in Operation::ALL {
            let micros = inner
                .durations_micros
                .get(&operation)
                .copied()
                .unwrap_or_default();
            let seconds = micros / 1_000_000;
            let fractional = micros % 1_000_000;
            let _ = writeln!(
                output,
                "impossible_operation_duration_seconds_total{{operation=\"{}\"}} {seconds}.{fractional:06}",
                operation.label()
            );
        }
        output.push_str("# TYPE impossible_queue_depth gauge\n");
        let _ = writeln!(output, "impossible_queue_depth {}", inner.queue_depth);
        output.push_str("# TYPE impossible_active_requests gauge\n");
        let _ = writeln!(output, "impossible_active_requests {}", inner.active);
        output
    }

    /// Maximum number of distinct labelled time series emitted by this version.
    #[must_use]
    pub const fn maximum_labelled_series() -> usize {
        24
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinality_is_bounded_and_values_accumulate() {
        let metrics = Metrics::default();
        for _ in 0..10_000 {
            metrics.observe(Operation::Embed, Outcome::Ok, Duration::from_micros(2));
        }
        let rendered = metrics.render();
        assert!(rendered.contains("operation=\"embed\",outcome=\"ok\"} 10000"));
        let labelled = rendered.lines().filter(|line| line.contains('{')).count();
        assert_eq!(labelled, Metrics::maximum_labelled_series());
        assert!(!rendered.contains("request_id"));
        assert!(!rendered.contains("model="));
    }
}

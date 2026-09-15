//! Process assembly placeholder for the foundation milestone.

use impossible_server_core::{LifecycleState, Readiness, ReadinessReason};

fn main() {
    let readiness = Readiness {
        state: LifecycleState::Starting,
        reason_code: Some(ReadinessReason::Starting),
    };
    println!(
        "Impossible Embedding foundation (ready: {})",
        readiness.is_ready()
    );
}

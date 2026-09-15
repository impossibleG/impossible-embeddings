//! Process assembly placeholder for the foundation milestone.

use impossible_server_core::{LifecycleState, Readiness};

fn main() {
    let readiness = Readiness {
        state: LifecycleState::Starting,
        reason_code: Some("foundation"),
    };
    println!(
        "Impossible Embedding foundation (ready: {})",
        readiness.is_ready()
    );
}

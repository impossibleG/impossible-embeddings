//! Cross-cutting policy integration tests.

use std::time::Duration;

use impossible_server_core::{
    ServerConfig,
    metrics::{Metrics, Operation, Outcome},
    security::{OriginPolicy, Secret, verify_bearer},
    telemetry::{RequestEvent, RequestId},
};

#[test]
fn validated_configuration_drives_security_and_privacy_safe_observability()
-> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::load(
        None,
        [
            ("IMPOSSIBLE_ALLOWED_ORIGINS", "https://console.example"),
            ("IMPOSSIBLE_AUTH_ENV", "SERVICE_TOKEN"),
        ],
        std::iter::empty::<&str>(),
    )?;
    let origins = OriginPolicy::new(config.allowed_origins);
    assert!(origins.allows(Some("https://console.example")));
    assert!(!origins.allows(Some("https://console.example.evil")));

    let sensitive = "integration-secret-sentinel";
    let secret = Secret::new(sensitive.as_bytes().to_vec())?;
    assert!(verify_bearer(
        Some("Bearer integration-secret-sentinel"),
        &secret
    ));

    let metrics = Metrics::default();
    metrics.observe(Operation::Embed, Outcome::Ok, Duration::from_millis(3));
    let event = RequestEvent {
        request_id: RequestId(9),
        operation: Operation::Embed,
        outcome: Outcome::Ok,
        input_count: 1,
        request_bytes: 24,
        elapsed: Duration::from_millis(3),
    };
    for observable in [metrics.render(), event.to_logfmt(), format!("{secret:?}")] {
        assert!(!observable.contains(sensitive));
        assert!(!observable.contains("console.example"));
    }
    Ok(())
}

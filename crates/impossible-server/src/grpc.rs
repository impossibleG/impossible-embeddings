//! Bounded, authenticated gRPC transport for the shared embedding application.

use std::{fmt, net::SocketAddr, time::Duration};

use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingTask, ErrorCode, PublicError,
    Truncation as CoreTruncation,
};
use impossible_protocol::{
    grpc_status_code, public_error_detail,
    v1::{
        self, DenseVector, EmbedRequest, EmbedResponse, EmbeddingTask as ProtoEmbeddingTask,
        ResolvedModelIdentity, TokenUsage, Truncation as ProtoTruncation,
        embedding_service_server::{EmbeddingService, EmbeddingServiceServer},
    },
};
use prost::Message;
use tokio::{net::TcpListener, sync::watch, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, metadata::MetadataMap, transport::Server};

use crate::{AppState, ApplicationError, EmbedCommand};

const AUTHORIZATION: &str = "authorization";
const AUTHORIZATION_BIN: &str = "authorization-bin";
const GRPC_TIMEOUT: &str = "grpc-timeout";
const GRPC_TIMEOUT_BIN: &str = "grpc-timeout-bin";
const HEALTH_REFRESH_INTERVAL: Duration = Duration::from_millis(50);

/// Sanitized gRPC host lifecycle failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrpcServerError {
    /// Reflection descriptors could not be assembled.
    Reflection,
    /// The transport server failed.
    Transport,
    /// The transport task terminated abnormally.
    Join,
    /// Graceful shutdown exceeded the configured bound and was aborted.
    ShutdownTimeout,
}

impl fmt::Display for GrpcServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Reflection => "gRPC reflection initialization failed",
            Self::Transport => "gRPC transport failed",
            Self::Join => "gRPC transport task failed",
            Self::ShutdownTimeout => "gRPC shutdown timed out",
        })
    }
}

impl std::error::Error for GrpcServerError {}

/// Running gRPC listener with bounded graceful shutdown.
pub struct GrpcServerHandle {
    local_addr: SocketAddr,
    shutdown_timeout: Duration,
    shutdown: watch::Sender<bool>,
    join: Option<JoinHandle<Result<(), GrpcServerError>>>,
}

impl fmt::Debug for GrpcServerHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrpcServerHandle")
            .field("local_addr", &"[REDACTED]")
            .field("running", &self.join.is_some())
            .finish_non_exhaustive()
    }
}

impl GrpcServerHandle {
    /// Bound address, including the operating-system-selected port.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Mark health not-serving, stop accepting connections, and wait within the configured bound.
    ///
    /// # Errors
    /// Returns a sanitized transport, join, or shutdown-timeout failure.
    pub async fn shutdown(mut self) -> Result<(), GrpcServerError> {
        let _ = self.shutdown.send(true);
        let Some(mut join) = self.join.take() else {
            return Ok(());
        };
        match tokio::time::timeout(self.shutdown_timeout, &mut join).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(GrpcServerError::Join),
            Err(_) => {
                join.abort();
                let _ = join.await;
                Err(GrpcServerError::ShutdownTimeout)
            }
        }
    }
}

impl Drop for GrpcServerHandle {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

/// Start a gRPC host on an already-bound listener.
///
/// Binding is intentionally performed by the caller so startup can reserve an ephemeral port and
/// report bind failures before any background task exists. The server observes both its local
/// shutdown handle and the process-wide [`crate::ShutdownTrigger`].
///
/// # Errors
/// Returns a sanitized error if reflection descriptors cannot be assembled.
pub async fn spawn_grpc_server(
    listener: TcpListener,
    state: AppState,
) -> Result<GrpcServerHandle, GrpcServerError> {
    let local_addr = listener
        .local_addr()
        .map_err(|_| GrpcServerError::Transport)?;
    let limits = state.limits().clone();
    let service = GrpcEmbeddingService::new(state.clone());
    let auth_state = state.clone();
    let embedding = EmbeddingServiceServer::new(service)
        .max_decoding_message_size(limits.max_body_bytes)
        .max_encoding_message_size(limits.max_body_bytes);
    let embedding =
        tonic::service::interceptor::InterceptedService::new(embedding, move |request| {
            authorize(request, &auth_state)
        });

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    publish_health(&health_reporter, state.application().readiness().is_ready()).await;
    let reflection = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(v1::FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
        .build_v1()
        .map_err(|_| GrpcServerError::Reflection)?;

    let (shutdown, mut shutdown_rx) = watch::channel(false);
    let health_shutdown_rx = shutdown_rx.clone();
    let process_shutdown = state.shutdown_trigger().clone();
    let health_process_shutdown = process_shutdown.clone();
    let health_state = state;
    let join = tokio::spawn(async move {
        let shutdown_signal = async move {
            tokio::select! {
                () = wait_for_shutdown(&mut shutdown_rx) => {}
                () = process_shutdown.cancelled() => {}
            }
        };
        let server = Server::builder()
            .add_service(reflection)
            .add_service(health_service)
            .add_service(embedding)
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown_signal);
        let health = track_health(
            health_reporter,
            health_state,
            health_shutdown_rx,
            health_process_shutdown,
        );
        let (server_result, ()) = tokio::join!(server, health);
        server_result.map_err(|_| GrpcServerError::Transport)
    });

    Ok(GrpcServerHandle {
        local_addr,
        shutdown_timeout: limits.shutdown_timeout,
        shutdown,
        join: Some(join),
    })
}

async fn publish_health(reporter: &tonic_health::server::HealthReporter, ready: bool) {
    let status = if ready {
        tonic_health::ServingStatus::Serving
    } else {
        tonic_health::ServingStatus::NotServing
    };
    reporter.set_service_status("", status).await;
    reporter
        .set_service_status(
            <EmbeddingServiceServer<GrpcEmbeddingService> as tonic::server::NamedService>::NAME,
            status,
        )
        .await;
}

async fn track_health(
    reporter: tonic_health::server::HealthReporter,
    state: AppState,
    mut shutdown: watch::Receiver<bool>,
    process_shutdown: crate::ShutdownTrigger,
) {
    let mut last_ready = state.application().readiness().is_ready();
    let mut interval = tokio::time::interval(HEALTH_REFRESH_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            () = wait_for_shutdown(&mut shutdown) => break,
            () = process_shutdown.cancelled() => break,
            _ = interval.tick() => {
                let ready = state.application().readiness().is_ready();
                if ready != last_ready {
                    publish_health(&reporter, ready).await;
                    last_ready = ready;
                }
            }
        }
    }
    publish_health(&reporter, false).await;
}

async fn wait_for_shutdown(receiver: &mut watch::Receiver<bool>) {
    while !*receiver.borrow() {
        if receiver.changed().await.is_err() {
            return;
        }
    }
}

#[derive(Clone, Debug)]
struct GrpcEmbeddingService {
    state: AppState,
}

impl GrpcEmbeddingService {
    const fn new(state: AppState) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl EmbeddingService for GrpcEmbeddingService {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let caller_timeout = parse_grpc_timeout(request.metadata())?;
        if caller_timeout == Some(Duration::ZERO) {
            return Err(status_for_public(&PublicError::for_code(
                ErrorCode::DeadlineExceeded,
            )));
        }
        let request = request.into_inner();
        let options = decode_options(&request)?;
        let cancellation = CancellationToken::default();
        let mut disconnect = DisconnectGuard::new(cancellation.clone());
        let effective_timeout =
            effective_timeout(caller_timeout, self.state.limits().request_timeout);
        let application = self.state.application().embed(EmbedCommand {
            model: request.model,
            input: request.input,
            options,
            cancellation: cancellation.clone(),
            timeout: caller_timeout,
        });
        tokio::pin!(application);
        let output = tokio::select! {
            biased;
            result = &mut application => result.map_err(|error| status_for_application(&error))?,
            () = tokio::time::sleep(effective_timeout) => {
                cancellation.cancel();
                disconnect.disarm();
                return Err(status_for_public(&PublicError::for_code(ErrorCode::DeadlineExceeded)));
            }
        };
        disconnect.disarm();
        Ok(Response::new(encode_response(output)))
    }
}

fn authorize(request: Request<()>, state: &AppState) -> Result<Request<()>, Status> {
    if request.metadata().contains_key(AUTHORIZATION_BIN) {
        return Err(Status::unauthenticated("authentication required"));
    }
    let mut values = request.metadata().get_all(AUTHORIZATION).iter();
    let value = values.next();
    if values.next().is_some() {
        return Err(Status::unauthenticated("authentication required"));
    }
    let Some(value) = value else {
        return if state.public_auth_required() {
            Err(Status::unauthenticated("authentication required"))
        } else {
            Ok(request)
        };
    };
    let value = value
        .to_str()
        .map_err(|_| Status::unauthenticated("authentication required"))?;
    let Some(token) = value.strip_prefix("Bearer ") else {
        return Err(Status::unauthenticated("authentication required"));
    };
    if token.is_empty()
        || (state.public_auth_required() && !state.verify_public_token(token.as_bytes()))
    {
        return Err(Status::unauthenticated("authentication required"));
    }
    Ok(request)
}

fn parse_grpc_timeout(metadata: &MetadataMap) -> Result<Option<Duration>, Status> {
    if metadata.contains_key(GRPC_TIMEOUT_BIN) {
        return Err(invalid_request_status());
    }
    let mut values = metadata.get_all(GRPC_TIMEOUT).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(invalid_request_status());
    }
    let value = value.to_str().map_err(|_| invalid_request_status())?;
    if value.len() < 2 || value.len() > 9 {
        return Err(invalid_request_status());
    }
    let (digits, unit) = value.split_at(value.len() - 1);
    if digits.len() > 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_request_status());
    }
    let amount = digits
        .parse::<u64>()
        .map_err(|_| invalid_request_status())?;
    let duration = match unit {
        "H" => Duration::from_secs(amount.saturating_mul(60 * 60)),
        "M" => Duration::from_secs(amount.saturating_mul(60)),
        "S" => Duration::from_secs(amount),
        "m" => Duration::from_millis(amount),
        "u" => Duration::from_micros(amount),
        "n" => Duration::from_nanos(amount),
        _ => return Err(invalid_request_status()),
    };
    Ok(Some(duration))
}

fn effective_timeout(caller: Option<Duration>, server: Duration) -> Duration {
    caller.unwrap_or(server).min(server)
}

fn decode_options(request: &EmbedRequest) -> Result<EmbedOptions, Status> {
    let task = match ProtoEmbeddingTask::try_from(request.task) {
        Ok(ProtoEmbeddingTask::Unspecified | ProtoEmbeddingTask::Document) => {
            EmbeddingTask::Document
        }
        Ok(ProtoEmbeddingTask::Query) => EmbeddingTask::Query,
        Err(_) => return Err(invalid_request_status()),
    };
    let truncation = match ProtoTruncation::try_from(request.truncation) {
        Ok(ProtoTruncation::Unspecified | ProtoTruncation::Reject) => CoreTruncation::Reject,
        Ok(ProtoTruncation::Truncate) => CoreTruncation::Truncate,
        Err(_) => return Err(invalid_request_status()),
    };
    let dimensions = request
        .dimensions
        .map(|value| usize::try_from(value).map_err(|_| invalid_request_status()))
        .transpose()?;
    Ok(EmbedOptions {
        task,
        truncation,
        dimensions,
        normalize: request.normalize,
    })
}

fn encode_response(output: impossible_embedding_core::EmbeddingOutput) -> EmbedResponse {
    let identity = output.model;
    let usage = output.usage;
    EmbedResponse {
        embeddings: output
            .vectors
            .into_iter()
            .map(|values| DenseVector { values })
            .collect(),
        model: Some(ResolvedModelIdentity {
            canonical_id: identity.canonical_id,
            revision: identity.revision,
            runtime: identity.runtime,
            artifact_fingerprint: identity.artifact_fingerprint,
            semantic_fingerprint: identity.semantic_fingerprint,
        }),
        usage: Some(TokenUsage {
            prompt_tokens: usage.total_tokens(),
            total_tokens: usage.total_tokens(),
            input_tokens: usage.input_tokens().to_vec(),
        }),
    }
}

fn invalid_request_status() -> Status {
    status_for_public(&PublicError::for_code(ErrorCode::InvalidRequest))
}

fn status_for_application(error: &ApplicationError) -> Status {
    status_for_public(error.public_error())
}

fn status_for_public(error: &PublicError) -> Status {
    let details = public_error_detail(error).encode_to_vec();
    Status::with_details(grpc_status_code(error.code), error.message, details.into())
}

struct DisconnectGuard {
    cancellation: Option<CancellationToken>,
}

impl DisconnectGuard {
    const fn new(cancellation: CancellationToken) -> Self {
        Self {
            cancellation: Some(cancellation),
        }
    }

    fn disarm(&mut self) {
        self.cancellation = None;
    }
}

impl Drop for DisconnectGuard {
    fn drop(&mut self) {
        if let Some(cancellation) = self.cancellation.take() {
            cancellation.cancel();
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use impossible_embedding_core::{EmbeddingOutput, EmbeddingUsage, ResolvedModelIdentity};
    use impossible_protocol::v1::{
        ErrorCode as ProtoErrorCode, embedding_service_client::EmbeddingServiceClient,
    };
    use impossible_server_core::{
        LifecycleState, ModelKey, ModelState, ReadinessReason, ServerConfig,
        config::CredentialSource,
    };
    use tonic::{Code, metadata::MetadataValue, transport::Endpoint};
    use tonic_health::pb::{
        HealthCheckRequest, health_check_response::ServingStatus, health_client::HealthClient,
    };
    use tonic_reflection::pb::v1::{
        ServerReflectionRequest, server_reflection_client::ServerReflectionClient,
        server_reflection_request::MessageRequest, server_reflection_response::MessageResponse,
    };

    use super::*;

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture_config(authenticated: bool) -> (ServerConfig, std::path::PathBuf) {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("impossible-grpc-test-{}-{id}", std::process::id()));
        fs::create_dir_all(&directory).expect("create fixture directory");
        let mut config = ServerConfig {
            cache_directory: directory.join("cache"),
            ..ServerConfig::default()
        };
        if authenticated {
            let credential = directory.join("public-token");
            fs::write(&credential, b"grpc-test-token").expect("write credential");
            config.auth = Some(CredentialSource::File(credential));
        }
        (config, directory)
    }

    async fn start(authenticated: bool) -> (GrpcServerHandle, std::path::PathBuf) {
        let (config, directory) = fixture_config(authenticated);
        let state = AppState::new(&config).expect("construct app state");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture");
        let server = spawn_grpc_server(listener, state)
            .await
            .expect("start server");
        (server, directory)
    }

    async fn channel(address: SocketAddr) -> tonic::transport::Channel {
        Endpoint::from_shared(format!("http://{address}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect")
    }

    fn request() -> EmbedRequest {
        EmbedRequest {
            model: "bge-small-en".into(),
            input: vec!["hello".into()],
            task: ProtoEmbeddingTask::Document as i32,
            truncation: ProtoTruncation::Reject as i32,
            dimensions: None,
            normalize: None,
        }
    }

    #[test]
    fn request_conversion_rejects_unknown_enums_and_maps_defaults() {
        let defaults = EmbedRequest {
            task: 0,
            truncation: 0,
            ..request()
        };
        assert_eq!(
            decode_options(&defaults).expect("defaults"),
            EmbedOptions::default()
        );
        let invalid_task = EmbedRequest {
            task: 99,
            ..request()
        };
        assert_eq!(
            decode_options(&invalid_task)
                .expect_err("unknown task")
                .code(),
            Code::InvalidArgument
        );
        let invalid_truncation = EmbedRequest {
            truncation: 99,
            ..request()
        };
        assert_eq!(
            decode_options(&invalid_truncation)
                .expect_err("unknown truncation")
                .code(),
            Code::InvalidArgument
        );
    }

    #[test]
    fn timeout_parser_is_strict_and_bounded() {
        let mut metadata = MetadataMap::new();
        metadata.insert(GRPC_TIMEOUT, "25m".parse().expect("metadata"));
        assert_eq!(
            parse_grpc_timeout(&metadata).expect("timeout"),
            Some(Duration::from_millis(25))
        );
        for invalid in ["", "1", "123456789S", "4x", "-1S", " 1S"] {
            let mut metadata = MetadataMap::new();
            if let Ok(value) = invalid.parse() {
                metadata.insert(GRPC_TIMEOUT, value);
                assert_eq!(
                    parse_grpc_timeout(&metadata)
                        .expect_err("invalid timeout")
                        .code(),
                    Code::InvalidArgument
                );
            }
        }
        let mut duplicate = MetadataMap::new();
        duplicate.append(GRPC_TIMEOUT, "1S".parse().expect("metadata"));
        duplicate.append(GRPC_TIMEOUT, "2S".parse().expect("metadata"));
        assert_eq!(
            parse_grpc_timeout(&duplicate)
                .expect_err("duplicate timeout")
                .code(),
            Code::InvalidArgument
        );
        let mut binary = MetadataMap::new();
        binary.insert_bin(GRPC_TIMEOUT_BIN, MetadataValue::from_bytes(b"1S"));
        assert_eq!(
            parse_grpc_timeout(&binary)
                .expect_err("binary timeout")
                .code(),
            Code::InvalidArgument
        );
        assert_eq!(
            effective_timeout(Some(Duration::from_millis(5)), Duration::from_secs(1)),
            Duration::from_millis(5)
        );
        assert_eq!(
            effective_timeout(Some(Duration::from_secs(2)), Duration::from_secs(1)),
            Duration::from_secs(1)
        );
        assert_eq!(
            effective_timeout(None, Duration::from_secs(1)),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn disconnect_guard_owns_only_its_private_token() {
        let private = CancellationToken::default();
        let unrelated = CancellationToken::default();
        drop(DisconnectGuard::new(private.clone()));
        assert!(private.is_cancelled());
        assert!(!unrelated.is_cancelled());

        let retained = CancellationToken::default();
        let mut guard = DisconnectGuard::new(retained.clone());
        guard.disarm();
        drop(guard);
        assert!(!retained.is_cancelled());
    }

    #[test]
    fn response_and_error_details_preserve_the_wire_contract() {
        let output = EmbeddingOutput {
            vectors: vec![vec![1.0, 2.0]],
            model: ResolvedModelIdentity::new("id", "revision", "runtime", "artifact", "semantic")
                .expect("identity"),
            usage: EmbeddingUsage::new(vec![3]).expect("usage"),
        };
        let response = encode_response(output);
        assert_eq!(response.embeddings[0].values, [1.0, 2.0]);
        assert_eq!(response.usage.expect("usage").input_tokens, [3]);

        let status = invalid_request_status();
        let detail = v1::PublicErrorDetail::decode(status.details()).expect("detail");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(detail.code, ProtoErrorCode::InvalidRequest as i32);
        assert_eq!(detail.message, "the embedding request is invalid");
    }

    #[tokio::test]
    async fn ephemeral_server_enforces_exact_auth_and_structured_errors() {
        let (server, directory) = start(true).await;
        let mut client = EmbeddingServiceClient::new(channel(server.local_addr()).await);

        let missing = client.embed(request()).await.expect_err("missing auth");
        assert_eq!(missing.code(), Code::Unauthenticated);

        let mut wrong_scheme = Request::new(request());
        wrong_scheme.metadata_mut().insert(
            AUTHORIZATION,
            "bearer grpc-test-token".parse().expect("metadata"),
        );
        assert_eq!(
            client
                .embed(wrong_scheme)
                .await
                .expect_err("wrong scheme")
                .code(),
            Code::Unauthenticated
        );

        let mut binary = Request::new(request());
        binary.metadata_mut().insert_bin(
            AUTHORIZATION_BIN,
            MetadataValue::from_bytes(b"grpc-test-token"),
        );
        assert_eq!(
            client.embed(binary).await.expect_err("binary auth").code(),
            Code::Unauthenticated
        );

        let mut duplicate = Request::new(request());
        duplicate.metadata_mut().append(
            AUTHORIZATION,
            "Bearer grpc-test-token".parse().expect("metadata"),
        );
        duplicate.metadata_mut().append(
            AUTHORIZATION,
            "Bearer grpc-test-token".parse().expect("metadata"),
        );
        assert_eq!(
            client
                .embed(duplicate)
                .await
                .expect_err("duplicate auth")
                .code(),
            Code::Unauthenticated
        );

        let mut valid = Request::new(request());
        valid.metadata_mut().insert(
            AUTHORIZATION,
            "Bearer grpc-test-token".parse().expect("metadata"),
        );
        let unavailable = client.embed(valid).await.expect_err("model is not loaded");
        assert_eq!(unavailable.code(), Code::Unavailable);
        let detail = v1::PublicErrorDetail::decode(unavailable.details()).expect("public detail");
        assert_eq!(detail.code, ProtoErrorCode::ModelUnavailable as i32);

        let mut unknown_enum = Request::new(EmbedRequest {
            task: 99,
            ..request()
        });
        unknown_enum.metadata_mut().insert(
            AUTHORIZATION,
            "Bearer grpc-test-token".parse().expect("metadata"),
        );
        let unknown = client.embed(unknown_enum).await.expect_err("unknown enum");
        assert_eq!(unknown.code(), Code::InvalidArgument);
        let detail = v1::PublicErrorDetail::decode(unknown.details()).expect("public detail");
        assert_eq!(detail.code, ProtoErrorCode::InvalidRequest as i32);

        let mut malformed_timeout = Request::new(request());
        malformed_timeout.metadata_mut().insert(
            AUTHORIZATION,
            "Bearer grpc-test-token".parse().expect("metadata"),
        );
        malformed_timeout
            .metadata_mut()
            .insert(GRPC_TIMEOUT, "broken".parse().expect("metadata"));
        let malformed = client
            .embed(malformed_timeout)
            .await
            .expect_err("malformed timeout");
        assert_eq!(malformed.code(), Code::InvalidArgument);

        server.shutdown().await.expect("graceful shutdown");
        fs::remove_dir_all(directory).expect("remove fixture");
    }

    #[tokio::test]
    async fn health_and_reflection_publish_standard_descriptors() -> Result<(), &'static str> {
        let (config, directory) = fixture_config(false);
        let state = AppState::new(&config).expect("construct app state");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture");
        let server = spawn_grpc_server(listener, state.clone())
            .await
            .expect("start server");
        let channel = channel(server.local_addr()).await;
        let mut health = HealthClient::new(channel.clone());
        let initial = health
            .check(HealthCheckRequest {
                service: "impossible.embedding.v1.EmbeddingService".into(),
            })
            .await
            .expect("health check")
            .into_inner();
        assert_eq!(initial.status, ServingStatus::NotServing as i32);

        state
            .application()
            .health()
            .set_model(ModelKey(1), ModelState::Ready);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        loop {
            let current = health
                .check(HealthCheckRequest {
                    service: "impossible.embedding.v1.EmbeddingService".into(),
                })
                .await
                .expect("health check")
                .into_inner();
            if current.status == ServingStatus::Serving as i32 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "health became ready"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let mut reflection = ServerReflectionClient::new(channel);
        let requests = tokio_stream::iter([ServerReflectionRequest {
            host: String::new(),
            message_request: Some(MessageRequest::ListServices(String::new())),
        }]);
        let mut responses = reflection
            .server_reflection_info(requests)
            .await
            .expect("reflection call")
            .into_inner();
        let response = responses
            .message()
            .await
            .expect("reflection stream")
            .expect("reflection response");
        let Some(MessageResponse::ListServicesResponse(services)) = response.message_response
        else {
            return Err("unexpected reflection response");
        };
        let mut names = services
            .service
            .into_iter()
            .map(|service| service.name)
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            [
                "grpc.health.v1.Health".to_owned(),
                "grpc.reflection.v1.ServerReflection".to_owned(),
                "impossible.embedding.v1.EmbeddingService".to_owned(),
            ]
        );

        assert!(
            state
                .application()
                .health()
                .transition(LifecycleState::Draining, Some(ReadinessReason::Draining))
        );
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        loop {
            let current = health
                .check(HealthCheckRequest {
                    service: String::new(),
                })
                .await
                .expect("overall health check")
                .into_inner();
            if current.status == ServingStatus::NotServing as i32 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "health reflected drain"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        server.shutdown().await.expect("graceful shutdown");
        fs::remove_dir_all(directory).expect("remove fixture");
        Ok(())
    }

    #[tokio::test]
    async fn configured_message_limit_rejects_oversized_requests() {
        let (mut config, directory) = fixture_config(false);
        config.limits.max_body_bytes = 64;
        config.limits.max_request_bytes = 64;
        config.limits.max_input_bytes = 64;
        let state = AppState::new(&config).expect("construct app state");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fixture");
        let server = spawn_grpc_server(listener, state)
            .await
            .expect("start server");
        let mut client = EmbeddingServiceClient::new(channel(server.local_addr()).await);
        let oversized = EmbedRequest {
            input: vec!["x".repeat(512)],
            ..request()
        };
        let status = client
            .embed(oversized)
            .await
            .expect_err("oversized request");
        assert!(matches!(
            status.code(),
            Code::OutOfRange | Code::ResourceExhausted
        ));
        server.shutdown().await.expect("graceful shutdown");
        fs::remove_dir_all(directory).expect("remove fixture");
    }
}

//! HTTP v1 adapter for the transport-independent embedding application.

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{
        HeaderMap, HeaderValue, Method, StatusCode,
        header::{
            ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS,
            ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_REQUEST_HEADERS,
            ACCESS_CONTROL_REQUEST_METHOD, AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, ORIGIN,
            VARY,
        },
    },
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingOutput, EmbeddingTask, ErrorCode,
    ModelVerificationStatus, PublicError, Retryability, Truncation,
};
use impossible_server_core::{
    ReadinessReason,
    metrics::{Operation, Outcome},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::{
    AppState, ApplicationError, ApplicationErrorKind, EmbedCommand, LoadStatus, ModelInfo,
};

const OPENAPI: &[u8] = include_bytes!("../../../docs/openapi-v1.json");
const JSON_CONTENT_TYPE: &str = "application/json";

/// Build the complete HTTP v1 adapter without opening a listener.
///
/// Administrative model routes are absent when disabled. Metrics always uses the separate
/// administrative credential, while inference and model listing use the public credential.
pub fn router(state: AppState) -> Router {
    let public = Router::new()
        .route("/v1/embeddings", post(openai_embeddings))
        .route("/v1/embed", post(native_embeddings))
        .route("/v1/models", get(list_models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_public_auth,
        ));

    let mut app = Router::new()
        .route("/", get(status_page))
        .route("/health/live", get(live_health))
        .route("/health/ready", get(ready_health))
        .route("/openapi.json", get(openapi))
        .merge(public)
        .route(
            "/metrics",
            get(metrics).route_layer(middleware::from_fn_with_state(
                state.clone(),
                require_admin_auth,
            )),
        );

    if state.admin_api_enabled() {
        let admin = Router::new()
            .route("/v1/admin/models/install", post(admin_install))
            .route("/v1/admin/models/load", post(admin_load))
            .route("/v1/admin/models/unload", post(admin_unload))
            .route("/v1/admin/models/delete", post(admin_delete))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                require_admin_auth,
            ));
        app = app.merge(admin);
    }

    app.with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, origin_layer))
}

async fn origin_layer(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let flavor = flavor_for_path(request.uri().path());
    let mut origins = request.headers().get_all(ORIGIN).iter();
    let origin_header = origins.next();
    if origins.next().is_some() {
        return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
    }
    let origin = origin_header.and_then(|value| value.to_str().ok());
    if origin_header.is_some() && origin.is_none() {
        return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
    }
    if !state.origins().allows(origin) {
        return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
    }

    let response_origin = state.origins().response_origin(origin).map(str::to_owned);
    if request.method() == Method::OPTIONS {
        let Some(origin) = response_origin.as_deref() else {
            return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
        };
        let Some(expected_method) = route_method(request.uri().path(), state.admin_api_enabled())
        else {
            return StatusCode::NOT_FOUND.into_response();
        };
        let Some(requested_method) =
            single_header(request.headers(), ACCESS_CONTROL_REQUEST_METHOD)
                .and_then(|value| value.to_str().ok())
        else {
            return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
        };
        if requested_method != expected_method.as_str()
            || !preflight_headers_allowed(request.headers())
        {
            return error_response(flavor, StatusCode::FORBIDDEN, ErrorCode::InvalidRequest);
        }
        let mut response = StatusCode::NO_CONTENT.into_response();
        apply_preflight(response.headers_mut(), origin, &expected_method);
        return response;
    }

    let mut response = next.run(request).await;
    apply_origin(response.headers_mut(), response_origin.as_deref());
    response
}

fn apply_origin(headers: &mut HeaderMap, origin: Option<&str>) {
    let Some(origin) = origin.and_then(|value| HeaderValue::from_str(value).ok()) else {
        return;
    };
    headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    headers.insert(VARY, HeaderValue::from_static("Origin"));
}

fn apply_preflight(headers: &mut HeaderMap, origin: &str, method: &Method) {
    apply_origin(headers, Some(origin));
    headers.insert(
        ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_str(method.as_str()).unwrap_or(HeaderValue::from_static("GET")),
    );
    headers.insert(
        ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Authorization, Content-Type"),
    );
}

fn route_method(path: &str, admin_enabled: bool) -> Option<Method> {
    match path {
        "/" | "/v1/models" | "/health/live" | "/health/ready" | "/metrics" | "/openapi.json" => {
            Some(Method::GET)
        }
        "/v1/embeddings" | "/v1/embed" => Some(Method::POST),
        "/v1/admin/models/install"
        | "/v1/admin/models/load"
        | "/v1/admin/models/unload"
        | "/v1/admin/models/delete"
            if admin_enabled =>
        {
            Some(Method::POST)
        }
        _ => None,
    }
}

fn single_header(
    headers: &HeaderMap,
    name: axum::http::header::HeaderName,
) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn preflight_headers_allowed(headers: &HeaderMap) -> bool {
    let values = headers.get_all(ACCESS_CONTROL_REQUEST_HEADERS);
    let mut found = false;
    for value in &values {
        if found {
            return false;
        }
        found = true;
        let Ok(value) = value.to_str() else {
            return false;
        };
        if value.split(',').any(|header| {
            !matches!(
                header.trim().to_ascii_lowercase().as_str(),
                "authorization" | "content-type"
            )
        }) {
            return false;
        }
    }
    true
}

async fn require_public_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if state.public_auth_required()
        && !bearer_candidate(request.headers())
            .is_some_and(|candidate| state.verify_public_token(candidate))
    {
        return error_response(
            flavor_for_path(request.uri().path()),
            StatusCode::UNAUTHORIZED,
            ErrorCode::InvalidRequest,
        );
    }
    next.run(request).await
}

async fn require_admin_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if state.admin_auth_required()
        && !bearer_candidate(request.headers())
            .is_some_and(|candidate| state.verify_admin_token(candidate))
    {
        return wire_error(StatusCode::UNAUTHORIZED, ErrorCode::InvalidRequest);
    }
    next.run(request).await
}

fn bearer_candidate(headers: &HeaderMap) -> Option<&[u8]> {
    single_header(headers, AUTHORIZATION)?
        .as_bytes()
        .strip_prefix(b"Bearer ")
        .filter(|candidate| !candidate.is_empty())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OpenAiInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiRequest {
    model: String,
    input: OpenAiInput,
    #[serde(default, rename = "encoding_format")]
    _encoding_format: Option<FloatEncoding>,
    #[serde(default)]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum FloatEncoding {
    Float,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRequest {
    model: String,
    input: Vec<String>,
    #[serde(default)]
    task: Task,
    #[serde(default)]
    truncation: TruncationMode,
    #[serde(default)]
    dimensions: Option<u32>,
    #[serde(default)]
    normalize: Option<bool>,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Task {
    Query,
    #[default]
    Document,
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TruncationMode {
    #[default]
    Reject,
    Truncate,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelCommand {
    model: String,
}

async fn openai_embeddings(State(state): State<AppState>, request: Request) -> Response {
    let request: OpenAiRequest =
        match decode_json(&state, request, ErrorFlavor::Compatibility).await {
            Ok(value) => value,
            Err(response) => return response,
        };
    let inputs = match request.input {
        OpenAiInput::One(value) => vec![value],
        OpenAiInput::Many(values) => values,
    };
    if let Err(response) =
        validate_inputs(&state, &request.model, &inputs, ErrorFlavor::Compatibility)
    {
        return response;
    }
    let Ok(dimensions) = request.dimensions.map(usize::try_from).transpose() else {
        return compatibility_error(StatusCode::BAD_REQUEST, ErrorCode::InvalidRequest);
    };
    let result = run_embed(
        &state,
        request.model,
        inputs,
        EmbedOptions {
            dimensions,
            ..EmbedOptions::default()
        },
    )
    .await;
    match result {
        Ok(output) => json_response(StatusCode::OK, &OpenAiResponse::from(output)),
        Err(error) => application_error(&error, ErrorFlavor::Compatibility),
    }
}

async fn native_embeddings(State(state): State<AppState>, request: Request) -> Response {
    let request: NativeRequest = match decode_json(&state, request, ErrorFlavor::Native).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    if let Err(response) =
        validate_inputs(&state, &request.model, &request.input, ErrorFlavor::Native)
    {
        return response;
    }
    let Ok(dimensions) = request.dimensions.map(usize::try_from).transpose() else {
        return wire_error(StatusCode::BAD_REQUEST, ErrorCode::InvalidRequest);
    };
    let options = EmbedOptions {
        task: match request.task {
            Task::Query => EmbeddingTask::Query,
            Task::Document => EmbeddingTask::Document,
        },
        truncation: match request.truncation {
            TruncationMode::Reject => Truncation::Reject,
            TruncationMode::Truncate => Truncation::Truncate,
        },
        dimensions,
        normalize: request.normalize,
    };
    let result = run_embed(&state, request.model, request.input, options).await;
    match result {
        Ok(output) => json_response(StatusCode::OK, &NativeResponse::from(output)),
        Err(error) => application_error(&error, ErrorFlavor::Native),
    }
}

async fn run_embed(
    state: &AppState,
    model: String,
    input: Vec<String>,
    options: EmbedOptions,
) -> Result<EmbeddingOutput, ApplicationError> {
    let cancellation = CancellationToken::default();
    let mut guard = CancelOnDrop(Some(cancellation.clone()));
    let result = state
        .application()
        .embed(EmbedCommand {
            model,
            input,
            options,
            cancellation,
            timeout: None,
        })
        .await;
    guard.0 = None;
    result
}

struct CancelOnDrop(Option<CancellationToken>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            token.cancel();
        }
    }
}

async fn decode_json<T: for<'de> Deserialize<'de>>(
    state: &AppState,
    request: Request,
    flavor: ErrorFlavor,
) -> Result<T, Response> {
    if single_header(request.headers(), CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_none_or(|value| {
            value
                .split(';')
                .next()
                .is_none_or(|media_type| !media_type.trim().eq_ignore_ascii_case(JSON_CONTENT_TYPE))
        })
    {
        return Err(error_response(
            flavor,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ErrorCode::InvalidRequest,
        ));
    }
    if request.headers().contains_key(CONTENT_LENGTH) {
        let Some(length) = single_header(request.headers(), CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        else {
            return Err(error_response(
                flavor,
                StatusCode::BAD_REQUEST,
                ErrorCode::InvalidRequest,
            ));
        };
        if length > state.limits().max_body_bytes as u64 {
            return Err(error_response(
                flavor,
                StatusCode::PAYLOAD_TOO_LARGE,
                ErrorCode::InvalidRequest,
            ));
        }
    }
    let body = to_bytes(request.into_body(), state.limits().max_body_bytes)
        .await
        .map_err(|_| {
            error_response(
                flavor,
                StatusCode::PAYLOAD_TOO_LARGE,
                ErrorCode::InvalidRequest,
            )
        })?;
    serde_json::from_slice(&body)
        .map_err(|_| error_response(flavor, StatusCode::BAD_REQUEST, ErrorCode::InvalidRequest))
}

fn validate_inputs(
    state: &AppState,
    model: &str,
    inputs: &[String],
    flavor: ErrorFlavor,
) -> Result<(), Response> {
    if model.is_empty() || inputs.is_empty() || inputs.len() > state.limits().max_items {
        return Err(error_response(
            flavor,
            StatusCode::BAD_REQUEST,
            ErrorCode::InvalidRequest,
        ));
    }
    let total = inputs.iter().try_fold(0_usize, |total, input| {
        if input.len() > state.limits().max_input_bytes {
            return None;
        }
        total.checked_add(input.len())
    });
    if total.is_none_or(|bytes| bytes > state.limits().max_request_bytes) {
        return Err(error_response(
            flavor,
            StatusCode::PAYLOAD_TOO_LARGE,
            ErrorCode::InvalidRequest,
        ));
    }
    Ok(())
}

async fn list_models(State(state): State<AppState>) -> Response {
    let started = Instant::now();
    let result = state.application().list_models();
    observe(&state, Operation::ListModels, &result, started);
    match result {
        Ok(models) => json_response(
            StatusCode::OK,
            &ModelList {
                data: models.into_iter().map(ModelSummary::from).collect(),
            },
        ),
        Err(error) => application_error(&error, ErrorFlavor::Native),
    }
}

async fn admin_install(State(state): State<AppState>, request: Request) -> Response {
    admin_command(state, request, AdminAction::Install).await
}

async fn admin_load(State(state): State<AppState>, request: Request) -> Response {
    admin_command(state, request, AdminAction::Load).await
}

async fn admin_unload(State(state): State<AppState>, request: Request) -> Response {
    admin_command(state, request, AdminAction::Unload).await
}

async fn admin_delete(State(state): State<AppState>, request: Request) -> Response {
    admin_command(state, request, AdminAction::Delete).await
}

#[derive(Clone, Copy)]
enum AdminAction {
    Install,
    Load,
    Unload,
    Delete,
}

async fn admin_command(state: AppState, request: Request, action: AdminAction) -> Response {
    let command: ModelCommand = match decode_json(&state, request, ErrorFlavor::Native).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    if command.model.is_empty() {
        return wire_error(StatusCode::BAD_REQUEST, ErrorCode::InvalidRequest);
    }
    let started = Instant::now();
    let result = match action {
        AdminAction::Install => state.application().install(&command.model).await,
        AdminAction::Load => state.application().load(&command.model).await,
        AdminAction::Unload => state.application().unload(&command.model).await,
        AdminAction::Delete => state.application().delete(&command.model).await,
    };
    observe(&state, Operation::AdminModel, &result, started);
    match result {
        Ok(model) => json_response(
            if matches!(action, AdminAction::Install) {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            },
            &ModelSummary::from(model),
        ),
        Err(error) => application_error(&error, ErrorFlavor::Native),
    }
}

async fn live_health(State(state): State<AppState>) -> Response {
    let started = Instant::now();
    let live = state.application().health().is_live();
    state.application().metrics().observe(
        Operation::Health,
        if live { Outcome::Ok } else { Outcome::Failed },
        started.elapsed(),
    );
    json_response(
        if live {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        &HealthResponse {
            status: if live { "ok" } else { "unavailable" },
            reason: None,
        },
    )
}

async fn ready_health(State(state): State<AppState>) -> Response {
    let started = Instant::now();
    let readiness = state.application().readiness();
    let ready = readiness.is_ready();
    state.application().metrics().observe(
        Operation::Health,
        if ready { Outcome::Ok } else { Outcome::Failed },
        started.elapsed(),
    );
    json_response(
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        &HealthResponse {
            status: if ready { "ok" } else { "unavailable" },
            reason: readiness.reason_code.map(ReadinessReason::code),
        },
    )
}

async fn metrics(State(state): State<AppState>) -> Response {
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        state.application().metrics().render(),
    )
        .into_response()
}

async fn openapi() -> Response {
    (
        StatusCode::OK,
        [(CONTENT_TYPE, "application/json")],
        Body::from(OPENAPI),
    )
        .into_response()
}

async fn status_page(State(state): State<AppState>) -> Html<String> {
    let readiness = state.application().readiness();
    let (ready_models, total_models) = state.application().health().model_counts();
    let status = if readiness.is_ready() {
        "Ready"
    } else {
        "Unavailable"
    };
    let reason = readiness.reason_code.map_or("none", ReadinessReason::code);
    Html(format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Impossible Embedding</title><style>body{{font:16px system-ui;margin:3rem;max-width:48rem;background:#101416;color:#edf7f4}}main{{border-left:.4rem solid #ff746c;padding:1rem 1.5rem;background:#182023}}dt{{color:#9cb0aa}}dd{{margin:0 0 1rem}}</style></head><body><main><h1>Impossible Embedding</h1><dl><dt>Status</dt><dd>{status}</dd><dt>Reason</dt><dd>{reason}</dd><dt>Ready models</dt><dd>{ready_models} of {total_models}</dd><dt>Version</dt><dd>{}</dd></dl></main></body></html>",
        state.version()
    ))
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: &'static str,
    retryable: bool,
}

#[derive(Clone, Copy)]
enum ErrorFlavor {
    Native,
    Compatibility,
}

fn flavor_for_path(path: &str) -> ErrorFlavor {
    if path == "/v1/embeddings" {
        ErrorFlavor::Compatibility
    } else {
        ErrorFlavor::Native
    }
}

fn error_response(flavor: ErrorFlavor, status: StatusCode, code: ErrorCode) -> Response {
    match flavor {
        ErrorFlavor::Native => wire_error(status, code),
        ErrorFlavor::Compatibility => compatibility_error(status, code),
    }
}

fn wire_error(status: StatusCode, code: ErrorCode) -> Response {
    let error = PublicError::for_code(code);
    json_response(
        status,
        &ErrorEnvelope {
            error: ErrorBody {
                code: error.code.as_str(),
                message: error.message,
                retryable: matches!(error.retryability, Retryability::Retryable),
            },
        },
    )
}

#[derive(Serialize)]
struct CompatibilityErrorEnvelope {
    error: CompatibilityErrorBody,
}

#[derive(Serialize)]
struct CompatibilityErrorBody {
    message: &'static str,
    r#type: &'static str,
    param: Option<&'static str>,
    code: &'static str,
}

fn compatibility_error(status: StatusCode, code: ErrorCode) -> Response {
    let public = PublicError::for_code(code);
    json_response(
        status,
        &CompatibilityErrorEnvelope {
            error: CompatibilityErrorBody {
                message: public.message,
                r#type: if code == ErrorCode::InvalidRequest {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                param: None,
                code: code.as_str(),
            },
        },
    )
}

fn application_error(error: &ApplicationError, flavor: ErrorFlavor) -> Response {
    let status = match error.kind() {
        ApplicationErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        ApplicationErrorKind::ModelUnavailable | ApplicationErrorKind::ShuttingDown => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        ApplicationErrorKind::Overloaded => StatusCode::TOO_MANY_REQUESTS,
        ApplicationErrorKind::Cancelled => {
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST)
        }
        ApplicationErrorKind::DeadlineExceeded => StatusCode::REQUEST_TIMEOUT,
        ApplicationErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    match flavor {
        ErrorFlavor::Native => {
            let public = error.public_error();
            json_response(
                status,
                &ErrorEnvelope {
                    error: ErrorBody {
                        code: public.code.as_str(),
                        message: public.message,
                        retryable: matches!(public.retryability, Retryability::Retryable),
                    },
                },
            )
        }
        ErrorFlavor::Compatibility => compatibility_error(status, error.public_error().code),
    }
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response {
    match serde_json::to_vec(value) {
        Ok(body) => (status, [(CONTENT_TYPE, JSON_CONTENT_TYPE)], body).into_response(),
        Err(_) => wire_error(StatusCode::INTERNAL_SERVER_ERROR, ErrorCode::Internal),
    }
}

fn observe<T>(
    state: &AppState,
    operation: Operation,
    result: &Result<T, ApplicationError>,
    started: Instant,
) {
    let outcome = match result {
        Ok(_) => Outcome::Ok,
        Err(error) => match error.kind() {
            ApplicationErrorKind::InvalidRequest => Outcome::Invalid,
            ApplicationErrorKind::Overloaded => Outcome::Overloaded,
            ApplicationErrorKind::Cancelled | ApplicationErrorKind::DeadlineExceeded => {
                Outcome::Cancelled
            }
            _ => Outcome::Failed,
        },
    };
    state
        .application()
        .metrics()
        .observe(operation, outcome, started.elapsed());
}

#[derive(Serialize)]
struct OpenAiResponse {
    object: &'static str,
    data: Vec<OpenAiEmbedding>,
    model: String,
    usage: OpenAiUsage,
}

#[derive(Serialize)]
struct OpenAiEmbedding {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct OpenAiUsage {
    prompt_tokens: u64,
    total_tokens: u64,
}

impl From<EmbeddingOutput> for OpenAiResponse {
    fn from(output: EmbeddingOutput) -> Self {
        let total = output.usage.total_tokens();
        Self {
            object: "list",
            data: output
                .vectors
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| OpenAiEmbedding {
                    object: "embedding",
                    index,
                    embedding,
                })
                .collect(),
            model: output.model.canonical_id,
            usage: OpenAiUsage {
                prompt_tokens: total,
                total_tokens: total,
            },
        }
    }
}

#[derive(Serialize)]
struct NativeResponse {
    embeddings: Vec<Vec<f32>>,
    model: NativeIdentity,
    usage: NativeUsage,
}

#[derive(Serialize)]
struct NativeIdentity {
    canonical_id: String,
    revision: String,
    runtime: String,
    artifact_fingerprint: String,
    semantic_fingerprint: String,
}

#[derive(Serialize)]
#[allow(clippy::struct_field_names)]
struct NativeUsage {
    prompt_tokens: u64,
    total_tokens: u64,
    input_tokens: Vec<u64>,
}

impl From<EmbeddingOutput> for NativeResponse {
    fn from(output: EmbeddingOutput) -> Self {
        let total = output.usage.total_tokens();
        Self {
            embeddings: output.vectors,
            model: NativeIdentity {
                canonical_id: output.model.canonical_id,
                revision: output.model.revision,
                runtime: output.model.runtime,
                artifact_fingerprint: output.model.artifact_fingerprint,
                semantic_fingerprint: output.model.semantic_fingerprint,
            },
            usage: NativeUsage {
                prompt_tokens: total,
                total_tokens: total,
                input_tokens: output.usage.input_tokens().to_vec(),
            },
        }
    }
}

#[derive(Serialize)]
struct ModelList {
    data: Vec<ModelSummary>,
}

#[derive(Serialize)]
struct ModelSummary {
    model: String,
    state: &'static str,
}

impl From<ModelInfo> for ModelSummary {
    fn from(model: ModelInfo) -> Self {
        let state = if model.load_status == LoadStatus::Loaded {
            "ready"
        } else {
            match model.installation_status {
                ModelVerificationStatus::Missing => "missing",
                ModelVerificationStatus::Invalid => "invalid",
                ModelVerificationStatus::IntegrityVerified | ModelVerificationStatus::Loadable => {
                    "installed"
                }
            }
        };
        Self {
            model: model.canonical_id,
            state,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use impossible_server_core::{
        ServerConfig,
        config::{CredentialSource, Limits},
    };
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    fn fixture_dir(label: &str) -> std::path::PathBuf {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "impossible-http-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create fixture directory");
        path
    }

    fn state(label: &str) -> AppState {
        AppState::new(&ServerConfig {
            cache_directory: fixture_dir(label),
            ..ServerConfig::default()
        })
        .expect("valid state")
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("JSON response")
    }

    fn json_request(path: &str, body: &'static str) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(CONTENT_TYPE, JSON_CONTENT_TYPE)
            .body(Body::from(body))
            .expect("request")
    }

    #[tokio::test]
    async fn served_openapi_bytes_are_exact_over_ephemeral_listener() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral listener");
        let address = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            axum::serve(listener, router(state("listener")))
                .await
                .expect("HTTP server");
        });
        let mut stream = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect");
        stream
            .write_all(b"GET /openapi.json HTTP/1.1\r\nHost: local\r\nConnection: close\r\n\r\n")
            .await
            .expect("write request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read response");
        server.abort();
        let separator = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("response separator");
        assert!(response.starts_with(b"HTTP/1.1 200 OK\r\n"));
        assert_eq!(&response[separator + 4..], OPENAPI);
    }

    #[tokio::test]
    async fn compatibility_and_native_failures_have_distinct_stable_envelopes() {
        let app = router(state("envelopes"));
        let compatibility = app
            .clone()
            .oneshot(json_request(
                "/v1/embeddings",
                r#"{"model":"bge-small-en","input":"hello","encoding_format":"base64"}"#,
            ))
            .await
            .expect("response");
        assert_eq!(compatibility.status(), StatusCode::BAD_REQUEST);
        let compatibility = body_json(compatibility).await;
        assert_eq!(compatibility["error"]["code"], "invalid_request");
        assert_eq!(compatibility["error"]["type"], "invalid_request_error");
        assert!(compatibility["error"]["param"].is_null());
        assert!(compatibility["error"].get("retryable").is_none());

        let native = app
            .oneshot(json_request(
                "/v1/embed",
                r#"{"model":"bge-small-en","input":["hello"],"unknown":true}"#,
            ))
            .await
            .expect("response");
        assert_eq!(native.status(), StatusCode::BAD_REQUEST);
        let native = body_json(native).await;
        assert_eq!(native["error"]["code"], "invalid_request");
        assert_eq!(native["error"]["retryable"], false);
        assert!(native["error"].get("type").is_none());
    }

    #[tokio::test]
    async fn encoded_body_limit_is_enforced_before_json_decoding() {
        let limits = Limits {
            max_body_bytes: 48,
            max_input_bytes: 32,
            max_request_bytes: 32,
            ..Limits::default()
        };
        let state = AppState::new(&ServerConfig {
            cache_directory: fixture_dir("body-limit"),
            limits,
            ..ServerConfig::default()
        })
        .expect("valid state");
        let response = router(state)
            .oneshot(json_request(
                "/v1/embed",
                r#"{"model":"bge-small-en","input":["this body is deliberately too large"]}"#,
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            body_json(response).await["error"]["code"],
            "invalid_request"
        );
    }

    #[tokio::test]
    async fn decoded_item_and_aggregate_limits_are_enforced() {
        let limits = Limits {
            max_items: 1,
            max_batch_items: 1,
            max_input_bytes: 4,
            max_request_bytes: 6,
            ..Limits::default()
        };
        let state = AppState::new(&ServerConfig {
            cache_directory: fixture_dir("input-limits"),
            limits,
            ..ServerConfig::default()
        })
        .expect("valid state");
        for (body, expected) in [
            (
                r#"{"model":"bge-small-en","input":["a","b"]}"#,
                StatusCode::BAD_REQUEST,
            ),
            (
                r#"{"model":"bge-small-en","input":["12345"]}"#,
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let response = router(state.clone())
                .oneshot(json_request("/v1/embed", body))
                .await
                .expect("response");
            assert_eq!(response.status(), expected);
        }
    }

    #[tokio::test]
    async fn public_and_admin_credentials_are_not_interchangeable() {
        let directory = fixture_dir("auth");
        let public_file = directory.join("public.token");
        let admin_file = directory.join("admin.token");
        fs::write(&public_file, b"public-fixture-token").expect("public credential");
        fs::write(&admin_file, b"admin-fixture-token").expect("admin credential");
        let state = AppState::new(&ServerConfig {
            cache_directory: directory.join("cache"),
            auth: Some(CredentialSource::File(public_file)),
            admin_auth: Some(CredentialSource::File(admin_file)),
            ..ServerConfig::default()
        })
        .expect("valid state");
        let app = router(state);

        let public_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(AUTHORIZATION, "Bearer admin-fixture-token")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(public_denied.status(), StatusCode::UNAUTHORIZED);

        let public_allowed = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(AUTHORIZATION, "Bearer public-fixture-token")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(public_allowed.status(), StatusCode::OK);

        let metrics_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(AUTHORIZATION, "Bearer public-fixture-token")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(metrics_denied.status(), StatusCode::UNAUTHORIZED);

        let metrics_allowed = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(AUTHORIZATION, "Bearer admin-fixture-token")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(metrics_allowed.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cors_is_exact_and_preflight_never_reflects_untrusted_origins() {
        let state = AppState::new(&ServerConfig {
            cache_directory: fixture_dir("cors"),
            allowed_origins: vec!["https://console.example".to_owned()],
            ..ServerConfig::default()
        })
        .expect("valid state");
        let app = router(state);
        let rejected = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header(ORIGIN, "https://console.example.evil")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(rejected.status(), StatusCode::FORBIDDEN);
        assert!(
            rejected
                .headers()
                .get(ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none()
        );

        let preflight = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/v1/embed")
                    .header(ORIGIN, "https://console.example")
                    .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(preflight.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            preflight.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("https://console.example"))
        );
    }

    #[tokio::test]
    async fn admin_routes_can_be_absent_and_status_is_aggregate_only() {
        let state = AppState::new(&ServerConfig {
            cache_directory: fixture_dir("disabled-admin"),
            admin_api_enabled: false,
            ..ServerConfig::default()
        })
        .expect("valid state");
        let app = router(state);
        let admin = app
            .clone()
            .oneshot(json_request(
                "/v1/admin/models/delete",
                r#"{"model":"bge-small-en"}"#,
            ))
            .await
            .expect("response");
        assert_eq!(admin.status(), StatusCode::NOT_FOUND);

        let page = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response")
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        let page = String::from_utf8(page.to_vec()).expect("UTF-8 page");
        assert!(page.contains("Ready models"));
        assert!(!page.contains("BAAI/"));
        assert!(!page.contains("nomic-ai/"));
        assert!(!page.contains("intfloat/"));
    }

    #[tokio::test]
    async fn health_and_models_are_deterministic_without_a_loaded_model() {
        let app = router(state("health-models"));
        let ready = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body_json(ready).await["reason"], "no_models_configured");

        let models = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(models.status(), StatusCode::OK);
        let models = body_json(models).await;
        assert_eq!(models["data"].as_array().map(Vec::len), Some(3));
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

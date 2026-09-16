//! Minimal MCP 2025-03-26 adapter for local embedding inference.
//!
//! The adapter deliberately exposes only read-only inference and model discovery. Model
//! installation and lifecycle administration remain outside the MCP trust boundary.

use axum::{
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
};
use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingOutput, EmbeddingTask, ModelVerificationStatus,
    Retryability, Truncation,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::io;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};

use crate::{
    AppState, ApplicationError, EmbedCommand, LoadStatus, ModelInfo, RuntimeStatus, SemanticStatus,
};

/// The single protocol revision implemented by this intentionally narrow adapter.
pub const PROTOCOL_VERSION: &str = "2025-03-26";
const JSON_CONTENT_TYPE: &str = "application/json";
const SESSION_HEADER: &str = "mcp-session-id";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    AwaitInitialize,
    AwaitInitialized,
    Ready,
    Stateless,
}

/// A connection-scoped MCP JSON-RPC dispatcher.
///
/// Use [`McpDispatcher::new`] for stateful stdio. Streamable HTTP creates a stateless dispatcher
/// for every POST, as this server intentionally does not issue or retain session identifiers.
pub struct McpDispatcher {
    state: AppState,
    phase: Phase,
}

impl McpDispatcher {
    /// Construct a dispatcher that enforces the MCP initialize/initialized sequence.
    #[must_use]
    pub const fn new(state: AppState) -> Self {
        Self {
            state,
            phase: Phase::AwaitInitialize,
        }
    }

    fn stateless(state: AppState) -> Self {
        Self {
            state,
            phase: Phase::Stateless,
        }
    }

    /// Dispatch exactly one UTF-8 JSON-RPC frame.
    pub async fn dispatch(&mut self, frame: &[u8]) -> DispatchResult {
        let Ok(value) = serde_json::from_slice::<Value>(frame) else {
            return DispatchResult::response(rpc_error(Value::Null, -32700, "Parse error"));
        };
        if value.is_array() {
            return DispatchResult::response(rpc_error(
                Value::Null,
                -32600,
                "JSON-RPC batches are not supported",
            ));
        }
        let incoming = match Incoming::parse(value) {
            Ok(incoming) => incoming,
            Err(id) => {
                return DispatchResult::response(rpc_error(id, -32600, "Invalid Request"));
            }
        };
        self.dispatch_incoming(incoming).await
    }

    async fn dispatch_incoming(&mut self, incoming: Incoming) -> DispatchResult {
        let is_notification = incoming.id.is_none();
        let id = incoming.id.clone().unwrap_or(Value::Null);
        let result = match incoming.method.as_str() {
            "initialize" if !is_notification => self.initialize(incoming.params),
            "notifications/initialized" if is_notification => {
                self.initialized(incoming.params);
                return DispatchResult::Notification;
            }
            "ping" if !is_notification => parse_empty(incoming.params).map(|()| json!({})),
            "tools/list" if !is_notification => {
                if self.ready() {
                    list_tools(incoming.params)
                } else {
                    Err(RpcFailure::new(-32002, "Server not initialized"))
                }
            }
            "tools/call" if !is_notification => {
                if self.ready() {
                    self.call_tool(incoming.params).await
                } else {
                    Err(RpcFailure::new(-32002, "Server not initialized"))
                }
            }
            _ if is_notification => return DispatchResult::Notification,
            _ => Err(RpcFailure::new(-32601, "Method not found")),
        };
        match result {
            Ok(result) => DispatchResult::response(rpc_success(id, result)),
            Err(error) => DispatchResult::response(rpc_error(id, error.code, error.message)),
        }
    }

    fn ready(&self) -> bool {
        matches!(self.phase, Phase::Ready | Phase::Stateless)
    }

    fn initialize(&mut self, params: Option<Value>) -> Result<Value, RpcFailure> {
        if !matches!(self.phase, Phase::AwaitInitialize | Phase::Stateless) {
            return Err(RpcFailure::new(-32600, "Invalid Request"));
        }
        let params: InitializeParams = parse_params(params)?;
        if params.protocol_version.trim().is_empty()
            || params.client_info.name.trim().is_empty()
            || params.client_info.version.trim().is_empty()
        {
            return Err(RpcFailure::invalid_params());
        }
        let _ = (params.capabilities, params.meta, params.client_info.title);
        if self.phase == Phase::AwaitInitialize {
            self.phase = Phase::AwaitInitialized;
        }
        Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": "impossible-embedding",
                "version": self.state.version()
            },
            "instructions": "Local embedding inference and model discovery. No model administration is exposed."
        }))
    }

    fn initialized(&mut self, params: Option<Value>) {
        if parse_empty(params).is_ok() && self.phase == Phase::AwaitInitialized {
            self.phase = Phase::Ready;
        }
    }

    async fn call_tool(&self, params: Option<Value>) -> Result<Value, RpcFailure> {
        let request: CallToolParams = parse_params(params)?;
        match request.name.as_str() {
            "embed" => self.embed(request.arguments).await,
            "list_models" => self.list_models(request.arguments),
            _ => Err(RpcFailure::invalid_params()),
        }
    }

    async fn embed(&self, arguments: Option<Value>) -> Result<Value, RpcFailure> {
        let arguments: EmbedArguments = parse_arguments(arguments)?;
        if arguments.model.is_empty() {
            return Err(RpcFailure::invalid_params());
        }
        let input = match arguments.input {
            EmbedInput::One(input) => vec![input],
            EmbedInput::Many(input) if !input.is_empty() => input,
            EmbedInput::Many(_) => return Err(RpcFailure::invalid_params()),
        };
        let dimensions = arguments
            .dimensions
            .map(usize::try_from)
            .transpose()
            .map_err(|_| RpcFailure::invalid_params())?;
        if dimensions == Some(0) {
            return Err(RpcFailure::invalid_params());
        }
        let result = self
            .state
            .application()
            .embed(EmbedCommand {
                model: arguments.model,
                input,
                options: EmbedOptions {
                    task: arguments.task.into(),
                    truncation: arguments.truncation.into(),
                    dimensions,
                    normalize: arguments.normalize,
                },
                cancellation: CancellationToken::default(),
                timeout: None,
            })
            .await;
        Ok(match result {
            Ok(output) => tool_success(&embedding_value(&output)),
            Err(error) => tool_error(&error),
        })
    }

    fn list_models(&self, arguments: Option<Value>) -> Result<Value, RpcFailure> {
        let _: EmptyArguments = parse_arguments(arguments)?;
        Ok(match self.state.application().list_models() {
            Ok(models) => tool_success(&json!({
                "models": models.iter().map(model_value).collect::<Vec<_>>()
            })),
            Err(error) => tool_error(&error),
        })
    }
}

/// Outcome of dispatching one MCP frame.
#[derive(Debug, PartialEq)]
pub enum DispatchResult {
    /// A request or malformed frame produced a JSON-RPC response.
    Response(Value),
    /// A valid notification produces no protocol response.
    Notification,
}

impl DispatchResult {
    fn response(value: Value) -> Self {
        Self::Response(value)
    }
}

struct Incoming {
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

impl Incoming {
    fn parse(value: Value) -> Result<Self, Value> {
        let Value::Object(mut object) = value else {
            return Err(Value::Null);
        };
        let prospective_id = object.get("id").cloned().unwrap_or(Value::Null);
        if object
            .keys()
            .any(|key| !matches!(key.as_str(), "jsonrpc" | "id" | "method" | "params"))
        {
            return Err(valid_error_id(&prospective_id));
        }
        if object.remove("jsonrpc") != Some(Value::String("2.0".into())) {
            return Err(valid_error_id(&prospective_id));
        }
        let Some(Value::String(method)) = object.remove("method") else {
            return Err(valid_error_id(&prospective_id));
        };
        if method.is_empty() {
            return Err(valid_error_id(&prospective_id));
        }
        let id = object.remove("id");
        if id.as_ref().is_some_and(|value| !valid_id(value)) {
            return Err(Value::Null);
        }
        Ok(Self {
            id,
            method,
            params: object.remove("params"),
        })
    }
}

fn valid_id(value: &Value) -> bool {
    matches!(value, Value::String(_))
        || value
            .as_number()
            .is_some_and(|number| number.is_i64() || number.is_u64())
}

fn valid_error_id(value: &Value) -> Value {
    valid_id(value)
        .then(|| value.clone())
        .unwrap_or(Value::Null)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct InitializeParams {
    protocol_version: String,
    capabilities: Map<String, Value>,
    client_info: ClientInfo,
    #[serde(default, rename = "_meta")]
    meta: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientInfo {
    name: String,
    version: String,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyParams {
    #[serde(default, rename = "_meta")]
    _meta: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListToolsParams {
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default, rename = "_meta")]
    _meta: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallToolParams {
    name: String,
    #[serde(default)]
    arguments: Option<Value>,
    #[serde(default, rename = "_meta")]
    _meta: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyArguments {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbedArguments {
    model: String,
    input: EmbedInput,
    #[serde(default)]
    task: ToolTask,
    #[serde(default)]
    truncation: ToolTruncation,
    #[serde(default)]
    dimensions: Option<u64>,
    #[serde(default)]
    normalize: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ToolTask {
    Query,
    #[default]
    Document,
}

impl From<ToolTask> for EmbeddingTask {
    fn from(value: ToolTask) -> Self {
        match value {
            ToolTask::Query => Self::Query,
            ToolTask::Document => Self::Document,
        }
    }
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ToolTruncation {
    #[default]
    Reject,
    Truncate,
}

impl From<ToolTruncation> for Truncation {
    fn from(value: ToolTruncation) -> Self {
        match value {
            ToolTruncation::Reject => Self::Reject,
            ToolTruncation::Truncate => Self::Truncate,
        }
    }
}

fn parse_params<T: for<'de> Deserialize<'de>>(params: Option<Value>) -> Result<T, RpcFailure> {
    serde_json::from_value(params.ok_or_else(RpcFailure::invalid_params)?)
        .map_err(|_| RpcFailure::invalid_params())
}

fn parse_arguments<T: for<'de> Deserialize<'de>>(
    arguments: Option<Value>,
) -> Result<T, RpcFailure> {
    serde_json::from_value(arguments.unwrap_or_else(|| json!({})))
        .map_err(|_| RpcFailure::invalid_params())
}

fn parse_empty(params: Option<Value>) -> Result<(), RpcFailure> {
    match params {
        None => Ok(()),
        Some(value) => serde_json::from_value::<EmptyParams>(value)
            .map(|_| ())
            .map_err(|_| RpcFailure::invalid_params()),
    }
}

fn list_tools(params: Option<Value>) -> Result<Value, RpcFailure> {
    if let Some(params) = params {
        let params: ListToolsParams =
            serde_json::from_value(params).map_err(|_| RpcFailure::invalid_params())?;
        if params.cursor.is_some() {
            return Err(RpcFailure::invalid_params());
        }
    }
    Ok(json!({
        "tools": [
            {
                "name": "embed",
                "description": "Create dense embeddings with a locally loaded open model.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["model", "input"],
                    "properties": {
                        "model": { "type": "string", "minLength": 1 },
                        "input": {
                            "oneOf": [
                                { "type": "string" },
                                { "type": "array", "minItems": 1, "items": { "type": "string" } }
                            ]
                        },
                        "task": { "type": "string", "enum": ["query", "document"], "default": "document" },
                        "truncation": { "type": "string", "enum": ["reject", "truncate"], "default": "reject" },
                        "dimensions": { "type": "integer", "minimum": 1 },
                        "normalize": { "type": "boolean" }
                    }
                },
                "annotations": {
                    "readOnlyHint": true,
                    "destructiveHint": false,
                    "idempotentHint": true,
                    "openWorldHint": false
                }
            },
            {
                "name": "list_models",
                "description": "List local model availability without revealing paths or host details.",
                "inputSchema": { "type": "object", "additionalProperties": false },
                "annotations": {
                    "readOnlyHint": true,
                    "destructiveHint": false,
                    "idempotentHint": true,
                    "openWorldHint": false
                }
            }
        ]
    }))
}

fn embedding_value(output: &EmbeddingOutput) -> Value {
    let total = output.usage.total_tokens();
    json!({
        "embeddings": output.vectors,
        "model": {
            "canonical_id": output.model.canonical_id,
            "revision": output.model.revision,
            "runtime": output.model.runtime,
            "artifact_fingerprint": output.model.artifact_fingerprint,
            "semantic_fingerprint": output.model.semantic_fingerprint
        },
        "usage": {
            "prompt_tokens": total,
            "total_tokens": total,
            "input_tokens": output.usage.input_tokens()
        }
    })
}

fn model_value(model: &ModelInfo) -> Value {
    json!({
        "model": model.canonical_id,
        "revision": model.revision,
        "aliases": model.aliases,
        "native_dimensions": model.native_dimensions,
        "dimensions": model.dimensions,
        "semantic_status": match model.semantic_status {
            SemanticStatus::Verified => "verified",
            SemanticStatus::Unverified => "unverified",
        },
        "installation_status": match model.installation_status {
            ModelVerificationStatus::Missing => "missing",
            ModelVerificationStatus::Invalid => "invalid",
            ModelVerificationStatus::IntegrityVerified => "integrity_verified",
            ModelVerificationStatus::Loadable => "loadable",
        },
        "runtime_status": match model.runtime_status {
            RuntimeStatus::CatalogOnly => "catalog_only",
            RuntimeStatus::Onnx => "onnx",
        },
        "load_status": match model.load_status {
            LoadStatus::Unloaded => "unloaded",
            LoadStatus::Loaded => "loaded",
        }
    })
}

fn tool_success(value: &Value) -> Value {
    tool_result(value, false)
}

fn tool_error(error: &ApplicationError) -> Value {
    let public = error.public_error();
    let value = json!({
        "error": {
            "code": public.code.as_str(),
            "message": public.message,
            "retryable": matches!(public.retryability, Retryability::Retryable)
        }
    });
    tool_result(&value, true)
}

fn tool_result(value: &Value, is_error: bool) -> Value {
    let text = serde_json::to_string(&value).unwrap_or_else(|_| {
        r#"{"error":{"code":"internal","message":"Internal server error","retryable":false}}"#
            .to_owned()
    });
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error
    })
}

#[derive(Clone, Copy)]
struct RpcFailure {
    code: i32,
    message: &'static str,
}

impl RpcFailure {
    const fn new(code: i32, message: &'static str) -> Self {
        Self { code, message }
    }

    const fn invalid_params() -> Self {
        Self::new(-32602, "Invalid params")
    }
}

fn rpc_success(id: Value, result: Value) -> Value {
    let mut response = Map::new();
    response.insert("jsonrpc".into(), Value::String("2.0".into()));
    response.insert("id".into(), id);
    response.insert("result".into(), result);
    Value::Object(response)
}

fn rpc_error(id: Value, code: i32, message: &'static str) -> Value {
    let mut response = Map::new();
    response.insert("jsonrpc".into(), Value::String("2.0".into()));
    response.insert("id".into(), id);
    response.insert("error".into(), json!({ "code": code, "message": message }));
    Value::Object(response)
}

/// Handle one stateless MCP Streamable HTTP POST.
pub(crate) async fn http_handler(State(state): State<AppState>, request: Request) -> Response {
    if request.headers().contains_key(SESSION_HEADER) {
        let error = rpc_error(Value::Null, -32600, "Sessions are not supported");
        return http_rpc_response(StatusCode::BAD_REQUEST, &error);
    }
    if !accepts_streamable_http(request.headers()) {
        return StatusCode::NOT_ACCEPTABLE.into_response();
    }
    if !has_json_content_type(request.headers()) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    if !valid_content_length(request.headers(), state.limits().max_body_bytes) {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let Ok(body) = to_bytes(request.into_body(), state.limits().max_body_bytes).await else {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    };
    let mut dispatcher = McpDispatcher::stateless(state);
    match dispatcher.dispatch(&body).await {
        DispatchResult::Notification => StatusCode::ACCEPTED.into_response(),
        DispatchResult::Response(value) => {
            let status = if value["error"]["code"] == -32700 || value["error"]["code"] == -32600 {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::OK
            };
            http_rpc_response(status, &value)
        }
    }
}

fn http_rpc_response(status: StatusCode, value: &Value) -> Response {
    match serde_json::to_vec(value) {
        Ok(body) => (
            status,
            [(CONTENT_TYPE, JSON_CONTENT_TYPE)],
            Body::from(body),
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

fn has_json_content_type(headers: &HeaderMap) -> bool {
    let mut values = headers.get_all(CONTENT_TYPE).iter();
    let Some(value) = values.next() else {
        return false;
    };
    if values.next().is_some() {
        return false;
    }
    value.to_str().ok().is_some_and(|value| {
        value
            .split(';')
            .next()
            .is_some_and(|media| media.trim().eq_ignore_ascii_case(JSON_CONTENT_TYPE))
    })
}

fn accepts_streamable_http(headers: &HeaderMap) -> bool {
    let mut json = false;
    let mut events = false;
    for value in headers.get_all(axum::http::header::ACCEPT) {
        let Ok(value) = value.to_str() else {
            return false;
        };
        for media in value.split(',').filter_map(|part| part.split(';').next()) {
            match media.trim().to_ascii_lowercase().as_str() {
                "application/json" => json = true,
                "text/event-stream" => events = true,
                _ => {}
            }
        }
    }
    json && events
}

fn valid_content_length(headers: &HeaderMap, limit: usize) -> bool {
    let mut values = headers.get_all(axum::http::header::CONTENT_LENGTH).iter();
    let Some(value) = values.next() else {
        return true;
    };
    if values.next().is_some() {
        return false;
    }
    value
        .to_str()
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length <= limit as u64)
}

/// Run a single MCP stdio connection using newline-delimited UTF-8 JSON-RPC frames.
///
/// The caller owns the process streams so tests and launchers can guarantee that stdout contains
/// protocol frames only. The configured HTTP body limit is also the maximum stdio frame size.
///
/// # Errors
/// Returns the underlying input or output error if a stdio stream cannot be read, written, or
/// flushed.
pub async fn run_stdio<R, W>(state: AppState, mut reader: R, mut writer: W) -> io::Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let limit = state.limits().max_body_bytes;
    let mut dispatcher = McpDispatcher::new(state);
    let mut line = Vec::new();
    loop {
        match read_capped_line(&mut reader, &mut line, limit).await? {
            LineResult::Eof => break,
            LineResult::Oversized => {
                write_frame(
                    &mut writer,
                    &rpc_error(Value::Null, -32600, "Invalid Request"),
                )
                .await?;
            }
            LineResult::Frame => {
                if let DispatchResult::Response(response) = dispatcher.dispatch(&line).await {
                    write_frame(&mut writer, &response).await?;
                }
            }
        }
    }
    writer.flush().await
}

enum LineResult {
    Frame,
    Oversized,
    Eof,
}

async fn read_capped_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
    limit: usize,
) -> io::Result<LineResult> {
    line.clear();
    let mut oversized = false;
    let mut saw_bytes = false;
    loop {
        let buffer = reader.fill_buf().await?;
        if buffer.is_empty() {
            if !saw_bytes {
                return Ok(LineResult::Eof);
            }
            break;
        }
        saw_bytes = true;
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(buffer.len(), |position| position + 1);
        let payload = newline.map_or(consumed, |position| position);
        if !oversized {
            if line.len().saturating_add(payload) > limit {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(&buffer[..payload]);
            }
        }
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if oversized {
        return Ok(LineResult::Oversized);
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(LineResult::Frame)
}

async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, value: &Value) -> io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
    writer.write_all(&bytes).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use axum::http::{Method, header::ACCEPT};
    use http_body_util::BodyExt;
    use impossible_server_core::{ServerConfig, config::Limits};
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
    use tower::ServiceExt;

    static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    fn state(label: &str) -> AppState {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "impossible-mcp-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create fixture");
        AppState::new(&ServerConfig {
            cache_directory: directory,
            ..ServerConfig::default()
        })
        .expect("valid state")
    }

    async fn dispatch(dispatcher: &mut McpDispatcher, input: &str) -> Value {
        match dispatcher.dispatch(input.as_bytes()).await {
            DispatchResult::Response(value) => value,
            DispatchResult::Notification => Value::Null,
        }
    }

    async fn initialize(dispatcher: &mut McpDispatcher) {
        let response = dispatch(
            dispatcher,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#,
        )
        .await;
        assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(
            dispatcher
                .dispatch(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .await,
            DispatchResult::Notification
        );
    }

    #[tokio::test]
    async fn lifecycle_is_enforced_and_ping_is_available_during_handshake() {
        let mut dispatcher = McpDispatcher::new(state("lifecycle"));
        let before = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":"a","method":"tools/list"}"#,
        )
        .await;
        assert_eq!(before["error"]["code"], -32002);
        let ping = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#,
        )
        .await;
        assert_eq!(ping["result"], json!({}));
        initialize(&mut dispatcher).await;
        let tools = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}}"#,
        )
        .await;
        assert_eq!(tools["result"]["tools"].as_array().map(Vec::len), Some(2));
        assert_eq!(tools["result"]["tools"][0]["name"], "embed");
        assert_eq!(tools["result"]["tools"][1]["name"], "list_models");
    }

    #[tokio::test]
    async fn strict_envelopes_params_ids_and_batches_return_stable_errors() {
        let mut dispatcher = McpDispatcher::new(state("strict"));
        for (input, code) in [
            ("not json", -32700),
            (r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#, -32600),
            (r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#, -32600),
            (r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#, -32600),
            (
                r#"{"jsonrpc":"2.0","id":1,"method":"ping","extra":true}"#,
                -32600,
            ),
            (
                r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{"extra":true}}"#,
                -32602,
            ),
        ] {
            assert_eq!(
                dispatch(&mut dispatcher, input).await["error"]["code"],
                code
            );
        }
    }

    #[tokio::test]
    async fn notifications_never_produce_responses_and_cannot_skip_initialization() {
        let mut dispatcher = McpDispatcher::new(state("notifications"));
        assert_eq!(
            dispatcher
                .dispatch(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .await,
            DispatchResult::Notification
        );
        assert_eq!(
            dispatch(
                &mut dispatcher,
                r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            )
            .await["error"]["code"],
            -32002
        );
        assert_eq!(
            dispatcher
                .dispatch(br#"{"jsonrpc":"2.0","method":"unknown"}"#)
                .await,
            DispatchResult::Notification
        );
    }

    #[tokio::test]
    async fn tool_schemas_and_arguments_are_closed_and_no_admin_tool_is_exposed() {
        let mut dispatcher = McpDispatcher::new(state("tools"));
        initialize(&mut dispatcher).await;
        let invalid = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"embed","arguments":{"model":"bge-small-en","input":"x","secret":"no"}}}"#,
        )
        .await;
        assert_eq!(invalid["error"]["code"], -32602);
        let unknown = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"install_model","arguments":{}}}"#,
        )
        .await;
        assert_eq!(unknown["error"]["code"], -32602);
        let models = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_models","arguments":{}}}"#,
        )
        .await;
        assert_eq!(models["result"]["isError"], false);
        let text = models["result"]["content"][0]["text"]
            .as_str()
            .expect("tool text");
        let value: Value = serde_json::from_str(text).expect("structured text");
        assert_eq!(value["models"].as_array().map(Vec::len), Some(3));
    }

    #[tokio::test]
    async fn application_failures_are_sanitized_tool_errors() {
        let mut dispatcher = McpDispatcher::new(state("tool-error"));
        initialize(&mut dispatcher).await;
        let result = dispatch(
            &mut dispatcher,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"embed","arguments":{"model":"bge-small-en","input":"hello"}}}"#,
        )
        .await;
        assert_eq!(result["result"]["isError"], true);
        let text = result["result"]["content"][0]["text"]
            .as_str()
            .expect("tool error text");
        let error: Value = serde_json::from_str(text).expect("tool error JSON");
        assert_eq!(error["error"]["code"], "model_unavailable");
        assert!(error.to_string().len() < 256);
        assert!(!error.to_string().contains("impossible-mcp"));
    }

    fn mcp_request(body: &'static str) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri("/mcp")
            .header(CONTENT_TYPE, JSON_CONTENT_TYPE)
            .header(ACCEPT, "application/json, text/event-stream")
            .body(Body::from(body))
            .expect("request")
    }

    async fn response_json(response: Response) -> Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("response JSON")
    }

    #[tokio::test]
    async fn streamable_http_is_stateless_and_notifications_return_empty_202() {
        let app = crate::http::router(state("http"));
        let initialize = app
            .clone()
            .oneshot(mcp_request(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#))
            .await
            .expect("response");
        assert_eq!(initialize.status(), StatusCode::OK);
        assert!(!initialize.headers().contains_key(SESSION_HEADER));
        let tools = app
            .clone()
            .oneshot(mcp_request(
                r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            ))
            .await
            .expect("response");
        assert_eq!(tools.status(), StatusCode::OK);
        assert_eq!(
            response_json(tools).await["result"]["tools"][0]["name"],
            "embed"
        );
        let notification = app
            .oneshot(mcp_request(
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            ))
            .await
            .expect("response");
        assert_eq!(notification.status(), StatusCode::ACCEPTED);
        assert_eq!(
            notification
                .into_body()
                .collect()
                .await
                .expect("body")
                .to_bytes()
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn streamable_http_rejects_wrong_media_sessions_and_unsupported_methods() {
        let app = crate::http::router(state("http-reject"));
        let mut missing_accept = mcp_request(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
        missing_accept.headers_mut().remove(ACCEPT);
        assert_eq!(
            app.clone()
                .oneshot(missing_accept)
                .await
                .expect("response")
                .status(),
            StatusCode::NOT_ACCEPTABLE
        );
        let mut session = mcp_request(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
        session
            .headers_mut()
            .insert(SESSION_HEADER, "opaque".parse().expect("header"));
        assert_eq!(
            app.clone()
                .oneshot(session)
                .await
                .expect("response")
                .status(),
            StatusCode::BAD_REQUEST
        );
        for method in [Method::GET, Method::DELETE] {
            let request = Request::builder()
                .method(method)
                .uri("/mcp")
                .body(Body::empty())
                .expect("request");
            assert_eq!(
                app.clone()
                    .oneshot(request)
                    .await
                    .expect("response")
                    .status(),
                StatusCode::METHOD_NOT_ALLOWED
            );
        }
    }

    #[tokio::test]
    async fn stdio_supports_crlf_eof_and_emits_no_notification_frame() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\r\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}"
        );
        let reader = BufReader::new(std::io::Cursor::new(input.as_bytes().to_vec()));
        let mut output = Vec::new();
        run_stdio(state("stdio"), reader, &mut output)
            .await
            .expect("stdio run");
        let frames = String::from_utf8(output).expect("UTF-8");
        let frames = frames.lines().collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(frames[0]).expect("JSON")["id"],
            1
        );
        assert_eq!(
            serde_json::from_str::<Value>(frames[1]).expect("JSON")["id"],
            2
        );
    }

    #[tokio::test]
    async fn stdio_discards_oversized_lines_and_recovers_at_the_next_frame() {
        let limits = Limits {
            max_body_bytes: 64,
            max_input_bytes: 32,
            max_request_bytes: 48,
            ..Limits::default()
        };
        let directory = std::env::temp_dir().join(format!(
            "impossible-mcp-cap-{}-{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).expect("fixture");
        let state = AppState::new(&ServerConfig {
            cache_directory: directory,
            limits,
            ..ServerConfig::default()
        })
        .expect("state");
        let (client, server) = tokio::io::duplex(4096);
        let (server_read, server_write) = tokio::io::split(server);
        let task = tokio::spawn(run_stdio(state, BufReader::new(server_read), server_write));
        let (mut client_read, mut client_write) = tokio::io::split(client);
        client_write
            .write_all(&[b'x'; 200])
            .await
            .expect("oversized");
        client_write
            .write_all(b"\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n")
            .await
            .expect("ping");
        client_write.shutdown().await.expect("close input");
        let mut output = String::new();
        client_read
            .read_to_string(&mut output)
            .await
            .expect("output");
        task.await.expect("join").expect("stdio");
        let frames = output
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("JSON"))
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0]["error"]["code"], -32600);
        assert_eq!(frames[1]["id"], 2);
    }
}

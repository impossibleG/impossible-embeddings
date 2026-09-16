//! Command-line parsing and process entry points.

use std::{
    env, fmt, io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use clap::{Args, Parser, Subcommand};
use impossible_server_core::{
    ServerConfig,
    config::{ConfigError, CredentialSource},
    doctor::DoctorReport,
    health::LifecycleState,
    security::load_credential,
};
use reqwest::{
    StatusCode, Url,
    header::{AUTHORIZATION, HeaderValue},
};
use serde_json::{Value, json};
use tokio::io::BufReader;

use crate::{AppState, AppStateError, ProcessError, ServerHost, mcp};

const EXIT_FAILURE: u8 = 1;
const EXIT_USAGE: u8 = 2;
const EXIT_UNAVAILABLE: u8 = 3;
const EXIT_AUTH: u8 = 4;
const EXIT_REQUEST: u8 = 5;
const EXIT_SERVER: u8 = 6;
const MAX_CLIENT_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

/// Impossible Embedding command-line interface.
#[derive(Debug, Parser)]
#[command(name = "impossible-embedding", version, about)]
pub struct Cli {
    /// Operation to run.
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the local HTTP and gRPC servers.
    Serve(ConfigArgs),
    /// Run an MCP transport.
    Mcp(McpArgs),
    /// Inspect or change models through a running server's HTTP API.
    Models(ModelsArgs),
    /// Validate local configuration and report privacy-safe aggregate readiness.
    Doctor(ConfigArgs),
}

/// MCP process options.
#[derive(Debug, Args)]
struct McpArgs {
    /// Use newline-delimited JSON-RPC over stdin and stdout.
    #[arg(long, required = true)]
    stdio: bool,
    /// Shared application configuration.
    #[command(flatten)]
    config: ConfigArgs,
}

/// Model administration command.
#[derive(Debug, Args)]
struct ModelsArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List the server's curated model catalog and local states.
    List(ClientArgs),
    /// Explicitly download and verify one curated model.
    Install(ModelActionArgs),
    /// Load one already-installed model.
    Load(ModelActionArgs),
    /// Unload one model.
    Unload(ModelActionArgs),
    /// Delete one unloaded model from the local cache.
    Delete(ModelActionArgs),
}

/// One remote model action.
#[derive(Debug, Args)]
struct ModelActionArgs {
    /// Curated alias or canonical model id.
    model: String,
    #[command(flatten)]
    client: ClientArgs,
}

/// Safe references used by the HTTP administration client.
#[derive(Clone, Debug, Args)]
struct ClientArgs {
    /// Base URL of the running local HTTP server.
    #[arg(
        long,
        env = "IMPOSSIBLE_SERVER_URL",
        default_value = "http://[::1]:8080"
    )]
    server: String,
    /// Name of an environment variable containing the endpoint credential.
    #[arg(long, conflicts_with = "auth_file")]
    auth_env: Option<String>,
    /// Path to a regular file containing the endpoint credential.
    #[arg(long, conflicts_with = "auth_env")]
    auth_file: Option<PathBuf>,
    /// Client-side request deadline in seconds.
    #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u64).range(1..=86_400))]
    timeout_seconds: u64,
}

/// Server configuration flags. Values are forwarded to the central typed configuration loader,
/// preserving CLI > environment > TOML > defaults precedence.
#[derive(Clone, Debug, Default, Args)]
struct ConfigArgs {
    /// Optional TOML configuration file.
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    http_bind: Option<String>,
    #[arg(long)]
    grpc_bind: Option<String>,
    #[arg(long)]
    allow_insecure_remote: bool,
    /// Name of the environment variable holding the public credential.
    #[arg(long, conflicts_with = "auth_file")]
    auth_env: Option<String>,
    /// Regular file holding the public credential.
    #[arg(long, conflicts_with = "auth_env")]
    auth_file: Option<PathBuf>,
    /// Name of the environment variable holding the separate admin credential.
    #[arg(long, conflicts_with = "admin_auth_file")]
    admin_auth_env: Option<String>,
    /// Regular file holding the separate admin credential.
    #[arg(long, conflicts_with = "admin_auth_env")]
    admin_auth_file: Option<PathBuf>,
    #[arg(long)]
    admin_api_enabled: Option<String>,
    #[arg(long)]
    allowed_origins: Option<String>,
    #[arg(long)]
    model_directories: Option<String>,
    #[arg(long)]
    cache_directory: Option<PathBuf>,
    #[arg(long)]
    preload_models: Option<String>,
    #[arg(long)]
    offline: bool,
    #[arg(long)]
    startup_policy: Option<String>,
    #[arg(long)]
    max_body_bytes: Option<String>,
    #[arg(long)]
    max_input_bytes: Option<String>,
    #[arg(long)]
    max_request_bytes: Option<String>,
    #[arg(long)]
    max_items: Option<String>,
    #[arg(long)]
    max_tokens: Option<String>,
    #[arg(long)]
    max_queue_depth: Option<String>,
    #[arg(long)]
    max_batch_items: Option<String>,
    #[arg(long)]
    max_batch_tokens: Option<String>,
    #[arg(long)]
    max_concurrency: Option<String>,
    #[arg(long)]
    request_timeout_ms: Option<String>,
    #[arg(long)]
    shutdown_timeout_ms: Option<String>,
}

impl ConfigArgs {
    fn load(&self) -> Result<ServerConfig, ConfigError> {
        let arguments = self.configuration_arguments();
        ServerConfig::load(self.config.as_deref(), env::vars(), arguments)
    }

    fn configuration_arguments(&self) -> Vec<String> {
        let mut arguments = Vec::new();
        push_value(&mut arguments, "--http-bind", self.http_bind.as_deref());
        push_value(&mut arguments, "--grpc-bind", self.grpc_bind.as_deref());
        if self.allow_insecure_remote {
            arguments.push("--allow-insecure-remote".to_owned());
        }
        push_value(&mut arguments, "--auth-env", self.auth_env.as_deref());
        push_path(&mut arguments, "--auth-file", self.auth_file.as_deref());
        push_value(
            &mut arguments,
            "--admin-auth-env",
            self.admin_auth_env.as_deref(),
        );
        push_path(
            &mut arguments,
            "--admin-auth-file",
            self.admin_auth_file.as_deref(),
        );
        push_value(
            &mut arguments,
            "--admin-api-enabled",
            self.admin_api_enabled.as_deref(),
        );
        push_value(
            &mut arguments,
            "--allowed-origins",
            self.allowed_origins.as_deref(),
        );
        push_value(
            &mut arguments,
            "--model-directories",
            self.model_directories.as_deref(),
        );
        push_path(
            &mut arguments,
            "--cache-directory",
            self.cache_directory.as_deref(),
        );
        push_value(
            &mut arguments,
            "--preload-models",
            self.preload_models.as_deref(),
        );
        if self.offline {
            arguments.push("--offline".to_owned());
        }
        push_value(
            &mut arguments,
            "--startup-policy",
            self.startup_policy.as_deref(),
        );
        for (name, value) in [
            ("--max-body-bytes", self.max_body_bytes.as_deref()),
            ("--max-input-bytes", self.max_input_bytes.as_deref()),
            ("--max-request-bytes", self.max_request_bytes.as_deref()),
            ("--max-items", self.max_items.as_deref()),
            ("--max-tokens", self.max_tokens.as_deref()),
            ("--max-queue-depth", self.max_queue_depth.as_deref()),
            ("--max-batch-items", self.max_batch_items.as_deref()),
            ("--max-batch-tokens", self.max_batch_tokens.as_deref()),
            ("--max-concurrency", self.max_concurrency.as_deref()),
            ("--request-timeout-ms", self.request_timeout_ms.as_deref()),
            ("--shutdown-timeout-ms", self.shutdown_timeout_ms.as_deref()),
        ] {
            push_value(&mut arguments, name, value);
        }
        arguments
    }
}

fn push_value(arguments: &mut Vec<String>, name: &str, value: Option<&str>) {
    if let Some(value) = value {
        arguments.push(name.to_owned());
        arguments.push(value.to_owned());
    }
}

fn push_path(arguments: &mut Vec<String>, name: &str, value: Option<&Path>) {
    if let Some(value) = value {
        arguments.push(name.to_owned());
        arguments.push(value.to_string_lossy().into_owned());
    }
}

/// Execute a parsed CLI and return a stable process exit status.
pub async fn run(cli: Cli) -> ExitCode {
    match run_inner(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(error.exit_code())
        }
    }
}

async fn run_inner(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Serve(arguments) => run_server(arguments).await,
        Command::Mcp(arguments) => run_mcp(arguments).await,
        Command::Models(arguments) => run_models(arguments).await,
        Command::Doctor(arguments) => run_doctor(arguments).await,
    }
}

async fn run_server(arguments: ConfigArgs) -> Result<(), CliError> {
    init_logging();
    let config = arguments.load()?;
    let host = ServerHost::start(config).await?;
    let loaded = host
        .preload_report()
        .outcomes
        .iter()
        .filter(|outcome| outcome.loaded)
        .count();
    let failed = host.preload_report().outcomes.len().saturating_sub(loaded);
    tracing::info!(
        event = "server_started",
        preload_loaded = loaded,
        preload_failed = failed
    );
    host.run_until_signal().await?;
    tracing::info!(event = "server_stopped");
    Ok(())
}

async fn run_mcp(arguments: McpArgs) -> Result<(), CliError> {
    let _ = arguments.stdio;
    init_logging();
    let config = arguments.config.load()?;
    let state = AppState::new(&config)?;
    state
        .application()
        .preload()
        .await
        .map_err(ProcessError::Preload)?;
    let input = BufReader::new(tokio::io::stdin());
    let output = tokio::io::stdout();
    let result = mcp::run_stdio(state.clone(), input, output).await;
    state.shutdown_trigger().trigger();
    let drained = state.application().shutdown().await;
    result.map_err(|_| CliError::Stdio)?;
    if !drained {
        return Err(CliError::Process(ProcessError::ShutdownTimeout));
    }
    Ok(())
}

async fn run_doctor(arguments: ConfigArgs) -> Result<(), CliError> {
    init_logging();
    let config = arguments.load()?;
    let state = AppState::new(&config)?;
    let _ = state
        .application()
        .preload()
        .await
        .map_err(ProcessError::Preload)?;
    let readiness = state.application().readiness();
    let (ready_models, model_count) = state.application().health().model_counts();
    let report = DoctorReport::from_readiness(
        env!("CARGO_PKG_VERSION"),
        &readiness,
        ready_models,
        model_count,
    );
    println!(
        "{}",
        json!({
            "version": report.version,
            "lifecycle": lifecycle_name(report.lifecycle),
            "reason_code": report.reason_code.map(impossible_server_core::ReadinessReason::code),
            "model_count": report.model_count,
            "ready_model_count": report.ready_model_count
        })
    );
    state.shutdown_trigger().trigger();
    let _ = state.application().shutdown().await;
    Ok(())
}

const fn lifecycle_name(state: LifecycleState) -> &'static str {
    match state {
        LifecycleState::Starting => "starting",
        LifecycleState::Ready => "ready",
        LifecycleState::Draining => "draining",
        LifecycleState::Stopped => "stopped",
    }
}

async fn run_models(arguments: ModelsArgs) -> Result<(), CliError> {
    match arguments.command {
        ModelCommand::List(client) => request_models(client, None).await,
        ModelCommand::Install(action) => {
            request_models(action.client, Some(("install", action.model))).await
        }
        ModelCommand::Load(action) => {
            request_models(action.client, Some(("load", action.model))).await
        }
        ModelCommand::Unload(action) => {
            request_models(action.client, Some(("unload", action.model))).await
        }
        ModelCommand::Delete(action) => {
            request_models(action.client, Some(("delete", action.model))).await
        }
    }
}

async fn request_models(
    arguments: ClientArgs,
    action: Option<(&'static str, String)>,
) -> Result<(), CliError> {
    let mut url = parse_server_url(&arguments.server)?;
    let administrative = action.is_some();
    url.set_path(
        action
            .as_ref()
            .map_or("/v1/models", |(name, _)| match *name {
                "install" => "/v1/admin/models/install",
                "load" => "/v1/admin/models/load",
                "unload" => "/v1/admin/models/unload",
                "delete" => "/v1/admin/models/delete",
                _ => "/v1/models",
            }),
    );
    url.set_query(None);
    url.set_fragment(None);

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(arguments.timeout_seconds))
        .build()
        .map_err(|_| ClientError::Unavailable)?;
    let mut request = if let Some((_, model)) = action {
        client.post(url).json(&json!({ "model": model }))
    } else {
        client.get(url)
    };
    if let Some(source) = client_credential_source(&arguments, administrative) {
        let secret = load_credential(&source).map_err(|_| ClientError::Credential)?;
        let header = secret
            .with_bytes(bearer_header)
            .map_err(|()| ClientError::Credential)?;
        request = request.header(AUTHORIZATION, header);
    }
    let response = request.send().await.map_err(|_| ClientError::Unavailable)?;
    let status = response.status();
    let body = bounded_response(response).await?;
    if !status.is_success() {
        return Err(ClientError::for_status(status).into());
    }
    let value: Value = serde_json::from_slice(&body).map_err(|_| ClientError::InvalidResponse)?;
    println!("{value}");
    Ok(())
}

fn parse_server_url(value: &str) -> Result<Url, CliError> {
    let url = Url::parse(value).map_err(|_| ClientError::InvalidUrl)?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ClientError::InvalidUrl.into());
    }
    Ok(url)
}

fn client_credential_source(
    arguments: &ClientArgs,
    administrative: bool,
) -> Option<CredentialSource> {
    if let Some(name) = &arguments.auth_env {
        return Some(CredentialSource::Environment(name.clone()));
    }
    if let Some(path) = &arguments.auth_file {
        return Some(CredentialSource::File(path.clone()));
    }
    let conventional = if administrative {
        "IMPOSSIBLE_ADMIN_AUTH_TOKEN"
    } else {
        "IMPOSSIBLE_AUTH_TOKEN"
    };
    env::var_os(conventional)
        .is_some()
        .then(|| CredentialSource::Environment(conventional.to_owned()))
}

fn bearer_header(secret: &[u8]) -> Result<HeaderValue, ()> {
    let mut value = Vec::with_capacity(7 + secret.len());
    value.extend_from_slice(b"Bearer ");
    value.extend_from_slice(secret);
    let header = HeaderValue::from_bytes(&value).map_err(|_| ());
    value.fill(0);
    header
}

async fn bounded_response(mut response: reqwest::Response) -> Result<Vec<u8>, CliError> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_CLIENT_RESPONSE_BYTES as u64)
    {
        return Err(ClientError::InvalidResponse.into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| ClientError::Unavailable)?
    {
        if body.len().saturating_add(chunk.len()) > MAX_CLIENT_RESPONSE_BYTES {
            return Err(ClientError::InvalidResponse.into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .json()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClientError {
    InvalidUrl,
    Credential,
    Unavailable,
    Authentication,
    Rejected,
    Server,
    InvalidResponse,
}

impl ClientError {
    fn for_status(status: StatusCode) -> Self {
        match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Self::Authentication,
            status if status.is_client_error() => Self::Rejected,
            _ => Self::Server,
        }
    }

    const fn exit_code(self) -> u8 {
        match self {
            Self::InvalidUrl => EXIT_USAGE,
            Self::Credential | Self::Authentication => EXIT_AUTH,
            Self::Unavailable | Self::InvalidResponse => EXIT_UNAVAILABLE,
            Self::Rejected => EXIT_REQUEST,
            Self::Server => EXIT_SERVER,
        }
    }
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidUrl => "server URL is invalid",
            Self::Credential => "client credential configuration is invalid",
            Self::Unavailable => "server is unavailable",
            Self::Authentication => "server authentication failed",
            Self::Rejected => "model operation was rejected",
            Self::Server => "server failed the model operation",
            Self::InvalidResponse => "server returned an invalid response",
        })
    }
}

#[derive(Debug)]
enum CliError {
    Config(ConfigError),
    State(AppStateError),
    Process(ProcessError),
    Stdio,
    Client(ClientError),
}

impl CliError {
    const fn exit_code(&self) -> u8 {
        match self {
            Self::Config(_) => EXIT_USAGE,
            Self::Client(error) => error.exit_code(),
            Self::State(_) | Self::Process(_) | Self::Stdio => EXIT_FAILURE,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(formatter, "configuration failed: {error}"),
            Self::State(error) => write!(formatter, "{error}"),
            Self::Process(error) => write!(formatter, "{error}"),
            Self::Stdio => formatter.write_str("MCP stdio transport failed"),
            Self::Client(error) => write!(formatter, "{error}"),
        }
    }
}

impl From<ConfigError> for CliError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<AppStateError> for CliError {
    fn from(error: AppStateError) -> Self {
        Self::State(error)
    }
}

impl From<ProcessError> for CliError {
    fn from(error: ProcessError) -> Self {
        Self::Process(error)
    }
}

impl From<ClientError> for CliError {
    fn from(error: ClientError) -> Self {
        Self::Client(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_parser_separates_commands_and_rejects_literal_secrets() {
        let parsed = Cli::try_parse_from([
            "impossible-embedding",
            "serve",
            "--http-bind",
            "127.0.0.1:8080",
            "--auth-env",
            "PUBLIC_TOKEN",
        ]);
        assert!(parsed.is_ok());
        for forbidden in ["--token", "--auth-token", "--password", "--api-key"] {
            let parsed = Cli::try_parse_from([
                "impossible-embedding",
                "serve",
                forbidden,
                "sentinel-secret",
            ]);
            assert!(parsed.is_err());
        }
    }

    #[test]
    fn server_url_rejects_embedded_credentials_and_non_http_schemes() {
        assert!(parse_server_url("http://localhost:8080").is_ok());
        assert!(parse_server_url("http://user:secret@localhost:8080").is_err());
        assert!(parse_server_url("file:///private/model").is_err());
    }

    #[test]
    fn client_exit_codes_are_stable() {
        assert_eq!(ClientError::InvalidUrl.exit_code(), EXIT_USAGE);
        assert_eq!(ClientError::Authentication.exit_code(), EXIT_AUTH);
        assert_eq!(ClientError::Rejected.exit_code(), EXIT_REQUEST);
        assert_eq!(ClientError::Server.exit_code(), EXIT_SERVER);
        assert_eq!(ClientError::Unavailable.exit_code(), EXIT_UNAVAILABLE);
    }
}

//! Typed configuration with deterministic precedence and secure defaults.

use std::{
    collections::BTreeMap,
    env, fmt, fs,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    time::Duration,
};
use url::{Origin, Url};

/// A non-secret reference to where a credential is stored.
#[derive(Clone, PartialEq, Eq)]
pub enum CredentialSource {
    /// Read the credential from this environment variable.
    Environment(String),
    /// Read the credential from this file at startup.
    File(PathBuf),
}

impl fmt::Debug for CredentialSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Environment(_) => formatter.write_str("Environment([REDACTED])"),
            Self::File(_) => formatter.write_str("File([REDACTED])"),
        }
    }
}

/// Bounded resource policy used before requests reach an inference engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Maximum encoded request body size.
    pub max_body_bytes: usize,
    /// Maximum text inputs in one request.
    pub max_items: usize,
    /// Maximum estimated tokens across a request.
    pub max_tokens: usize,
    /// Maximum requests waiting for admission.
    pub max_queue_depth: usize,
    /// Maximum inputs combined by dynamic batching.
    pub max_batch_items: usize,
    /// Maximum concurrent inference calls.
    pub max_concurrency: usize,
    /// End-to-end request timeout.
    pub request_timeout: Duration,
    /// Maximum graceful shutdown drain interval.
    pub shutdown_timeout: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body_bytes: 2 * 1024 * 1024,
            max_items: 128,
            max_tokens: 32_768,
            max_queue_depth: 256,
            max_batch_items: 128,
            max_concurrency: 4,
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(20),
        }
    }
}

/// Fully merged server configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct ServerConfig {
    /// Listener address. Loopback is the safe default.
    pub bind: SocketAddr,
    /// Explicit acknowledgement allowing an unauthenticated non-loopback bind.
    pub allow_insecure_remote: bool,
    /// Credential for inference APIs.
    pub auth: Option<CredentialSource>,
    /// Separate credential for administrative APIs.
    pub admin_auth: Option<CredentialSource>,
    /// Exact allowed browser origins. Wildcards are prohibited.
    pub allowed_origins: Vec<String>,
    /// Paths explicitly selected for model discovery.
    pub model_directories: Vec<PathBuf>,
    /// Resource and time limits.
    pub limits: Limits,
}

impl fmt::Debug for ServerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerConfig")
            .field("bind", &"[REDACTED]")
            .field("allow_insecure_remote", &self.allow_insecure_remote)
            .field("auth", &self.auth)
            .field("admin_auth", &self.admin_auth)
            .field("allowed_origin_count", &self.allowed_origins.len())
            .field("model_directory_count", &self.model_directories.len())
            .field("limits", &self.limits)
            .finish()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 8080),
            allow_insecure_remote: false,
            auth: None,
            admin_auth: None,
            allowed_origins: Vec::new(),
            model_directories: Vec::new(),
            limits: Limits::default(),
        }
    }
}

/// A closed identifier for a supported configuration field.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfigKey {
    Bind,
    AllowInsecureRemote,
    AuthEnv,
    AuthTokenPresent,
    AuthFile,
    AdminAuthEnv,
    AdminAuthTokenPresent,
    AdminAuthFile,
    AllowedOrigins,
    ModelDirectories,
    MaxBodyBytes,
    MaxItems,
    MaxTokens,
    MaxQueueDepth,
    MaxBatchItems,
    MaxConcurrency,
    RequestTimeout,
    ShutdownTimeout,
    Toml,
    Path,
}

impl ConfigKey {
    /// Return the stable, non-sensitive public name for this field.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bind => "bind",
            Self::AllowInsecureRemote => "allow_insecure_remote",
            Self::AuthEnv => "auth_env",
            Self::AuthTokenPresent => "auth_token_present",
            Self::AuthFile => "auth_file",
            Self::AdminAuthEnv => "admin_auth_env",
            Self::AdminAuthTokenPresent => "admin_auth_token_present",
            Self::AdminAuthFile => "admin_auth_file",
            Self::AllowedOrigins => "allowed_origins",
            Self::ModelDirectories => "model_directories",
            Self::MaxBodyBytes => "limits.max_body_bytes",
            Self::MaxItems => "limits.max_items",
            Self::MaxTokens => "limits.max_tokens",
            Self::MaxQueueDepth => "limits.max_queue_depth",
            Self::MaxBatchItems => "limits.max_batch_items",
            Self::MaxConcurrency => "limits.max_concurrency",
            Self::RequestTimeout => "limits.request_timeout_ms",
            Self::ShutdownTimeout => "limits.shutdown_timeout_ms",
            Self::Toml => "toml",
            Self::Path => "path",
        }
    }
}

/// Configuration input category, deliberately excluding attacker-controlled names.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    Toml,
    Environment,
    CommandLine,
    Validation,
}

/// Stable diagnostic metadata available only through an explicit call.
#[allow(missing_docs)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigDiagnostic {
    pub code: &'static str,
    pub source: ConfigSource,
    pub key: Option<ConfigKey>,
    pub reason: Option<&'static str>,
}

/// Configuration failure that never retains credential values, paths, or arbitrary field names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// An option or TOML key is not supported.
    UnknownKey {
        /// Input category containing the unsupported name.
        source: ConfigSource,
    },
    /// A known option has an invalid value.
    InvalidValue {
        /// Public option or key name.
        key: ConfigKey,
        /// Stable explanation that excludes the rejected value.
        reason: &'static str,
    },
    /// A CLI option requires a following value.
    MissingValue(ConfigKey),
    /// A positional argument was supplied where only an option is valid.
    UnexpectedArgument,
    /// A configuration file could not be accessed.
    ConfigFileUnavailable,
    /// A remote listener was configured without protection.
    UnsafeRemoteBind,
    /// Administrative authentication aliases public authentication.
    InvalidAdminCredentialPolicy,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownKey { source } => {
                write!(f, "unknown configuration key in {}", source.as_str())
            }
            Self::InvalidValue { key, reason } => {
                write!(f, "invalid value for {}: {reason}", key.as_str())
            }
            Self::MissingValue(key) => write!(f, "missing value for {}", key.as_str()),
            Self::UnexpectedArgument => f.write_str("unexpected positional argument"),
            Self::ConfigFileUnavailable => f.write_str("configuration file is unavailable"),
            Self::UnsafeRemoteBind => f.write_str(
                "non-loopback binding requires authentication or an explicit insecure override",
            ),
            Self::InvalidAdminCredentialPolicy => {
                f.write_str("administrative APIs require a separate credential source")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl ConfigSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Toml => "TOML",
            Self::Environment => "environment",
            Self::CommandLine => "command line",
            Self::Validation => "validation",
        }
    }
}

impl ConfigError {
    /// Return stable, non-sensitive diagnostic metadata for structured logging.
    #[must_use]
    pub const fn diagnostic(&self) -> ConfigDiagnostic {
        match self {
            Self::UnknownKey { source } => ConfigDiagnostic {
                code: "config_unknown_key",
                source: *source,
                key: None,
                reason: None,
            },
            Self::InvalidValue { key, reason } => ConfigDiagnostic {
                code: "config_invalid_value",
                source: ConfigSource::Validation,
                key: Some(*key),
                reason: Some(reason),
            },
            Self::MissingValue(key) => ConfigDiagnostic {
                code: "config_missing_value",
                source: ConfigSource::CommandLine,
                key: Some(*key),
                reason: None,
            },
            Self::UnexpectedArgument => ConfigDiagnostic {
                code: "config_unexpected_argument",
                source: ConfigSource::CommandLine,
                key: None,
                reason: None,
            },
            Self::ConfigFileUnavailable => ConfigDiagnostic {
                code: "config_file_unavailable",
                source: ConfigSource::Toml,
                key: None,
                reason: None,
            },
            Self::UnsafeRemoteBind => ConfigDiagnostic {
                code: "config_unsafe_remote_bind",
                source: ConfigSource::Validation,
                key: Some(ConfigKey::Bind),
                reason: None,
            },
            Self::InvalidAdminCredentialPolicy => ConfigDiagnostic {
                code: "config_invalid_admin_credential_policy",
                source: ConfigSource::Validation,
                key: Some(ConfigKey::AdminAuthEnv),
                reason: None,
            },
        }
    }
}

#[derive(Default)]
struct Patch(BTreeMap<ConfigKey, PatchValue>);

enum PatchValue {
    Scalar(String),
    List(Vec<String>),
}

impl Patch {
    fn set(&mut self, key: ConfigKey, value: PatchValue) -> Result<(), ConfigError> {
        if self.0.insert(key, value).is_some() {
            return Err(ConfigError::InvalidValue {
                key,
                reason: "duplicate key in one configuration source",
            });
        }
        Ok(())
    }

    fn apply(self, target: &mut ServerConfig) -> Result<(), ConfigError> {
        for alternatives in [
            [
                ConfigKey::AuthEnv,
                ConfigKey::AuthFile,
                ConfigKey::AuthTokenPresent,
            ],
            [
                ConfigKey::AdminAuthEnv,
                ConfigKey::AdminAuthFile,
                ConfigKey::AdminAuthTokenPresent,
            ],
        ] {
            if alternatives
                .iter()
                .filter(|key| self.0.contains_key(key))
                .count()
                > 1
            {
                return Err(ConfigError::InvalidValue {
                    key: alternatives[0],
                    reason: "configure exactly one credential source",
                });
            }
        }
        for (key, value) in self.0 {
            apply_value(target, key, value)?;
        }
        Ok(())
    }
}

impl ServerConfig {
    /// Load configuration with precedence CLI > environment > TOML > defaults.
    ///
    /// Secret values are deliberately unsupported on the command line. Only
    /// environment-variable names and credential file paths may be configured.
    ///
    /// # Errors
    ///
    /// Returns a sanitized [`ConfigError`] when a source cannot be parsed,
    /// includes an unknown key, violates a bound, or creates an unsafe listener.
    pub fn load<I, K, V, A>(
        toml_path: Option<&Path>,
        environment: I,
        cli: A,
    ) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
        A: IntoIterator,
        A::Item: AsRef<str>,
    {
        let mut config = Self::default();
        if let Some(path) = toml_path {
            let contents =
                fs::read_to_string(path).map_err(|_| ConfigError::ConfigFileUnavailable)?;
            parse_toml(&contents)?.apply(&mut config)?;
        }
        parse_environment(environment)?.apply(&mut config)?;
        parse_cli(cli)?.apply(&mut config)?;
        config.validate()?;
        Ok(config)
    }

    /// Load from the current process environment and argument list.
    ///
    /// # Errors
    ///
    /// Returns a sanitized [`ConfigError`] under the same conditions as [`Self::load`].
    pub fn load_process(toml_path: Option<&Path>) -> Result<Self, ConfigError> {
        Self::load(toml_path, env::vars(), env::args().skip(1))
    }

    /// Validate network safety, origins, credential separation, and resource bounds.
    ///
    /// # Errors
    ///
    /// Returns a sanitized [`ConfigError`] for an unsafe or unsupported policy.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.bind.ip().is_loopback() && self.auth.is_none() && !self.allow_insecure_remote {
            return Err(ConfigError::UnsafeRemoteBind);
        }
        if self.admin_auth.is_some() && self.admin_auth == self.auth {
            return Err(ConfigError::InvalidAdminCredentialPolicy);
        }
        validate_bounds(&self.limits)?;
        if self
            .allowed_origins
            .iter()
            .any(|origin| !valid_origin(origin))
        {
            return Err(ConfigError::InvalidValue {
                key: ConfigKey::AllowedOrigins,
                reason: "origins must be exact HTTP(S) origins without wildcards",
            });
        }
        Ok(())
    }
}

fn parse_toml(input: &str) -> Result<Patch, ConfigError> {
    let mut patch = Patch::default();
    let table = input
        .parse::<toml::Table>()
        .map_err(|_| ConfigError::InvalidValue {
            key: ConfigKey::Toml,
            reason: "invalid TOML document",
        })?;
    for (key, value) in table {
        if key == "limits" {
            let toml::Value::Table(limits) = value else {
                return Err(ConfigError::InvalidValue {
                    key: ConfigKey::Toml,
                    reason: "expected table",
                });
            };
            for (limit_key, limit_value) in limits {
                let raw_key = format!("limits.{limit_key}");
                let key = parse_key(&raw_key).ok_or(ConfigError::UnknownKey {
                    source: ConfigSource::Toml,
                })?;
                insert_toml_value(&mut patch, key, limit_value)?;
            }
        } else {
            let key = parse_key(&key).ok_or(ConfigError::UnknownKey {
                source: ConfigSource::Toml,
            })?;
            insert_toml_value(&mut patch, key, value)?;
        }
    }
    Ok(patch)
}

fn insert_toml_value(
    patch: &mut Patch,
    key: ConfigKey,
    value: toml::Value,
) -> Result<(), ConfigError> {
    let value = match value {
        toml::Value::String(value) => PatchValue::Scalar(value),
        toml::Value::Integer(value) => PatchValue::Scalar(value.to_string()),
        toml::Value::Boolean(value) => PatchValue::Scalar(value.to_string()),
        toml::Value::Array(values) => PatchValue::List(
            values
                .into_iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(ToOwned::to_owned)
                        .ok_or(ConfigError::InvalidValue {
                            key,
                            reason: "expected an array of strings",
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => {
            return Err(ConfigError::InvalidValue {
                key,
                reason: "unsupported TOML value type",
            });
        }
    };
    patch.set(key, value)
}

fn parse_environment<I, K, V>(environment: I) -> Result<Patch, ConfigError>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut patch = Patch::default();
    for (key, value) in environment {
        let key = key.as_ref();
        let Some(suffix) = key.strip_prefix("IMPOSSIBLE_") else {
            continue;
        };
        let normalized = match suffix {
            "BIND" => ConfigKey::Bind,
            "ALLOW_INSECURE_REMOTE" => ConfigKey::AllowInsecureRemote,
            "AUTH_ENV" => ConfigKey::AuthEnv,
            "AUTH_TOKEN" => ConfigKey::AuthTokenPresent,
            "AUTH_FILE" => ConfigKey::AuthFile,
            "ADMIN_AUTH_ENV" => ConfigKey::AdminAuthEnv,
            "ADMIN_AUTH_TOKEN" => ConfigKey::AdminAuthTokenPresent,
            "ADMIN_AUTH_FILE" => ConfigKey::AdminAuthFile,
            "ALLOWED_ORIGINS" => ConfigKey::AllowedOrigins,
            "MODEL_DIRECTORIES" => ConfigKey::ModelDirectories,
            "MAX_BODY_BYTES" => ConfigKey::MaxBodyBytes,
            "MAX_ITEMS" => ConfigKey::MaxItems,
            "MAX_TOKENS" => ConfigKey::MaxTokens,
            "MAX_QUEUE_DEPTH" => ConfigKey::MaxQueueDepth,
            "MAX_BATCH_ITEMS" => ConfigKey::MaxBatchItems,
            "MAX_CONCURRENCY" => ConfigKey::MaxConcurrency,
            "REQUEST_TIMEOUT_MS" => ConfigKey::RequestTimeout,
            "SHUTDOWN_TIMEOUT_MS" => ConfigKey::ShutdownTimeout,
            _ => {
                return Err(ConfigError::UnknownKey {
                    source: ConfigSource::Environment,
                });
            }
        };
        if matches!(
            normalized,
            ConfigKey::AuthTokenPresent | ConfigKey::AdminAuthTokenPresent
        ) {
            if value.as_ref().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: normalized,
                    reason: "credential must not be empty",
                });
            }
            patch.set(normalized, PatchValue::Scalar(key.to_owned()))?;
        } else {
            patch.set(normalized, PatchValue::Scalar(value.as_ref().to_owned()))?;
        }
    }
    Ok(patch)
}

fn parse_cli<A>(arguments: A) -> Result<Patch, ConfigError>
where
    A: IntoIterator,
    A::Item: AsRef<str>,
{
    let mut patch = Patch::default();
    let mut args = arguments.into_iter();
    while let Some(argument) = args.next() {
        let argument = argument.as_ref();
        let Some(option) = argument.strip_prefix("--") else {
            return Err(ConfigError::UnexpectedArgument);
        };
        let option_name = option.split_once('=').map_or(option, |(name, _)| name);
        let normalized_name = option_name.replace('-', "_");
        if credential_like_option(&normalized_name) {
            return Err(ConfigError::UnknownKey {
                source: ConfigSource::CommandLine,
            });
        }
        let normalized_name = match normalized_name.as_str() {
            "max_body_bytes" => "limits.max_body_bytes",
            "max_items" => "limits.max_items",
            "max_tokens" => "limits.max_tokens",
            "max_queue_depth" => "limits.max_queue_depth",
            "max_batch_items" => "limits.max_batch_items",
            "max_concurrency" => "limits.max_concurrency",
            "request_timeout_ms" => "limits.request_timeout_ms",
            "shutdown_timeout_ms" => "limits.shutdown_timeout_ms",
            other => other,
        };
        let key = parse_key(normalized_name).ok_or(ConfigError::UnknownKey {
            source: ConfigSource::CommandLine,
        })?;
        if option.contains('=') {
            return Err(ConfigError::InvalidValue {
                key,
                reason: "inline option values are not supported",
            });
        }
        if key == ConfigKey::AllowInsecureRemote {
            patch.set(key, PatchValue::Scalar("true".to_owned()))?;
            continue;
        }
        let value = args.next().ok_or(ConfigError::MissingValue(key))?;
        patch.set(key, PatchValue::Scalar(value.as_ref().to_owned()))?;
    }
    Ok(patch)
}

fn credential_like_option(option: &str) -> bool {
    option == "auth"
        || option.contains("token")
        || option.contains("secret")
        || option.contains("password")
        || option.contains("credential")
        || option.contains("api_key")
}

fn parse_key(key: &str) -> Option<ConfigKey> {
    match key {
        "bind" => Some(ConfigKey::Bind),
        "allow_insecure_remote" => Some(ConfigKey::AllowInsecureRemote),
        "auth_env" => Some(ConfigKey::AuthEnv),
        "auth_token_present" => Some(ConfigKey::AuthTokenPresent),
        "auth_file" => Some(ConfigKey::AuthFile),
        "admin_auth_env" => Some(ConfigKey::AdminAuthEnv),
        "admin_auth_token_present" => Some(ConfigKey::AdminAuthTokenPresent),
        "admin_auth_file" => Some(ConfigKey::AdminAuthFile),
        "allowed_origins" => Some(ConfigKey::AllowedOrigins),
        "model_directories" => Some(ConfigKey::ModelDirectories),
        "limits.max_body_bytes" => Some(ConfigKey::MaxBodyBytes),
        "limits.max_items" => Some(ConfigKey::MaxItems),
        "limits.max_tokens" => Some(ConfigKey::MaxTokens),
        "limits.max_queue_depth" => Some(ConfigKey::MaxQueueDepth),
        "limits.max_batch_items" => Some(ConfigKey::MaxBatchItems),
        "limits.max_concurrency" => Some(ConfigKey::MaxConcurrency),
        "limits.request_timeout_ms" => Some(ConfigKey::RequestTimeout),
        "limits.shutdown_timeout_ms" => Some(ConfigKey::ShutdownTimeout),
        _ => None,
    }
}

fn apply_value(
    target: &mut ServerConfig,
    key: ConfigKey,
    value: PatchValue,
) -> Result<(), ConfigError> {
    let invalid = |reason| ConfigError::InvalidValue { key, reason };
    let scalar = || match &value {
        PatchValue::Scalar(value) => Ok(value.as_str()),
        PatchValue::List(_) => Err(invalid("expected scalar value")),
    };
    match key {
        ConfigKey::Bind => {
            target.bind = scalar()?
                .parse()
                .map_err(|_| invalid("expected socket address"))?;
        }
        ConfigKey::AllowInsecureRemote => {
            target.allow_insecure_remote =
                parse_bool(scalar()?).ok_or_else(|| invalid("expected boolean"))?;
        }
        ConfigKey::AuthEnv => {
            target.auth = Some(CredentialSource::Environment(validate_env_name(
                scalar()?,
                key,
            )?));
        }
        ConfigKey::AuthTokenPresent => {
            target.auth = Some(CredentialSource::Environment(scalar()?.to_owned()));
        }
        ConfigKey::AuthFile => target.auth = Some(CredentialSource::File(PathBuf::from(scalar()?))),
        ConfigKey::AdminAuthEnv => {
            target.admin_auth = Some(CredentialSource::Environment(validate_env_name(
                scalar()?,
                key,
            )?));
        }
        ConfigKey::AdminAuthTokenPresent => {
            target.admin_auth = Some(CredentialSource::Environment(scalar()?.to_owned()));
        }
        ConfigKey::AdminAuthFile => {
            target.admin_auth = Some(CredentialSource::File(PathBuf::from(scalar()?)));
        }
        ConfigKey::AllowedOrigins => {
            target.allowed_origins = list_value(value, key)?;
        }
        ConfigKey::ModelDirectories => {
            target.model_directories = list_value(value, key)?
                .into_iter()
                .map(PathBuf::from)
                .collect();
        }
        ConfigKey::MaxBodyBytes => target.limits.max_body_bytes = parse_usize(scalar()?, key)?,
        ConfigKey::MaxItems => target.limits.max_items = parse_usize(scalar()?, key)?,
        ConfigKey::MaxTokens => target.limits.max_tokens = parse_usize(scalar()?, key)?,
        ConfigKey::MaxQueueDepth => target.limits.max_queue_depth = parse_usize(scalar()?, key)?,
        ConfigKey::MaxBatchItems => target.limits.max_batch_items = parse_usize(scalar()?, key)?,
        ConfigKey::MaxConcurrency => target.limits.max_concurrency = parse_usize(scalar()?, key)?,
        ConfigKey::RequestTimeout => {
            target.limits.request_timeout = parse_duration(scalar()?, key)?;
        }
        ConfigKey::ShutdownTimeout => {
            target.limits.shutdown_timeout = parse_duration(scalar()?, key)?;
        }
        ConfigKey::Toml | ConfigKey::Path => {
            return Err(invalid("field is not directly configurable"));
        }
    }
    Ok(())
}

fn list_value(value: PatchValue, key: ConfigKey) -> Result<Vec<String>, ConfigError> {
    let values = match value {
        PatchValue::Scalar(value) => value
            .split(',')
            .map(str::trim)
            .map(ToOwned::to_owned)
            .collect(),
        PatchValue::List(values) => values,
    };
    if values.iter().any(String::is_empty) {
        return Err(ConfigError::InvalidValue {
            key,
            reason: "list entries must not be empty",
        });
    }
    Ok(values)
}

fn valid_origin(origin: &str) -> bool {
    if origin.is_empty() || origin.contains('*') {
        return false;
    }
    let Ok(url) = Url::parse(origin) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return false;
    }
    let Origin::Tuple(_, _, _) = url.origin() else {
        return false;
    };
    url.origin().ascii_serialization() == origin
}
fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}
fn parse_usize(value: &str, key: ConfigKey) -> Result<usize, ConfigError> {
    value.parse().map_err(|_| ConfigError::InvalidValue {
        key,
        reason: "expected positive integer",
    })
}
fn parse_duration(value: &str, key: ConfigKey) -> Result<Duration, ConfigError> {
    value
        .parse::<u64>()
        .map(Duration::from_millis)
        .map_err(|_| ConfigError::InvalidValue {
            key,
            reason: "expected milliseconds",
        })
}
fn validate_env_name(value: &str, key: ConfigKey) -> Result<String, ConfigError> {
    if value.is_empty()
        || !value
            .chars()
            .all(|c| c == '_' || c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return Err(ConfigError::InvalidValue {
            key,
            reason: "expected an uppercase environment variable name",
        });
    }
    Ok(value.to_owned())
}

fn validate_bounds(limits: &Limits) -> Result<(), ConfigError> {
    const MIB: usize = 1024 * 1024;
    for (key, value, min, max) in [
        (ConfigKey::MaxBodyBytes, limits.max_body_bytes, 1, 64 * MIB),
        (ConfigKey::MaxItems, limits.max_items, 1, 4096),
        (ConfigKey::MaxTokens, limits.max_tokens, 1, 1_000_000),
        (ConfigKey::MaxQueueDepth, limits.max_queue_depth, 1, 100_000),
        (ConfigKey::MaxBatchItems, limits.max_batch_items, 1, 4096),
        (ConfigKey::MaxConcurrency, limits.max_concurrency, 1, 1024),
    ] {
        if !(min..=max).contains(&value) {
            return Err(ConfigError::InvalidValue {
                key,
                reason: "outside supported bounds",
            });
        }
    }
    for (key, value) in [
        (ConfigKey::RequestTimeout, limits.request_timeout),
        (ConfigKey::ShutdownTimeout, limits.shutdown_timeout),
    ] {
        if value < Duration::from_millis(10) || value > Duration::from_secs(600) {
            return Err(ConfigError::InvalidValue {
                key,
                reason: "outside supported bounds",
            });
        }
    }
    Ok(())
}

/// Resolve and canonicalize an existing explicitly configured path.
///
/// # Errors
///
/// Returns a sanitized error when the path does not exist or cannot be resolved.
pub fn canonical_existing_path(path: &Path) -> Result<PathBuf, ConfigError> {
    fs::canonicalize(path).map_err(|_| ConfigError::InvalidValue {
        key: ConfigKey::Path,
        reason: "path does not exist or cannot be resolved",
    })
}

/// Resolve a child and ensure symlinks cannot escape its configured root.
///
/// # Errors
///
/// Returns a sanitized error when either path cannot be resolved or the child
/// escapes the configured root after resolving symlinks.
pub fn canonical_path_within(root: &Path, child: &Path) -> Result<PathBuf, ConfigError> {
    let root = canonical_existing_path(root)?;
    let candidate = canonical_existing_path(&root.join(child))?;
    if !candidate.starts_with(&root) {
        return Err(ConfigError::InvalidValue {
            key: ConfigKey::Path,
            reason: "path escapes configured root",
        });
    }
    Ok(candidate)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn precedence_is_cli_then_environment_then_toml_then_defaults() {
        let path = std::env::temp_dir().join(format!("impossible-config-{}", std::process::id()));
        fs::write(
            &path,
            "bind = \"127.0.0.1:1001\"\n[limits]\nmax_items = 4\n",
        )
        .expect("fixture");
        let config = ServerConfig::load(
            Some(&path),
            [
                ("IMPOSSIBLE_BIND", "127.0.0.1:1002"),
                ("IMPOSSIBLE_MAX_ITEMS", "8"),
            ],
            ["--bind", "127.0.0.1:1003", "--max-items", "16"],
        )
        .expect("valid config");
        let _ = fs::remove_file(path);
        assert_eq!(config.bind, "127.0.0.1:1003".parse().expect("address"));
        assert_eq!(config.limits.max_items, 16);
        assert_eq!(config.limits.max_tokens, Limits::default().max_tokens);
    }

    #[test]
    fn strict_unknown_keys_across_sources() {
        assert!(matches!(
            parse_toml("mystery = 1"),
            Err(ConfigError::UnknownKey { .. })
        ));
        assert!(matches!(
            parse_environment([("IMPOSSIBLE_MYSTERY", "1")]),
            Err(ConfigError::UnknownKey { .. })
        ));
        assert!(matches!(
            parse_cli(["--mystery", "1"]),
            Err(ConfigError::UnknownKey { .. })
        ));
    }

    #[test]
    fn display_and_debug_never_echo_untrusted_configuration_text() {
        const SENTINEL: &str = "SENTINEL_SECRET_PATH";
        let errors = [
            parse_toml("\"SENTINEL_SECRET_PATH\\n\\u001b[31m\" = 1")
                .err()
                .expect("unknown TOML key"),
            parse_environment([("IMPOSSIBLE_SENTINEL_SECRET_PATH\n\u{1b}[31m", "secret")])
                .err()
                .expect("unknown environment key"),
            parse_cli(["--SENTINEL_SECRET_PATH\n\u{1b}[31m", "secret"])
                .err()
                .expect("unknown command-line key"),
            parse_cli(["--bind", "SENTINEL_SECRET_PATH\n\u{1b}[31m"])
                .and_then(|patch| {
                    let mut config = ServerConfig::default();
                    patch.apply(&mut config)
                })
                .expect_err("invalid command-line value"),
            ServerConfig::load(
                Some(Path::new("C:/SENTINEL_SECRET_PATH/private/config.toml")),
                std::iter::empty::<(&str, &str)>(),
                std::iter::empty::<&str>(),
            )
            .expect_err("unavailable path"),
        ];

        for error in errors {
            let rendered = format!("{error} {error:?}");
            assert!(!rendered.contains(SENTINEL), "leaked sentinel: {rendered}");
            assert!(!rendered.contains('\n'), "leaked newline: {rendered:?}");
            assert!(
                !rendered.contains('\u{1b}'),
                "leaked ANSI escape: {rendered:?}"
            );
            assert!(error.diagnostic().code.starts_with("config_"));
        }
    }

    #[test]
    fn loopback_ipv4_and_ipv6_are_safe_but_remote_requires_policy() {
        for bind in ["127.0.0.1:8080", "[::1]:8080"] {
            let config = ServerConfig {
                bind: bind.parse().expect("address"),
                ..ServerConfig::default()
            };
            assert_eq!(config.validate(), Ok(()));
        }
        let remote = ServerConfig {
            bind: "0.0.0.0:8080".parse().expect("address"),
            ..ServerConfig::default()
        };
        assert_eq!(remote.validate(), Err(ConfigError::UnsafeRemoteBind));
        let protected = ServerConfig {
            auth: Some(CredentialSource::Environment("IMPOSSIBLE_TOKEN".to_owned())),
            ..remote
        };
        assert_eq!(protected.validate(), Ok(()));
    }

    #[test]
    fn boundaries_reject_zero_and_tiny_timeout() {
        for key in [
            "max-body-bytes",
            "max-items",
            "max-tokens",
            "max-queue-depth",
            "max-batch-items",
            "max-concurrency",
        ] {
            let option = format!("--{key}");
            assert!(
                ServerConfig::load(
                    None,
                    std::iter::empty::<(&str, &str)>(),
                    [option.as_str(), "0"]
                )
                .is_err()
            );
        }
        assert!(
            ServerConfig::load(
                None,
                std::iter::empty::<(&str, &str)>(),
                ["--request-timeout-ms", "1"]
            )
            .is_err()
        );
    }

    #[test]
    fn cli_rejects_inline_secrets_and_admin_aliasing() {
        let separated = ServerConfig::load(
            None,
            std::iter::empty::<(&str, &str)>(),
            ["--token=sentinel-secret"],
        )
        .expect_err("inline secret must fail");
        let adjacent = ServerConfig::load(
            None,
            std::iter::empty::<(&str, &str)>(),
            ["--api-key=sentinel-api-key"],
        )
        .expect_err("inline API key must fail");
        let debug = format!("{separated:?} {adjacent:?}");
        let display = format!("{separated} {adjacent}");
        assert!(!debug.contains("sentinel"));
        assert!(!display.contains("sentinel"));
        assert!(
            ServerConfig::load(
                None,
                std::iter::empty::<(&str, &str)>(),
                ["--token", "sentinel-secret"]
            )
            .is_err()
        );
        let source = CredentialSource::Environment("IMPOSSIBLE_TOKEN".to_owned());
        let config = ServerConfig {
            auth: Some(source.clone()),
            admin_auth: Some(source),
            ..ServerConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(ConfigError::InvalidAdminCredentialPolicy)
        );
    }

    #[test]
    fn debug_redacts_credentials_network_and_configured_paths() {
        let config = ServerConfig {
            bind: "192.0.2.40:9123".parse().expect("address"),
            auth: Some(CredentialSource::Environment(
                "SENTINEL_PRIVATE_ENV".to_owned(),
            )),
            admin_auth: Some(CredentialSource::File(PathBuf::from(
                "C:/sentinel/private/admin-token",
            ))),
            allowed_origins: vec!["https://sentinel.internal".to_owned()],
            model_directories: vec![PathBuf::from("C:/sentinel/private/models")],
            ..ServerConfig::default()
        };
        let debug = format!("{config:?}");
        for sensitive in [
            "192.0.2.40",
            "SENTINEL_PRIVATE_ENV",
            "admin-token",
            "sentinel.internal",
            "private/models",
        ] {
            assert!(!debug.contains(sensitive), "leaked {sensitive:?}: {debug}");
        }
        assert!(debug.contains("allowed_origin_count: 1"));
        assert!(debug.contains("model_directory_count: 1"));
        assert!(debug.contains("max_body_bytes"));
    }

    #[test]
    fn direct_environment_secrets_are_referenced_but_not_retained() {
        let config = ServerConfig::load(
            None,
            [
                ("IMPOSSIBLE_AUTH_TOKEN", "sentinel-public-secret"),
                ("IMPOSSIBLE_ADMIN_AUTH_TOKEN", "sentinel-admin-secret"),
            ],
            std::iter::empty::<&str>(),
        )
        .expect("valid config");
        let debug = format!("{config:?}");
        assert!(!debug.contains("sentinel-public-secret"));
        assert!(!debug.contains("sentinel-admin-secret"));
        assert_eq!(
            config.auth,
            Some(CredentialSource::Environment(
                "IMPOSSIBLE_AUTH_TOKEN".to_owned()
            ))
        );
    }

    #[test]
    fn toml_lists_and_hashes_in_strings_are_supported() {
        let patch = parse_toml(
            "allowed_origins = [\"http://localhost:3000\", \"https://example.test\"]\nmodel_directories = \"models#one\" # comment",
        )
        .expect("valid subset");
        let mut config = ServerConfig::default();
        patch.apply(&mut config).expect("valid values");
        assert_eq!(config.allowed_origins.len(), 2);
        assert_eq!(config.model_directories, vec![PathBuf::from("models#one")]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn toml_model_directory_arrays_preserve_commas_inside_entries() {
        let patch = parse_toml("model_directories = [\"C:/models,private\", \"D:/other models\"]")
            .expect("valid typed array");
        let mut config = ServerConfig::default();
        patch.apply(&mut config).expect("valid values");
        assert_eq!(
            config.model_directories,
            [
                PathBuf::from("C:/models,private"),
                PathBuf::from("D:/other models")
            ]
        );
    }

    #[test]
    fn list_sources_reject_empty_entries_but_allow_an_empty_toml_array() {
        for input in [
            "model_directories = [\"\"]",
            "model_directories = [\"models\", \"\"]",
        ] {
            let patch = parse_toml(input).expect("well-typed TOML");
            assert!(patch.apply(&mut ServerConfig::default()).is_err());
        }
        for value in ["", ",", "models,", ",models", "models,,other"] {
            let patch = parse_environment([("IMPOSSIBLE_MODEL_DIRECTORIES", value)])
                .expect("known environment key");
            assert!(
                patch.apply(&mut ServerConfig::default()).is_err(),
                "accepted {value:?}"
            );
        }

        let patch = parse_toml("model_directories = []").expect("empty typed array");
        let mut config = ServerConfig {
            model_directories: vec![PathBuf::from("old")],
            ..ServerConfig::default()
        };
        patch.apply(&mut config).expect("empty array clears list");
        assert!(config.model_directories.is_empty());
    }

    #[test]
    fn list_precedence_replaces_typed_toml_values_without_reinterpreting_them() {
        let path = std::env::temp_dir().join(format!(
            "impossible-list-precedence-{}.toml",
            std::process::id()
        ));
        fs::write(&path, "model_directories = [\"toml,one\", \"toml-two\"]\n").expect("fixture");
        let config = ServerConfig::load(
            Some(&path),
            [("IMPOSSIBLE_MODEL_DIRECTORIES", "environment")],
            ["--model-directories", "command-line"],
        )
        .expect("valid precedence");
        let _ = fs::remove_file(path);
        assert_eq!(config.model_directories, [PathBuf::from("command-line")]);
    }

    #[test]
    fn origin_validation_rejects_paths_wildcards_and_injection() {
        for origin in [
            "*",
            "https://example.test/path",
            "https://example.test?x=1",
            "https://example.test#fragment",
            "https://user@example.test",
            "https://user:password@example.test",
            "https://*.example.test",
            "https://example.test:443",
            "https://EXAMPLE.test",
            "https://example.test/",
            "https://example.test:",
            "https://example.test:99999",
            "https://example.test%2f.evil",
            "https://[::1",
            "https:///missing-authority",
            "http:example.test",
            "file://local",
            "https://ok.test\r\nInjected: yes",
        ] {
            let config = ServerConfig {
                allowed_origins: vec![origin.to_owned()],
                ..ServerConfig::default()
            };
            assert!(config.validate().is_err(), "accepted {origin:?}");
        }

        for origin in [
            "https://example.test",
            "https://example.test:8443",
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
        ] {
            let config = ServerConfig {
                allowed_origins: vec![origin.to_owned()],
                ..ServerConfig::default()
            };
            assert_eq!(config.validate(), Ok(()), "rejected {origin:?}");
        }
    }

    #[test]
    fn duplicate_and_conflicting_keys_fail_closed() {
        assert!(parse_toml("bind = \"[::1]:1\"\nbind = \"[::1]:2\"").is_err());
        assert!(
            ServerConfig::load(
                None,
                [
                    ("IMPOSSIBLE_AUTH_ENV", "TOKEN"),
                    ("IMPOSSIBLE_AUTH_FILE", "token.txt")
                ],
                std::iter::empty::<&str>(),
            )
            .is_err()
        );
    }

    #[test]
    fn canonical_child_cannot_use_parent_traversal() {
        let fixture = std::env::temp_dir().join(format!("impossible-path-{}", std::process::id()));
        let root = fixture.join("root");
        let outside = fixture.join("outside");
        fs::create_dir_all(&root).expect("root fixture");
        fs::create_dir_all(&outside).expect("outside fixture");
        assert_eq!(
            canonical_path_within(&root, Path::new(".")),
            canonical_existing_path(&root)
        );
        assert!(canonical_path_within(&root, Path::new("../outside")).is_err());
        let _ = fs::remove_dir_all(fixture);
    }
}

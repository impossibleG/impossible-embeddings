//! Typed configuration with deterministic precedence and secure defaults.

use std::{
    collections::BTreeMap,
    env, fmt, fs,
    net::{IpAddr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    time::Duration,
};

/// A non-secret reference to where a credential is stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialSource {
    /// Read the credential from this environment variable.
    Environment(String),
    /// Read the credential from this file at startup.
    File(PathBuf),
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

/// Configuration failure that never includes credential values or file contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// An option or TOML key is not supported.
    UnknownKey(String),
    /// A known option has an invalid value.
    InvalidValue {
        /// Public option or key name.
        key: String,
        /// Stable explanation that excludes the rejected value.
        reason: &'static str,
    },
    /// A CLI option requires a following value.
    MissingValue(String),
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
            Self::UnknownKey(key) => write!(f, "unknown configuration key: {key}"),
            Self::InvalidValue { key, reason } => write!(f, "invalid value for {key}: {reason}"),
            Self::MissingValue(key) => write!(f, "missing value for {key}"),
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

#[derive(Default)]
struct Patch(BTreeMap<String, String>);

impl Patch {
    fn set(&mut self, key: &str, value: String) -> Result<(), ConfigError> {
        if !KNOWN_KEYS.contains(&key) {
            return Err(ConfigError::UnknownKey(key.to_owned()));
        }
        if self.0.insert(key.to_owned(), value).is_some() {
            return Err(ConfigError::InvalidValue {
                key: key.to_owned(),
                reason: "duplicate key in one configuration source",
            });
        }
        Ok(())
    }

    fn apply(self, target: &mut ServerConfig) -> Result<(), ConfigError> {
        for alternatives in [
            ["auth_env", "auth_file", "auth_token_present"],
            [
                "admin_auth_env",
                "admin_auth_file",
                "admin_auth_token_present",
            ],
        ] {
            if alternatives
                .iter()
                .filter(|key| self.0.contains_key(**key))
                .count()
                > 1
            {
                return Err(ConfigError::InvalidValue {
                    key: alternatives[0].to_owned(),
                    reason: "configure exactly one credential source",
                });
            }
        }
        for (key, value) in self.0 {
            apply_value(target, &key, &value)?;
        }
        Ok(())
    }
}

const KNOWN_KEYS: &[&str] = &[
    "bind",
    "allow_insecure_remote",
    "auth_env",
    "auth_token_present",
    "auth_file",
    "admin_auth_env",
    "admin_auth_token_present",
    "admin_auth_file",
    "allowed_origins",
    "model_directories",
    "limits.max_body_bytes",
    "limits.max_items",
    "limits.max_tokens",
    "limits.max_queue_depth",
    "limits.max_batch_items",
    "limits.max_concurrency",
    "limits.request_timeout_ms",
    "limits.shutdown_timeout_ms",
];

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
                key: "allowed_origins".to_owned(),
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
            key: "toml".to_owned(),
            reason: "invalid TOML document",
        })?;
    for (key, value) in table {
        if key == "limits" {
            let toml::Value::Table(limits) = value else {
                return Err(ConfigError::InvalidValue {
                    key,
                    reason: "expected table",
                });
            };
            for (limit_key, limit_value) in limits {
                insert_toml_value(&mut patch, &format!("limits.{limit_key}"), limit_value)?;
            }
        } else {
            insert_toml_value(&mut patch, &key, value)?;
        }
    }
    Ok(patch)
}

fn insert_toml_value(patch: &mut Patch, key: &str, value: toml::Value) -> Result<(), ConfigError> {
    let value = match value {
        toml::Value::String(value) => value,
        toml::Value::Integer(value) => value.to_string(),
        toml::Value::Boolean(value) => value.to_string(),
        toml::Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| ConfigError::InvalidValue {
                        key: key.to_owned(),
                        reason: "expected an array of strings",
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(","),
        _ => {
            return Err(ConfigError::InvalidValue {
                key: key.to_owned(),
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
            "BIND" => "bind",
            "ALLOW_INSECURE_REMOTE" => "allow_insecure_remote",
            "AUTH_ENV" => "auth_env",
            "AUTH_TOKEN" => "auth_token_present",
            "AUTH_FILE" => "auth_file",
            "ADMIN_AUTH_ENV" => "admin_auth_env",
            "ADMIN_AUTH_TOKEN" => "admin_auth_token_present",
            "ADMIN_AUTH_FILE" => "admin_auth_file",
            "ALLOWED_ORIGINS" => "allowed_origins",
            "MODEL_DIRECTORIES" => "model_directories",
            "MAX_BODY_BYTES" => "limits.max_body_bytes",
            "MAX_ITEMS" => "limits.max_items",
            "MAX_TOKENS" => "limits.max_tokens",
            "MAX_QUEUE_DEPTH" => "limits.max_queue_depth",
            "MAX_BATCH_ITEMS" => "limits.max_batch_items",
            "MAX_CONCURRENCY" => "limits.max_concurrency",
            "REQUEST_TIMEOUT_MS" => "limits.request_timeout_ms",
            "SHUTDOWN_TIMEOUT_MS" => "limits.shutdown_timeout_ms",
            _ => return Err(ConfigError::UnknownKey(key.to_owned())),
        };
        if normalized.ends_with("_token_present") {
            if value.as_ref().is_empty() {
                return Err(ConfigError::InvalidValue {
                    key: key.to_owned(),
                    reason: "credential must not be empty",
                });
            }
            patch.set(normalized, key.to_owned())?;
        } else {
            patch.set(normalized, value.as_ref().to_owned())?;
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
            return Err(ConfigError::UnknownKey(argument.to_owned()));
        };
        if option.contains("token") || option.contains("secret") || option == "auth" {
            return Err(ConfigError::UnknownKey(option.to_owned()));
        }
        let normalized = option.replace('-', "_");
        let key = match normalized.as_str() {
            "max_body_bytes"
            | "max_items"
            | "max_tokens"
            | "max_queue_depth"
            | "max_batch_items"
            | "max_concurrency"
            | "request_timeout_ms"
            | "shutdown_timeout_ms" => format!("limits.{normalized}"),
            _ => normalized,
        };
        if key == "allow_insecure_remote" {
            patch.set(&key, "true".to_owned())?;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| ConfigError::MissingValue(option.to_owned()))?;
        patch.set(&key, value.as_ref().to_owned())?;
    }
    Ok(patch)
}

fn apply_value(target: &mut ServerConfig, key: &str, value: &str) -> Result<(), ConfigError> {
    let invalid = |reason| ConfigError::InvalidValue {
        key: key.to_owned(),
        reason,
    };
    match key {
        "bind" => {
            target.bind = value
                .parse()
                .map_err(|_| invalid("expected socket address"))?;
        }
        "allow_insecure_remote" => {
            target.allow_insecure_remote =
                parse_bool(value).ok_or_else(|| invalid("expected boolean"))?;
        }
        "auth_env" => {
            target.auth = Some(CredentialSource::Environment(validate_env_name(
                value, key,
            )?));
        }
        "auth_token_present" => {
            target.auth = Some(CredentialSource::Environment(value.to_owned()));
        }
        "auth_file" => target.auth = Some(CredentialSource::File(PathBuf::from(value))),
        "admin_auth_env" => {
            target.admin_auth = Some(CredentialSource::Environment(validate_env_name(
                value, key,
            )?));
        }
        "admin_auth_token_present" => {
            target.admin_auth = Some(CredentialSource::Environment(value.to_owned()));
        }
        "admin_auth_file" => target.admin_auth = Some(CredentialSource::File(PathBuf::from(value))),
        "allowed_origins" => target.allowed_origins = split_list(value),
        "model_directories" => {
            target.model_directories = split_list(value).into_iter().map(PathBuf::from).collect();
        }
        "limits.max_body_bytes" => target.limits.max_body_bytes = parse_usize(value, key)?,
        "limits.max_items" => target.limits.max_items = parse_usize(value, key)?,
        "limits.max_tokens" => target.limits.max_tokens = parse_usize(value, key)?,
        "limits.max_queue_depth" => target.limits.max_queue_depth = parse_usize(value, key)?,
        "limits.max_batch_items" => target.limits.max_batch_items = parse_usize(value, key)?,
        "limits.max_concurrency" => target.limits.max_concurrency = parse_usize(value, key)?,
        "limits.request_timeout_ms" => target.limits.request_timeout = parse_duration(value, key)?,
        "limits.shutdown_timeout_ms" => {
            target.limits.shutdown_timeout = parse_duration(value, key)?;
        }
        _ => return Err(ConfigError::UnknownKey(key.to_owned())),
    }
    Ok(())
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn valid_origin(origin: &str) -> bool {
    let authority = origin
        .strip_prefix("https://")
        .or_else(|| origin.strip_prefix("http://"));
    authority.is_some_and(|value| {
        !value.is_empty()
            && !value.contains(['/', '?', '#', '@', '\r', '\n', ' ', '\t'])
            && value != "*"
    })
}
fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}
fn parse_usize(value: &str, key: &str) -> Result<usize, ConfigError> {
    value.parse().map_err(|_| ConfigError::InvalidValue {
        key: key.to_owned(),
        reason: "expected positive integer",
    })
}
fn parse_duration(value: &str, key: &str) -> Result<Duration, ConfigError> {
    value
        .parse::<u64>()
        .map(Duration::from_millis)
        .map_err(|_| ConfigError::InvalidValue {
            key: key.to_owned(),
            reason: "expected milliseconds",
        })
}
fn validate_env_name(value: &str, key: &str) -> Result<String, ConfigError> {
    if value.is_empty()
        || !value
            .chars()
            .all(|c| c == '_' || c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return Err(ConfigError::InvalidValue {
            key: key.to_owned(),
            reason: "expected an uppercase environment variable name",
        });
    }
    Ok(value.to_owned())
}

fn validate_bounds(limits: &Limits) -> Result<(), ConfigError> {
    const MIB: usize = 1024 * 1024;
    for (key, value, min, max) in [
        ("limits.max_body_bytes", limits.max_body_bytes, 1, 64 * MIB),
        ("limits.max_items", limits.max_items, 1, 4096),
        ("limits.max_tokens", limits.max_tokens, 1, 1_000_000),
        ("limits.max_queue_depth", limits.max_queue_depth, 1, 100_000),
        ("limits.max_batch_items", limits.max_batch_items, 1, 4096),
        ("limits.max_concurrency", limits.max_concurrency, 1, 1024),
    ] {
        if !(min..=max).contains(&value) {
            return Err(ConfigError::InvalidValue {
                key: key.to_owned(),
                reason: "outside supported bounds",
            });
        }
    }
    for (key, value) in [
        ("limits.request_timeout_ms", limits.request_timeout),
        ("limits.shutdown_timeout_ms", limits.shutdown_timeout),
    ] {
        if value < Duration::from_millis(10) || value > Duration::from_secs(600) {
            return Err(ConfigError::InvalidValue {
                key: key.to_owned(),
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
        key: "path".to_owned(),
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
            key: "path".to_owned(),
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
            Err(ConfigError::UnknownKey(_))
        ));
        assert!(matches!(
            parse_environment([("IMPOSSIBLE_MYSTERY", "1")]),
            Err(ConfigError::UnknownKey(_))
        ));
        assert!(matches!(
            parse_cli(["--mystery", "1"]),
            Err(ConfigError::UnknownKey(_))
        ));
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
    fn origin_validation_rejects_paths_wildcards_and_injection() {
        for origin in [
            "*",
            "https://example.test/path",
            "https://example.test?x=1",
            "file://local",
            "https://ok.test\r\nInjected: yes",
        ] {
            let config = ServerConfig {
                allowed_origins: vec![origin.to_owned()],
                ..ServerConfig::default()
            };
            assert!(config.validate().is_err(), "accepted {origin:?}");
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

//! Credential loading and request-origin primitives.

use std::{env, fmt, fs, path::Path};

use crate::config::CredentialSource;

/// Owned secret bytes. Debug and Display never expose the value; memory is wiped on drop.
pub struct Secret(Vec<u8>);

impl Secret {
    /// Construct a non-empty secret.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::Invalid`] for empty or unreasonably large values.
    pub fn new(value: impl Into<Vec<u8>>) -> Result<Self, CredentialError> {
        let value = value.into();
        if value.is_empty() || value.len() > 16 * 1024 {
            return Err(CredentialError::Invalid);
        }
        Ok(Self(value))
    }

    /// Verify a candidate without early-returning at the first unequal byte.
    #[must_use]
    pub fn verify(&self, candidate: &[u8]) -> bool {
        constant_time_eq(&self.0, candidate)
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret([REDACTED])")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

/// Sanitized credential-loading failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialError {
    /// The configured source could not be read.
    Unavailable,
    /// The source contained an empty or unreasonably large credential.
    Invalid,
}

impl fmt::Display for CredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => f.write_str("credential source is unavailable"),
            Self::Invalid => f.write_str("credential source is invalid"),
        }
    }
}

impl std::error::Error for CredentialError {}

/// Load a credential without retaining its value in the typed configuration.
///
/// # Errors
///
/// Returns a sanitized error when the source is missing, unreadable, empty, or too large.
pub fn load_credential(source: &CredentialSource) -> Result<Secret, CredentialError> {
    let bytes = match source {
        CredentialSource::Environment(name) => env::var_os(name)
            .map(|value| value.to_string_lossy().into_owned().into_bytes())
            .ok_or(CredentialError::Unavailable)?,
        CredentialSource::File(path) => read_secret_file(path)?,
    };
    let trimmed = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
    let trimmed = trimmed.strip_suffix(b"\r").unwrap_or(trimmed);
    Secret::new(trimmed.to_vec())
}

fn read_secret_file(path: &Path) -> Result<Vec<u8>, CredentialError> {
    let metadata = fs::metadata(path).map_err(|_| CredentialError::Unavailable)?;
    if !metadata.is_file() || metadata.len() > 16 * 1024 {
        return Err(CredentialError::Invalid);
    }
    fs::read(path).map_err(|_| CredentialError::Unavailable)
}

fn constant_time_eq(expected: &[u8], candidate: &[u8]) -> bool {
    let max_len = expected.len().max(candidate.len());
    let mut difference = expected.len() ^ candidate.len();
    for index in 0..max_len {
        let left = expected.get(index).copied().unwrap_or_default();
        let right = candidate.get(index).copied().unwrap_or_default();
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

/// Exact-match browser origin allowlist.
#[derive(Clone, Debug, Default)]
pub struct OriginPolicy {
    allowed: Vec<String>,
}

impl OriginPolicy {
    /// Create a policy from configuration that has already been validated.
    #[must_use]
    pub fn new(allowed: Vec<String>) -> Self {
        Self { allowed }
    }

    /// Return whether an Origin header is allowed. Missing origins represent non-browser clients.
    #[must_use]
    pub fn allows(&self, origin: Option<&str>) -> bool {
        origin.is_none_or(|value| {
            self.allowed
                .iter()
                .any(|allowed| constant_time_eq(allowed.as_bytes(), value.as_bytes()))
        })
    }

    /// Value suitable for `Access-Control-Allow-Origin`, never a wildcard.
    #[must_use]
    pub fn response_origin<'a>(&self, origin: Option<&'a str>) -> Option<&'a str> {
        origin.filter(|value| self.allows(Some(value)))
    }
}

/// Verify an RFC 6750-style `Authorization: Bearer` header.
#[must_use]
pub fn verify_bearer(header: Option<&str>, secret: &Secret) -> bool {
    let Some(value) = header else {
        return false;
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return false;
    };
    !token.is_empty() && secret.verify(token.as_bytes())
}

/// Check at runtime that public and administrative credentials are distinct.
///
/// # Errors
///
/// Returns [`CredentialError::Invalid`] when both credentials have the same value.
pub fn verify_distinct_credentials(public: &Secret, admin: &Secret) -> Result<(), CredentialError> {
    if public.verify(&admin.0) {
        Err(CredentialError::Invalid)
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn bearer_is_exact_and_debug_is_redacted() {
        let secret = Secret::new(b"sentinel-secret-value".to_vec()).expect("secret");
        assert!(verify_bearer(Some("Bearer sentinel-secret-value"), &secret));
        assert!(!verify_bearer(
            Some("bearer sentinel-secret-value"),
            &secret
        ));
        assert!(!verify_bearer(
            Some("Bearer sentinel-secret-valuE"),
            &secret
        ));
        assert!(!verify_bearer(
            Some("Bearer sentinel-secret-value-extra"),
            &secret
        ));
        assert!(!format!("{secret:?}").contains("sentinel-secret-value"));
    }

    #[test]
    fn origins_are_exact_and_missing_origin_is_allowed() {
        let policy = OriginPolicy::new(vec!["https://console.example".to_owned()]);
        assert!(policy.allows(None));
        assert!(policy.allows(Some("https://console.example")));
        assert!(!policy.allows(Some("https://console.example.evil")));
        assert!(!policy.allows(Some("https://CONSOLE.example")));
    }

    #[test]
    fn admin_credential_value_must_differ() {
        let public = Secret::new(b"same".to_vec()).expect("secret");
        let admin = Secret::new(b"same".to_vec()).expect("secret");
        assert_eq!(
            verify_distinct_credentials(&public, &admin),
            Err(CredentialError::Invalid)
        );
    }
}

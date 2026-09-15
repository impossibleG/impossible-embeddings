//! Credential loading and request-origin primitives.

use std::{
    env, fmt,
    fs::{File, OpenOptions},
    io::Read,
    path::Path,
};

use zeroize::Zeroizing;

use crate::config::CredentialSource;

/// Owned secret bytes. Debug and Display never expose the value; memory is wiped on drop.
pub struct Secret(Zeroizing<Vec<u8>>);

const MAX_CREDENTIAL_BYTES: usize = 16 * 1024;

impl Secret {
    /// Construct a non-empty secret.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::Invalid`] for empty or unreasonably large values.
    pub fn new(value: impl Into<Vec<u8>>) -> Result<Self, CredentialError> {
        let value = Zeroizing::new(value.into());
        if value.is_empty() || value.len() > MAX_CREDENTIAL_BYTES {
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

impl zeroize::ZeroizeOnDrop for Secret {}

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
    let mut bytes = match source {
        CredentialSource::Environment(name) => Zeroizing::new(
            env::var_os(name)
                .map(|value| value.to_string_lossy().into_owned().into_bytes())
                .ok_or(CredentialError::Unavailable)?,
        ),
        CredentialSource::File(path) => read_secret_file(path)?,
    };
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    if bytes.is_empty() || bytes.len() > MAX_CREDENTIAL_BYTES {
        return Err(CredentialError::Invalid);
    }
    Ok(Secret(bytes))
}

fn read_secret_file(path: &Path) -> Result<Zeroizing<Vec<u8>>, CredentialError> {
    let file = open_secret_file(path)?;
    read_secret_handle(file)
}

fn open_secret_file(path: &Path) -> Result<File, CredentialError> {
    let mut options = OpenOptions::new();
    options.read(true);
    configure_no_follow(&mut options);
    options.open(path).map_err(map_open_error)
}

#[cfg(unix)]
fn configure_no_follow(options: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;

    // Opening a FIFO read-only normally waits indefinitely for a writer. Open every
    // candidate non-blocking so handle metadata can reject FIFOs and other special
    // files without letting a configured credential path stall startup. O_NONBLOCK
    // does not change reads from regular files.
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
}

#[cfg(windows)]
fn configure_no_follow(options: &mut OpenOptions) {
    use std::os::windows::fs::OpenOptionsExt;

    // Open the reparse point itself so its handle metadata can be rejected instead of following it.
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
}

#[cfg(not(any(unix, windows)))]
fn configure_no_follow(_options: &mut OpenOptions) {}

fn map_open_error(_error: std::io::Error) -> CredentialError {
    #[cfg(unix)]
    if _error.raw_os_error() == Some(libc::ELOOP) {
        return CredentialError::Invalid;
    }
    CredentialError::Unavailable
}

fn read_secret_handle(file: File) -> Result<Zeroizing<Vec<u8>>, CredentialError> {
    read_secret_handle_with_hook(file, || {})
}

fn read_secret_handle_with_hook(
    mut file: File,
    after_metadata: impl FnOnce(),
) -> Result<Zeroizing<Vec<u8>>, CredentialError> {
    let initial = file.metadata().map_err(|_| CredentialError::Unavailable)?;
    if !is_safe_regular_file(&initial) || initial.len() > MAX_CREDENTIAL_BYTES as u64 {
        return Err(CredentialError::Invalid);
    }
    let initial_len = usize::try_from(initial.len()).map_err(|_| CredentialError::Invalid)?;

    after_metadata();

    // Take one extra byte so a growing file cannot silently bypass the hard limit.
    let mut bytes = Zeroizing::new(Vec::with_capacity(initial_len));
    file.by_ref()
        .take(MAX_CREDENTIAL_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| CredentialError::Unavailable)?;

    let final_metadata = file.metadata().map_err(|_| CredentialError::Unavailable)?;
    if !is_safe_regular_file(&final_metadata)
        || bytes.len() > MAX_CREDENTIAL_BYTES
        || final_metadata.len() != initial.len()
        || final_metadata.len() != bytes.len() as u64
    {
        return Err(CredentialError::Invalid);
    }
    Ok(bytes)
}

fn is_safe_regular_file(metadata: &std::fs::Metadata) -> bool {
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return false;
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return false;
        }
    }

    true
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
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture_dir() -> std::path::PathBuf {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "impossible-credential-test-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create fixture directory");
        path
    }

    #[test]
    fn bearer_is_exact_and_debug_is_redacted() {
        fn assert_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}

        assert_zeroize_on_drop::<Secret>();
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
    fn credential_file_enforces_exact_size_cap() {
        let directory = fixture_dir();
        let maximum = directory.join("maximum");
        let oversized = directory.join("oversized");
        fs::write(&maximum, vec![b'x'; MAX_CREDENTIAL_BYTES]).expect("write maximum fixture");
        fs::write(&oversized, vec![b'x'; MAX_CREDENTIAL_BYTES + 1])
            .expect("write oversized fixture");

        let secret = load_credential(&CredentialSource::File(maximum)).expect("maximum accepted");
        assert!(secret.verify(&vec![b'x'; MAX_CREDENTIAL_BYTES]));
        assert!(matches!(
            load_credential(&CredentialSource::File(oversized)),
            Err(CredentialError::Invalid)
        ));

        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn credential_read_is_bound_to_the_open_handle() {
        let directory = fixture_dir();
        let configured = directory.join("credential");
        let moved = directory.join("original");
        fs::write(&configured, b"first-value").expect("write original fixture");

        let file = open_secret_file(&configured).expect("open original fixture");
        fs::rename(&configured, &moved).expect("move original fixture");
        fs::write(&configured, b"replacement").expect("write replacement fixture");

        let bytes = read_secret_handle(file).expect("read opened handle");
        assert_eq!(bytes.as_slice(), b"first-value");

        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn credential_read_rejects_growth_after_metadata_check() {
        let directory = fixture_dir();
        let path = directory.join("credential");
        fs::write(&path, b"value").expect("write fixture");
        let file = open_secret_file(&path).expect("open fixture");

        let result = read_secret_handle_with_hook(file, || {
            let mut writer = OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("open fixture for append");
            writer.write_all(b"x").expect("grow fixture");
        });
        assert_eq!(result, Err(CredentialError::Invalid));

        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[test]
    fn credential_file_must_be_regular() {
        let directory = fixture_dir();
        assert!(matches!(
            load_credential(&CredentialSource::File(directory.clone())),
            Err(CredentialError::Invalid)
        ));
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[cfg(unix)]
    #[test]
    fn credential_file_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = fixture_dir();
        let target = directory.join("target");
        let link = directory.join("link");
        fs::write(&target, b"value").expect("write fixture");
        symlink(&target, &link).expect("create symlink fixture");
        assert!(matches!(
            load_credential(&CredentialSource::File(link)),
            Err(CredentialError::Invalid)
        ));
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[cfg(unix)]
    #[test]
    fn credential_file_rejects_fifo_without_waiting_for_a_writer() {
        use std::{process::Command, sync::mpsc, time::Duration};

        let directory = fixture_dir();
        let fifo = directory.join("credential-fifo");
        let status = Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("run mkfifo");
        assert!(status.success(), "create FIFO fixture");

        let (sender, receiver) = mpsc::channel();
        let credential_path = fifo.clone();
        std::thread::spawn(move || {
            let result = load_credential(&CredentialSource::File(credential_path));
            let _ = sender.send(result);
        });

        let result = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("FIFO credential validation must not wait for a writer");
        assert!(matches!(result, Err(CredentialError::Invalid)));

        fs::remove_file(fifo).expect("remove FIFO fixture");
        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[cfg(unix)]
    #[test]
    fn credential_file_rejects_unix_socket_promptly() {
        use std::{os::unix::net::UnixListener, time::Instant};

        let directory = fixture_dir();
        let socket = directory.join("credential-socket");
        let _listener = UnixListener::bind(&socket).expect("create socket fixture");
        let started = Instant::now();

        assert!(load_credential(&CredentialSource::File(socket)).is_err());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "socket credential validation must return promptly"
        );

        fs::remove_dir_all(directory).expect("remove fixture directory");
    }

    #[cfg(windows)]
    #[test]
    fn credential_file_rejects_reparse_symlinks_when_supported() {
        use std::os::windows::fs::symlink_file;

        let directory = fixture_dir();
        let target = directory.join("target");
        let link = directory.join("link");
        fs::write(&target, b"value").expect("write fixture");
        if symlink_file(&target, &link).is_ok() {
            assert!(matches!(
                load_credential(&CredentialSource::File(link)),
                Err(CredentialError::Invalid)
            ));
        }
        fs::remove_dir_all(directory).expect("remove fixture directory");
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

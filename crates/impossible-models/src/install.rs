use std::{
    fs::{self, OpenOptions},
    io,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use fs2::FileExt;
use reqwest::{
    Client, StatusCode, Url,
    header::{CONTENT_LENGTH, LOCATION},
    redirect::Policy,
};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::{
    Error, Manifest, ModelStatus, ModelStore, Result,
    store::{prepare_staging, promote},
};

/// Cheap cloneable cancellation signal checked during locks and streamed transfers.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Creates a live token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Requests cooperative cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Network policy for model installation.
#[derive(Debug, Clone)]
pub struct InstallOptions {
    /// Prohibits every network request when true.
    pub offline: bool,
    /// Exact URL origins that manifests and redirects may use.
    pub allowed_origins: Vec<Url>,
    /// Maximum redirect hops.
    pub max_redirects: usize,
    /// Absolute defense-in-depth bound per artifact.
    pub max_artifact_bytes: u64,
}

impl Default for InstallOptions {
    fn default() -> Self {
        Self {
            offline: false,
            allowed_origins: ["https://huggingface.co"]
                .into_iter()
                .filter_map(|value| Url::parse(value).ok())
                .collect(),
            max_redirects: 3,
            max_artifact_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// Downloads immutable artifacts into an application cache.
#[derive(Debug, Clone)]
pub struct Installer {
    store: ModelStore,
    options: InstallOptions,
    client: Client,
}

impl Installer {
    /// Constructs an installer with redirects disabled at the client layer so every hop is checked.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty/unsafe allowlist or if the HTTP client cannot be built.
    pub fn new(store: ModelStore, options: InstallOptions) -> Result<Self> {
        if options.allowed_origins.is_empty() {
            return Err(Error::Invalid(
                "at least one download origin must be explicitly allowed".into(),
            ));
        }
        for origin in &options.allowed_origins {
            let safe_loopback = origin.scheme() == "http"
                && origin
                    .host_str()
                    .is_some_and(|host| matches!(host, "127.0.0.1" | "::1" | "localhost"));
            if origin.host_str().is_none()
                || (origin.scheme() != "https" && !safe_loopback)
                || origin.username() != ""
                || origin.password().is_some()
            {
                return Err(Error::Invalid(
                    "allowed origins must use HTTPS (HTTP is test-only for loopback)".into(),
                ));
            }
        }
        let client = Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(15))
            .read_timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            store,
            options,
            client,
        })
    }

    /// Installs a manifest atomically, or returns the existing verified status.
    ///
    /// # Errors
    ///
    /// Returns a validation, policy, cancellation, network, integrity, or filesystem error. Failed
    /// transfers are never promoted into the installed-model directory.
    pub async fn install(&self, manifest: &Manifest, cancel: &CancelToken) -> Result<ModelStatus> {
        manifest.validate()?;
        let existing = self.store.status(manifest)?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            return Ok(existing);
        }
        if self.options.offline {
            return Err(Error::Offline);
        }
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let lock = acquire_lock(self.store.layout().lock_file(manifest), cancel).await?;
        let existing = self.store.status(manifest)?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            drop(lock);
            return Ok(existing);
        }
        let staging = self.store.layout().staging_dir(manifest);
        prepare_staging(self.store.layout(), &staging)?;
        for artifact in &manifest.artifacts {
            if artifact.size > self.options.max_artifact_bytes {
                return Err(Error::SizeLimit {
                    expected: self.options.max_artifact_bytes,
                    actual: artifact.size,
                });
            }
            let target = staging.join(&artifact.path);
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await?;
                crate::store::ensure_contained_directory(&staging, parent)?;
            }
            self.download(
                &artifact.url,
                &target,
                artifact.size,
                &artifact.sha256,
                cancel,
            )
            .await?;
        }
        if cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let manifest_path = staging.join("manifest.json");
        tokio::fs::write(&manifest_path, manifest.to_json()?).await?;
        let manifest_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&manifest_path)?;
        manifest_file.sync_all()?;
        drop(manifest_file);
        promote(self.store.layout(), manifest, &staging)?;
        drop(lock);
        self.store.status(manifest)
    }

    async fn download(
        &self,
        source: &str,
        target: &Path,
        expected_size: u64,
        expected_hash: &str,
        cancel: &CancelToken,
    ) -> Result<()> {
        let mut url =
            Url::parse(source).map_err(|_| Error::Invalid("invalid artifact URL".into()))?;
        let mut redirects = 0_usize;
        let mut response = loop {
            self.ensure_allowed(&url)?;
            if cancel.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let request = self.client.get(url.clone()).send();
            let response = tokio::select! {
                result = request => result?,
                () = wait_for_cancel(cancel) => return Err(Error::Cancelled),
            };
            if response.status().is_redirection() {
                if redirects >= self.options.max_redirects {
                    return Err(Error::Invalid("redirect limit exceeded".into()));
                }
                let location = response
                    .headers()
                    .get(LOCATION)
                    .ok_or_else(|| Error::Invalid("redirect omitted Location".into()))?
                    .to_str()
                    .map_err(|_| Error::Invalid("redirect Location is invalid".into()))?;
                url = url
                    .join(location)
                    .map_err(|_| Error::Invalid("redirect URL is invalid".into()))?;
                redirects += 1;
                continue;
            }
            if response.status() != StatusCode::OK {
                return Err(Error::Invalid(format!(
                    "artifact server returned HTTP {}",
                    response.status().as_u16()
                )));
            }
            break response;
        };
        if let Some(length) = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
        {
            if length > expected_size {
                return Err(Error::SizeLimit {
                    expected: expected_size,
                    actual: length,
                });
            }
        }
        let mut file = tokio::fs::File::create(target).await?;
        let mut digest = Sha256::new();
        let mut received = 0_u64;
        loop {
            let chunk = tokio::select! {
                result = response.chunk() => result?,
                () = wait_for_cancel(cancel) => {
                    file.flush().await?;
                    return Err(Error::Cancelled);
                },
            };
            let Some(chunk) = chunk else {
                break;
            };
            received = received
                .checked_add(
                    u64::try_from(chunk.len())
                        .map_err(|_| Error::Invalid("artifact chunk length overflow".into()))?,
                )
                .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
            if received > expected_size || received > self.options.max_artifact_bytes {
                return Err(Error::SizeLimit {
                    expected: expected_size.min(self.options.max_artifact_bytes),
                    actual: received,
                });
            }
            digest.update(&chunk);
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        if received != expected_size {
            return Err(Error::SizeLimit {
                expected: expected_size,
                actual: received,
            });
        }
        let actual = format!("{:x}", digest.finalize());
        if actual != expected_hash {
            return Err(Error::HashMismatch {
                expected: expected_hash.into(),
                actual,
            });
        }
        Ok(())
    }

    fn ensure_allowed(&self, candidate: &Url) -> Result<()> {
        let allowed = self.options.allowed_origins.iter().any(|origin| {
            origin.scheme() == candidate.scheme()
                && origin.host_str() == candidate.host_str()
                && origin.port_or_known_default() == candidate.port_or_known_default()
        });
        if allowed {
            Ok(())
        } else {
            Err(Error::OriginNotAllowed(
                candidate.origin().ascii_serialization(),
            ))
        }
    }
}

async fn wait_for_cancel(cancel: &CancelToken) {
    while !cancel.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn acquire_lock(path: impl AsRef<Path>, cancel: &CancelToken) -> Result<fs::File> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
        let root = parent
            .parent()
            .ok_or_else(|| Error::Invalid("lock directory has no cache root".into()))?;
        crate::store::reject_reparse_components(root, Path::new("locks"))?;
    }
    if fs::symlink_metadata(path.as_ref())
        .is_ok_and(|metadata| metadata.file_type().is_symlink() || is_windows_reparse(&metadata))
    {
        return Err(Error::Invalid(
            "lock file cannot be a symlink or reparse point".into(),
        ));
    }
    loop {
        let file = match OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path.as_ref())
        {
            Ok(file) => file,
            Err(error) if is_lock_contention(&error) => {
                if cancel.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(error) if is_lock_contention(&error) => {
                if cancel.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(windows)]
fn is_windows_reparse(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x400 != 0
}

#[cfg(not(windows))]
const fn is_windows_reparse(_metadata: &fs::Metadata) -> bool {
    false
}

fn is_lock_contention(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock || error.raw_os_error() == Some(33)
}

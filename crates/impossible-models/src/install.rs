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
#[derive(Clone)]
pub struct InstallOptions {
    /// Prohibits every network request when true.
    pub offline: bool,
    /// Exact URL origins that manifests and redirects may use.
    pub allowed_origins: Vec<Url>,
    /// Maximum redirect hops.
    pub max_redirects: usize,
    /// Absolute defense-in-depth bound per artifact.
    pub max_artifact_bytes: u64,
    /// Maximum number of artifacts accepted by one installation.
    pub max_artifacts: usize,
    /// Maximum checked sum of declared artifact bytes for one installation.
    pub max_total_artifact_bytes: u64,
}

impl std::fmt::Debug for InstallOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InstallOptions")
            .field("offline", &self.offline)
            .field("allowed_origin_count", &self.allowed_origins.len())
            .field("max_redirects", &self.max_redirects)
            .field("max_artifact_bytes", &self.max_artifact_bytes)
            .field("max_artifacts", &self.max_artifacts)
            .field("max_total_artifact_bytes", &self.max_total_artifact_bytes)
            .finish()
    }
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
            max_artifacts: crate::MAX_ARTIFACTS,
            max_total_artifact_bytes: 8 * 1024 * 1024 * 1024,
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
        if options.max_artifact_bytes == 0
            || options.max_artifacts == 0
            || options.max_total_artifact_bytes == 0
        {
            return Err(Error::Invalid(
                "installation limits must be non-zero".into(),
            ));
        }
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
            let canonical = format!("{}/", origin.origin().ascii_serialization());
            if origin.host_str().is_none()
                || (origin.scheme() != "https" && !safe_loopback)
                || origin.username() != ""
                || origin.password().is_some()
                || origin.path() != "/"
                || origin.query().is_some()
                || origin.fragment().is_some()
                || origin.as_str() != canonical
            {
                return Err(Error::Invalid(
                    "allowed origins must be exact canonical HTTPS origins (HTTP is test-only for loopback)".into(),
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
        self.install_with_commit_observer(manifest, cancel, || {})
            .await
    }

    async fn install_with_commit_observer(
        &self,
        manifest: &Manifest,
        cancel: &CancelToken,
        after_commit_boundary: impl FnOnce(),
    ) -> Result<ModelStatus> {
        manifest.validate()?;
        self.validate_install_size(manifest)?;
        let existing = self.status(manifest, cancel).await?;
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
        let lock = acquire_lock(self.store.layout().lock_file(manifest)?, cancel).await?;
        self.store.reconcile_repair(manifest)?;
        let existing = self.status(manifest, cancel).await?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            drop(lock);
            return Ok(existing);
        }
        let staging = self.store.layout().staging_dir(manifest)?;
        prepare_staging(self.store.layout(), &staging)?;
        for artifact in &manifest.artifacts {
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
        // This final cancellation observation is the commit point. Once all downloaded bytes have
        // been authenticated, cancellation must not turn a durable promotion into a false failure.
        after_commit_boundary();
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
        self.status_non_cancellable(manifest).await
    }

    fn validate_install_size(&self, manifest: &Manifest) -> Result<()> {
        if manifest.artifacts.len() > self.options.max_artifacts {
            return Err(Error::Invalid(
                "manifest exceeds configured artifact count".into(),
            ));
        }
        let mut total = 0_u64;
        for artifact in &manifest.artifacts {
            if artifact.size > self.options.max_artifact_bytes {
                return Err(Error::SizeLimit {
                    expected: self.options.max_artifact_bytes,
                    actual: artifact.size,
                });
            }
            total = total
                .checked_add(artifact.size)
                .ok_or_else(|| Error::Invalid("aggregate artifact size overflow".into()))?;
        }
        if total > self.options.max_total_artifact_bytes {
            return Err(Error::SizeLimit {
                expected: self.options.max_total_artifact_bytes,
                actual: total,
            });
        }
        Ok(())
    }

    async fn status(&self, manifest: &Manifest, cancel: &CancelToken) -> Result<ModelStatus> {
        let store = self.store.clone();
        let manifest = manifest.clone();
        let cancel = cancel.clone();
        tokio::task::spawn_blocking(move || {
            store.status_with_cancel(&manifest, || cancel.is_cancelled())
        })
        .await
        .map_err(|_| Error::Invalid("model verification worker failed".into()))?
    }

    async fn status_non_cancellable(&self, manifest: &Manifest) -> Result<ModelStatus> {
        let store = self.store.clone();
        let manifest = manifest.clone();
        tokio::task::spawn_blocking(move || store.status(&manifest))
            .await
            .map_err(|_| Error::Invalid("model verification worker failed".into()))?
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
    match fs::symlink_metadata(path.as_ref()) {
        Ok(metadata) if metadata.file_type().is_symlink() || is_windows_reparse(&metadata) => {
            return Err(Error::Invalid(
                "lock file cannot be a symlink or reparse point".into(),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let deadline = tokio::time::Instant::now() + crate::store::LOCK_WAIT_LIMIT;
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
                if tokio::time::Instant::now() >= deadline {
                    return Err(Error::Busy);
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
                if tokio::time::Instant::now() >= deadline {
                    return Err(Error::Busy);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Artifact, Dimensions, License, Pooling, Prefixes, RuntimeMetadata, SemanticVerification,
        TensorMetadata, TokenizerMetadata,
    };
    use tempfile::TempDir;
    use tokio::{io::AsyncReadExt, net::TcpListener};

    fn fixture_manifest(url: String, body: &[u8]) -> Manifest {
        Manifest {
            schema_version: 1,
            canonical_id: "tests/commit-boundary".into(),
            revision: "0123456789abcdef0123456789abcdef01234567".into(),
            license: License {
                spdx: "MIT".into(),
                source_url: "https://example.invalid/license".into(),
            },
            semantic_verification: SemanticVerification::Unverified {
                reason: "test fixture".into(),
            },
            tokenizer: TokenizerMetadata {
                kind: "fixture".into(),
                max_tokens: 8,
                lowercase: false,
            },
            pooling: Pooling::Mean,
            prefixes: Prefixes {
                query: String::new(),
                document: String::new(),
            },
            dimensions: Dimensions {
                native: 2,
                matryoshka: vec![],
            },
            tensors: TensorMetadata {
                format: "fixture".into(),
                dtype: "bytes".into(),
                architecture: "identity".into(),
            },
            runtime: RuntimeMetadata::CatalogOnly,
            artifacts: vec![Artifact {
                path: "weights/model.bin".into(),
                url,
                sha256: format!("{:x}", Sha256::digest(body)),
                size: u64::try_from(body.len()).unwrap_or_default(),
            }],
        }
    }

    async fn fixture_server(body: Vec<u8>) -> Result<(Url, tokio::task::JoinHandle<()>)> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let handle = tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request).await;
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes()).await;
                let _ = stream.write_all(&body).await;
            }
        });
        Ok((
            Url::parse(&format!("http://{address}/artifact"))
                .map_err(|_| Error::Invalid("test URL did not parse".into()))?,
            handle,
        ))
    }

    #[test]
    fn rejects_non_origin_allowlist_entries_and_redacts_debug() -> Result<()> {
        let temp = TempDir::new()?;
        let store = ModelStore::new(temp.path())?;
        for value in [
            "https://example.test/path",
            "https://example.test/?token=sentinel-query-token",
            "https://example.test/#fragment",
            "https://user@example.test/",
            "https://example.test//",
        ] {
            let options = InstallOptions {
                allowed_origins: vec![
                    Url::parse(value)
                        .map_err(|_| Error::Invalid("test URL did not parse".into()))?,
                ],
                ..InstallOptions::default()
            };
            let debug = format!("{options:?}");
            assert!(!debug.contains("example.test"));
            assert!(!debug.contains("sentinel-query-token"));
            assert!(
                Installer::new(store.clone(), options).is_err(),
                "accepted {value}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn installer_lock_wait_is_bounded_and_retryable() -> Result<()> {
        let temp = TempDir::new()?;
        let locks = temp.path().join("locks");
        fs::create_dir(&locks)?;
        let path = locks.join("external.lock");
        let external = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        external.lock_exclusive()?;
        let started = tokio::time::Instant::now();
        let Err(error) = acquire_lock(&path, &CancelToken::new()).await else {
            return Err(Error::Invalid("external lock unexpectedly acquired".into()));
        };
        assert!(matches!(error, Error::Busy));
        assert!(error.is_retryable());
        assert!(started.elapsed() < Duration::from_secs(2));
        FileExt::unlock(&external)?;
        assert!(acquire_lock(&path, &CancelToken::new()).await.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn installer_lock_wait_honors_cancellation_before_bound() -> Result<()> {
        let temp = TempDir::new()?;
        let locks = temp.path().join("locks");
        fs::create_dir(&locks)?;
        let path = locks.join("external.lock");
        let external = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        external.lock_exclusive()?;
        let cancel = CancelToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let started = tokio::time::Instant::now();
        assert!(matches!(
            acquire_lock(&path, &cancel).await,
            Err(Error::Cancelled)
        ));
        assert!(started.elapsed() < Duration::from_millis(400));
        FileExt::unlock(&external)?;
        Ok(())
    }

    #[tokio::test]
    async fn configured_aggregate_limit_fails_before_cache_mutation() -> Result<()> {
        let temp = TempDir::new()?;
        let store = ModelStore::new(temp.path())?;
        let manifest = crate::curated_manifests()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invalid("missing test manifest".into()))?;
        let installer = Installer::new(
            store,
            InstallOptions {
                max_total_artifact_bytes: manifest.artifacts[0].size - 1,
                ..InstallOptions::default()
            },
        )?;
        assert!(matches!(
            installer.install(&manifest, &CancelToken::new()).await,
            Err(Error::SizeLimit { .. })
        ));
        assert!(!temp.path().join("models").exists());
        assert!(!temp.path().join("staging").exists());
        assert!(!temp.path().join("locks").exists());
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_after_commit_boundary_cannot_turn_install_into_false_failure()
    -> Result<()> {
        let temp = TempDir::new()?;
        let body = b"authenticated fixture bytes".to_vec();
        let (url, server) = fixture_server(body.clone()).await?;
        let origin = Url::parse(&format!("{}/", url.origin().ascii_serialization()))
            .map_err(|_| Error::Invalid("test origin did not parse".into()))?;
        let manifest = fixture_manifest(url.to_string(), &body);
        let store = ModelStore::new(temp.path())?;
        let installer = Installer::new(
            store.clone(),
            InstallOptions {
                allowed_origins: vec![origin],
                max_artifact_bytes: 1024,
                max_total_artifact_bytes: 1024,
                ..InstallOptions::default()
            },
        )?;
        let cancel = CancelToken::new();
        let trigger = cancel.clone();
        let result = installer
            .install_with_commit_observer(&manifest, &cancel, move || trigger.cancel())
            .await;

        assert_eq!(result?, ModelStatus::IntegrityVerified);
        assert!(cancel.is_cancelled());
        assert_eq!(store.status(&manifest)?, ModelStatus::IntegrityVerified);
        server
            .await
            .map_err(|_| Error::Invalid("test server failed".into()))?;
        Ok(())
    }
}

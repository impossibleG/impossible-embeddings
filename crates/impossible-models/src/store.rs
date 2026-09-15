use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::{Error, Manifest, ModelStatus, Result};

const MANIFEST_FILE: &str = "manifest.json";
static QUARANTINE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Independently supplied semantic-verification evidence.
///
/// The fingerprint is calculated from the complete inference contract. Merely setting a custom
/// manifest's `semantic_verification` field never adds it to this trust root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedSemanticEvidence {
    /// Canonical fingerprint returned by [`Manifest::semantic_fingerprint`].
    pub manifest_fingerprint: String,
    /// Auditable, non-empty reference to the independent evidence.
    pub evidence: String,
}

/// Explicit trust roots used to decide semantic loadability.
#[derive(Debug, Clone, Default)]
pub struct SemanticTrustRoot {
    fingerprints: BTreeSet<String>,
}

impl SemanticTrustRoot {
    /// Builds the release trust root from repository-curated manifests only.
    ///
    /// # Errors
    /// Returns an error if committed catalog data is invalid.
    pub fn curated() -> Result<Self> {
        let mut root = Self::default();
        for manifest in crate::curated_manifests()? {
            if manifest.semantics_verified() {
                root.fingerprints.insert(manifest.semantic_fingerprint()?);
            }
        }
        Ok(root)
    }

    /// Builds an explicit operator trust root from independently authenticated records.
    ///
    /// # Errors
    /// Returns an error for malformed fingerprints or empty evidence references.
    pub fn from_evidence(
        evidence: impl IntoIterator<Item = TrustedSemanticEvidence>,
    ) -> Result<Self> {
        let mut root = Self::default();
        for record in evidence {
            if record.evidence.trim().is_empty()
                || record.manifest_fingerprint.len() != 71
                || !record.manifest_fingerprint.starts_with("sha256:")
                || !record.manifest_fingerprint[7..]
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                return Err(Error::Invalid(
                    "semantic trust evidence is malformed".into(),
                ));
            }
            root.fingerprints.insert(record.manifest_fingerprint);
        }
        Ok(root)
    }

    fn trusts(&self, manifest: &Manifest) -> Result<bool> {
        Ok(manifest.semantics_verified()
            && self
                .fingerprints
                .contains(&manifest.semantic_fingerprint()?))
    }
}

/// Deterministic application-owned cache layout.
#[derive(Debug, Clone)]
pub struct CacheLayout {
    root: PathBuf,
}

impl CacheLayout {
    /// Creates the cache root and rejects a symlink root.
    ///
    /// # Errors
    ///
    /// Returns an error if the root cannot be created/canonicalized or is a symbolic link.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        fs::create_dir_all(root)?;
        if is_reparse(&fs::symlink_metadata(root)?) {
            return Err(Error::Invalid("cache root cannot be a symlink".into()));
        }
        Ok(Self {
            root: fs::canonicalize(root)?,
        })
    }

    /// Canonical application cache root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn key(manifest: &Manifest) -> String {
        let mut digest = Sha256::new();
        digest.update(manifest.canonical_id.as_bytes());
        digest.update([0]);
        digest.update(manifest.revision.as_bytes());
        format!("{:x}", digest.finalize())
    }

    /// Final directory for an exact immutable identity.
    #[must_use]
    pub fn model_dir(&self, manifest: &Manifest) -> PathBuf {
        self.root.join("models").join(Self::key(manifest))
    }
    pub(crate) fn staging_dir(&self, manifest: &Manifest) -> PathBuf {
        self.root
            .join("staging")
            .join(format!("{}.partial", Self::key(manifest)))
    }
    pub(crate) fn lock_file(&self, manifest: &Manifest) -> PathBuf {
        self.root
            .join("locks")
            .join(format!("{}.lock", Self::key(manifest)))
    }
}

/// An explicit directory the caller authorizes for discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryRoot(PathBuf);

impl DiscoveryRoot {
    /// Registers one directory. No parent, sibling, or full-disk search is performed.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }
}

/// A model identity proven loadable by [`ModelStore`].
///
/// Construction is deliberately private so runtime adapters cannot accidentally bypass artifact
/// integrity and semantic-readiness checks.
#[derive(Debug, Clone)]
pub struct VerifiedModel {
    manifest: Manifest,
    root: PathBuf,
    // A shared filesystem lease prevents an exact identity from being deleted or replaced while
    // any runtime retains this capability.
    _lease: Arc<fs::File>,
}

impl VerifiedModel {
    /// Returns the validated canonical manifest.
    #[must_use]
    pub const fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Returns the verified, application-owned artifact directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns a stable fingerprint derived from every pinned artifact digest.
    #[must_use]
    pub fn artifact_fingerprint(&self) -> String {
        self.manifest.artifact_fingerprint()
    }

    /// Resolves one declared artifact while rejecting every symlink/junction component.
    ///
    /// # Errors
    /// Returns an error if the path is undeclared, unsafe, missing, or escapes the leased root.
    pub fn artifact_path(&self, relative: &str) -> Result<PathBuf> {
        if !self
            .manifest
            .artifacts
            .iter()
            .any(|artifact| artifact.path == relative)
        {
            return Err(Error::Invalid(
                "runtime requested an undeclared artifact".into(),
            ));
        }
        reject_reparse_components(&self.root, Path::new(relative))?;
        let path = self.root.join(relative);
        let canonical = fs::canonicalize(&path)?;
        if !canonical.starts_with(fs::canonicalize(&self.root)?) {
            return Err(Error::Invalid(
                "artifact escaped verified model root".into(),
            ));
        }
        Ok(canonical)
    }

    /// Rehashes the stored manifest and every artifact while the in-use lease is held.
    ///
    /// # Errors
    /// Returns an error if any byte or filesystem invariant changed since verification.
    pub fn revalidate_integrity(&self) -> Result<()> {
        let stored = Manifest::from_json(&fs::read(self.root.join(MANIFEST_FILE))?)?;
        if stored != self.manifest {
            return Err(Error::Invalid(
                "stored manifest changed after verification".into(),
            ));
        }
        for artifact in &stored.artifacts {
            let path = self.artifact_path(&artifact.path)?;
            let metadata = fs::symlink_metadata(&path)?;
            if is_reparse(&metadata)
                || !metadata.is_file()
                || metadata.len() != artifact.size
                || hash_file(&path)? != artifact.sha256
            {
                return Err(Error::Invalid("artifact changed after verification".into()));
            }
        }
        Ok(())
    }
}

impl Manifest {
    /// Returns the canonical content fingerprint for all pinned artifacts.
    #[must_use]
    pub fn artifact_fingerprint(&self) -> String {
        artifact_fingerprint(self)
    }

    /// Fingerprints the complete inference contract independently of its self-declared trust state.
    /// Artifact ordering is canonicalized so equivalent manifests have one stable trust identity.
    ///
    /// # Errors
    /// Returns an error if the validated manifest cannot be serialized.
    pub fn semantic_fingerprint(&self) -> Result<String> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.artifacts.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then(left.sha256.cmp(&right.sha256))
                .then(left.size.cmp(&right.size))
                .then(left.url.cmp(&right.url))
        });
        canonical.semantic_verification = crate::SemanticVerification::Unverified {
            reason: "trust-state-excluded-from-fingerprint".into(),
        };
        let encoded = serde_json::to_vec(&canonical)?;
        Ok(format!("sha256:{:x}", Sha256::digest(encoded)))
    }
}

fn artifact_fingerprint(manifest: &Manifest) -> String {
    let mut artifacts = manifest.artifacts.iter().collect::<Vec<_>>();
    artifacts.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.sha256.cmp(&right.sha256))
            .then(left.size.cmp(&right.size))
    });
    let mut digest = Sha256::new();
    digest.update(manifest.canonical_id.as_bytes());
    digest.update([0]);
    digest.update(manifest.revision.as_bytes());
    for artifact in artifacts {
        digest.update([0]);
        digest.update(artifact.path.as_bytes());
        digest.update([0]);
        digest.update(artifact.sha256.as_bytes());
        digest.update([0]);
        digest.update(artifact.size.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

/// Safe local store operations independent from the network installer.
#[derive(Debug, Clone)]
pub struct ModelStore {
    layout: CacheLayout,
    trust_root: Arc<SemanticTrustRoot>,
}

impl ModelStore {
    /// Opens an application cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache root is unsafe or unavailable.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::with_trust_root(root, SemanticTrustRoot::curated()?)
    }

    /// Opens an application cache with an explicit operator-authenticated trust root.
    ///
    /// # Errors
    /// Returns an error if the cache root is unsafe or unavailable.
    pub fn with_trust_root(root: impl AsRef<Path>, trust_root: SemanticTrustRoot) -> Result<Self> {
        Ok(Self {
            layout: CacheLayout::new(root)?,
            trust_root: Arc::new(trust_root),
        })
    }
    /// Returns cache layout details for integration with installers.
    #[must_use]
    pub const fn layout(&self) -> &CacheLayout {
        &self.layout
    }

    /// Verifies the exact manifest and every artifact without following symlinks.
    ///
    /// # Errors
    ///
    /// Returns an error when validation or filesystem inspection cannot be completed safely.
    pub fn status(&self, manifest: &Manifest) -> Result<ModelStatus> {
        manifest.validate()?;
        let directory = self.layout.model_dir(manifest);
        if reject_reparse_components(
            &self.layout.root,
            Path::new("models")
                .join(CacheLayout::key(manifest))
                .as_path(),
        )
        .is_err()
        {
            return Ok(ModelStatus::Invalid);
        }
        let metadata = match fs::symlink_metadata(&directory) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(ModelStatus::Missing);
            }
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Ok(ModelStatus::Invalid);
        }
        let stored = match fs::read(directory.join(MANIFEST_FILE))
            .ok()
            .and_then(|bytes| Manifest::from_json(&bytes).ok())
        {
            Some(value) if value == *manifest => value,
            _ => return Ok(ModelStatus::Invalid),
        };
        for artifact in &stored.artifacts {
            let path = directory.join(&artifact.path);
            if reject_reparse_components(&directory, Path::new(&artifact.path)).is_err() {
                return Ok(ModelStatus::Invalid);
            }
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                return Ok(ModelStatus::Invalid);
            };
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != artifact.size
                || hash_file(&path)? != artifact.sha256
            {
                return Ok(ModelStatus::Invalid);
            }
        }
        Ok(if self.trust_root.trusts(&stored)? {
            ModelStatus::Loadable
        } else {
            ModelStatus::IntegrityVerified
        })
    }

    /// Performs a complete integrity and semantic-readiness verification.
    ///
    /// # Errors
    ///
    /// Returns an error when validation or filesystem inspection cannot be completed safely.
    pub fn verify(&self, manifest: &Manifest) -> Result<ModelStatus> {
        self.status(manifest)
    }

    /// Returns a capability to load an exact identity only after complete verification.
    ///
    /// # Errors
    ///
    /// Returns an error unless integrity and semantic verification both make the model loadable.
    pub fn verified_model(&self, manifest: &Manifest) -> Result<VerifiedModel> {
        manifest.validate()?;
        let lease = acquire_shared_store_lock(&self.layout, manifest)?;
        match self.verify(manifest)? {
            ModelStatus::Loadable => Ok(VerifiedModel {
                manifest: manifest.clone(),
                root: self.layout.model_dir(manifest),
                _lease: Arc::new(lease),
            }),
            status => Err(Error::Invalid(format!(
                "model is not loadable after verification: {status:?}"
            ))),
        }
    }

    /// Imports artifacts from a caller-authorized directory after full integrity validation.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, identity/integrity failures, or filesystem failures.
    pub fn import(&self, manifest: &Manifest, source: impl AsRef<Path>) -> Result<ModelStatus> {
        manifest.validate()?;
        let _lock = acquire_store_lock(&self.layout, manifest)?;
        let existing = self.status(manifest)?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            return Ok(existing);
        }
        let source = fs::canonicalize(source)?;
        if fs::symlink_metadata(&source)?.file_type().is_symlink() || !source.is_dir() {
            return Err(Error::Invalid(
                "import source must be a real directory".into(),
            ));
        }
        let staging = self.layout.staging_dir(manifest);
        prepare_staging(&self.layout, &staging)?;
        let result = (|| {
            for artifact in &manifest.artifacts {
                let from = source.join(&artifact.path);
                reject_reparse_components(&source, Path::new(&artifact.path))?;
                let metadata = fs::symlink_metadata(&from)?;
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(Error::Invalid(format!(
                        "artifact is not a regular file: {}",
                        artifact.path
                    )));
                }
                let canonical = fs::canonicalize(&from)?;
                if !canonical.starts_with(&source) {
                    return Err(Error::Invalid(
                        "import artifact escaped configured source".into(),
                    ));
                }
                if metadata.len() != artifact.size {
                    return Err(Error::SizeLimit {
                        expected: artifact.size,
                        actual: metadata.len(),
                    });
                }
                let actual = hash_file(&canonical)?;
                if actual != artifact.sha256 {
                    return Err(Error::HashMismatch {
                        expected: artifact.sha256.clone(),
                        actual,
                    });
                }
                let target = staging.join(&artifact.path);
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                    ensure_contained_directory(&staging, parent)?;
                }
                fs::copy(canonical, target)?;
            }
            fs::write(staging.join(MANIFEST_FILE), manifest.to_json()?)?;
            promote(&self.layout, manifest, &staging)
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result?;
        self.status(manifest)
    }

    /// Finds valid identity manifests only inside explicitly configured roots.
    ///
    /// # Errors
    ///
    /// Returns an error if an authorized root cannot be inspected.
    pub fn discover(&self, roots: &[DiscoveryRoot]) -> Result<Vec<Manifest>> {
        let mut found = Vec::new();
        for root in roots {
            let metadata = match fs::symlink_metadata(&root.0) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&root.0)? {
                let entry = entry?;
                if entry.file_type()?.is_symlink() || !entry.file_type()?.is_dir() {
                    continue;
                }
                let path = entry.path().join(MANIFEST_FILE);
                if let Ok(bytes) = fs::read(path) {
                    if let Ok(manifest) = Manifest::from_json(&bytes) {
                        found.push(manifest);
                    }
                }
            }
        }
        Ok(found)
    }

    /// Deletes one exact identity and nothing else.
    ///
    /// # Errors
    ///
    /// Returns an error if containment or identity checks fail or removal cannot complete.
    pub fn delete(&self, manifest: &Manifest) -> Result<bool> {
        manifest.validate()?;
        let _lock = acquire_delete_lock(&self.layout, manifest)?;
        let target = self.layout.model_dir(manifest);
        let metadata = match fs::symlink_metadata(&target) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Invalid(
                "refusing to delete non-directory or symlink target".into(),
            ));
        }
        let parent = target
            .parent()
            .ok_or_else(|| Error::Invalid("model target has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let canonical_parent = fs::canonicalize(parent)?;
        let expected_parent = self.layout.root.join("models");
        if canonical_parent != fs::canonicalize(expected_parent)? {
            return Err(Error::Invalid(
                "model target escaped cache containment".into(),
            ));
        }
        let stored = Manifest::from_json(&fs::read(target.join(MANIFEST_FILE))?)?;
        if stored != *manifest || stored.artifact_fingerprint() != manifest.artifact_fingerprint() {
            return Err(Error::Invalid(
                "stored manifest does not exactly match deletion request".into(),
            ));
        }
        fs::remove_dir_all(target)?;
        Ok(true)
    }
}

fn acquire_store_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    let path = layout.lock_file(manifest);
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("lock path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    reject_reparse_components(&layout.root, Path::new("locks"))?;
    if is_reparse(&fs::symlink_metadata(parent)?)
        || fs::symlink_metadata(&path).is_ok_and(|metadata| is_reparse(&metadata))
    {
        return Err(Error::Invalid("model lock path cannot be a symlink".into()));
    }
    loop {
        match OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
        {
            Ok(file) => match file.try_lock_exclusive() {
                Ok(()) => return Ok(file),
                Err(error) if is_lock_contention(&error) => {}
                Err(error) => return Err(error.into()),
            },
            Err(error) if is_lock_contention(&error) => {}
            Err(error) => return Err(error.into()),
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn acquire_shared_store_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    let path = layout.lock_file(manifest);
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("lock path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    reject_reparse_components(&layout.root, Path::new("locks"))?;
    match fs::symlink_metadata(&path) {
        Ok(metadata) if is_reparse(&metadata) => {
            return Err(Error::Invalid(
                "model lock path cannot be a reparse point".into(),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    FileExt::lock_shared(&file)?;
    Ok(file)
}

fn acquire_delete_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    let path = layout.lock_file(manifest);
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("lock path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    reject_reparse_components(&layout.root, Path::new("locks"))?;
    match fs::symlink_metadata(&path) {
        Ok(metadata) if is_reparse(&metadata) => {
            return Err(Error::Invalid(
                "model lock path cannot be a reparse point".into(),
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    match file.try_lock_exclusive() {
        Ok(()) => Ok(file),
        Err(error) if is_lock_contention(&error) => Err(Error::InUse),
        Err(error) => Err(error.into()),
    }
}

fn is_lock_contention(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock || error.raw_os_error() == Some(33)
}

pub(crate) fn prepare_staging(layout: &CacheLayout, staging: &Path) -> Result<()> {
    for directory in [
        layout.root.join("models"),
        layout.root.join("staging"),
        layout.root.join("locks"),
    ] {
        fs::create_dir_all(&directory)?;
        ensure_contained_directory(&layout.root, &directory)?;
    }
    if let Ok(metadata) = fs::symlink_metadata(staging) {
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Invalid("unsafe staging path".into()));
        }
        fs::remove_dir_all(staging)?;
    }
    fs::create_dir_all(staging)?;
    ensure_contained_directory(&layout.root, staging)?;
    Ok(())
}

pub(crate) fn ensure_contained_directory(base: &Path, directory: &Path) -> Result<()> {
    if is_reparse(&fs::symlink_metadata(directory)?) {
        return Err(Error::Invalid("cache directory cannot be a symlink".into()));
    }
    let canonical_base = fs::canonicalize(base)?;
    let canonical_directory = fs::canonicalize(directory)?;
    if !canonical_directory.starts_with(canonical_base) {
        return Err(Error::Invalid("cache directory escaped containment".into()));
    }
    Ok(())
}

pub(crate) fn promote(layout: &CacheLayout, manifest: &Manifest, staging: &Path) -> Result<()> {
    let final_path = layout.model_dir(manifest);
    let displaced = if fs::symlink_metadata(&final_path).is_ok() {
        reject_reparse_components(
            &layout.root,
            Path::new("models")
                .join(CacheLayout::key(manifest))
                .as_path(),
        )?;
        let quarantine_root = layout.root.join("quarantine");
        fs::create_dir_all(&quarantine_root)?;
        ensure_contained_directory(&layout.root, &quarantine_root)?;
        let quarantine = loop {
            let sequence = QUARANTINE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = quarantine_root.join(format!(
                "{}.{}.{}.invalid",
                CacheLayout::key(manifest),
                std::process::id(),
                sequence
            ));
            match fs::symlink_metadata(&candidate) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break candidate,
                Ok(_) => {}
                Err(error) => return Err(error.into()),
            }
        };
        fs::rename(&final_path, &quarantine)?;
        Some(quarantine)
    } else {
        None
    };
    if let Err(error) = fs::rename(staging, &final_path) {
        if let Some(quarantine) = displaced {
            let _ = fs::rename(quarantine, &final_path);
        }
        return Err(error.into());
    }
    Ok(())
}

pub(crate) fn reject_reparse_components(base: &Path, relative: &Path) -> Result<()> {
    let mut current = base.to_path_buf();
    if is_reparse(&fs::symlink_metadata(&current)?) {
        return Err(Error::Invalid(
            "path contains a symlink or reparse point".into(),
        ));
    }
    for component in relative.components() {
        if !matches!(component, std::path::Component::Normal(_)) {
            return Err(Error::Invalid("path contains unsafe components".into()));
        }
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if is_reparse(&metadata) => {
                return Err(Error::Invalid(
                    "path contains a symlink or reparse point".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn is_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

pub(crate) fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

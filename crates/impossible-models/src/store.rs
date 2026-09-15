use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::{
    Artifact, Dimensions, Error, Manifest, ModelStatus, Pooling, Prefixes, Result, RuntimeMetadata,
    TensorMetadata, TokenizerMetadata,
};

const MANIFEST_FILE: &str = "manifest.json";
pub(crate) const LOCK_WAIT_LIMIT: Duration = Duration::from_millis(500);
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

    pub(crate) fn key(manifest: &Manifest) -> Result<String> {
        Ok(manifest
            .semantic_fingerprint()?
            .trim_start_matches("sha256:")
            .to_owned())
    }

    /// Final directory for an exact immutable identity.
    ///
    /// # Errors
    /// Returns an error when the manifest is invalid or cannot be canonically fingerprinted.
    pub fn model_dir(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self.root.join("models").join(Self::key(manifest)?))
    }
    pub(crate) fn staging_dir(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self
            .root
            .join("staging")
            .join(format!("{}.partial", Self::key(manifest)?)))
    }
    pub(crate) fn lock_file(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self
            .root
            .join("locks")
            .join(format!("{}.lock", Self::key(manifest)?)))
    }
    fn repair_marker(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self
            .root
            .join("transactions")
            .join(format!("{}.repair", Self::key(manifest)?)))
    }
    fn repair_backup(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self
            .root
            .join("transactions")
            .join(format!("{}.previous", Self::key(manifest)?)))
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

    /// Reads one declared artifact from a single, non-following handle and verifies those exact
    /// bytes against the immutable manifest before returning them.
    ///
    /// The returned allocation is independent from the cache path. Replacing a path after this
    /// method returns therefore cannot change the bytes consumed by a runtime adapter.
    ///
    /// # Errors
    /// Returns an error if the artifact is undeclared, unsafe, replaced, or fails integrity.
    pub fn artifact_bytes(&self, relative: &str) -> Result<Vec<u8>> {
        let artifact = self
            .manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.path == relative)
            .ok_or_else(|| Error::Invalid("runtime requested an undeclared artifact".into()))?;
        reject_reparse_components(&self.root, Path::new(relative))?;
        let path = self.root.join(relative);
        let mut options = OpenOptions::new();
        options.read(true);
        configure_no_follow(&mut options);
        let mut file = options.open(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != artifact.size {
            return Err(Error::Invalid("artifact changed after verification".into()));
        }
        let capacity = usize::try_from(artifact.size)
            .map_err(|_| Error::Invalid("artifact is too large for this platform".into()))?;
        let mut bytes = Vec::with_capacity(capacity);
        file.read_to_end(&mut bytes)?;
        if bytes.len() != capacity || format!("{:x}", Sha256::digest(&bytes)) != artifact.sha256 {
            return Err(Error::Invalid("artifact changed after verification".into()));
        }
        reject_reparse_components(&self.root, Path::new(relative))?;
        let canonical = fs::canonicalize(&path)?;
        if !canonical.starts_with(fs::canonicalize(&self.root)?) {
            return Err(Error::Invalid(
                "artifact escaped verified model root".into(),
            ));
        }
        Ok(bytes)
    }

    /// Rehashes the stored manifest and every artifact while the in-use lease is held.
    ///
    /// # Errors
    /// Returns an error if any byte or filesystem invariant changed since verification.
    pub fn revalidate_integrity(&self) -> Result<()> {
        let stored = read_contained_manifest(&self.root)?;
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
        let mut artifacts = self.artifacts.iter().collect::<Vec<_>>();
        artifacts.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then(left.sha256.cmp(&right.sha256))
                .then(left.size.cmp(&right.size))
        });
        let canonical = SemanticContract {
            schema_version: self.schema_version,
            canonical_id: &self.canonical_id,
            revision: &self.revision,
            tokenizer: &self.tokenizer,
            pooling: self.pooling,
            prefixes: &self.prefixes,
            dimensions: &self.dimensions,
            tensors: &self.tensors,
            runtime: &self.runtime,
            artifacts: artifacts.into_iter().map(SemanticArtifact::from).collect(),
        };
        let encoded = serde_json::to_vec(&canonical)?;
        Ok(format!("sha256:{:x}", Sha256::digest(encoded)))
    }
}

#[derive(serde::Serialize)]
struct SemanticContract<'a> {
    schema_version: u32,
    canonical_id: &'a str,
    revision: &'a str,
    tokenizer: &'a TokenizerMetadata,
    pooling: Pooling,
    prefixes: &'a Prefixes,
    dimensions: &'a Dimensions,
    tensors: &'a TensorMetadata,
    runtime: &'a RuntimeMetadata,
    artifacts: Vec<SemanticArtifact<'a>>,
}

#[derive(serde::Serialize)]
struct SemanticArtifact<'a> {
    path: &'a str,
    sha256: &'a str,
    size: u64,
}

impl<'a> From<&'a Artifact> for SemanticArtifact<'a> {
    fn from(artifact: &'a Artifact) -> Self {
        Self {
            path: &artifact.path,
            sha256: &artifact.sha256,
            size: artifact.size,
        }
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
        self.status_with_cancel(manifest, || false)
    }

    pub(crate) fn status_with_cancel(
        &self,
        manifest: &Manifest,
        cancelled: impl Fn() -> bool,
    ) -> Result<ModelStatus> {
        manifest.validate()?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        match fs::symlink_metadata(self.layout.repair_marker(manifest)?) {
            Ok(_) => {
                let _lock = acquire_store_lock_with_cancel(&self.layout, manifest, &cancelled)?;
                reconcile_repair_inner(&self.layout, manifest)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        self.status_inner(manifest, &cancelled)
    }

    fn status_inner(
        &self,
        manifest: &Manifest,
        cancelled: &impl Fn() -> bool,
    ) -> Result<ModelStatus> {
        let directory = self.layout.model_dir(manifest)?;
        if reject_reparse_components(
            &self.layout.root,
            Path::new("models")
                .join(CacheLayout::key(manifest)?)
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
        if is_reparse(&metadata) || !metadata.is_dir() {
            return Ok(ModelStatus::Invalid);
        }
        let stored = match read_contained_manifest(&directory) {
            Ok(value) if value == *manifest => value,
            _ => return Ok(ModelStatus::Invalid),
        };
        for artifact in &stored.artifacts {
            let path = directory.join(&artifact.path);
            if reject_reparse_components(&directory, Path::new(&artifact.path)).is_err() {
                return Ok(ModelStatus::Invalid);
            }
            let metadata = match fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(ModelStatus::Invalid);
                }
                Err(error) => return Err(error.into()),
            };
            if metadata.file_type().is_symlink()
                || !metadata.is_file()
                || metadata.len() != artifact.size
                || hash_file_with_cancel(&path, cancelled)? != artifact.sha256
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

    pub(crate) fn reconcile_repair(&self, manifest: &Manifest) -> Result<()> {
        reconcile_repair_inner(&self.layout, manifest)
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
        loop {
            // Repair promotion is serialized by the exclusive model lock. Reconcile before
            // retaining the shared runtime lease: attempting reconciliation while already
            // holding that lease would contend with our own lock on some platforms.
            match fs::symlink_metadata(self.layout.repair_marker(manifest)?) {
                Ok(_) => {
                    let repair_lock = acquire_store_lock(&self.layout, manifest)?;
                    reconcile_repair_inner(&self.layout, manifest)?;
                    drop(repair_lock);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            let lease = acquire_shared_store_lock(&self.layout, manifest)?;
            // Close the gap between dropping the repair lock and taking the shared lease. A
            // competing promoter can only leave a marker before our lease is granted; after it
            // is granted, no legitimate promotion can begin.
            match fs::symlink_metadata(self.layout.repair_marker(manifest)?) {
                Ok(_) => {
                    drop(lease);
                    continue;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
            return match self.status_inner(manifest, &|| false)? {
                ModelStatus::Loadable => Ok(VerifiedModel {
                    manifest: manifest.clone(),
                    root: self.layout.model_dir(manifest)?,
                    _lease: Arc::new(lease),
                }),
                status => Err(Error::Invalid(format!(
                    "model is not loadable after verification: {status:?}"
                ))),
            };
        }
    }

    /// Imports artifacts from a caller-authorized directory after full integrity validation.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, identity/integrity failures, or filesystem failures.
    pub fn import(&self, manifest: &Manifest, source: impl AsRef<Path>) -> Result<ModelStatus> {
        self.import_with_observer(manifest, source.as_ref(), |_| Ok(()))
    }

    fn import_with_observer(
        &self,
        manifest: &Manifest,
        source: &Path,
        mut source_opened: impl FnMut(&Path) -> Result<()>,
    ) -> Result<ModelStatus> {
        manifest.validate()?;
        let existing = self.status(manifest)?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            return Ok(existing);
        }
        let _lock = acquire_store_lock(&self.layout, manifest)?;
        reconcile_repair_inner(&self.layout, manifest)?;
        let existing = self.status_inner(manifest, &|| false)?;
        if matches!(
            existing,
            ModelStatus::IntegrityVerified | ModelStatus::Loadable
        ) {
            return Ok(existing);
        }
        let source = validate_import_source(source)?;
        if !source.is_dir() {
            return Err(Error::Invalid(
                "import source must be a real directory".into(),
            ));
        }
        let staging = self.layout.staging_dir(manifest)?;
        prepare_staging(&self.layout, &staging)?;
        let result = (|| {
            for artifact in &manifest.artifacts {
                let from = source.join(&artifact.path);
                reject_reparse_components(&source, Path::new(&artifact.path))?;
                let canonical = fs::canonicalize(&from)?;
                if !canonical.starts_with(&source) {
                    return Err(Error::Invalid(
                        "import artifact escaped configured source".into(),
                    ));
                }
                let target = staging.join(&artifact.path);
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)?;
                    ensure_contained_directory(&staging, parent)?;
                }
                copy_import_artifact(&canonical, &target, artifact, || source_opened(&from))?;
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
            if is_reparse(&metadata) || !metadata.is_dir() {
                continue;
            }
            let canonical_root = fs::canonicalize(&root.0)?;
            for entry in fs::read_dir(&root.0)? {
                let entry = entry?;
                let metadata = fs::symlink_metadata(entry.path())?;
                if is_reparse(&metadata) || !metadata.is_dir() {
                    continue;
                }
                let directory = entry.path();
                let Ok(canonical_directory) = fs::canonicalize(&directory) else {
                    continue;
                };
                if !canonical_directory.starts_with(&canonical_root) {
                    continue;
                }
                if let Ok(manifest) = read_contained_manifest(&directory) {
                    found.push(manifest);
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
        let target = self.layout.model_dir(manifest)?;
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
        let stored = read_contained_manifest(&target)?;
        if stored != *manifest || stored.artifact_fingerprint() != manifest.artifact_fingerprint() {
            return Err(Error::Invalid(
                "stored manifest does not exactly match deletion request".into(),
            ));
        }
        fs::remove_dir_all(target)?;
        Ok(true)
    }
}

/// Reads an identity record only after both the containing directory and manifest file have been
/// proven to be regular, non-reparse filesystem objects within the same canonical directory.
fn read_contained_manifest(directory: &Path) -> Result<Manifest> {
    let directory_metadata = fs::symlink_metadata(directory)?;
    if is_reparse(&directory_metadata) || !directory_metadata.is_dir() {
        return Err(Error::Invalid(
            "manifest directory cannot be a symlink or reparse point".into(),
        ));
    }
    let canonical_directory = fs::canonicalize(directory)?;
    let path = directory.join(MANIFEST_FILE);
    let metadata = fs::symlink_metadata(&path)?;
    if is_reparse(&metadata) || !metadata.is_file() {
        return Err(Error::Invalid(
            "manifest must be a regular non-reparse file".into(),
        ));
    }
    let canonical_path = fs::canonicalize(&path)?;
    if canonical_path.parent() != Some(canonical_directory.as_path()) {
        return Err(Error::Invalid(
            "manifest escaped its containing directory".into(),
        ));
    }
    Manifest::from_json(&fs::read(canonical_path)?)
}

fn acquire_store_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    acquire_store_lock_with_cancel(layout, manifest, &|| false)
}

fn acquire_store_lock_with_cancel(
    layout: &CacheLayout,
    manifest: &Manifest,
    cancelled: &impl Fn() -> bool,
) -> Result<fs::File> {
    let path = layout.lock_file(manifest)?;
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("lock path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    reject_reparse_components(&layout.root, Path::new("locks"))?;
    let lock_metadata = symlink_metadata_if_exists(&path)?;
    if is_reparse(&fs::symlink_metadata(parent)?) || lock_metadata.as_ref().is_some_and(is_reparse)
    {
        return Err(Error::Invalid("model lock path cannot be a symlink".into()));
    }
    let deadline = Instant::now() + LOCK_WAIT_LIMIT;
    loop {
        if cancelled() {
            return Err(Error::Cancelled);
        }
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
        if Instant::now() >= deadline {
            return Err(Error::Busy);
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn acquire_shared_store_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    let path = layout.lock_file(manifest)?;
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
    let deadline = Instant::now() + LOCK_WAIT_LIMIT;
    loop {
        match FileExt::try_lock_shared(&file) {
            Ok(()) => return Ok(file),
            Err(error) if is_lock_contention(&error) && Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(error) if is_lock_contention(&error) => return Err(Error::Busy),
            Err(error) => return Err(error.into()),
        }
    }
}

fn acquire_delete_lock(layout: &CacheLayout, manifest: &Manifest) -> Result<fs::File> {
    let path = layout.lock_file(manifest)?;
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

fn configure_no_follow(options: &mut OpenOptions) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
}

fn copy_import_artifact(
    source: &Path,
    target: &Path,
    artifact: &Artifact,
    after_open: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let mut source_options = OpenOptions::new();
    source_options.read(true);
    configure_no_follow(&mut source_options);
    let mut source_file = source_options.open(source)?;
    let source_metadata = source_file.metadata()?;
    if !source_metadata.is_file() {
        return Err(Error::Invalid(format!(
            "artifact is not a regular file: {}",
            artifact.path
        )));
    }
    if source_metadata.len() != artifact.size {
        return Err(Error::SizeLimit {
            expected: artifact.size,
            actual: source_metadata.len(),
        });
    }

    // Tests use this boundary to replace or mutate the pathname deterministically. All reads
    // below remain bound to the single handle opened above.
    after_open()?;

    let mut target_options = OpenOptions::new();
    target_options.write(true).create_new(true);
    configure_no_follow(&mut target_options);
    let mut target_file = target_options.open(target)?;
    let mut digest = Sha256::new();
    let mut copied = 0_u64;
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = source_file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        copied = copied
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| Error::Invalid("artifact length overflow".into()))?,
            )
            .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
        if copied > artifact.size {
            return Err(Error::SizeLimit {
                expected: artifact.size,
                actual: copied,
            });
        }
        digest.update(&buffer[..read]);
        target_file.write_all(&buffer[..read])?;
    }
    target_file.flush()?;
    target_file.sync_all()?;
    drop(target_file);

    if copied != artifact.size {
        return Err(Error::SizeLimit {
            expected: artifact.size,
            actual: copied,
        });
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != artifact.sha256 {
        return Err(Error::HashMismatch {
            expected: artifact.sha256.clone(),
            actual,
        });
    }

    // Verify the exact bytes at the staging pathname after the copy is durable. Promotion never
    // relies only on the source hash or on metadata observed before the copy.
    let mut staged_options = OpenOptions::new();
    staged_options.read(true);
    configure_no_follow(&mut staged_options);
    let mut staged_file = staged_options.open(target)?;
    let staged_metadata = staged_file.metadata()?;
    if !staged_metadata.is_file() || staged_metadata.len() != artifact.size {
        return Err(Error::Invalid(
            "staged artifact changed during import".into(),
        ));
    }
    let mut staged_digest = Sha256::new();
    let mut staged_size = 0_u64;
    loop {
        let read = staged_file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        staged_size = staged_size
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| Error::Invalid("artifact length overflow".into()))?,
            )
            .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
        if staged_size > artifact.size {
            return Err(Error::SizeLimit {
                expected: artifact.size,
                actual: staged_size,
            });
        }
        staged_digest.update(&buffer[..read]);
    }
    let staged_actual = format!("{:x}", staged_digest.finalize());
    if staged_size != artifact.size || staged_actual != artifact.sha256 {
        return Err(Error::HashMismatch {
            expected: artifact.sha256.clone(),
            actual: staged_actual,
        });
    }
    Ok(())
}

pub(crate) fn prepare_staging(layout: &CacheLayout, staging: &Path) -> Result<()> {
    for directory in [
        layout.root.join("models"),
        layout.root.join("staging"),
        layout.root.join("locks"),
        layout.root.join("transactions"),
    ] {
        fs::create_dir_all(&directory)?;
        ensure_contained_directory(&layout.root, &directory)?;
    }
    if let Some(metadata) = symlink_metadata_if_exists(staging)? {
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
    promote_with_observer(layout, manifest, staging, |_| Ok(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromotionPhase {
    MarkerDurable,
    PreviousDisplaced,
    ReplacementPromoted,
}

fn promote_with_observer(
    layout: &CacheLayout,
    manifest: &Manifest,
    staging: &Path,
    mut observe: impl FnMut(PromotionPhase) -> Result<()>,
) -> Result<()> {
    promote_with_operations(
        layout,
        manifest,
        staging,
        &mut observe,
        |from, to| fs::rename(from, to),
        sync_directory,
    )
}

fn promote_with_operations(
    layout: &CacheLayout,
    manifest: &Manifest,
    staging: &Path,
    observe: &mut impl FnMut(PromotionPhase) -> Result<()>,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
    mut sync_dir: impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let final_path = layout.model_dir(manifest)?;
    reconcile_repair_inner(layout, manifest)?;
    if symlink_metadata_if_exists(&final_path)?.is_none() {
        durable_rename(staging, &final_path, &mut rename, &mut sync_dir)?;
        return Ok(());
    }
    reject_reparse_components(
        &layout.root,
        Path::new("models")
            .join(CacheLayout::key(manifest)?)
            .as_path(),
    )?;
    let transaction_root = layout.root.join("transactions");
    fs::create_dir_all(&transaction_root)?;
    ensure_contained_directory(&layout.root, &transaction_root)?;
    sync_dir(&layout.root)?;
    let marker = layout.repair_marker(manifest)?;
    let backup = layout.repair_backup(manifest)?;
    if symlink_metadata_if_exists(&backup)?.is_some()
        || symlink_metadata_if_exists(&marker)?.is_some()
    {
        return Err(Error::Invalid(
            "repair transaction state is not clean".into(),
        ));
    }
    let marker_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&marker)?;
    marker_file.sync_all()?;
    sync_dir(&transaction_root)?;
    observe(PromotionPhase::MarkerDurable)?;
    if let Err(error) = durable_rename(&final_path, &backup, &mut rename, &mut sync_dir) {
        verify_transaction_final(&final_path)?;
        fs::remove_file(&marker)?;
        sync_dir(&transaction_root)?;
        return Err(error.into());
    }
    observe(PromotionPhase::PreviousDisplaced)?;
    if durable_rename(staging, &final_path, &mut rename, &mut sync_dir).is_err() {
        if durable_rename(&backup, &final_path, &mut rename, &mut sync_dir).is_err() {
            // The durable marker is intentionally retained. A later status/load/install call can
            // now reconcile the unambiguous (missing final, complete staging, complete backup)
            // transaction instead of silently losing the only recovery signal.
            return Err(Error::Invalid(
                "repair promotion was interrupted and requires reconciliation".into(),
            ));
        }
        verify_transaction_final(&final_path)?;
        fs::remove_file(&marker)?;
        sync_dir(&transaction_root)?;
        return Err(Error::Invalid(
            "replacement promotion failed; the previous model was restored".into(),
        ));
    }
    verify_transaction_final(&final_path)?;
    observe(PromotionPhase::ReplacementPromoted)?;
    quarantine_backup_with_operations(layout, manifest, &backup, &mut rename, &mut sync_dir)?;
    fs::remove_file(marker)?;
    sync_dir(&transaction_root)?;
    Ok(())
}

fn durable_rename(
    from: &Path,
    to: &Path,
    rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let from_parent = from.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "source has no parent")
    })?;
    let to_parent = to.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "destination has no parent",
        )
    })?;
    // The model directory contains already-synced files, but its own entries also need a barrier
    // before the directory is made reachable under a durable name.
    sync_dir(from)?;
    sync_dir(from_parent)?;
    if from_parent != to_parent {
        sync_dir(to_parent)?;
    }
    rename(from, to)?;
    sync_dir(to)?;
    sync_dir(from_parent)?;
    if from_parent != to_parent {
        sync_dir(to_parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    fs::File::open(path)?.sync_all()
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;

    // FILE_FLAG_BACKUP_SEMANTICS is required to obtain a directory handle. `sync_all` maps to
    // FlushFileBuffers, giving supported Windows filesystems the strongest available metadata
    // persistence barrier without relying on host-specific native APIs.
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const FILE_SHARE_DELETE: u32 = 0x0000_0004;
    OpenOptions::new()
        .read(true)
        .write(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?
        .sync_all()
}

fn quarantine_backup_with_operations(
    layout: &CacheLayout,
    manifest: &Manifest,
    backup: &Path,
    rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    if symlink_metadata_if_exists(backup)?.is_none() {
        return Ok(());
    }
    let quarantine_root = layout.root.join("quarantine");
    fs::create_dir_all(&quarantine_root)?;
    ensure_contained_directory(&layout.root, &quarantine_root)?;
    sync_dir(&layout.root)?;
    let quarantine = loop {
        let sequence = QUARANTINE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = quarantine_root.join(format!(
            "{}.{}.{}.invalid",
            CacheLayout::key(manifest)?,
            std::process::id(),
            sequence
        ));
        match fs::symlink_metadata(&candidate) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break candidate,
            Ok(_) => {}
            Err(error) => return Err(error.into()),
        }
    };
    durable_rename(backup, &quarantine, rename, sync_dir)?;
    Ok(())
}

fn reconcile_repair_inner(layout: &CacheLayout, manifest: &Manifest) -> Result<()> {
    reconcile_repair_with_operations(
        layout,
        manifest,
        &mut |from, to| fs::rename(from, to),
        &mut sync_directory,
    )
}

fn reconcile_repair_with_operations(
    layout: &CacheLayout,
    manifest: &Manifest,
    rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let marker = layout.repair_marker(manifest)?;
    let marker_metadata = match fs::symlink_metadata(&marker) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if is_reparse(&marker_metadata) || !marker_metadata.is_file() {
        return Err(Error::Invalid("repair marker is not a regular file".into()));
    }
    let final_path = layout.model_dir(manifest)?;
    let staging = layout.staging_dir(manifest)?;
    let backup = layout.repair_backup(manifest)?;
    let final_exists = symlink_metadata_if_exists(&final_path)?.is_some();
    let staging_exists = symlink_metadata_if_exists(&staging)?.is_some();
    let backup_exists = symlink_metadata_if_exists(&backup)?.is_some();

    match (final_exists, staging_exists, backup_exists) {
        // Crash after marker creation: resume the intended replacement.
        (true, true, false) => {
            durable_rename(&final_path, &backup, rename, sync_dir)?;
            durable_rename(&staging, &final_path, rename, sync_dir)?;
        }
        // Crash after displacement: finish promoting the fully prepared replacement.
        (false, true, true) => durable_rename(&staging, &final_path, rename, sync_dir)?,
        // The replacement vanished: restore the previous state rather than leave a hole.
        (false, false, true) => durable_rename(&backup, &final_path, rename, sync_dir)?,
        // New state is already visible, or the marker preceded any mutation.
        (true, _, _) => {}
        _ => {
            return Err(Error::Invalid(
                "repair transaction cannot be reconciled safely".into(),
            ));
        }
    }
    verify_transaction_final(&final_path)?;
    quarantine_backup_with_operations(layout, manifest, &backup, rename, sync_dir)?;
    fs::remove_file(marker)?;
    let transaction_root = layout.root.join("transactions");
    sync_dir(&transaction_root)?;
    Ok(())
}

fn symlink_metadata_if_exists(path: &Path) -> Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn verify_transaction_final(path: &Path) -> Result<()> {
    match symlink_metadata_if_exists(path)? {
        Some(metadata) if metadata.is_dir() && !is_reparse(&metadata) => Ok(()),
        _ => Err(Error::Invalid(
            "repair transaction did not restore a safe installed state".into(),
        )),
    }
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
    hash_file_with_cancel(path, &|| false)
}

fn hash_file_with_cancel(path: &Path, cancelled: &impl Fn() -> bool) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_import_source(source: &Path) -> Result<PathBuf> {
    if source.components().any(|component| {
        matches!(
            component,
            std::path::Component::CurDir | std::path::Component::ParentDir
        )
    }) {
        return Err(Error::Invalid(
            "import source contains unsafe components".into(),
        ));
    }
    let metadata = fs::symlink_metadata(source)?;
    if is_reparse(&metadata) || !metadata.is_dir() {
        return Err(Error::Invalid(
            "import source must be a real non-reparse directory".into(),
        ));
    }
    // Inspect each existing spelling component before canonicalization, including parent
    // components. Canonicalization alone would erase evidence that the authorized path traversed
    // a symlink or Windows junction.
    let mut current = PathBuf::new();
    for component in source.components() {
        current.push(component.as_os_str());
        if current.as_os_str().is_empty() {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if is_reparse(&metadata) => {
                return Err(Error::Invalid(
                    "import source contains a symlink or reparse point".into(),
                ));
            }
            Ok(_) => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(fs::canonicalize(source)?)
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };
    use tempfile::TempDir;

    fn local_manifest(body: &[u8]) -> Manifest {
        Manifest {
            schema_version: 1,
            canonical_id: "tests/store-transaction".into(),
            revision: "0123456789abcdef0123456789abcdef01234567".into(),
            license: crate::License {
                spdx: "MIT".into(),
                source_url: "https://example.invalid/license".into(),
            },
            semantic_verification: crate::SemanticVerification::Verified {
                evidence: "store transaction fixture".into(),
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
                matryoshka: Vec::new(),
            },
            tensors: TensorMetadata {
                format: "fixture".into(),
                dtype: "bytes".into(),
                architecture: "identity".into(),
            },
            runtime: RuntimeMetadata::CatalogOnly,
            artifacts: vec![Artifact {
                path: "weights/model.bin".into(),
                url: "https://example.invalid/model".into(),
                sha256: format!("{:x}", Sha256::digest(body)),
                size: u64::try_from(body.len()).unwrap_or_default(),
            }],
        }
    }

    fn trusted_store(root: &Path, manifest: &Manifest) -> Result<ModelStore> {
        let trust = SemanticTrustRoot::from_evidence([TrustedSemanticEvidence {
            manifest_fingerprint: manifest.semantic_fingerprint()?,
            evidence: "store unit test".into(),
        }])?;
        ModelStore::with_trust_root(root, trust)
    }

    fn write_valid_model(directory: &Path, manifest: &Manifest, body: &[u8]) -> Result<()> {
        fs::create_dir_all(directory.join("weights"))?;
        fs::write(directory.join("weights/model.bin"), body)?;
        fs::write(directory.join(MANIFEST_FILE), manifest.to_json()?)?;
        Ok(())
    }

    fn manifest() -> Result<Manifest> {
        crate::curated_manifests()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invalid("missing curated test manifest".into()))
    }

    #[test]
    fn hashing_checks_cancellation_between_chunks() -> Result<()> {
        let temp = TempDir::new()?;
        let path = temp.path().join("large.bin");
        fs::write(&path, vec![7_u8; 256 * 1024])?;
        let checks = Cell::new(0_u8);
        let result = hash_file_with_cancel(&path, &|| {
            checks.set(checks.get().saturating_add(1));
            checks.get() > 2
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        Ok(())
    }

    #[test]
    fn every_repair_promotion_phase_is_reconciled_deterministically() -> Result<()> {
        for interrupted_after in [
            PromotionPhase::MarkerDurable,
            PromotionPhase::PreviousDisplaced,
            PromotionPhase::ReplacementPromoted,
        ] {
            let temp = TempDir::new()?;
            let layout = CacheLayout::new(temp.path())?;
            let manifest = manifest()?;
            let final_path = layout.model_dir(&manifest)?;
            let staging = layout.staging_dir(&manifest)?;
            fs::create_dir_all(&final_path)?;
            fs::write(final_path.join("state"), b"old")?;
            prepare_staging(&layout, &staging)?;
            fs::write(staging.join("state"), b"new")?;

            let result = promote_with_observer(&layout, &manifest, &staging, |phase| {
                if phase == interrupted_after {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(Error::Cancelled)));
            reconcile_repair_inner(&layout, &manifest)?;
            assert_eq!(fs::read(final_path.join("state"))?, b"new");
            assert!(!layout.repair_marker(&manifest)?.exists());
            assert!(!layout.repair_backup(&manifest)?.exists());
        }
        Ok(())
    }

    #[test]
    fn failed_promotion_and_failed_rollback_retain_marker_for_later_recovery() -> Result<()> {
        let temp = TempDir::new()?;
        let layout = CacheLayout::new(temp.path())?;
        let manifest = manifest()?;
        let final_path = layout.model_dir(&manifest)?;
        let staging = layout.staging_dir(&manifest)?;
        fs::create_dir_all(&final_path)?;
        fs::write(final_path.join("state"), b"old")?;
        prepare_staging(&layout, &staging)?;
        fs::write(staging.join("state"), b"new")?;

        let mut rename_count = 0_u8;
        let mut observer = |_| Ok(());
        let result = promote_with_operations(
            &layout,
            &manifest,
            &staging,
            &mut observer,
            |from, to| {
                rename_count = rename_count.saturating_add(1);
                if matches!(rename_count, 2 | 3) {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "injected rename failure",
                    ))
                } else {
                    fs::rename(from, to)
                }
            },
            sync_directory,
        );
        assert!(matches!(result, Err(Error::Invalid(_))));
        assert!(layout.repair_marker(&manifest)?.is_file());
        assert!(!final_path.exists());
        assert!(staging.is_dir());
        assert!(layout.repair_backup(&manifest)?.is_dir());

        reconcile_repair_inner(&layout, &manifest)?;
        assert_eq!(fs::read(final_path.join("state"))?, b"new");
        assert!(!layout.repair_marker(&manifest)?.exists());
        assert!(!layout.repair_backup(&manifest)?.exists());
        Ok(())
    }

    #[test]
    fn repair_promotion_persists_each_metadata_transition_in_order() -> Result<()> {
        let temp = TempDir::new()?;
        let layout = CacheLayout::new(temp.path())?;
        let manifest = manifest()?;
        let final_path = layout.model_dir(&manifest)?;
        let staging = layout.staging_dir(&manifest)?;
        fs::create_dir_all(&final_path)?;
        fs::write(final_path.join("state"), b"old")?;
        prepare_staging(&layout, &staging)?;
        fs::write(staging.join("state"), b"new")?;

        let events = Rc::new(RefCell::new(Vec::new()));
        let rename_events = Rc::clone(&events);
        let sync_events = Rc::clone(&events);
        promote_with_operations(
            &layout,
            &manifest,
            &staging,
            &mut |_| Ok(()),
            move |from, to| {
                rename_events.borrow_mut().push(format!(
                    "rename:{}->{}",
                    from.parent()
                        .and_then(Path::file_name)
                        .and_then(|v| v.to_str())
                        .unwrap_or("root"),
                    to.parent()
                        .and_then(Path::file_name)
                        .and_then(|v| v.to_str())
                        .unwrap_or("root")
                ));
                fs::rename(from, to)
            },
            move |path| {
                sync_events.borrow_mut().push(format!(
                    "sync:{}",
                    path.file_name().and_then(|v| v.to_str()).unwrap_or("root")
                ));
                Ok(())
            },
        )?;

        let events = events.borrow();
        assert!(
            events
                .first()
                .is_some_and(|event| event.starts_with("sync:"))
        );
        assert_eq!(events.get(1).map(String::as_str), Some("sync:transactions"));
        let metadata_events = events
            .iter()
            .filter(|event| {
                matches!(
                    event.as_str(),
                    "sync:models"
                        | "sync:transactions"
                        | "sync:staging"
                        | "rename:models->transactions"
                        | "rename:staging->models"
                )
            })
            .map(String::as_str)
            .collect::<Vec<_>>();
        assert!(metadata_events.windows(5).any(|window| window
            == [
                "sync:models",
                "sync:transactions",
                "rename:models->transactions",
                "sync:models",
                "sync:transactions"
            ]));
        assert!(metadata_events.windows(5).any(|window| window
            == [
                "sync:staging",
                "sync:models",
                "rename:staging->models",
                "sync:staging",
                "sync:models"
            ]));
        assert_eq!(events.last().map(String::as_str), Some("sync:transactions"));
        Ok(())
    }

    #[test]
    fn failed_marker_directory_barrier_retains_repair_signal() -> Result<()> {
        let temp = TempDir::new()?;
        let layout = CacheLayout::new(temp.path())?;
        let manifest = manifest()?;
        let final_path = layout.model_dir(&manifest)?;
        let staging = layout.staging_dir(&manifest)?;
        fs::create_dir_all(&final_path)?;
        fs::write(final_path.join("state"), b"old")?;
        prepare_staging(&layout, &staging)?;
        fs::write(staging.join("state"), b"new")?;
        let calls = Cell::new(0_u8);

        let result = promote_with_operations(
            &layout,
            &manifest,
            &staging,
            &mut |_| Ok(()),
            |from, to| fs::rename(from, to),
            |_| {
                calls.set(calls.get() + 1);
                if calls.get() == 2 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "injected sync failure",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(Error::Io(_))));
        assert!(layout.repair_marker(&manifest)?.is_file());
        assert!(final_path.is_dir());
        assert!(staging.is_dir());

        reconcile_repair_inner(&layout, &manifest)?;
        assert_eq!(fs::read(final_path.join("state"))?, b"new");
        assert!(!layout.repair_marker(&manifest)?.exists());
        Ok(())
    }

    #[test]
    fn reconciliation_sync_failure_keeps_marker_until_safe_retry() -> Result<()> {
        let temp = TempDir::new()?;
        let layout = CacheLayout::new(temp.path())?;
        let manifest = manifest()?;
        let final_path = layout.model_dir(&manifest)?;
        let staging = layout.staging_dir(&manifest)?;
        fs::create_dir_all(&final_path)?;
        fs::write(final_path.join("state"), b"old")?;
        prepare_staging(&layout, &staging)?;
        fs::write(staging.join("state"), b"new")?;
        let result = promote_with_observer(&layout, &manifest, &staging, |phase| {
            if phase == PromotionPhase::PreviousDisplaced {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(Error::Cancelled)));

        let calls = Cell::new(0_u8);
        let result = reconcile_repair_with_operations(
            &layout,
            &manifest,
            &mut |from, to| fs::rename(from, to),
            &mut |_| {
                calls.set(calls.get() + 1);
                if calls.get() == 4 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "injected reconciliation sync failure",
                    ))
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(Error::Io(_))));
        assert!(layout.repair_marker(&manifest)?.is_file());

        reconcile_repair_inner(&layout, &manifest)?;
        assert_eq!(fs::read(final_path.join("state"))?, b"new");
        assert!(!layout.repair_marker(&manifest)?.exists());
        assert!(!layout.repair_backup(&manifest)?.exists());
        Ok(())
    }

    #[test]
    fn verified_model_reconciles_every_interrupted_promotion_before_shared_lease() -> Result<()> {
        let body = b"verified transaction bytes";
        for interrupted_after in [
            PromotionPhase::MarkerDurable,
            PromotionPhase::PreviousDisplaced,
            PromotionPhase::ReplacementPromoted,
        ] {
            let temp = TempDir::new()?;
            let manifest = local_manifest(body);
            let store = trusted_store(temp.path(), &manifest)?;
            let final_path = store.layout.model_dir(&manifest)?;
            let staging = store.layout.staging_dir(&manifest)?;
            write_valid_model(&final_path, &manifest, body)?;
            prepare_staging(&store.layout, &staging)?;
            write_valid_model(&staging, &manifest, body)?;

            let result = promote_with_observer(&store.layout, &manifest, &staging, |phase| {
                if phase == interrupted_after {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(Error::Cancelled)));

            let verified = store.verified_model(&manifest)?;
            assert_eq!(verified.artifact_bytes("weights/model.bin")?, body);
            assert!(!store.layout.repair_marker(&manifest)?.exists());
            assert!(!store.layout.repair_backup(&manifest)?.exists());
        }
        Ok(())
    }

    #[test]
    fn import_hashes_the_open_handle_and_rejects_synchronized_mutation() -> Result<()> {
        let cache = TempDir::new()?;
        let source = TempDir::new()?;
        let body = b"expected import bytes";
        let manifest = local_manifest(body);
        fs::create_dir_all(source.path().join("weights"))?;
        fs::write(source.path().join("weights/model.bin"), body)?;
        let store = trusted_store(cache.path(), &manifest)?;

        let result = store.import_with_observer(&manifest, source.path(), |opened| {
            fs::write(opened, b"malicious import byte")?;
            Ok(())
        });
        assert!(matches!(result, Err(Error::HashMismatch { .. })));
        assert_eq!(store.status(&manifest)?, ModelStatus::Missing);
        Ok(())
    }

    #[test]
    fn import_reads_exact_open_handle_when_source_path_is_replaced() -> Result<()> {
        let cache = TempDir::new()?;
        let source = TempDir::new()?;
        let body = b"expected import bytes";
        let replacement = b"malicious import byte";
        assert_eq!(body.len(), replacement.len());
        let manifest = local_manifest(body);
        fs::create_dir_all(source.path().join("weights"))?;
        let source_path = source.path().join("weights/model.bin");
        fs::write(&source_path, body)?;
        let store = trusted_store(cache.path(), &manifest)?;

        let displaced = source.path().join("weights/original.bin");
        let status = store.import_with_observer(&manifest, source.path(), |opened| {
            fs::rename(opened, &displaced)?;
            fs::write(opened, replacement)?;
            Ok(())
        })?;
        assert_eq!(status, ModelStatus::Loadable);
        let verified = store.verified_model(&manifest)?;
        assert_eq!(verified.artifact_bytes("weights/model.bin")?, body);
        Ok(())
    }
}

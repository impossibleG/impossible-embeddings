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

use crate::manifest::MAX_MANIFEST_BYTES;
use crate::{
    Artifact, Dimensions, Error, Manifest, ModelStatus, Pooling, Prefixes, Result, RuntimeMetadata,
    TensorMetadata, TokenizerMetadata,
};

const MANIFEST_FILE: &str = "manifest.json";
// Durable promotion can include several metadata flushes on high-latency local filesystems. Give
// a coalescing peer enough time to observe the committed model while keeping contention bounded.
pub(crate) const LOCK_WAIT_LIMIT: Duration = Duration::from_secs(1);
const MAX_DISCOVERY_ROOTS: usize = 64;
const MAX_DISCOVERY_ENTRIES: usize = 1_024;
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
    fn deletion_tombstone(&self, manifest: &Manifest) -> Result<PathBuf> {
        Ok(self
            .root
            .join("transactions")
            .join(format!("{}.deleted", Self::key(manifest)?)))
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
        self.artifact_bytes_with_observer(relative, || Ok(()))
    }

    fn artifact_bytes_with_observer(
        &self,
        relative: &str,
        after_open: impl FnOnce() -> Result<()>,
    ) -> Result<Vec<u8>> {
        let artifact = self
            .manifest
            .artifacts
            .iter()
            .find(|artifact| artifact.path == relative)
            .ok_or_else(|| Error::Invalid("runtime requested an undeclared artifact".into()))?;
        let mut file = open_contained_regular_file(&self.root, Path::new(relative), artifact.size)?;
        after_open()?;
        let capacity = usize::try_from(artifact.size)
            .map_err(|_| Error::Invalid("artifact is too large for this platform".into()))?;
        let read_limit = artifact
            .size
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
        let mut bytes = Vec::with_capacity(capacity);
        Read::by_ref(&mut file)
            .take(read_limit)
            .read_to_end(&mut bytes)?;
        let final_metadata = file.metadata()?;
        if bytes.len() != capacity
            || is_reparse(&final_metadata)
            || !final_metadata.is_file()
            || final_metadata.len() != artifact.size
            || format!("{:x}", Sha256::digest(&bytes)) != artifact.sha256
        {
            return Err(Error::Invalid("artifact changed after verification".into()));
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
            if !verify_contained_artifact(&self.root, artifact, &|| false)? {
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
            if !verify_contained_artifact(&directory, artifact, cancelled)? {
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
            let manifest_path = staging.join(MANIFEST_FILE);
            let mut manifest_options = OpenOptions::new();
            manifest_options.write(true).create_new(true);
            configure_no_follow(&mut manifest_options);
            let mut manifest_file = manifest_options.open(&manifest_path)?;
            manifest_file.write_all(&manifest.to_json()?)?;
            manifest_file.flush()?;
            manifest_file.sync_all()?;
            drop(manifest_file);
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
        if roots.len() > MAX_DISCOVERY_ROOTS {
            return Err(Error::Invalid("model discovery root limit exceeded".into()));
        }
        let mut found = Vec::new();
        let mut inspected_entries = 0_usize;
        for root in roots {
            let metadata = match fs::symlink_metadata(&root.0) {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => {
                    return Err(Error::Invalid("model discovery root is unavailable".into()));
                }
            };
            if is_reparse(&metadata) || !metadata.is_dir() {
                continue;
            }
            let canonical_root = fs::canonicalize(&root.0)
                .map_err(|_| Error::Invalid("model discovery root is unavailable".into()))?;
            let entries = fs::read_dir(&root.0)
                .map_err(|_| Error::Invalid("model discovery root is unavailable".into()))?;
            for entry in entries {
                inspected_entries = inspected_entries
                    .checked_add(1)
                    .ok_or_else(|| Error::Invalid("model discovery entry limit exceeded".into()))?;
                if inspected_entries > MAX_DISCOVERY_ENTRIES {
                    return Err(Error::Invalid(
                        "model discovery entry limit exceeded".into(),
                    ));
                }
                let entry = entry
                    .map_err(|_| Error::Invalid("model discovery entry is unavailable".into()))?;
                let metadata = fs::symlink_metadata(entry.path())
                    .map_err(|_| Error::Invalid("model discovery entry is unavailable".into()))?;
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
        self.delete_locked_with_operations(
            manifest,
            &mut |_| Ok(()),
            &mut |from, to| fs::rename(from, to),
            &mut |path| fs::remove_dir_all(path),
            &mut sync_directory,
        )
    }

    fn delete_locked_with_operations(
        &self,
        manifest: &Manifest,
        observe: &mut impl FnMut(DeletionPhase) -> Result<()>,
        rename: &mut impl FnMut(&Path, &Path) -> std::io::Result<()>,
        remove_dir_all: &mut impl FnMut(&Path) -> std::io::Result<()>,
        sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
    ) -> Result<bool> {
        // A replacement transaction and deletion use the same exclusive identity lock. Complete
        // any durable repair record first so no later recovery can recreate a model that this
        // operation has successfully removed.
        reconcile_repair_with_operations(&self.layout, manifest, rename, sync_dir)?;
        observe(DeletionPhase::RepairReconciled)?;

        let transaction_root = self.layout.root.join("transactions");
        fs::create_dir_all(&transaction_root)?;
        ensure_contained_directory(&self.layout.root, &transaction_root)?;
        sync_dir(&transaction_root)?;
        sync_dir(&self.layout.root)?;
        let tombstone = self.layout.deletion_tombstone(manifest)?;
        let pending_deletion = symlink_metadata_if_exists(&tombstone)?.is_some();
        if pending_deletion {
            remove_deletion_tombstone(&tombstone, &transaction_root, remove_dir_all, sync_dir)?;
        }

        let target = self.layout.model_dir(manifest)?;
        let metadata = match fs::symlink_metadata(&target) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(pending_deletion);
            }
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
        durable_rename(&target, &tombstone, rename, sync_dir)?;
        observe(DeletionPhase::TombstoneDurable)?;
        remove_deletion_tombstone(&tombstone, &transaction_root, remove_dir_all, sync_dir)?;
        observe(DeletionPhase::TombstoneRemoved)?;
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeletionPhase {
    RepairReconciled,
    TombstoneDurable,
    TombstoneRemoved,
}

fn remove_deletion_tombstone(
    tombstone: &Path,
    transaction_root: &Path,
    remove_dir_all: &mut impl FnMut(&Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let metadata = match fs::symlink_metadata(tombstone) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if is_reparse(&metadata) || !metadata.is_dir() {
        return Err(Error::Invalid(
            "deletion tombstone is not a regular directory".into(),
        ));
    }
    remove_dir_all(tombstone)?;
    sync_dir(transaction_root)?;
    Ok(())
}

/// Reads an identity record only after both the containing directory and manifest file have been
/// proven to be regular, non-reparse filesystem objects within the same canonical directory.
fn read_contained_manifest(directory: &Path) -> Result<Manifest> {
    let directory_metadata = fs::symlink_metadata(directory)
        .map_err(|_| Error::Invalid("manifest directory is unavailable".into()))?;
    if is_reparse(&directory_metadata) || !directory_metadata.is_dir() {
        return Err(Error::Invalid(
            "manifest directory cannot be a symlink or reparse point".into(),
        ));
    }
    let canonical_directory = fs::canonicalize(directory)
        .map_err(|_| Error::Invalid("manifest directory is unavailable".into()))?;
    let path = directory.join(MANIFEST_FILE);
    let mut options = OpenOptions::new();
    options.read(true);
    configure_no_follow(&mut options);
    let mut file = options
        .open(&path)
        .map_err(|_| Error::Invalid("manifest is unavailable".into()))?;
    let initial = file
        .metadata()
        .map_err(|_| Error::Invalid("manifest is unavailable".into()))?;
    if is_reparse(&initial) || !initial.is_file() || initial.len() > MAX_MANIFEST_BYTES as u64 {
        return Err(Error::Invalid(
            "manifest must be a bounded regular non-reparse file".into(),
        ));
    }

    // The file is read through the same no-follow handle whose metadata was checked. One extra
    // byte detects growth past the hard bound without allocating from attacker-controlled size.
    let initial_len = usize::try_from(initial.len())
        .map_err(|_| Error::Invalid("manifest exceeds the supported size".into()))?;
    let mut bytes = Vec::with_capacity(initial_len);
    Read::by_ref(&mut file)
        .take(MAX_MANIFEST_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Invalid("manifest is unavailable".into()))?;
    let final_metadata = file
        .metadata()
        .map_err(|_| Error::Invalid("manifest is unavailable".into()))?;
    if is_reparse(&final_metadata)
        || !final_metadata.is_file()
        || bytes.len() > MAX_MANIFEST_BYTES
        || final_metadata.len() != initial.len()
        || final_metadata.len() != bytes.len() as u64
    {
        return Err(Error::Invalid(
            "manifest changed while it was being read".into(),
        ));
    }
    // Retain the directory containment proof for the authorized parent. The opened leaf itself
    // is never canonicalized or reopened, avoiding a check/open race.
    if fs::canonicalize(directory)
        .map_err(|_| Error::Invalid("manifest directory is unavailable".into()))?
        != canonical_directory
    {
        return Err(Error::Invalid(
            "manifest directory changed while it was being read".into(),
        ));
    }
    Manifest::from_json(&bytes)
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
        // O_NONBLOCK prevents a regular-file-to-FIFO swap from stalling the process at open.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        const FILE_SHARE_READ: u32 = 0x0000_0001;
        const FILE_SHARE_WRITE: u32 = 0x0000_0002;
        const FILE_SHARE_DELETE: u32 = 0x0000_0004;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        options
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS);
    }
}

/// Creates each missing directory component under an already trusted base and proves after every
/// operation that the component is still a real directory below that base. This is intentionally
/// component-at-a-time: `create_dir_all` may otherwise traverse an attacker-replaced intermediate
/// symlink or Windows junction.
pub(crate) fn create_contained_parent_directories(base: &Path, relative_file: &Path) -> Result<()> {
    let parent = relative_file.parent().unwrap_or_else(|| Path::new(""));
    validate_relative_path(parent)?;
    securely_create_parent_directories(base, parent)?;
    revalidate_contained_directory(base, parent)
}

/// Opens a brand-new regular leaf below `base` without following a leaf symlink. The handle is
/// returned only after a second component and canonical-containment pass, so callers never write
/// through a path that was replaced at the create boundary. Existing leaves are never truncated.
pub(crate) fn create_new_contained_file(base: &Path, relative: &Path) -> Result<fs::File> {
    validate_relative_path(relative)?;
    if relative.as_os_str().is_empty() {
        return Err(Error::Invalid("file path cannot be empty".into()));
    }
    let file = securely_create_new_file(base, relative)?;
    let metadata = file.metadata()?;
    if is_reparse(&metadata) || !metadata.is_file() {
        return Err(Error::Invalid(
            "cache leaf is not a regular non-reparse file".into(),
        ));
    }
    revalidate_contained_path(base, relative)?;
    Ok(file)
}

#[cfg(windows)]
fn locked_windows_directory_chain(base: &Path, relative: &Path) -> Result<Vec<fs::File>> {
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;

    fn open_locked_directory(path: &Path) -> Result<fs::File> {
        use std::os::windows::fs::OpenOptionsExt;
        let file = OpenOptions::new()
            .read(true)
            // Deliberately omit FILE_SHARE_DELETE. While this handle lives, Windows cannot
            // rename/delete this component and substitute a junction before the child opens.
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?;
        let metadata = file.metadata()?;
        if is_reparse(&metadata) || !metadata.is_dir() {
            return Err(Error::Invalid(
                "cache path contains a junction or non-directory".into(),
            ));
        }
        Ok(file)
    }

    let mut handles = vec![open_locked_directory(base)?];
    let mut current = base.to_path_buf();
    for component in relative.components() {
        let std::path::Component::Normal(name) = component else {
            return Err(Error::Invalid("path contains unsafe components".into()));
        };
        current.push(name);
        match fs::create_dir(&current) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        handles.push(open_locked_directory(&current)?);
    }
    Ok(handles)
}

#[cfg(windows)]
fn securely_create_parent_directories(base: &Path, relative: &Path) -> Result<()> {
    let _locked = locked_windows_directory_chain(base, relative)?;
    Ok(())
}

#[cfg(windows)]
fn securely_create_new_file(base: &Path, relative: &Path) -> Result<fs::File> {
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let _locked = locked_windows_directory_chain(base, parent)?;
    let path = base.join(relative);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    configure_no_follow(&mut options);
    options.open(path).map_err(Into::into)
}

#[cfg(windows)]
fn securely_open_existing_file(base: &Path, relative: &Path) -> Result<fs::File> {
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let _locked = locked_windows_directory_chain(base, parent)?;
    let mut options = OpenOptions::new();
    options.read(true);
    configure_no_follow(&mut options);
    options.open(base.join(relative)).map_err(Into::into)
}

#[cfg(unix)]
fn securely_open_unix_parent(base: &Path, relative: &Path) -> Result<fs::File> {
    use rustix::fs::{Mode, OFlags, mkdirat, open, openat};

    let mut directory = fs::File::from(open(
        base,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )?);
    if is_reparse(&directory.metadata()?) || !directory.metadata()?.is_dir() {
        return Err(Error::Invalid("cache directory cannot be a symlink".into()));
    }
    for component in relative.components() {
        let std::path::Component::Normal(name) = component else {
            return Err(Error::Invalid("path contains unsafe components".into()));
        };
        if let Err(error) = mkdirat(&directory, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
            if error != rustix::io::Errno::EXIST {
                return Err(std::io::Error::from(error).into());
            }
        }
        directory = fs::File::from(openat(
            &directory,
            name,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::CLOEXEC
                | OFlags::NONBLOCK,
            Mode::empty(),
        )?);
    }
    Ok(directory)
}

#[cfg(unix)]
fn securely_create_parent_directories(base: &Path, relative: &Path) -> Result<()> {
    let _directory = securely_open_unix_parent(base, relative)?;
    Ok(())
}

#[cfg(unix)]
fn securely_create_new_file(base: &Path, relative: &Path) -> Result<fs::File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let directory = securely_open_unix_parent(base, parent)?;
    let leaf = relative
        .file_name()
        .ok_or_else(|| Error::Invalid("file path has no leaf".into()))?;
    Ok(fs::File::from(openat(
        &directory,
        leaf,
        OFlags::WRONLY
            | OFlags::CREATE
            | OFlags::EXCL
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK,
        Mode::RUSR | Mode::WUSR,
    )?))
}

#[cfg(unix)]
fn securely_open_existing_file(base: &Path, relative: &Path) -> Result<fs::File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let directory = securely_open_unix_parent(base, parent)?;
    let leaf = relative
        .file_name()
        .ok_or_else(|| Error::Invalid("file path has no leaf".into()))?;
    Ok(fs::File::from(openat(
        &directory,
        leaf,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )?))
}

#[cfg(not(any(unix, windows)))]
fn securely_create_parent_directories(base: &Path, relative: &Path) -> Result<()> {
    let mut current = base.to_path_buf();
    for component in relative.components() {
        current.push(component.as_os_str());
        match fs::create_dir(&current) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        let metadata = fs::symlink_metadata(&current)?;
        if is_reparse(&metadata) || !metadata.is_dir() {
            return Err(Error::Invalid("cache path is unsafe".into()));
        }
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn securely_create_new_file(base: &Path, relative: &Path) -> Result<fs::File> {
    securely_create_parent_directories(base, relative.parent().unwrap_or_else(|| Path::new("")))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    configure_no_follow(&mut options);
    options.open(base.join(relative)).map_err(Into::into)
}

#[cfg(not(any(unix, windows)))]
fn securely_open_existing_file(base: &Path, relative: &Path) -> Result<fs::File> {
    let mut options = OpenOptions::new();
    options.read(true);
    configure_no_follow(&mut options);
    options.open(base.join(relative)).map_err(Into::into)
}

fn validate_relative_path(relative: &Path) -> Result<()> {
    if relative
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
        && !relative.as_os_str().is_empty()
    {
        return Err(Error::Invalid("path contains unsafe components".into()));
    }
    Ok(())
}

fn canonical_directory(directory: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(directory)?;
    if is_reparse(&metadata) || !metadata.is_dir() {
        return Err(Error::Invalid(
            "cache directory cannot be a symlink or reparse point".into(),
        ));
    }
    Ok(fs::canonicalize(directory)?)
}

fn revalidate_contained_directory(base: &Path, relative: &Path) -> Result<()> {
    validate_relative_path(relative)?;
    reject_reparse_components(base, relative)?;
    let canonical_base = canonical_directory(base)?;
    let directory = base.join(relative);
    let metadata = fs::symlink_metadata(&directory)?;
    if is_reparse(&metadata) || !metadata.is_dir() {
        return Err(Error::Invalid(
            "cache directory cannot be a symlink or reparse point".into(),
        ));
    }
    let canonical = fs::canonicalize(&directory)?;
    if !canonical.starts_with(canonical_base) {
        return Err(Error::Invalid("cache directory escaped containment".into()));
    }
    Ok(())
}

pub(crate) fn revalidate_contained_path(base: &Path, relative: &Path) -> Result<()> {
    validate_relative_path(relative)?;
    reject_reparse_components(base, relative)?;
    let canonical_base = canonical_directory(base)?;
    let path = base.join(relative);
    let metadata = fs::symlink_metadata(&path)?;
    if is_reparse(&metadata) || !metadata.is_file() {
        return Err(Error::Invalid(
            "cache leaf is not a regular non-reparse file".into(),
        ));
    }
    let canonical = fs::canonicalize(&path)?;
    if !canonical.starts_with(canonical_base) {
        return Err(Error::Invalid("cache file escaped containment".into()));
    }
    Ok(())
}

fn open_contained_regular_file(
    base: &Path,
    relative: &Path,
    expected_size: u64,
) -> Result<fs::File> {
    validate_relative_path(relative)?;
    reject_reparse_components(base, relative)?;
    let file = securely_open_existing_file(base, relative)?;
    let metadata = file.metadata()?;
    if is_reparse(&metadata) || !metadata.is_file() || metadata.len() != expected_size {
        return Err(Error::Invalid("artifact changed after verification".into()));
    }
    Ok(file)
}

/// Hashes exactly the declared bytes from one no-follow, nonblocking handle. The `+1` cap detects
/// growth without allowing an unbounded regular file, FIFO, or device to consume memory or time.
fn verify_contained_artifact(
    base: &Path,
    artifact: &Artifact,
    cancelled: &impl Fn() -> bool,
) -> Result<bool> {
    let relative = Path::new(&artifact.path);
    let mut file = match open_contained_regular_file(base, relative, artifact.size) {
        Ok(file) => file,
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(Error::Invalid(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut read_total = 0_u64;
    let read_limit = artifact
        .size
        .checked_add(1)
        .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
    loop {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let remaining = read_limit.saturating_sub(read_total);
        if remaining == 0 {
            return Ok(false);
        }
        let limit = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| Error::Invalid("artifact length overflow".into()))?;
        let read = file.read(&mut buffer[..limit])?;
        if read == 0 {
            break;
        }
        read_total = read_total
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| Error::Invalid("artifact length overflow".into()))?,
            )
            .ok_or_else(|| Error::Invalid("artifact length overflow".into()))?;
        if read_total > artifact.size {
            return Ok(false);
        }
        digest.update(&buffer[..read]);
    }
    let final_metadata = file.metadata()?;
    if is_reparse(&final_metadata)
        || !final_metadata.is_file()
        || final_metadata.len() != artifact.size
        || read_total != artifact.size
        || format!("{:x}", digest.finalize()) != artifact.sha256
    {
        return Ok(false);
    }
    Ok(true)
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
        sync_directory(&directory)?;
    }
    sync_directory(&layout.root)?;
    if let Some(metadata) = symlink_metadata_if_exists(staging)? {
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(Error::Invalid("unsafe staging path".into()));
        }
        fs::remove_dir_all(staging)?;
        sync_directory(
            staging
                .parent()
                .ok_or_else(|| Error::Invalid("staging path has no parent".into()))?,
        )?;
    }
    fs::create_dir_all(staging)?;
    ensure_contained_directory(&layout.root, staging)?;
    sync_directory(staging)?;
    sync_directory(
        staging
            .parent()
            .ok_or_else(|| Error::Invalid("staging path has no parent".into()))?,
    )?;
    sync_directory(&layout.root)?;
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
        sync_regular_file,
        sync_directory,
    )
}

fn promote_with_operations(
    layout: &CacheLayout,
    manifest: &Manifest,
    staging: &Path,
    observe: &mut impl FnMut(PromotionPhase) -> Result<()>,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
    mut sync_file: impl FnMut(&Path) -> std::io::Result<()>,
    mut sync_dir: impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let final_path = layout.model_dir(manifest)?;
    reconcile_repair_inner(layout, manifest)?;
    // Promotion is the commit boundary. Persist every file and every directory entry in the
    // staged tree before making it reachable from `models`, then persist both rename parents.
    // This remains deliberately centralized here so future import/install paths cannot omit a
    // durability barrier by mistake.
    sync_staged_tree_with_operations(layout, staging, &mut sync_file, &mut sync_dir)?;
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

fn sync_staged_directory_with_operations(
    directory: &Path,
    sync_file: &mut impl FnMut(&Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let metadata = fs::symlink_metadata(directory)?;
    if is_reparse(&metadata) || !metadata.is_dir() {
        return Err(Error::Invalid(
            "staged tree contains an unsafe directory".into(),
        ));
    }
    let mut entries = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if is_reparse(&metadata) {
            return Err(Error::Invalid(
                "staged tree contains a symlink or reparse point".into(),
            ));
        }
        if metadata.is_dir() {
            sync_staged_directory_with_operations(&path, sync_file, sync_dir)?;
        } else if metadata.is_file() {
            sync_file(&path)?;
        } else {
            return Err(Error::Invalid(
                "staged tree contains a special filesystem entry".into(),
            ));
        }
    }
    sync_dir(directory)?;
    Ok(())
}

fn sync_staged_tree_with_operations(
    layout: &CacheLayout,
    staging: &Path,
    sync_file: &mut impl FnMut(&Path) -> std::io::Result<()>,
    sync_dir: &mut impl FnMut(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let staging_parent = staging
        .parent()
        .ok_or_else(|| Error::Invalid("staging path has no parent".into()))?;
    if staging_parent != layout.root.join("staging") {
        return Err(Error::Invalid("staging path escaped cache layout".into()));
    }
    ensure_contained_directory(&layout.root, staging)?;

    sync_staged_directory_with_operations(staging, sync_file, sync_dir)?;
    sync_dir(staging_parent)?;
    sync_dir(&layout.root)?;
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

fn sync_regular_file(path: &Path) -> std::io::Result<()> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    configure_no_follow(&mut options);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "durability target is not a regular file",
        ));
    }
    file.sync_all()
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
        let body = vec![7_u8; 256 * 1024];
        fs::write(temp.path().join("large.bin"), &body)?;
        let artifact = Artifact {
            path: "large.bin".into(),
            url: "https://example.invalid/large.bin".into(),
            sha256: format!("{:x}", Sha256::digest(&body)),
            size: u64::try_from(body.len())
                .map_err(|_| Error::Invalid("test body length overflow".into()))?,
        };
        let checks = Cell::new(0_u8);
        let result = verify_contained_artifact(temp.path(), &artifact, &|| {
            checks.set(checks.get().saturating_add(1));
            checks.get() > 2
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        Ok(())
    }

    #[test]
    fn status_hashing_is_capped_and_rejects_growth_on_the_open_handle() -> Result<()> {
        let temp = TempDir::new()?;
        let body = vec![3_u8; 128 * 1024];
        let path = temp.path().join("artifact.bin");
        fs::write(&path, &body)?;
        let artifact = Artifact {
            path: "artifact.bin".into(),
            url: "https://example.invalid/artifact.bin".into(),
            sha256: format!("{:x}", Sha256::digest(&body)),
            size: u64::try_from(body.len())
                .map_err(|_| Error::Invalid("test body length overflow".into()))?,
        };
        let mutated = Cell::new(false);
        let mutation_succeeded = Cell::new(false);
        let valid = verify_contained_artifact(temp.path(), &artifact, &|| {
            if !mutated.replace(true) {
                if let Ok(mut growing) = OpenOptions::new().append(true).open(&path) {
                    mutation_succeeded
                        .set(growing.write_all(b"x").is_ok() && growing.flush().is_ok());
                }
            }
            false
        })?;
        assert!(mutation_succeeded.get());
        assert!(!valid);
        Ok(())
    }

    #[test]
    fn artifact_bytes_caps_growth_to_declared_size_plus_one() -> Result<()> {
        let cache = TempDir::new()?;
        let body = vec![5_u8; 128 * 1024];
        let manifest = local_manifest(&body);
        let store = trusted_store(cache.path(), &manifest)?;
        let model = store.layout.model_dir(&manifest)?;
        write_valid_model(&model, &manifest, &body)?;
        let verified = store.verified_model(&manifest)?;
        let path = model.join("weights/model.bin");
        let result = verified.artifact_bytes_with_observer("weights/model.bin", || {
            let mut growing = OpenOptions::new().append(true).open(&path)?;
            growing.write_all(b"attacker-controlled-growth")?;
            growing.flush()?;
            Ok(())
        });
        assert!(matches!(result, Err(Error::Invalid(_))));
        Ok(())
    }

    #[test]
    fn create_new_never_replaces_an_existing_leaf() -> Result<()> {
        let temp = TempDir::new()?;
        fs::create_dir(temp.path().join("nested"))?;
        let leaf = temp.path().join("nested/artifact.bin");
        fs::write(&leaf, b"sentinel")?;
        let result = create_new_contained_file(temp.path(), Path::new("nested/artifact.bin"));
        assert!(result.is_err());
        assert_eq!(fs::read(leaf)?, b"sentinel");
        Ok(())
    }

    #[test]
    fn component_swap_between_preparation_and_create_is_rejected() -> Result<()> {
        let cache = TempDir::new()?;
        let outside = TempDir::new()?;
        let relative = Path::new("nested/artifact.bin");
        create_contained_parent_directories(cache.path(), relative)?;
        fs::remove_dir(cache.path().join("nested"))?;
        make_directory_link(outside.path(), &cache.path().join("nested"))?;

        let result = create_new_contained_file(cache.path(), relative);
        assert!(result.is_err());
        assert!(!outside.path().join("artifact.bin").exists());
        remove_directory_link(&cache.path().join("nested"))?;
        Ok(())
    }

    #[cfg(unix)]
    fn make_directory_link(target: &Path, link: &Path) -> Result<()> {
        std::os::unix::fs::symlink(target, link)?;
        Ok(())
    }

    #[cfg(unix)]
    fn remove_directory_link(link: &Path) -> Result<()> {
        fs::remove_file(link)?;
        Ok(())
    }

    #[cfg(windows)]
    fn make_directory_link(target: &Path, link: &Path) -> Result<()> {
        let output = std::process::Command::new("cmd")
            .args(["/d", "/c", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()?;
        if !output.status.success() {
            return Err(Error::Invalid(
                "test environment could not create a directory junction".into(),
            ));
        }
        Ok(())
    }

    #[cfg(windows)]
    fn remove_directory_link(link: &Path) -> Result<()> {
        fs::remove_dir(link)?;
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
            sync_regular_file,
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
            sync_regular_file,
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
    fn staged_tree_is_persisted_bottom_up_before_first_rename() -> Result<()> {
        let temp = TempDir::new()?;
        let layout = CacheLayout::new(temp.path())?;
        let manifest = manifest()?;
        let staging = layout.staging_dir(&manifest)?;
        prepare_staging(&layout, &staging)?;
        fs::create_dir_all(staging.join("nested/deep"))?;
        fs::write(staging.join(MANIFEST_FILE), b"manifest")?;
        fs::write(staging.join("nested/deep/artifact.bin"), b"artifact")?;

        let events = Rc::new(RefCell::new(Vec::new()));
        let rename_events = Rc::clone(&events);
        let file_events = Rc::clone(&events);
        let directory_events = Rc::clone(&events);
        let root_for_files = layout.root.clone();
        let root_for_directories = layout.root.clone();
        promote_with_operations(
            &layout,
            &manifest,
            &staging,
            &mut |_| Ok(()),
            move |from, to| {
                rename_events.borrow_mut().push("rename".to_owned());
                fs::rename(from, to)
            },
            move |path| {
                file_events.borrow_mut().push(format!(
                    "file:{}",
                    path.strip_prefix(&root_for_files)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/")
                ));
                Ok(())
            },
            move |path| {
                directory_events.borrow_mut().push(format!(
                    "dir:{}",
                    path.strip_prefix(&root_for_directories)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/")
                ));
                Ok(())
            },
        )?;

        let events = events.borrow();
        let rename_index = events
            .iter()
            .position(|event| event == "rename")
            .ok_or_else(|| Error::Invalid("missing rename event".into()))?;
        let before_rename = &events[..rename_index];
        let expected_manifest = format!(
            "file:staging/{}/manifest.json",
            staging
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
        );
        assert!(
            before_rename
                .iter()
                .any(|event| event == &expected_manifest)
        );
        let artifact_index = before_rename
            .iter()
            .position(|event| event.ends_with("/nested/deep/artifact.bin"))
            .ok_or_else(|| Error::Invalid("missing artifact barrier".into()))?;
        let deep_index = before_rename
            .iter()
            .position(|event| event.ends_with("/nested/deep"))
            .ok_or_else(|| Error::Invalid("missing deep directory barrier".into()))?;
        let nested_index = before_rename
            .iter()
            .position(|event| event.ends_with("/nested"))
            .ok_or_else(|| Error::Invalid("missing nested directory barrier".into()))?;
        assert!(artifact_index < deep_index && deep_index < nested_index);
        assert!(before_rename.iter().any(|event| event == "dir:staging"));
        assert!(before_rename.iter().any(|event| event == "dir:"));
        Ok(())
    }

    #[test]
    fn staged_tree_barrier_failure_prevents_promotion() -> Result<()> {
        for fail_file in [true, false] {
            let temp = TempDir::new()?;
            let layout = CacheLayout::new(temp.path())?;
            let manifest = manifest()?;
            let staging = layout.staging_dir(&manifest)?;
            prepare_staging(&layout, &staging)?;
            fs::create_dir_all(staging.join("nested"))?;
            fs::write(staging.join("nested/artifact.bin"), b"artifact")?;
            let renamed = Cell::new(false);

            let result = promote_with_operations(
                &layout,
                &manifest,
                &staging,
                &mut |_| Ok(()),
                |from, to| {
                    renamed.set(true);
                    fs::rename(from, to)
                },
                |_| {
                    if fail_file {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "injected file sync failure",
                        ))
                    } else {
                        Ok(())
                    }
                },
                |path| {
                    if !fail_file && path.ends_with("nested") {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "injected directory sync failure",
                        ))
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(matches!(result, Err(Error::Io(_))));
            assert!(!renamed.get());
            assert!(staging.is_dir());
            assert!(!layout.model_dir(&manifest)?.exists());
        }
        Ok(())
    }

    #[test]
    fn delete_reconciles_every_interrupted_repair_phase_before_removal() -> Result<()> {
        let body = b"delete repair transaction bytes";
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
            assert!(store.delete(&manifest)?);
            assert!(!final_path.exists());
            assert!(!store.layout.repair_marker(&manifest)?.exists());
            assert!(!store.layout.repair_backup(&manifest)?.exists());
            assert!(!store.layout.deletion_tombstone(&manifest)?.exists());
            assert_eq!(store.status(&manifest)?, ModelStatus::Missing);
        }
        Ok(())
    }

    #[test]
    fn every_interrupted_delete_phase_has_an_idempotent_safe_retry() -> Result<()> {
        let body = b"delete interruption bytes";
        for interrupted_after in [
            DeletionPhase::RepairReconciled,
            DeletionPhase::TombstoneDurable,
            DeletionPhase::TombstoneRemoved,
        ] {
            let temp = TempDir::new()?;
            let manifest = local_manifest(body);
            let store = trusted_store(temp.path(), &manifest)?;
            let final_path = store.layout.model_dir(&manifest)?;
            write_valid_model(&final_path, &manifest, body)?;

            let result = store.delete_locked_with_operations(
                &manifest,
                &mut |phase| {
                    if phase == interrupted_after {
                        Err(Error::Cancelled)
                    } else {
                        Ok(())
                    }
                },
                &mut |from, to| fs::rename(from, to),
                &mut |path| fs::remove_dir_all(path),
                &mut sync_directory,
            );
            assert!(matches!(result, Err(Error::Cancelled)));
            let retry_deleted = store.delete(&manifest)?;
            assert_eq!(
                retry_deleted,
                interrupted_after != DeletionPhase::TombstoneRemoved
            );
            assert!(!final_path.exists());
            assert!(!store.layout.repair_marker(&manifest)?.exists());
            assert!(!store.layout.repair_backup(&manifest)?.exists());
            assert!(!store.layout.deletion_tombstone(&manifest)?.exists());
        }
        Ok(())
    }

    #[test]
    fn failed_tombstone_cleanup_is_recoverable_without_resurrection() -> Result<()> {
        let body = b"delete cleanup failure bytes";
        let temp = TempDir::new()?;
        let manifest = local_manifest(body);
        let store = trusted_store(temp.path(), &manifest)?;
        let final_path = store.layout.model_dir(&manifest)?;
        let tombstone = store.layout.deletion_tombstone(&manifest)?;
        write_valid_model(&final_path, &manifest, body)?;

        let result = store.delete_locked_with_operations(
            &manifest,
            &mut |_| Ok(()),
            &mut |from, to| fs::rename(from, to),
            &mut |_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "injected tombstone cleanup failure",
                ))
            },
            &mut sync_directory,
        );
        assert!(matches!(result, Err(Error::Io(_))));
        assert!(!final_path.exists());
        assert!(tombstone.is_dir());

        assert!(store.delete(&manifest)?);
        assert!(!final_path.exists());
        assert!(!tombstone.exists());
        assert_eq!(store.status(&manifest)?, ModelStatus::Missing);
        Ok(())
    }

    #[test]
    fn delete_persists_models_parent_before_tombstone_cleanup() -> Result<()> {
        let body = b"delete ordering bytes";
        let temp = TempDir::new()?;
        let manifest = local_manifest(body);
        let store = trusted_store(temp.path(), &manifest)?;
        let final_path = store.layout.model_dir(&manifest)?;
        write_valid_model(&final_path, &manifest, body)?;
        let events = Rc::new(RefCell::new(Vec::new()));
        let rename_events = Rc::clone(&events);
        let remove_events = Rc::clone(&events);
        let sync_events = Rc::clone(&events);

        assert!(store.delete_locked_with_operations(
            &manifest,
            &mut |_| Ok(()),
            &mut move |from, to| {
                rename_events.borrow_mut().push("rename".to_owned());
                fs::rename(from, to)
            },
            &mut move |path| {
                remove_events.borrow_mut().push("remove".to_owned());
                fs::remove_dir_all(path)
            },
            &mut move |path| {
                sync_events.borrow_mut().push(format!(
                    "sync:{}",
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("root")
                ));
                Ok(())
            },
        )?);

        let events = events.borrow();
        let rename = events
            .iter()
            .position(|event| event == "rename")
            .unwrap_or(usize::MAX);
        let remove = events
            .iter()
            .position(|event| event == "remove")
            .unwrap_or(0);
        assert!(rename < remove);
        assert!(
            events[rename + 1..remove]
                .iter()
                .any(|event| event == "sync:models")
        );
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
        let failed_transaction_barrier = Cell::new(false);
        let transaction_root = layout.root.join("transactions");

        let result = promote_with_operations(
            &layout,
            &manifest,
            &staging,
            &mut |_| Ok(()),
            |from, to| fs::rename(from, to),
            sync_regular_file,
            |path| {
                if path == transaction_root && !failed_transaction_barrier.replace(true) {
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

    #[test]
    fn manifest_reads_are_bounded_and_discovery_work_is_capped() -> Result<()> {
        let oversized = TempDir::new()?;
        fs::write(
            oversized.path().join(MANIFEST_FILE),
            vec![b'x'; MAX_MANIFEST_BYTES + 1],
        )?;
        let Err(error) = read_contained_manifest(oversized.path()) else {
            return Err(Error::Invalid(
                "test fixture unexpectedly accepted oversized manifest".into(),
            ));
        };
        assert!(matches!(error, Error::Invalid(_)));
        let rendered = format!("{error} {error:?}");
        assert!(!rendered.contains(oversized.path().to_string_lossy().as_ref()));

        let cache = TempDir::new()?;
        let store = ModelStore::new(cache.path().join("cache"))?;
        let discovery = cache.path().join("discovery");
        fs::create_dir(&discovery)?;
        for index in 0..=MAX_DISCOVERY_ENTRIES {
            fs::write(discovery.join(format!("entry-{index}")), [])?;
        }
        let Err(error) = store.discover(&[DiscoveryRoot::new(&discovery)]) else {
            return Err(Error::Invalid(
                "test fixture unexpectedly exceeded discovery work bound".into(),
            ));
        };
        assert_eq!(
            error.to_string(),
            "invalid model data: model discovery entry limit exceeded"
        );
        assert!(
            !error
                .to_string()
                .contains(discovery.to_string_lossy().as_ref())
        );
        Ok(())
    }

    #[test]
    fn discovery_root_count_is_bounded_before_filesystem_work() -> Result<()> {
        let cache = TempDir::new()?;
        let store = ModelStore::new(cache.path().join("cache"))?;
        let roots = (0..=MAX_DISCOVERY_ROOTS)
            .map(|index| DiscoveryRoot::new(cache.path().join(format!("missing-{index}"))))
            .collect::<Vec<_>>();
        let Err(error) = store.discover(&roots) else {
            return Err(Error::Invalid(
                "test fixture unexpectedly exceeded discovery root bound".into(),
            ));
        };
        assert_eq!(
            error.to_string(),
            "invalid model data: model discovery root limit exceeded"
        );
        Ok(())
    }
}

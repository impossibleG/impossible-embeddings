use std::{
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::{Error, Manifest, Result};

const MANIFEST_FILE: &str = "manifest.json";

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
        if fs::symlink_metadata(root)?.file_type().is_symlink() {
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

/// Integrity and semantic readiness for an exact identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// No final installation exists.
    Missing,
    /// Files exist but do not match the manifest or cannot safely be inspected.
    Invalid,
    /// Every byte matches, but semantic verification is pending.
    IntegrityVerified,
    /// Integrity and semantic behavior are both verified.
    Loadable,
}

/// Safe local store operations independent from the network installer.
#[derive(Debug, Clone)]
pub struct ModelStore {
    layout: CacheLayout,
}

impl ModelStore {
    /// Opens an application cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache root is unsafe or unavailable.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            layout: CacheLayout::new(root)?,
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
        Ok(if stored.semantics_verified() {
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
        let _lock = acquire_store_lock(&self.layout, manifest)?;
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
        if stored.canonical_id != manifest.canonical_id || stored.revision != manifest.revision {
            return Err(Error::Invalid(
                "stored identity does not match deletion request".into(),
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
    if fs::symlink_metadata(parent)?.file_type().is_symlink()
        || fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink())
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
    if fs::symlink_metadata(directory)?.file_type().is_symlink() {
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
    if final_path.exists() {
        return Err(Error::Invalid(
            "exact model identity is already installed".into(),
        ));
    }
    fs::rename(staging, final_path)?;
    Ok(())
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

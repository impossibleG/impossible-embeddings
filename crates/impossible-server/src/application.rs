//! Transport-independent application, catalog, and shared state boundaries.

use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingOutput, EngineFailure, ErrorCode, PublicError,
};
use impossible_models::{
    InstallOptions, Installer, Manifest, ModelStatus, ModelStore, RuntimeMetadata,
    SemanticVerification, curated_manifests,
};
use impossible_server_core::{
    HealthRegistry, Readiness, ServerConfig, StartupPolicy,
    config::Limits,
    metrics::Metrics,
    security::{
        CredentialError, OriginPolicy, Secret, load_credential, verify_distinct_credentials,
    },
    telemetry::RequestId,
};
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, Notify};

#[cfg(test)]
use std::{future::Future, pin::Pin};

use crate::runtime::{ApplicationRuntime, BatchPolicy, LifecycleError};

const DEFAULT_BATCH_WAIT: Duration = Duration::from_millis(4);
const MAX_MODEL_NAME_BYTES: usize = 512;

/// Catalog construction failure with no manifest or filesystem detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogError {
    /// The catalog is empty or contains invalid model data.
    InvalidEntry,
    /// A canonical id or alias is ambiguous.
    NameCollision,
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidEntry => "the model catalog contains an invalid entry",
            Self::NameCollision => "the model catalog contains an ambiguous name",
        })
    }
}

impl std::error::Error for CatalogError {}

/// Immutable, collision-checked catalog of exact model revisions and public aliases.
#[derive(Clone)]
pub struct ModelCatalog {
    manifests: Arc<BTreeMap<String, Manifest>>,
    aliases: Arc<BTreeMap<String, String>>,
    aliases_by_model: Arc<BTreeMap<String, Vec<String>>>,
}

impl fmt::Debug for ModelCatalog {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelCatalog")
            .field("model_count", &self.manifests.len())
            .field("alias_count", &self.aliases.len())
            .finish_non_exhaustive()
    }
}

impl ModelCatalog {
    /// Construct the built-in immutable catalog.
    ///
    /// # Errors
    /// Returns a closed catalog error if committed data or aliases are inconsistent.
    pub fn curated() -> Result<Self, CatalogError> {
        Self::new(
            curated_manifests().map_err(|_| CatalogError::InvalidEntry)?,
            [
                ("bge-small-en", "BAAI/bge-small-en-v1.5"),
                ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
                ("multilingual-e5-small", "intfloat/multilingual-e5-small"),
                ("nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5"),
                ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
            ],
        )
    }

    /// Construct a catalog from validated manifests and exact, case-sensitive aliases.
    ///
    /// Alias names may not collide with any canonical id, including ids declared after the
    /// alias target. Every target must be an exact canonical id in the same catalog.
    ///
    /// # Errors
    /// Returns [`CatalogError`] for invalid entries, duplicate names, or unknown targets.
    pub fn new<M, A, S1, S2>(manifests: M, aliases: A) -> Result<Self, CatalogError>
    where
        M: IntoIterator<Item = Manifest>,
        A: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        let mut manifest_map = BTreeMap::new();
        for manifest in manifests {
            manifest
                .validate()
                .map_err(|_| CatalogError::InvalidEntry)?;
            let canonical = manifest.canonical_id.clone();
            if !valid_model_name(&canonical) || manifest_map.insert(canonical, manifest).is_some() {
                return Err(CatalogError::NameCollision);
            }
        }
        if manifest_map.is_empty() {
            return Err(CatalogError::InvalidEntry);
        }

        let mut alias_map = BTreeMap::new();
        let mut aliases_by_model = BTreeMap::<String, Vec<String>>::new();
        for (alias, canonical) in aliases {
            let alias = alias.into();
            let canonical = canonical.into();
            if !valid_model_name(&alias)
                || manifest_map.contains_key(&alias)
                || !manifest_map.contains_key(&canonical)
                || alias_map.insert(alias.clone(), canonical.clone()).is_some()
            {
                return Err(CatalogError::NameCollision);
            }
            aliases_by_model.entry(canonical).or_default().push(alias);
        }
        for aliases in aliases_by_model.values_mut() {
            aliases.sort();
        }
        Ok(Self {
            manifests: Arc::new(manifest_map),
            aliases: Arc::new(alias_map),
            aliases_by_model: Arc::new(aliases_by_model),
        })
    }

    /// Resolve an exact canonical id or alias.
    #[must_use]
    pub fn resolve(&self, requested: &str) -> Option<&Manifest> {
        if !valid_model_name(requested) {
            return None;
        }
        self.manifests.get(requested).or_else(|| {
            self.aliases
                .get(requested)
                .and_then(|canonical| self.manifests.get(canonical))
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&str, &Manifest)> {
        self.manifests
            .iter()
            .map(|(canonical, manifest)| (canonical.as_str(), manifest))
    }

    fn aliases_for(&self, canonical: &str) -> Vec<String> {
        self.aliases_by_model
            .get(canonical)
            .cloned()
            .unwrap_or_default()
    }
}

fn valid_model_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_MODEL_NAME_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

/// Public semantic validation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticStatus {
    /// Golden-vector behavior is trusted by the application catalog.
    Verified,
    /// Artifact identity is pinned but embedding semantics are not yet certified.
    Unverified,
}

/// Public runtime adapter availability declared by a model manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStatus {
    /// The catalog entry has no executable in-process contract.
    CatalogOnly,
    /// The catalog entry declares an ONNX Runtime contract.
    Onnx,
}

/// Public loaded state maintained by the application lifecycle boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStatus {
    /// No runtime is registered for this exact identity.
    Unloaded,
    /// The exact identity is loaded and accepting work.
    Loaded,
}

/// Safe model listing data. No local path, URL, diagnostic, or secret is retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    /// Registry-qualified model identity.
    pub canonical_id: String,
    /// Immutable upstream revision.
    pub revision: String,
    /// Stable aliases accepted by every transport.
    pub aliases: Vec<String>,
    /// Native vector width.
    pub native_dimensions: u32,
    /// Truncation widths declared by the immutable manifest; consult `semantic_status` before use.
    pub dimensions: Vec<u32>,
    /// Semantic certification status.
    pub semantic_status: SemanticStatus,
    /// Local artifact verification state.
    pub installation_status: ModelStatus,
    /// Executable adapter declaration.
    pub runtime_status: RuntimeStatus,
    /// Current application registration state.
    pub load_status: LoadStatus,
}

/// Typed transport-independent embedding command.
#[derive(Clone, Debug)]
pub struct EmbedCommand {
    /// Canonical model id or exact catalog alias.
    pub model: String,
    /// Ordered text inputs.
    pub input: Vec<String>,
    /// Explicit embedding semantics.
    pub options: EmbedOptions,
    /// Caller cancellation that the application never mutates.
    pub cancellation: CancellationToken,
    /// Optional caller timeout, capped by server policy.
    pub timeout: Option<Duration>,
}

/// Closed application failure category used by every transport adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplicationErrorKind {
    /// Caller input or selected lifecycle transition is invalid.
    InvalidRequest,
    /// The selected model cannot currently serve the operation.
    ModelUnavailable,
    /// Bounded capacity is exhausted.
    Overloaded,
    /// The operation was cancelled.
    Cancelled,
    /// The effective deadline expired.
    DeadlineExceeded,
    /// The process is draining and accepts no new work.
    ShuttingDown,
    /// An implementation or infrastructure operation failed.
    Internal,
}

/// Privacy-safe application failure. Private sources are discarded at this boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplicationError {
    kind: ApplicationErrorKind,
    public: PublicError,
}

impl ApplicationError {
    /// Stable category for transport status mapping.
    #[must_use]
    pub const fn kind(&self) -> ApplicationErrorKind {
        self.kind
    }

    /// The only error payload a transport may serialize.
    #[must_use]
    pub const fn public_error(&self) -> &PublicError {
        &self.public
    }

    fn new(kind: ApplicationErrorKind, code: ErrorCode) -> Self {
        Self {
            kind,
            public: PublicError::for_code(code),
        }
    }
}

impl fmt::Display for ApplicationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.public.message)
    }
}

impl std::error::Error for ApplicationError {}

impl From<EngineFailure> for ApplicationError {
    fn from(failure: EngineFailure) -> Self {
        let public = failure.public_error().clone();
        let kind = match public.code {
            ErrorCode::InvalidRequest => ApplicationErrorKind::InvalidRequest,
            ErrorCode::ModelUnavailable => ApplicationErrorKind::ModelUnavailable,
            ErrorCode::QueueFull => ApplicationErrorKind::Overloaded,
            ErrorCode::Cancelled => ApplicationErrorKind::Cancelled,
            ErrorCode::DeadlineExceeded => ApplicationErrorKind::DeadlineExceeded,
            _ => ApplicationErrorKind::Internal,
        };
        Self { kind, public }
    }
}

impl From<LifecycleError> for ApplicationError {
    fn from(error: LifecycleError) -> Self {
        match error {
            LifecycleError::ShuttingDown => Self::new(
                ApplicationErrorKind::ShuttingDown,
                ErrorCode::ModelUnavailable,
            ),
            LifecycleError::NotFound | LifecycleError::LoadFailed => Self::new(
                ApplicationErrorKind::ModelUnavailable,
                ErrorCode::ModelUnavailable,
            ),
            LifecycleError::InvalidPolicy
            | LifecycleError::AlreadyLoaded
            | LifecycleError::InUse => Self::new(
                ApplicationErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
            ),
            LifecycleError::InitializationFailed
            | LifecycleError::InstallFailed
            | LifecycleError::DeleteFailed => {
                Self::new(ApplicationErrorKind::Internal, ErrorCode::Internal)
            }
        }
    }
}

fn catalog_request_error() -> ApplicationError {
    ApplicationError::new(
        ApplicationErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
    )
}

fn infrastructure_error() -> ApplicationError {
    ApplicationError::new(ApplicationErrorKind::Internal, ErrorCode::Internal)
}

struct ApplicationInner {
    runtime: ApplicationRuntime,
    store: ModelStore,
    installer: Installer,
    catalog: ModelCatalog,
    loaded: RwLock<HashSet<String>>,
    preload_pending: RwLock<HashSet<String>>,
    shutting_down: AtomicBool,
    lifecycle: Mutex<()>,
    preload_models: Vec<String>,
    startup_policy: StartupPolicy,
    request_timeout: Duration,
    shutdown_timeout: Duration,
    #[cfg(test)]
    load_hook: RwLock<Option<TestLoadHook>>,
}

#[cfg(test)]
type TestLoadHook = Arc<
    dyn Fn(
            ApplicationRuntime,
            ModelStore,
            Manifest,
        ) -> Pin<Box<dyn Future<Output = Result<(), LifecycleError>> + Send>>
        + Send
        + Sync,
>;

struct StrictPreloadTransaction {
    application: EmbeddingApplication,
    gated: Vec<String>,
    newly_loaded: Vec<String>,
    committed: bool,
}

impl StrictPreloadTransaction {
    fn begin(application: &EmbeddingApplication) -> Result<Self, ApplicationError> {
        let gated = {
            let loaded = application
                .0
                .loaded
                .read()
                .map_err(|_| infrastructure_error())?;
            application
                .0
                .preload_models
                .iter()
                .filter(|canonical| !loaded.contains(*canonical))
                .cloned()
                .collect::<Vec<_>>()
        };
        application
            .0
            .preload_pending
            .write()
            .map_err(|_| infrastructure_error())?
            .extend(gated.iter().cloned());
        Ok(Self {
            application: application.clone(),
            gated,
            newly_loaded: Vec::new(),
            committed: false,
        })
    }

    fn record_loaded(&mut self, canonical: String) {
        self.newly_loaded.push(canonical);
    }

    fn commit(mut self) -> Result<(), ApplicationError> {
        let mut pending = self
            .application
            .0
            .preload_pending
            .write()
            .map_err(|_| infrastructure_error())?;
        for canonical in &self.gated {
            pending.remove(canonical);
        }
        drop(pending);
        self.committed = true;
        Ok(())
    }
}

impl Drop for StrictPreloadTransaction {
    fn drop(&mut self) {
        if !self.committed {
            for canonical in self.newly_loaded.iter().rev() {
                let _ = self.application.0.runtime.unload(canonical);
            }
            if let Ok(mut loaded) = self.application.0.loaded.write() {
                for canonical in &self.newly_loaded {
                    loaded.remove(canonical);
                }
            }
        }
        if let Ok(mut pending) = self.application.0.preload_pending.write() {
            for canonical in &self.gated {
                pending.remove(canonical);
            }
        }
    }
}

/// Shared transport-independent embedding and model lifecycle application.
#[derive(Clone)]
pub struct EmbeddingApplication(Arc<ApplicationInner>);

impl fmt::Debug for EmbeddingApplication {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = self.0.runtime.snapshot();
        formatter
            .debug_struct("EmbeddingApplication")
            .field("catalog_size", &self.0.catalog.manifests.len())
            .field("ready_models", &snapshot.ready_models)
            .field("draining", &snapshot.draining)
            .finish()
    }
}

impl EmbeddingApplication {
    /// Build the application from validated server configuration and the curated catalog.
    ///
    /// # Errors
    /// Returns a sanitized application failure if configuration, catalog, cache, installer, or
    /// runtime initialization fails.
    pub fn new(config: &ServerConfig) -> Result<Self, ApplicationError> {
        config.validate().map_err(|_| catalog_request_error())?;
        let catalog = ModelCatalog::curated().map_err(|_| infrastructure_error())?;
        Self::with_catalog(config, catalog)
    }

    fn with_catalog(
        config: &ServerConfig,
        catalog: ModelCatalog,
    ) -> Result<Self, ApplicationError> {
        let mut resolved_preloads = Vec::with_capacity(config.preload_models.len());
        let mut seen = HashSet::new();
        for requested in &config.preload_models {
            let manifest = catalog
                .resolve(requested)
                .ok_or_else(catalog_request_error)?;
            if !seen.insert(manifest.canonical_id.clone()) {
                return Err(catalog_request_error());
            }
            resolved_preloads.push(manifest.canonical_id.clone());
        }
        let store = ModelStore::new(&config.cache_directory).map_err(|_| infrastructure_error())?;
        let install_options = InstallOptions {
            offline: config.offline,
            ..InstallOptions::default()
        };
        let installer =
            Installer::new(store.clone(), install_options).map_err(|_| infrastructure_error())?;
        let runtime =
            ApplicationRuntime::new(BatchPolicy::from_limits(&config.limits, DEFAULT_BATCH_WAIT))?;
        Ok(Self(Arc::new(ApplicationInner {
            runtime,
            store,
            installer,
            catalog,
            loaded: RwLock::new(HashSet::new()),
            preload_pending: RwLock::new(HashSet::new()),
            shutting_down: AtomicBool::new(false),
            lifecycle: Mutex::new(()),
            preload_models: resolved_preloads,
            startup_policy: config.startup_policy,
            request_timeout: config.limits.request_timeout,
            shutdown_timeout: config.limits.shutdown_timeout,
            #[cfg(test)]
            load_hook: RwLock::new(None),
        })))
    }

    /// Embed ordered text input using an exact catalog identity or alias.
    ///
    /// # Errors
    /// Returns only the centralized privacy-safe application taxonomy.
    pub async fn embed(&self, command: EmbedCommand) -> Result<EmbeddingOutput, ApplicationError> {
        let manifest = self
            .0
            .catalog
            .resolve(&command.model)
            .ok_or_else(catalog_request_error)?;
        if self.0.shutting_down.load(Ordering::Acquire)
            || self
                .0
                .preload_pending
                .read()
                .map_err(|_| infrastructure_error())?
                .contains(&manifest.canonical_id)
        {
            return Err(ApplicationError::new(
                ApplicationErrorKind::ModelUnavailable,
                ErrorCode::ModelUnavailable,
            ));
        }
        let timeout = command
            .timeout
            .unwrap_or(self.0.request_timeout)
            .min(self.0.request_timeout);
        if timeout.is_zero() {
            return Err(catalog_request_error());
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(infrastructure_error)?;
        self.0
            .runtime
            .embed(
                &manifest.canonical_id,
                command.input,
                command.options,
                command.cancellation,
                Some(deadline),
            )
            .await
            .map_err(Into::into)
    }

    /// List the complete curated catalog with safe local status and no filesystem paths.
    ///
    /// # Errors
    /// Returns an internal public failure when local status cannot be inspected safely.
    pub fn list_models(&self) -> Result<Vec<ModelInfo>, ApplicationError> {
        self.0
            .catalog
            .iter()
            .map(|(_, manifest)| self.model_info(manifest))
            .collect()
    }

    /// Explicitly install one curated exact identity. Startup never calls this method.
    ///
    /// # Errors
    /// Returns a centralized application error for unknown names, shutdown, or installation.
    pub async fn install(&self, model: &str) -> Result<ModelInfo, ApplicationError> {
        let manifest = self.resolve_owned(model)?;
        self.ensure_running()?;
        let _lifecycle = self.0.lifecycle.lock().await;
        self.ensure_running()?;
        self.0
            .runtime
            .install_model(&self.0.installer, &manifest)
            .await?;
        self.model_info(&manifest)
    }

    /// Load one already-installed, semantically verified ONNX identity.
    ///
    /// # Errors
    /// Never downloads. Returns a centralized application failure if verification or load fails.
    pub async fn load(&self, model: &str) -> Result<ModelInfo, ApplicationError> {
        let manifest = self.resolve_owned(model)?;
        self.ensure_running()?;
        let _lifecycle = self.0.lifecycle.lock().await;
        self.ensure_running()?;
        self.load_locked(&manifest).await?;
        self.model_info(&manifest)
    }

    async fn load_locked(&self, manifest: &Manifest) -> Result<(), ApplicationError> {
        if self.is_loaded(&manifest.canonical_id) {
            return Err(LifecycleError::AlreadyLoaded.into());
        }
        #[cfg(test)]
        let hook = self
            .0
            .load_hook
            .read()
            .map_err(|_| infrastructure_error())?
            .clone();
        #[cfg(test)]
        if let Some(hook) = hook {
            hook(
                self.0.runtime.clone(),
                self.0.store.clone(),
                manifest.clone(),
            )
            .await?;
        } else {
            self.0
                .runtime
                .load_onnx(self.0.store.clone(), manifest.clone())
                .await?;
        }
        #[cfg(not(test))]
        self.0
            .runtime
            .load_onnx(self.0.store.clone(), manifest.clone())
            .await?;
        let mut loaded = self.0.loaded.write().map_err(|_| infrastructure_error())?;
        if self.0.shutting_down.load(Ordering::Acquire) {
            drop(loaded);
            let _ = self.0.runtime.unload(&manifest.canonical_id);
            return Err(LifecycleError::ShuttingDown.into());
        }
        loaded.insert(manifest.canonical_id.clone());
        Ok(())
    }

    /// Unload an idle exact identity.
    ///
    /// # Errors
    /// Returns a centralized application error when the model is unknown, busy, or draining.
    pub async fn unload(&self, model: &str) -> Result<ModelInfo, ApplicationError> {
        let manifest = self.resolve_owned(model)?;
        self.ensure_running()?;
        let _lifecycle = self.0.lifecycle.lock().await;
        self.ensure_running()?;
        self.0.runtime.unload(&manifest.canonical_id)?;
        self.0
            .loaded
            .write()
            .map_err(|_| infrastructure_error())?
            .remove(&manifest.canonical_id);
        self.model_info(&manifest)
    }

    /// Delete one unloaded exact identity from the local cache.
    ///
    /// # Errors
    /// Returns a centralized application error when the identity is loaded, busy, or deletion
    /// fails. No caller-supplied manifest or path crosses this boundary.
    pub async fn delete(&self, model: &str) -> Result<ModelInfo, ApplicationError> {
        let manifest = self.resolve_owned(model)?;
        self.ensure_running()?;
        let _lifecycle = self.0.lifecycle.lock().await;
        self.ensure_running()?;
        self.0
            .runtime
            .delete(&manifest.canonical_id, &self.0.store, &manifest)?;
        self.model_info(&manifest)
    }

    /// Load only explicitly configured preload identities already present in the local cache.
    ///
    /// Best-effort policy reports each stable outcome. Strict policy rolls back models loaded by
    /// this call before returning the first privacy-safe failure. This method never installs.
    ///
    /// # Errors
    /// Returns the first preload failure under strict policy.
    pub async fn preload(&self) -> Result<StartupReport, ApplicationError> {
        self.ensure_running()?;
        let _lifecycle = self.0.lifecycle.lock().await;
        self.ensure_running()?;
        let mut outcomes = Vec::with_capacity(self.0.preload_models.len());
        let mut strict_transaction = if self.0.startup_policy == StartupPolicy::Strict {
            Some(StrictPreloadTransaction::begin(self)?)
        } else {
            None
        };
        for canonical in &self.0.preload_models {
            let manifest = self
                .0
                .catalog
                .resolve(canonical)
                .ok_or_else(infrastructure_error)?
                .clone();
            if self.is_loaded(canonical) {
                outcomes.push(PreloadOutcome {
                    canonical_id: canonical.clone(),
                    loaded: true,
                    error: None,
                });
                continue;
            }
            match self.load_locked(&manifest).await {
                Ok(()) => {
                    if let Some(transaction) = strict_transaction.as_mut() {
                        transaction.record_loaded(canonical.clone());
                    }
                    outcomes.push(PreloadOutcome {
                        canonical_id: canonical.clone(),
                        loaded: true,
                        error: None,
                    });
                }
                Err(error) if self.0.startup_policy == StartupPolicy::BestEffort => {
                    outcomes.push(PreloadOutcome {
                        canonical_id: canonical.clone(),
                        loaded: false,
                        error: Some(error.kind()),
                    });
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }
        if let Some(transaction) = strict_transaction {
            transaction.commit()?;
        }
        Ok(StartupReport { outcomes })
    }

    /// Current liveness registry.
    #[must_use]
    pub fn health(&self) -> &HealthRegistry {
        self.0.runtime.health()
    }

    /// Current aggregate readiness.
    #[must_use]
    pub fn readiness(&self) -> Readiness {
        self.0.runtime.health().readiness()
    }

    /// Bounded-cardinality metrics registry.
    #[must_use]
    pub fn metrics(&self) -> &Metrics {
        self.0.runtime.metrics()
    }

    /// Begin bounded shutdown and stop accepting new lifecycle or inference operations.
    pub async fn shutdown(&self) -> bool {
        self.shutdown_with_timeout(self.0.shutdown_timeout).await
    }

    /// Begin shutdown with a caller bound that can only shorten configured policy.
    pub async fn shutdown_with_timeout(&self, timeout: Duration) -> bool {
        let bounded = timeout.min(self.0.shutdown_timeout);
        // Fence every application publication before polling runtime shutdown. In particular, do
        // not await the lifecycle mutex here: its owner may itself be the stalled operation that
        // the runtime deadline must cancel. `load_locked` checks this flag while holding the
        // loaded-registry write lock, so the final clear cannot race a late `Loaded` insertion.
        self.0.shutting_down.store(true, Ordering::Release);
        let result = self.0.runtime.shutdown(bounded).await;
        if let Ok(mut loaded) = self.0.loaded.write() {
            loaded.clear();
        }
        if let Ok(mut pending) = self.0.preload_pending.write() {
            pending.clear();
        }
        result
    }

    fn resolve_owned(&self, requested: &str) -> Result<Manifest, ApplicationError> {
        self.0
            .catalog
            .resolve(requested)
            .cloned()
            .ok_or_else(catalog_request_error)
    }

    fn is_loaded(&self, canonical: &str) -> bool {
        !self.0.shutting_down.load(Ordering::Acquire)
            && !self
                .0
                .preload_pending
                .read()
                .is_ok_and(|pending| pending.contains(canonical))
            && self
                .0
                .loaded
                .read()
                .is_ok_and(|loaded| loaded.contains(canonical))
    }

    fn ensure_running(&self) -> Result<(), ApplicationError> {
        if self.0.shutting_down.load(Ordering::Acquire) {
            Err(LifecycleError::ShuttingDown.into())
        } else {
            Ok(())
        }
    }

    fn model_info(&self, manifest: &Manifest) -> Result<ModelInfo, ApplicationError> {
        let installation_status = self
            .0
            .store
            .status(manifest)
            .map_err(|_| infrastructure_error())?;
        Ok(ModelInfo {
            canonical_id: manifest.canonical_id.clone(),
            revision: manifest.revision.clone(),
            aliases: self.0.catalog.aliases_for(&manifest.canonical_id),
            native_dimensions: manifest.dimensions.native,
            dimensions: manifest.dimensions.matryoshka.clone(),
            semantic_status: match manifest.semantic_verification {
                SemanticVerification::Verified { .. } => SemanticStatus::Verified,
                SemanticVerification::Unverified { .. } => SemanticStatus::Unverified,
            },
            installation_status,
            runtime_status: match manifest.runtime {
                RuntimeMetadata::CatalogOnly => RuntimeStatus::CatalogOnly,
                RuntimeMetadata::Onnx { .. } => RuntimeStatus::Onnx,
            },
            load_status: if self.is_loaded(&manifest.canonical_id) {
                LoadStatus::Loaded
            } else {
                LoadStatus::Unloaded
            },
        })
    }
}

/// One explicit preload outcome without local diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreloadOutcome {
    /// Exact canonical identity attempted.
    pub canonical_id: String,
    /// Whether the identity is now loaded.
    pub loaded: bool,
    /// Stable failure category when loading failed under best-effort policy.
    pub error: Option<ApplicationErrorKind>,
}

/// Results for every explicitly configured preload identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartupReport {
    /// Outcomes in configuration order.
    pub outcomes: Vec<PreloadOutcome>,
}

/// Monotonic process-local request id source.
#[derive(Debug, Default)]
pub struct RequestIdSource(AtomicU64);

impl RequestIdSource {
    /// Return the next non-zero id. Wraparound skips zero.
    #[must_use]
    pub fn next(&self) -> RequestId {
        loop {
            let value = self.0.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
            if value != 0 {
                return RequestId(value);
            }
        }
    }
}

/// Cloneable one-way trigger used by signal and transport hosts to request process shutdown.
#[derive(Clone, Default)]
pub struct ShutdownTrigger(Arc<ShutdownSignal>);

#[derive(Default)]
struct ShutdownSignal {
    triggered: AtomicBool,
    notify: Notify,
}

impl fmt::Debug for ShutdownTrigger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShutdownTrigger")
            .field("triggered", &self.is_triggered())
            .finish()
    }
}

impl ShutdownTrigger {
    /// Publish shutdown exactly once. Repeated calls are harmless.
    pub fn trigger(&self) {
        if !self.0.triggered.swap(true, Ordering::AcqRel) {
            self.0.notify.notify_waiters();
        }
    }

    /// Whether shutdown has been requested.
    #[must_use]
    pub fn is_triggered(&self) -> bool {
        self.0.triggered.load(Ordering::Acquire)
    }

    /// Wait without a missed-wakeup window for shutdown to be requested.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_triggered() {
                return;
            }
            notified.await;
        }
    }
}

/// Shared state cloned into HTTP, gRPC, and MCP transport adapters.
#[derive(Clone)]
pub struct AppState {
    application: EmbeddingApplication,
    public_credential: Option<Arc<Secret>>,
    admin_credential: Option<Arc<Secret>>,
    origins: OriginPolicy,
    request_ids: Arc<RequestIdSource>,
    version: &'static str,
    shutdown: ShutdownTrigger,
    limits: Limits,
    admin_api_enabled: bool,
}

impl fmt::Debug for AppState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppState")
            .field("application", &self.application)
            .field("has_public_credential", &self.public_credential.is_some())
            .field("has_admin_credential", &self.admin_credential.is_some())
            .field("version", &self.version)
            .field("shutdown", &self.shutdown)
            .field("limits", &self.limits)
            .field("admin_api_enabled", &self.admin_api_enabled)
            .finish_non_exhaustive()
    }
}

impl AppState {
    /// Load credentials once and construct immutable shared process state.
    ///
    /// # Errors
    /// Returns a sanitized failure for invalid config, credentials, catalog, or application setup.
    pub fn new(config: &ServerConfig) -> Result<Self, AppStateError> {
        config
            .validate()
            .map_err(|_| AppStateError::Configuration)?;
        let public_credential = config
            .auth
            .as_ref()
            .map(load_credential)
            .transpose()
            .map_err(AppStateError::Credential)?
            .map(Arc::new);
        let admin_credential = config
            .admin_auth
            .as_ref()
            .map(load_credential)
            .transpose()
            .map_err(AppStateError::Credential)?
            .map(Arc::new);
        if let (Some(public), Some(admin)) = (&public_credential, &admin_credential) {
            verify_distinct_credentials(public, admin).map_err(AppStateError::Credential)?;
        }
        Ok(Self {
            application: EmbeddingApplication::new(config).map_err(AppStateError::Application)?,
            public_credential,
            admin_credential,
            origins: OriginPolicy::new(config.allowed_origins.clone()),
            request_ids: Arc::new(RequestIdSource::default()),
            version: env!("CARGO_PKG_VERSION"),
            shutdown: ShutdownTrigger::default(),
            limits: config.limits.clone(),
            admin_api_enabled: config.admin_api_enabled,
        })
    }

    /// Shared application boundary.
    #[must_use]
    pub const fn application(&self) -> &EmbeddingApplication {
        &self.application
    }

    /// Exact browser-origin policy.
    #[must_use]
    pub const fn origins(&self) -> &OriginPolicy {
        &self.origins
    }

    /// Generate a process-local request id.
    #[must_use]
    pub fn next_request_id(&self) -> RequestId {
        self.request_ids.next()
    }

    /// Server package version.
    #[must_use]
    pub const fn version(&self) -> &'static str {
        self.version
    }

    /// One-way process shutdown request.
    #[must_use]
    pub const fn shutdown_trigger(&self) -> &ShutdownTrigger {
        &self.shutdown
    }

    /// Transport body, concurrency, and deadline limits.
    #[must_use]
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Whether transport hosts should expose administrative lifecycle routes.
    #[must_use]
    pub const fn admin_api_enabled(&self) -> bool {
        self.admin_api_enabled
    }

    /// Whether public endpoints require bearer authentication.
    #[must_use]
    pub const fn public_auth_required(&self) -> bool {
        self.public_credential.is_some()
    }

    /// Whether administrative endpoints require bearer authentication.
    #[must_use]
    pub const fn admin_auth_required(&self) -> bool {
        self.admin_credential.is_some()
    }

    /// Whether the metrics endpoint requires bearer authentication.
    ///
    /// Metrics uses the dedicated administrative credential when one is configured, otherwise it
    /// falls back to the public credential. No credential is accepted only when the central
    /// configuration validator has admitted a loopback-only listener or the operator's explicit
    /// insecure-remote acknowledgement.
    #[must_use]
    pub fn metrics_auth_required(&self) -> bool {
        self.metrics_credential().is_some()
    }

    /// Verify a public bearer token candidate without exposing credential bytes.
    #[must_use]
    pub fn verify_public_token(&self, candidate: &[u8]) -> bool {
        self.public_credential
            .as_ref()
            .is_none_or(|secret| secret.verify(candidate))
    }

    /// Verify an administrative bearer token candidate. A missing credential is accepted only for
    /// the safe loopback configuration validated before this state is created.
    #[must_use]
    pub fn verify_admin_token(&self, candidate: &[u8]) -> bool {
        self.admin_credential
            .as_ref()
            .is_none_or(|secret| secret.verify(candidate))
    }

    /// Verify a metrics bearer token using the centralized admin-then-public fallback policy.
    #[must_use]
    pub fn verify_metrics_token(&self, candidate: &[u8]) -> bool {
        self.metrics_credential()
            .is_none_or(|secret| secret.verify(candidate))
    }

    fn metrics_credential(&self) -> Option<&Secret> {
        self.admin_credential
            .as_deref()
            .or(self.public_credential.as_deref())
    }
}

/// Sanitized shared-state construction failure.
#[derive(Debug)]
pub enum AppStateError {
    /// Typed configuration failed validation.
    Configuration,
    /// A configured credential could not be loaded or credentials were equal.
    Credential(CredentialError),
    /// Application construction failed.
    Application(ApplicationError),
}

impl fmt::Display for AppStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Configuration => "server configuration is invalid",
            Self::Credential(_) => "server credential configuration is invalid",
            Self::Application(_) => "embedding application initialization failed",
        })
    }
}

impl std::error::Error for AppStateError {}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runtime::{RuntimeEngine, RuntimeModelContract};
    use impossible_embedding_core::{
        EmbeddingBatch, EmbeddingUsage, ExecutionControl, RequestedModel, ResolvedModelIdentity,
    };
    use impossible_server_core::config::CredentialSource;
    use std::{
        fs,
        sync::{Mutex as StdMutex, atomic::AtomicUsize, mpsc},
        thread,
    };

    static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    fn fixture_path(label: &str) -> std::path::PathBuf {
        let id = FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "impossible-application-{label}-{}-{id}",
            std::process::id()
        ))
    }

    fn config(label: &str) -> ServerConfig {
        ServerConfig {
            cache_directory: fixture_path(label),
            ..ServerConfig::default()
        }
    }

    fn manifests() -> Vec<Manifest> {
        curated_manifests().expect("committed catalog")
    }

    fn set_load_hook(app: &EmbeddingApplication, hook: TestLoadHook) {
        *app.0.load_hook.write().expect("load hook") = Some(hook);
    }

    fn embed_command(model: impl Into<String>) -> EmbedCommand {
        EmbedCommand {
            model: model.into(),
            input: vec!["hello".to_owned()],
            options: EmbedOptions::default(),
            cancellation: CancellationToken::default(),
            timeout: None,
        }
    }

    #[derive(Debug)]
    struct MockEngine {
        canonical: String,
        delay: Duration,
    }

    impl MockEngine {
        fn identity(&self) -> ResolvedModelIdentity {
            ResolvedModelIdentity::new(
                self.canonical.clone(),
                "fixture-revision",
                "fixture-runtime",
                "sha256:fixture-artifacts",
                "sha256:fixture-semantics",
            )
            .expect("complete fixture identity")
        }
    }

    impl RuntimeEngine for MockEngine {
        fn model_contract(
            &self,
            _model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            RuntimeModelContract::new(self.identity(), 2)
        }

        fn embed(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            control.ensure_active()?;
            if !self.delay.is_zero() {
                thread::sleep(self.delay);
            }
            control.ensure_active()?;
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0, 2.0]).collect(),
                model: self.identity(),
                usage: EmbeddingUsage::new(
                    batch
                        .inputs()
                        .iter()
                        .map(|input| u64::try_from(input.len() + 2).unwrap_or(u64::MAX))
                        .collect(),
                )?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    #[test]
    fn curated_aliases_resolve_exactly_and_listing_is_deterministic() {
        let catalog = ModelCatalog::curated().expect("curated catalog");
        assert_eq!(
            catalog.resolve("bge-small-en").expect("alias").canonical_id,
            "BAAI/bge-small-en-v1.5"
        );
        assert!(catalog.resolve("BGE-SMALL-EN").is_none());
        assert!(catalog.resolve(" bge-small-en").is_none());
        assert!(catalog.resolve("bge-small-en\n").is_none());
        let canonical = catalog
            .iter()
            .map(|(canonical, _)| canonical)
            .collect::<Vec<_>>();
        assert_eq!(
            canonical,
            [
                "BAAI/bge-small-en-v1.5",
                "intfloat/multilingual-e5-small",
                "nomic-ai/nomic-embed-text-v1.5"
            ]
        );
    }

    #[test]
    fn catalog_rejects_every_ambiguous_or_invalid_alias_shape() {
        let entries = manifests();
        let first = entries[0].canonical_id.clone();
        let second = entries[1].canonical_id.clone();
        assert!(matches!(
            ModelCatalog::new(entries.clone(), [(first.as_str(), second.as_str())]),
            Err(CatalogError::NameCollision)
        ));
        assert!(matches!(
            ModelCatalog::new(entries.clone(), [("short", "missing/model")]),
            Err(CatalogError::NameCollision)
        ));
        assert!(matches!(
            ModelCatalog::new(
                entries.clone(),
                [
                    ("duplicate", first.as_str()),
                    ("duplicate", second.as_str())
                ]
            ),
            Err(CatalogError::NameCollision)
        ));
        assert!(matches!(
            ModelCatalog::new(entries.clone(), [("bad\nname", first.as_str())]),
            Err(CatalogError::NameCollision)
        ));
        let mut duplicated = entries;
        duplicated.push(duplicated[0].clone());
        assert!(matches!(
            ModelCatalog::new(duplicated, std::iter::empty::<(&str, &str)>()),
            Err(CatalogError::NameCollision)
        ));
    }

    #[test]
    fn model_listing_contains_status_but_never_cache_paths_or_urls() {
        let config = config("listing-private-sentinel");
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let models = app.list_models().expect("listing");
        assert_eq!(models.len(), 3);
        assert!(
            models
                .iter()
                .all(|model| model.installation_status == ModelStatus::Missing)
        );
        let rendered = format!("{app:?} {models:?}");
        assert!(!rendered.contains("listing-private-sentinel"));
        assert!(!rendered.contains("huggingface.co"));
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn aliases_share_one_lifecycle_and_embedding_identity() {
        let config = config("alias-lifecycle");
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let canonical = "BAAI/bge-small-en-v1.5".to_owned();
        app.0
            .runtime
            .register_engine(
                canonical.clone(),
                Arc::new(MockEngine {
                    canonical: canonical.clone(),
                    delay: Duration::ZERO,
                }),
            )
            .expect("register fixture");
        app.0
            .loaded
            .write()
            .expect("loaded state")
            .insert(canonical.clone());

        let output = app
            .embed(EmbedCommand {
                model: "bge-small-en".to_owned(),
                input: vec!["hello".to_owned(), "world".to_owned()],
                options: EmbedOptions::default(),
                cancellation: CancellationToken::default(),
                timeout: None,
            })
            .await
            .expect("embed by alias");
        assert_eq!(output.model.canonical_id, canonical);
        assert_eq!(output.vectors.len(), 2);
        assert_eq!(
            app.list_models()
                .expect("listing")
                .into_iter()
                .find(|model| model.canonical_id == canonical)
                .expect("catalog entry")
                .load_status,
            LoadStatus::Loaded
        );

        let unloaded = app.unload("bge-small-en-v1.5").await.expect("unload");
        assert_eq!(unloaded.load_status, LoadStatus::Unloaded);
        assert_eq!(
            app.embed(EmbedCommand {
                model: canonical,
                input: vec!["hello".to_owned()],
                options: EmbedOptions::default(),
                cancellation: CancellationToken::default(),
                timeout: None,
            })
            .await
            .expect_err("unloaded model"),
            ApplicationError::new(
                ApplicationErrorKind::ModelUnavailable,
                ErrorCode::ModelUnavailable
            )
        );
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn caller_timeout_is_capped_by_server_deadline() {
        let mut config = config("deadline-cap");
        config.limits.request_timeout = Duration::from_millis(10);
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let canonical = "BAAI/bge-small-en-v1.5".to_owned();
        app.0
            .runtime
            .register_engine(
                canonical.clone(),
                Arc::new(MockEngine {
                    canonical: canonical.clone(),
                    delay: Duration::from_millis(40),
                }),
            )
            .expect("register fixture");
        app.0
            .loaded
            .write()
            .expect("loaded state")
            .insert(canonical.clone());
        let error = app
            .embed(EmbedCommand {
                model: canonical,
                input: vec!["slow".to_owned()],
                options: EmbedOptions::default(),
                cancellation: CancellationToken::default(),
                timeout: Some(Duration::from_secs(60)),
            })
            .await
            .expect_err("server deadline must win");
        assert_eq!(error.kind(), ApplicationErrorKind::DeadlineExceeded);
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn preload_is_explicit_offline_and_policy_controlled() {
        let mut best_effort = config("preload-best-effort");
        best_effort.offline = true;
        best_effort.preload_models = vec!["bge-small-en".to_owned()];
        let best_cache = best_effort.cache_directory.clone();
        let app = EmbeddingApplication::new(&best_effort).expect("application");
        let report = app.preload().await.expect("best effort report");
        assert_eq!(report.outcomes.len(), 1);
        assert!(!report.outcomes[0].loaded);
        assert_eq!(
            report.outcomes[0].error,
            Some(ApplicationErrorKind::ModelUnavailable)
        );
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(best_cache);

        let mut strict = config("preload-strict");
        strict.offline = true;
        strict.startup_policy = StartupPolicy::Strict;
        strict.preload_models = vec!["bge-small-en".to_owned()];
        let strict_cache = strict.cache_directory.clone();
        let app = EmbeddingApplication::new(&strict).expect("application");
        assert_eq!(
            app.preload().await.expect_err("strict failure").kind(),
            ApplicationErrorKind::ModelUnavailable
        );
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(strict_cache);
    }

    #[tokio::test]
    async fn strict_preload_hides_partial_success_and_rolls_it_back() {
        let first = "BAAI/bge-small-en-v1.5".to_owned();
        let second = "intfloat/multilingual-e5-small".to_owned();
        let mut config = config("strict-preload-transaction");
        config.offline = true;
        config.startup_policy = StartupPolicy::Strict;
        config.preload_models = vec![first.clone(), second.clone()];
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let second_started = Arc::new(Notify::new());
        let second_release = Arc::new(Notify::new());
        let hook_first = first.clone();
        let hook_second = second.clone();
        let started = Arc::clone(&second_started);
        let release = Arc::clone(&second_release);
        set_load_hook(
            &app,
            Arc::new(move |runtime, _store, manifest| {
                let first = hook_first.clone();
                let second = hook_second.clone();
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                Box::pin(async move {
                    if manifest.canonical_id == first {
                        let canonical = manifest.canonical_id;
                        runtime.register_engine(
                            canonical.clone(),
                            Arc::new(MockEngine {
                                canonical,
                                delay: Duration::ZERO,
                            }),
                        )
                    } else if manifest.canonical_id == second {
                        started.notify_one();
                        release.notified().await;
                        Err(LifecycleError::LoadFailed)
                    } else {
                        Err(LifecycleError::LoadFailed)
                    }
                })
            }),
        );

        let preload = {
            let app = app.clone();
            tokio::spawn(async move { app.preload().await })
        };
        second_started.notified().await;
        assert_eq!(
            app.embed(embed_command(first.clone()))
                .await
                .expect_err("partial strict preload must remain private")
                .kind(),
            ApplicationErrorKind::ModelUnavailable
        );
        assert_eq!(
            app.list_models()
                .expect("model listing")
                .into_iter()
                .find(|model| model.canonical_id == first)
                .expect("first model")
                .load_status,
            LoadStatus::Unloaded
        );
        second_release.notify_one();
        assert_eq!(
            preload
                .await
                .expect("preload task")
                .expect_err("second model failure")
                .kind(),
            ApplicationErrorKind::ModelUnavailable
        );
        assert_eq!(app.0.runtime.snapshot().registered_models, 0);
        assert_eq!(
            app.embed(embed_command(first.clone()))
                .await
                .expect_err("rolled-back model")
                .kind(),
            ApplicationErrorKind::ModelUnavailable
        );
        assert!(
            app.list_models()
                .expect("model listing")
                .into_iter()
                .all(|model| model.installation_status == ModelStatus::Missing)
        );
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn best_effort_preload_keeps_success_visible_when_a_later_model_fails() {
        let first = "BAAI/bge-small-en-v1.5".to_owned();
        let second = "intfloat/multilingual-e5-small".to_owned();
        let mut config = config("best-effort-preload-visibility");
        config.offline = true;
        config.startup_policy = StartupPolicy::BestEffort;
        config.preload_models = vec![first.clone(), second.clone()];
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let second_started = Arc::new(Notify::new());
        let second_release = Arc::new(Notify::new());
        let hook_first = first.clone();
        let hook_second = second.clone();
        let started = Arc::clone(&second_started);
        let release = Arc::clone(&second_release);
        set_load_hook(
            &app,
            Arc::new(move |runtime, _store, manifest| {
                let first = hook_first.clone();
                let second = hook_second.clone();
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                Box::pin(async move {
                    if manifest.canonical_id == first {
                        let canonical = manifest.canonical_id;
                        runtime.register_engine(
                            canonical.clone(),
                            Arc::new(MockEngine {
                                canonical,
                                delay: Duration::ZERO,
                            }),
                        )
                    } else if manifest.canonical_id == second {
                        started.notify_one();
                        release.notified().await;
                        Err(LifecycleError::LoadFailed)
                    } else {
                        Err(LifecycleError::LoadFailed)
                    }
                })
            }),
        );

        let preload = {
            let app = app.clone();
            tokio::spawn(async move { app.preload().await })
        };
        second_started.notified().await;
        assert_eq!(
            app.embed(embed_command(first.clone()))
                .await
                .expect("best-effort success remains visible")
                .vectors
                .len(),
            1
        );
        second_release.notify_one();
        let report = preload.await.expect("preload task").expect("report");
        assert_eq!(report.outcomes.len(), 2);
        assert!(report.outcomes[0].loaded);
        assert!(!report.outcomes[1].loaded);
        assert_eq!(
            report.outcomes[1].error,
            Some(ApplicationErrorKind::ModelUnavailable)
        );
        assert!(app.shutdown().await);
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn shutdown_bound_includes_a_stalled_install_holding_lifecycle() {
        let mut config = config("bounded-stalled-install");
        config.limits.shutdown_timeout = Duration::from_millis(25);
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let started = Arc::new(Notify::new());
        let install = {
            let app = app.clone();
            let started = Arc::clone(&started);
            tokio::spawn(async move {
                let _lifecycle = app.0.lifecycle.lock().await;
                app.0
                    .runtime
                    .install_with(move |_, _| async move {
                        started.notify_one();
                        std::future::pending::<Result<(), LifecycleError>>().await
                    })
                    .await
            })
        };
        started.notified().await;
        let began = Instant::now();
        let drained = tokio::time::timeout(
            Duration::from_millis(250),
            app.shutdown_with_timeout(Duration::from_millis(25)),
        )
        .await
        .expect("application shutdown must be bounded");
        assert!(!drained);
        assert!(began.elapsed() < Duration::from_millis(200));
        assert_eq!(
            tokio::time::timeout(Duration::from_millis(100), install)
                .await
                .expect("stalled install caller must be released")
                .expect("install task"),
            Err(LifecycleError::ShuttingDown)
        );
        assert!(
            app.list_models()
                .expect("model listing")
                .iter()
                .all(|model| model.load_status == LoadStatus::Unloaded)
        );
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[tokio::test]
    async fn shutdown_bound_fences_a_blocked_load_and_late_publication() {
        let canonical = "BAAI/bge-small-en-v1.5".to_owned();
        let mut config = config("bounded-blocked-load");
        config.limits.shutdown_timeout = Duration::from_millis(25);
        let cache = config.cache_directory.clone();
        let app = EmbeddingApplication::new(&config).expect("application");
        let (entered_send, entered_receive) = tokio::sync::oneshot::channel();
        let entered_send = Arc::new(StdMutex::new(Some(entered_send)));
        let (release_send, release_receive) = mpsc::sync_channel(0);
        let release_receive = Arc::new(StdMutex::new(release_receive));
        let hook_entered = Arc::clone(&entered_send);
        let hook_release = Arc::clone(&release_receive);
        set_load_hook(
            &app,
            Arc::new(move |runtime, _store, manifest| {
                let entered = Arc::clone(&hook_entered);
                let release = Arc::clone(&hook_release);
                Box::pin(async move {
                    let canonical = manifest.canonical_id;
                    let engine_canonical = canonical.clone();
                    runtime
                        .load_with_test(canonical, move || {
                            if let Some(send) = entered.lock().expect("entered lock").take() {
                                let _ = send.send(());
                            }
                            let _ = release.lock().expect("release lock").recv();
                            Ok(Arc::new(MockEngine {
                                canonical: engine_canonical,
                                delay: Duration::ZERO,
                            }) as Arc<dyn RuntimeEngine>)
                        })
                        .await
                })
            }),
        );
        let load = {
            let app = app.clone();
            let canonical = canonical.clone();
            tokio::spawn(async move { app.load(&canonical).await })
        };
        entered_receive.await.expect("blocking load entered");
        let began = Instant::now();
        let drained = tokio::time::timeout(
            Duration::from_millis(250),
            app.shutdown_with_timeout(Duration::from_millis(25)),
        )
        .await
        .expect("application shutdown must be bounded");
        assert!(!drained);
        assert!(began.elapsed() < Duration::from_millis(200));
        assert_eq!(
            app.list_models()
                .expect("model listing")
                .into_iter()
                .find(|model| model.canonical_id == canonical)
                .expect("blocked model")
                .load_status,
            LoadStatus::Unloaded
        );
        release_send.send(()).expect("release native load");
        assert_eq!(
            tokio::time::timeout(Duration::from_millis(100), load)
                .await
                .expect("blocked load caller must be released")
                .expect("load task")
                .expect_err("load cannot publish after shutdown")
                .kind(),
            ApplicationErrorKind::ShuttingDown
        );
        assert_eq!(app.0.runtime.snapshot().registered_models, 0);
        drop(app);
        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn duplicate_preload_aliases_for_one_identity_fail_before_runtime_setup() {
        let mut config = config("duplicate-preload");
        config.preload_models = vec![
            "bge-small-en".to_owned(),
            "BAAI/bge-small-en-v1.5".to_owned(),
        ];
        let error = EmbeddingApplication::new(&config).expect_err("ambiguous preload");
        assert_eq!(error.kind(), ApplicationErrorKind::InvalidRequest);
        assert!(!config.cache_directory.exists());
    }

    #[test]
    fn app_state_debug_and_verification_never_expose_credentials() {
        let config = config("state-debug");
        let cache = config.cache_directory.clone();
        let state = AppState {
            application: EmbeddingApplication::new(&config).expect("application"),
            public_credential: Some(Arc::new(
                Secret::new(b"sentinel-public-secret".to_vec()).expect("secret"),
            )),
            admin_credential: Some(Arc::new(
                Secret::new(b"sentinel-admin-secret".to_vec()).expect("secret"),
            )),
            origins: OriginPolicy::new(vec!["https://example.test".to_owned()]),
            request_ids: Arc::new(RequestIdSource::default()),
            version: "fixture-version",
            shutdown: ShutdownTrigger::default(),
            limits: Limits::default(),
            admin_api_enabled: true,
        };
        let debug = format!("{state:?}");
        assert!(!debug.contains("sentinel"));
        assert!(state.public_auth_required());
        assert!(state.admin_auth_required());
        assert!(state.verify_public_token(b"sentinel-public-secret"));
        assert!(state.verify_admin_token(b"sentinel-admin-secret"));
        assert!(!state.verify_admin_token(b"sentinel-public-secret"));
        assert!(state.origins().allows(Some("https://example.test")));
        drop(state);
        let _ = fs::remove_dir_all(cache);
    }

    #[test]
    fn app_state_rejects_equal_credential_values_from_distinct_sources() {
        let base = fixture_path("equal-credentials");
        fs::create_dir_all(&base).expect("fixture directory");
        let public = base.join("public-token");
        let admin = base.join("admin-token");
        fs::write(&public, b"same-secret").expect("public fixture");
        fs::write(&admin, b"same-secret").expect("admin fixture");
        let config = ServerConfig {
            cache_directory: base.join("cache"),
            auth: Some(CredentialSource::File(public)),
            admin_auth: Some(CredentialSource::File(admin)),
            ..ServerConfig::default()
        };
        assert!(matches!(
            AppState::new(&config),
            Err(AppStateError::Credential(CredentialError::Invalid))
        ));
        fs::remove_dir_all(base).expect("fixture cleanup");
    }

    #[tokio::test]
    async fn shutdown_trigger_is_idempotent_and_has_no_missed_wakeup() {
        let before = ShutdownTrigger::default();
        before.trigger();
        tokio::time::timeout(Duration::from_millis(50), before.cancelled())
            .await
            .expect("already-triggered waiter");

        let after = ShutdownTrigger::default();
        let waiter = {
            let after = after.clone();
            tokio::spawn(async move { after.cancelled().await })
        };
        tokio::task::yield_now().await;
        after.trigger();
        after.trigger();
        tokio::time::timeout(Duration::from_millis(50), waiter)
            .await
            .expect("live waiter")
            .expect("wait task");
    }

    #[test]
    fn request_ids_are_nonzero_and_shared_across_state_clones() {
        let source = Arc::new(RequestIdSource::default());
        let first = source.next();
        let clone = Arc::clone(&source);
        let second = clone.next();
        assert_eq!(first, RequestId(1));
        assert_eq!(second, RequestId(2));
    }
}

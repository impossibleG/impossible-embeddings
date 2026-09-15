//! Bounded, fair application scheduling independent of HTTP, gRPC, and MCP types.

use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingBatch, EmbeddingOutput, EngineFailure, ErrorCode,
    ExecutionControl, RequestedModel, ResolvedModelIdentity,
};
use impossible_embedding_onnx::OnnxEmbeddingEngine;
use impossible_models::{
    CancelToken as InstallCancelToken, Installer, Manifest, ModelStatus, ModelStore,
};
use impossible_server_core::{
    HealthRegistry, LifecycleState, ModelFailureReason, ModelKey, ModelState, ReadinessReason,
    ShutdownCoordinator,
    metrics::{Metrics, Operation, Outcome},
};
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    fmt,
    future::Future,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{SyncSender, TrySendError, sync_channel},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};

/// Engine operations required by the application layer.
pub trait RuntimeEngine: Send + Sync + 'static {
    /// Embed a batch with all semantics made explicit.
    ///
    /// # Errors
    /// Returns a stable engine failure when validation or inference fails.
    fn embed(
        &self,
        model: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        control: &ExecutionControl,
        options: EmbedOptions,
    ) -> Result<EmbeddingOutput, EngineFailure>;

    /// Execute a private warmup without exposing its contents.
    ///
    /// # Errors
    /// Returns a stable engine failure when the runtime cannot be warmed.
    fn warm(&self) -> Result<(), EngineFailure>;
}

impl RuntimeEngine for OnnxEmbeddingEngine {
    fn embed(
        &self,
        model: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        control: &ExecutionControl,
        options: EmbedOptions,
    ) -> Result<EmbeddingOutput, EngineFailure> {
        self.embed_with_options(model, batch, control, options)
    }

    fn warm(&self) -> Result<(), EngineFailure> {
        self.warm()
    }
}

/// Scheduler resource limits. All bounds are enforced before native inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPolicy {
    /// Waiting requests accepted per loaded model.
    pub queue_depth: usize,
    /// Inputs in one native call.
    pub max_batch_items: usize,
    /// Estimated tokens in one native call.
    pub max_batch_tokens: usize,
    /// Maximum time the oldest compatible request waits for batching.
    pub max_batch_wait: Duration,
    /// Number of dedicated native execution threads and queued native jobs.
    pub blocking_concurrency: usize,
    /// End-to-end default deadline.
    pub request_timeout: Duration,
}

impl Default for BatchPolicy {
    fn default() -> Self {
        Self {
            queue_depth: 256,
            max_batch_items: 128,
            max_batch_tokens: 32_768,
            max_batch_wait: Duration::from_millis(4),
            blocking_concurrency: 4,
            request_timeout: Duration::from_secs(30),
        }
    }
}

impl BatchPolicy {
    fn validate(self) -> Result<Self, LifecycleError> {
        if self.queue_depth == 0
            || self.max_batch_items == 0
            || self.max_batch_tokens == 0
            || self.blocking_concurrency == 0
            || self.request_timeout.is_zero()
        {
            return Err(LifecycleError::InvalidPolicy);
        }
        Ok(self)
    }
}

/// Stable administrative failures. Display output contains no paths or model identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleError {
    /// Runtime bounds were invalid.
    InvalidPolicy,
    /// The service no longer accepts lifecycle operations.
    ShuttingDown,
    /// The requested model is not registered.
    NotFound,
    /// A model with the same public id is already registered.
    AlreadyLoaded,
    /// Active requests or runtime leases prevent mutation.
    InUse,
    /// Artifact verification or runtime initialization failed.
    LoadFailed,
    /// Model installation failed.
    InstallFailed,
    /// Model deletion failed.
    DeleteFailed,
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidPolicy => "runtime policy is invalid",
            Self::ShuttingDown => "the service is shutting down",
            Self::NotFound => "the model is not registered",
            Self::AlreadyLoaded => "the model is already loaded",
            Self::InUse => "the model is in use",
            Self::LoadFailed => "model loading failed",
            Self::InstallFailed => "model installation failed",
            Self::DeleteFailed => "model deletion failed",
        })
    }
}

impl std::error::Error for LifecycleError {}

type BlockingJob = Box<dyn FnOnce() + Send + 'static>;

#[derive(Clone)]
struct BlockingPool {
    sender: SyncSender<BlockingJob>,
}

impl BlockingPool {
    fn new(size: usize, queue_depth: usize) -> Self {
        let (sender, receiver) = sync_channel::<BlockingJob>(queue_depth);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..size {
            let receiver = Arc::clone(&receiver);
            let _ = thread::Builder::new()
                .name(format!("impossible-inference-{index}"))
                .spawn(move || {
                    loop {
                        let job = receiver.lock().ok().and_then(|guard| guard.recv().ok());
                        match job {
                            Some(job) => job(),
                            None => break,
                        }
                    }
                });
        }
        Self { sender }
    }

    fn try_execute(&self, job: BlockingJob) -> Result<(), BlockingJob> {
        self.sender.try_send(job).map_err(|error| match error {
            TrySendError::Full(job) | TrySendError::Disconnected(job) => job,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CompatibilityKey(EmbedOptions);

struct Request {
    inputs: Vec<String>,
    token_estimate: usize,
    options: EmbedOptions,
    control: ExecutionControl,
    response: oneshot::Sender<Result<EmbeddingOutput, EngineFailure>>,
    _lease: RuntimeLease,
}

struct ModelSlot {
    key: ModelKey,
    sender: mpsc::Sender<Request>,
    leases: Arc<AtomicUsize>,
    accepting: Arc<AtomicBool>,
}

/// RAII capability preventing unload/delete while application work refers to a model.
#[derive(Debug)]
pub struct RuntimeLease(Arc<AtomicUsize>);

impl Clone for RuntimeLease {
    fn clone(&self) -> Self {
        self.0.fetch_add(1, Ordering::AcqRel);
        Self(Arc::clone(&self.0))
    }
}

impl Drop for RuntimeLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Aggregate state safe for health/status surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    /// Loaded models that currently accept work.
    pub ready_models: usize,
    /// Total registered models.
    pub registered_models: usize,
    /// Whether shutdown has started.
    pub draining: bool,
}

struct Inner {
    policy: BatchPolicy,
    models: RwLock<HashMap<String, Arc<ModelSlot>>>,
    loading: Mutex<HashMap<String, ModelKey>>,
    installs: Mutex<HashMap<usize, InstallCancelToken>>,
    health: HealthRegistry,
    shutdown: ShutdownCoordinator,
    pool: BlockingPool,
    next_key: AtomicUsize,
    draining: AtomicBool,
    metrics: Metrics,
    active: AtomicUsize,
}

/// Cloneable application service shared by every transport adapter.
#[derive(Clone)]
pub struct ApplicationRuntime(Arc<Inner>);

impl ApplicationRuntime {
    /// Construct a runtime with validated finite resource bounds.
    ///
    /// # Errors
    /// Returns [`LifecycleError::InvalidPolicy`] when any required bound is zero.
    pub fn new(policy: BatchPolicy) -> Result<Self, LifecycleError> {
        let policy = policy.validate()?;
        let health = HealthRegistry::default();
        let _ = health.transition(LifecycleState::Ready, None);
        Ok(Self(Arc::new(Inner {
            policy,
            models: RwLock::new(HashMap::new()),
            loading: Mutex::new(HashMap::new()),
            installs: Mutex::new(HashMap::new()),
            health,
            shutdown: ShutdownCoordinator::default(),
            pool: BlockingPool::new(policy.blocking_concurrency, policy.queue_depth),
            next_key: AtomicUsize::new(1),
            draining: AtomicBool::new(false),
            metrics: Metrics::default(),
            active: AtomicUsize::new(0),
        })))
    }

    /// Shared health registry for transport-specific health presentation.
    #[must_use]
    pub fn health(&self) -> &HealthRegistry {
        &self.0.health
    }

    /// Bounded-cardinality metrics shared by all transport adapters.
    #[must_use]
    pub fn metrics(&self) -> &Metrics {
        &self.0.metrics
    }

    /// Return only aggregate, privacy-safe runtime state.
    #[must_use]
    pub fn snapshot(&self) -> RuntimeSnapshot {
        let registered = self.0.models.read().map_or(0, |models| models.len());
        let (ready, _) = self.0.health.model_counts();
        RuntimeSnapshot {
            ready_models: ready,
            registered_models: registered,
            draining: self.0.draining.load(Ordering::Acquire),
        }
    }

    /// Register an already initialized engine and start its per-model bounded scheduler.
    ///
    /// # Errors
    /// Returns a lifecycle error for invalid, duplicate, or post-shutdown registration.
    pub fn register_engine(
        &self,
        model_id: impl Into<String>,
        engine: Arc<dyn RuntimeEngine>,
    ) -> Result<(), LifecycleError> {
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let model_id = model_id.into();
        if model_id.trim().is_empty() {
            return Err(LifecycleError::LoadFailed);
        }
        let mut models = self
            .0
            .models
            .write()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if models.contains_key(&model_id) {
            return Err(LifecycleError::AlreadyLoaded);
        }
        let key_value = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let key = ModelKey(u64::try_from(key_value).unwrap_or(u64::MAX));
        let (sender, receiver) = mpsc::channel(self.0.policy.queue_depth);
        let leases = Arc::new(AtomicUsize::new(0));
        let accepting = Arc::new(AtomicBool::new(true));
        let slot = Arc::new(ModelSlot {
            key,
            sender,
            leases,
            accepting: Arc::clone(&accepting),
        });
        models.insert(model_id.clone(), Arc::clone(&slot));
        drop(models);
        self.0.health.set_model(key, ModelState::Ready);
        let policy = self.0.policy;
        let pool = self.0.pool.clone();
        tokio::spawn(run_scheduler(
            receiver, model_id, engine, policy, pool, accepting,
        ));
        Ok(())
    }

    /// Verify, initialize, and warm an ONNX model on the dedicated blocking pool before publish.
    ///
    /// # Errors
    /// Returns a sanitized lifecycle error when verification, loading, warmup, or publication fails.
    pub async fn load_onnx(
        &self,
        store: ModelStore,
        manifest: Manifest,
    ) -> Result<(), LifecycleError> {
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let model_id = manifest.canonical_id.clone();
        let key_value = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let key = ModelKey(u64::try_from(key_value).unwrap_or(u64::MAX));
        {
            let models = self
                .0
                .models
                .read()
                .map_err(|_| LifecycleError::LoadFailed)?;
            let mut loading = self
                .0
                .loading
                .lock()
                .map_err(|_| LifecycleError::LoadFailed)?;
            if models.contains_key(&model_id) || loading.contains_key(&model_id) {
                return Err(LifecycleError::AlreadyLoaded);
            }
            loading.insert(model_id.clone(), key);
        }
        self.0.health.set_model(key, ModelState::Loading);
        let (tx, rx) = oneshot::channel();
        let job = Box::new(move || {
            let result = store
                .verified_model(&manifest)
                .map_err(|_| LifecycleError::LoadFailed)
                .and_then(|verified| {
                    let engine = OnnxEmbeddingEngine::new();
                    engine
                        .load(&verified)
                        .map_err(|_| LifecycleError::LoadFailed)?;
                    engine.warm().map_err(|_| LifecycleError::LoadFailed)?;
                    Ok(Arc::new(engine) as Arc<dyn RuntimeEngine>)
                });
            let _ = tx.send(result);
        }) as BlockingJob;
        if self.0.pool.try_execute(job).is_err() {
            self.finish_failed_load(&model_id, key);
            return Err(LifecycleError::InUse);
        }
        let engine = match rx.await {
            Ok(Ok(engine)) => engine,
            Ok(Err(error)) => {
                self.finish_failed_load(&model_id, key);
                return Err(error);
            }
            Err(_) => {
                self.finish_failed_load(&model_id, key);
                return Err(LifecycleError::LoadFailed);
            }
        };
        if self.0.draining.load(Ordering::Acquire) {
            self.finish_failed_load(&model_id, key);
            return Err(LifecycleError::ShuttingDown);
        }
        let result = self.register_reserved_engine(model_id.clone(), key, engine);
        if let Ok(mut loading) = self.0.loading.lock() {
            loading.remove(&model_id);
        }
        result
    }

    fn finish_failed_load(&self, model_id: &str, key: ModelKey) {
        if let Ok(mut loading) = self.0.loading.lock() {
            loading.remove(model_id);
        }
        self.0.health.set_model(
            key,
            ModelState::Failed(ModelFailureReason::RuntimeInitializationFailed),
        );
    }

    fn register_reserved_engine(
        &self,
        model_id: String,
        key: ModelKey,
        engine: Arc<dyn RuntimeEngine>,
    ) -> Result<(), LifecycleError> {
        let mut models = self
            .0
            .models
            .write()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if models.contains_key(&model_id) {
            return Err(LifecycleError::AlreadyLoaded);
        }
        let (sender, receiver) = mpsc::channel(self.0.policy.queue_depth);
        let leases = Arc::new(AtomicUsize::new(0));
        let accepting = Arc::new(AtomicBool::new(true));
        models.insert(
            model_id.clone(),
            Arc::new(ModelSlot {
                key,
                sender,
                leases,
                accepting: Arc::clone(&accepting),
            }),
        );
        drop(models);
        self.0.health.set_model(key, ModelState::Ready);
        tokio::spawn(run_scheduler(
            receiver,
            model_id,
            engine,
            self.0.policy,
            self.0.pool.clone(),
            accepting,
        ));
        Ok(())
    }

    /// Run a model installation operation while shutdown tracks and can reject its admission.
    ///
    /// # Errors
    /// Returns shutdown rejection or the operation's sanitized lifecycle failure.
    pub async fn install_with<F, T>(&self, operation: F) -> Result<T, LifecycleError>
    where
        F: Future<Output = Result<T, LifecycleError>>,
    {
        let _permit = self
            .0
            .shutdown
            .admit()
            .ok_or(LifecycleError::ShuttingDown)?;
        operation.await
    }

    /// Install one exact manifest with cooperative cancellation on forced shutdown.
    ///
    /// # Errors
    /// Returns shutdown rejection or a sanitized installation failure.
    pub async fn install_model(
        &self,
        installer: &Installer,
        manifest: &Manifest,
    ) -> Result<ModelStatus, LifecycleError> {
        let _permit = self
            .0
            .shutdown
            .admit()
            .ok_or(LifecycleError::ShuttingDown)?;
        let operation = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let cancellation = InstallCancelToken::new();
        self.0
            .installs
            .lock()
            .map_err(|_| LifecycleError::InstallFailed)?
            .insert(operation, cancellation.clone());
        let result = installer.install(manifest, &cancellation).await;
        if let Ok(mut installs) = self.0.installs.lock() {
            installs.remove(&operation);
        }
        result.map_err(|_| LifecycleError::InstallFailed)
    }

    /// Acquire an explicit model lease for coordinated external administrative work.
    ///
    /// # Errors
    /// Returns [`LifecycleError::NotFound`] unless the model currently accepts work.
    pub fn lease(&self, model_id: &str) -> Result<RuntimeLease, LifecycleError> {
        let models = self.0.models.read().map_err(|_| LifecycleError::NotFound)?;
        let slot = models.get(model_id).ok_or(LifecycleError::NotFound)?;
        if !slot.accepting.load(Ordering::Acquire) {
            return Err(LifecycleError::NotFound);
        }
        slot.leases.fetch_add(1, Ordering::AcqRel);
        Ok(RuntimeLease(Arc::clone(&slot.leases)))
    }

    /// Submit an embedding request. Queue-full is immediate and deterministic.
    ///
    /// # Errors
    /// Returns stable request, availability, overload, cancellation, deadline, or engine failures.
    pub async fn embed(
        &self,
        model_id: &str,
        inputs: Vec<String>,
        options: EmbedOptions,
        cancellation: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<EmbeddingOutput, EngineFailure> {
        let started = Instant::now();
        let active = self.0.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.0.metrics.set_active(active);
        let result = self
            .embed_inner(model_id, inputs, options, cancellation, deadline)
            .await;
        let remaining = self
            .0
            .active
            .fetch_sub(1, Ordering::AcqRel)
            .saturating_sub(1);
        self.0.metrics.set_active(remaining);
        let outcome = match &result {
            Ok(_) => Outcome::Ok,
            Err(error) => match error.public_error().code {
                ErrorCode::InvalidRequest => Outcome::Invalid,
                ErrorCode::QueueFull => Outcome::Overloaded,
                ErrorCode::Cancelled | ErrorCode::DeadlineExceeded => Outcome::Cancelled,
                ErrorCode::ModelUnavailable | ErrorCode::InferenceFailed | ErrorCode::Internal => {
                    Outcome::Failed
                }
                _ => Outcome::Failed,
            },
        };
        self.0
            .metrics
            .observe(Operation::Embed, outcome, started.elapsed());
        result
    }

    async fn embed_inner(
        &self,
        model_id: &str,
        inputs: Vec<String>,
        options: EmbedOptions,
        cancellation: CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<EmbeddingOutput, EngineFailure> {
        let _permit = self
            .0
            .shutdown
            .admit()
            .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        if inputs.is_empty() || inputs.len() > self.0.policy.max_batch_items {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        let token_estimate = estimate_tokens(&inputs);
        if token_estimate > self.0.policy.max_batch_tokens {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        let slot = {
            let models = self
                .0
                .models
                .read()
                .map_err(|_| EngineFailure::public(ErrorCode::Internal))?;
            Arc::clone(
                models
                    .get(model_id)
                    .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?,
            )
        };
        if !slot.accepting.load(Ordering::Acquire) {
            return Err(EngineFailure::public(ErrorCode::ModelUnavailable));
        }
        let lease = self
            .lease(model_id)
            .map_err(|_| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        let effective_deadline =
            deadline.or_else(|| Instant::now().checked_add(self.0.policy.request_timeout));
        let control = ExecutionControl::new(cancellation.clone(), effective_deadline);
        control.ensure_active()?;
        let (response, receive) = oneshot::channel();
        let request = Request {
            inputs,
            token_estimate,
            options,
            control,
            response,
            _lease: lease,
        };
        slot.sender.try_send(request).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => EngineFailure::public(ErrorCode::QueueFull),
            mpsc::error::TrySendError::Closed(_) => {
                EngineFailure::public(ErrorCode::ModelUnavailable)
            }
        })?;
        let wait = effective_deadline.map_or(self.0.policy.request_timeout, |value| {
            value.saturating_duration_since(Instant::now())
        });
        match tokio::time::timeout(wait, receive).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(EngineFailure::public(ErrorCode::ModelUnavailable)),
            Err(_) => {
                cancellation.cancel();
                Err(EngineFailure::public(ErrorCode::DeadlineExceeded))
            }
        }
    }

    /// Unregister an idle model. Existing leases make the operation fail without partial mutation.
    ///
    /// # Errors
    /// Returns not-found or in-use without changing the registry.
    pub fn unload(&self, model_id: &str) -> Result<(), LifecycleError> {
        let mut models = self
            .0
            .models
            .write()
            .map_err(|_| LifecycleError::NotFound)?;
        let slot = models.get(model_id).ok_or(LifecycleError::NotFound)?;
        if slot.leases.load(Ordering::Acquire) != 0 {
            return Err(LifecycleError::InUse);
        }
        slot.accepting.store(false, Ordering::Release);
        let slot = models.remove(model_id).ok_or(LifecycleError::NotFound)?;
        self.0.health.set_model(slot.key, ModelState::Unloaded);
        Ok(())
    }

    /// Delete an unloaded model through the verified store boundary.
    ///
    /// # Errors
    /// Returns in-use for a registered model or a sanitized store deletion failure.
    pub fn delete(
        &self,
        model_id: &str,
        store: &ModelStore,
        manifest: &Manifest,
    ) -> Result<bool, LifecycleError> {
        if self
            .0
            .models
            .read()
            .is_ok_and(|models| models.contains_key(model_id))
        {
            return Err(LifecycleError::InUse);
        }
        store
            .delete(manifest)
            .map_err(|_| LifecycleError::DeleteFailed)
    }

    /// Reject new work, allow accepted work to drain, then cancel and discard remaining work.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        if self.0.draining.swap(true, Ordering::AcqRel) {
            return self.0.shutdown.wait(Duration::ZERO);
        }
        let _ = self
            .0
            .health
            .transition(LifecycleState::Draining, Some(ReadinessReason::Draining));
        self.0.shutdown.begin();
        let started = Instant::now();
        while started.elapsed() < timeout {
            if self.0.shutdown.wait(Duration::ZERO) {
                break;
            }
            tokio::task::yield_now().await;
        }
        let drained = self.0.shutdown.wait(Duration::ZERO);
        if !drained {
            if let Ok(installs) = self.0.installs.lock() {
                for cancellation in installs.values() {
                    cancellation.cancel();
                }
            }
        }
        if let Ok(models) = self.0.models.read() {
            for slot in models.values() {
                slot.accepting.store(false, Ordering::Release);
            }
        }
        let _ = self
            .0
            .health
            .transition(LifecycleState::Stopped, Some(ReadinessReason::Stopped));
        drained
    }
}

fn estimate_tokens(inputs: &[String]) -> usize {
    inputs
        .iter()
        // UTF-8 bytes plus a small special-token allowance is deliberately conservative for the
        // supported subword tokenizers; a character/word heuristic could under-admit adversarial
        // Unicode and defeat the native batch memory bound.
        .map(|input| input.len().saturating_add(2).max(1))
        .fold(0_usize, usize::saturating_add)
}

async fn run_scheduler(
    mut receiver: mpsc::Receiver<Request>,
    model_id: String,
    engine: Arc<dyn RuntimeEngine>,
    policy: BatchPolicy,
    pool: BlockingPool,
    accepting: Arc<AtomicBool>,
) {
    let mut pending = VecDeque::new();
    loop {
        if pending.is_empty() {
            match receiver.recv().await {
                Some(request) => pending.push_back(request),
                None => break,
            }
        }
        let wait = tokio::time::sleep(policy.max_batch_wait);
        tokio::pin!(wait);
        loop {
            tokio::select! {
                () = &mut wait => break,
                request = receiver.recv() => match request {
                    Some(request) => pending.push_back(request),
                    None => break,
                }
            }
            if pending.len() >= policy.queue_depth {
                break;
            }
        }
        while !pending.is_empty() {
            let Some(first) = pending.pop_front() else {
                break;
            };
            if let Err(error) = first.control.ensure_active() {
                let _ = first.response.send(Err(error));
                continue;
            }
            let key = CompatibilityKey(first.options);
            let mut requests = vec![first];
            let mut items = requests[0].inputs.len();
            let mut tokens = requests[0].token_estimate;
            let scan = pending.len();
            for _ in 0..scan {
                let Some(candidate) = pending.pop_front() else {
                    break;
                };
                let compatible = CompatibilityKey(candidate.options) == key
                    && items.saturating_add(candidate.inputs.len()) <= policy.max_batch_items
                    && tokens.saturating_add(candidate.token_estimate) <= policy.max_batch_tokens;
                if compatible {
                    items += candidate.inputs.len();
                    tokens += candidate.token_estimate;
                    requests.push(candidate);
                } else {
                    pending.push_back(candidate);
                }
            }
            dispatch_batch(
                &pool,
                Arc::clone(&engine),
                model_id.clone(),
                requests,
                Arc::clone(&accepting),
            );
            // Rotating incompatible requests through the deque provides bounded round-robin
            // fairness across compatibility keys instead of draining one hot key first.
            tokio::task::yield_now().await;
        }
        if !accepting.load(Ordering::Acquire) && receiver.is_empty() {
            break;
        }
    }
}

fn dispatch_batch(
    pool: &BlockingPool,
    engine: Arc<dyn RuntimeEngine>,
    model_id: String,
    requests: Vec<Request>,
    accepting: Arc<AtomicBool>,
) {
    let requests = Arc::new(Mutex::new(Some(requests)));
    let job_requests = Arc::clone(&requests);
    let job = Box::new(move || {
        let owned = job_requests
            .lock()
            .ok()
            .and_then(|mut requests| requests.take());
        if let Some(owned) = owned {
            execute_batch(engine.as_ref(), &model_id, owned, &accepting);
        }
    }) as BlockingJob;
    if let Err(job) = pool.try_execute(job) {
        // The native queue is bounded. Running this closure would violate that bound, so recover
        // ownership and fail every request deterministically without invoking inference.
        drop(job);
        if let Some(requests) = requests
            .lock()
            .ok()
            .and_then(|mut requests| requests.take())
        {
            fail_requests(requests, ErrorCode::QueueFull);
        }
    }
}

fn execute_batch(
    engine: &dyn RuntimeEngine,
    model_id: &str,
    requests: Vec<Request>,
    accepting: &AtomicBool,
) {
    let mut active = Vec::with_capacity(requests.len());
    for request in requests {
        if !accepting.load(Ordering::Acquire) {
            let _ = request
                .response
                .send(Err(EngineFailure::public(ErrorCode::ModelUnavailable)));
        } else if let Err(error) = request.control.ensure_active() {
            let _ = request.response.send(Err(error));
        } else {
            active.push(request);
        }
    }
    let requests = active;
    if requests.is_empty() {
        return;
    }
    let mut inputs = Vec::new();
    let mut sizes = Vec::with_capacity(requests.len());
    for request in &requests {
        sizes.push(request.inputs.len());
        inputs.extend(request.inputs.iter().cloned());
    }
    let Ok(batch) = EmbeddingBatch::new(inputs.into_iter().map(Cow::Owned)) else {
        fail_requests(requests, ErrorCode::InvalidRequest);
        return;
    };
    let Ok(model) = RequestedModel::new(model_id.to_owned()) else {
        fail_requests(requests, ErrorCode::ModelUnavailable);
        return;
    };
    let batch_control = ExecutionControl::new(CancellationToken::default(), None);
    let result = engine.embed(&model, &batch, &batch_control, requests[0].options);
    match result {
        Ok(output) if output.vectors.len() == sizes.iter().sum::<usize>() => {
            publish_results(requests, sizes, output.vectors, &output.model, accepting);
        }
        Ok(_) => fail_requests(requests, ErrorCode::InferenceFailed),
        Err(error) => fail_requests(requests, error.public_error().code),
    }
}

fn publish_results(
    requests: Vec<Request>,
    sizes: Vec<usize>,
    vectors: Vec<Vec<f32>>,
    model: &ResolvedModelIdentity,
    accepting: &AtomicBool,
) {
    let mut vectors = vectors.into_iter();
    for (request, size) in requests.into_iter().zip(sizes) {
        if !accepting.load(Ordering::Acquire) {
            let _ = request
                .response
                .send(Err(EngineFailure::public(ErrorCode::ModelUnavailable)));
            for _ in 0..size {
                let _ = vectors.next();
            }
            continue;
        }
        if let Err(error) = request.control.ensure_active() {
            let _ = request.response.send(Err(error));
            for _ in 0..size {
                let _ = vectors.next();
            }
            continue;
        }
        let response_vectors = vectors.by_ref().take(size).collect();
        let _ = request.response.send(Ok(EmbeddingOutput {
            vectors: response_vectors,
            model: model.clone(),
        }));
    }
}

fn fail_requests(requests: Vec<Request>, code: ErrorCode) {
    for request in requests {
        let effective = request
            .control
            .ensure_active()
            .err()
            .unwrap_or_else(|| EngineFailure::public(code));
        let _ = request.response.send(Err(effective));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impossible_embedding_core::EmbeddingTask;
    use std::sync::atomic::AtomicUsize;

    struct FakeEngine {
        calls: AtomicUsize,
        block: Duration,
        fail: bool,
    }

    impl FakeEngine {
        fn new(block: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                block,
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                block: Duration::ZERO,
                fail: true,
            }
        }
    }

    impl RuntimeEngine for FakeEngine {
        fn embed(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            thread::sleep(self.block);
            if self.fail {
                return Err(EngineFailure::public(ErrorCode::InferenceFailed));
            }
            let marker = match options.task {
                EmbeddingTask::Query => 1.0,
                EmbeddingTask::Document => 2.0,
            };
            Ok(EmbeddingOutput {
                vectors: batch
                    .inputs()
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let index = u16::try_from(i).unwrap_or(u16::MAX);
                        vec![marker, f32::from(index)]
                    })
                    .collect(),
                model: ResolvedModelIdentity::new("fake", "rev", "fake@1", "artifact", "semantic")?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    fn policy() -> BatchPolicy {
        BatchPolicy {
            queue_depth: 8,
            max_batch_items: 8,
            max_batch_tokens: 32,
            max_batch_wait: Duration::from_millis(10),
            blocking_concurrency: 1,
            request_timeout: Duration::from_secs(2),
        }
    }

    #[tokio::test]
    async fn batches_compatible_requests_and_preserves_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let engine = Arc::new(FakeEngine::new(Duration::ZERO));
        runtime.register_engine("fake", engine.clone())?;
        let a = runtime.embed(
            "fake",
            vec!["a".into(), "b".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let b = runtime.embed(
            "fake",
            vec!["c".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let (a, b) = tokio::join!(a, b);
        assert_eq!(a?.vectors, vec![vec![2.0, 0.0], vec![2.0, 1.0]]);
        assert_eq!(b?.vectors, vec![vec![2.0, 2.0]]);
        assert_eq!(engine.calls.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn never_batches_incompatible_options() -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let engine = Arc::new(FakeEngine::new(Duration::ZERO));
        runtime.register_engine("fake", engine.clone())?;
        let query = EmbedOptions {
            task: EmbeddingTask::Query,
            ..EmbedOptions::default()
        };
        let a = runtime.embed(
            "fake",
            vec!["a".into()],
            query,
            CancellationToken::default(),
            None,
        );
        let b = runtime.embed(
            "fake",
            vec!["b".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let (a, b) = tokio::join!(a, b);
        assert!((a?.vectors[0][0] - 1.0).abs() < f32::EPSILON);
        assert!((b?.vectors[0][0] - 2.0).abs() < f32::EPSILON);
        assert_eq!(engine.calls.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[tokio::test]
    async fn rejects_request_boundaries_and_unknown_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
        let empty = runtime
            .embed(
                "fake",
                vec![],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await;
        assert_eq!(
            empty.err().ok_or("expected error")?.public_error().code,
            ErrorCode::InvalidRequest
        );
        let unknown = runtime
            .embed(
                "missing",
                vec!["x".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await;
        assert_eq!(
            unknown.err().ok_or("expected error")?.public_error().code,
            ErrorCode::ModelUnavailable
        );
        let long = runtime
            .embed(
                "fake",
                vec!["x".repeat(200)],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await;
        assert_eq!(
            long.err().ok_or("expected error")?.public_error().code,
            ErrorCode::InvalidRequest
        );
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_and_deadline_discard_late_native_results()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::from_millis(40))))?;
        let cancellation = CancellationToken::default();
        let cancel_copy = cancellation.clone();
        let request = runtime.embed(
            "fake",
            vec!["x".into()],
            EmbedOptions::default(),
            cancellation,
            None,
        );
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(15)).await;
            cancel_copy.cancel();
        });
        let error = request.await.err().ok_or("expected error")?;
        assert_eq!(error.public_error().code, ErrorCode::Cancelled);
        let error = runtime
            .embed(
                "fake",
                vec!["x".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                Some(Instant::now() + Duration::from_millis(5)),
            )
            .await
            .err()
            .ok_or("expected error")?;
        assert_eq!(error.public_error().code, ErrorCode::DeadlineExceeded);
        Ok(())
    }

    #[tokio::test]
    async fn leases_make_unload_atomic_and_health_tracks_lifecycle()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
        assert!(runtime.health().readiness().is_ready());
        let lease = runtime.lease("fake")?;
        assert_eq!(runtime.unload("fake"), Err(LifecycleError::InUse));
        drop(lease);
        runtime.unload("fake")?;
        assert!(!runtime.health().readiness().is_ready());
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_rejects_new_work_and_stops_health() -> Result<(), Box<dyn std::error::Error>>
    {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
        assert!(runtime.shutdown(Duration::from_millis(50)).await);
        assert!(!runtime.health().is_live());
        let error = runtime
            .embed(
                "fake",
                vec!["x".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("expected error")?;
        assert_eq!(error.public_error().code, ErrorCode::ModelUnavailable);
        assert_eq!(
            runtime.register_engine("other", Arc::new(FakeEngine::new(Duration::ZERO))),
            Err(LifecycleError::ShuttingDown)
        );
        Ok(())
    }

    #[tokio::test]
    async fn one_failed_native_batch_fails_every_constituent_request()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let engine = Arc::new(FakeEngine::failing());
        runtime.register_engine("fake", engine.clone())?;
        let a = runtime.embed(
            "fake",
            vec!["a".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let b = runtime.embed(
            "fake",
            vec!["b".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let (a, b) = tokio::join!(a, b);
        assert_eq!(
            a.err().ok_or("expected failure")?.public_error().code,
            ErrorCode::InferenceFailed
        );
        assert_eq!(
            b.err().ok_or("expected failure")?.public_error().code,
            ErrorCode::InferenceFailed
        );
        assert_eq!(engine.calls.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn bounded_native_queue_reports_retryable_overload()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::ZERO;
        limits.queue_depth = 1;
        let runtime = ApplicationRuntime::new(limits)?;
        let slow = Arc::new(FakeEngine::new(Duration::from_millis(150)));
        runtime.register_engine("one", slow.clone())?;
        runtime.register_engine("two", Arc::new(FakeEngine::new(Duration::from_millis(150))))?;
        runtime.register_engine("three", Arc::new(FakeEngine::new(Duration::ZERO)))?;

        let first_runtime = runtime.clone();
        let first = tokio::spawn(async move {
            first_runtime
                .embed(
                    "one",
                    vec!["a".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        while slow.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let second_runtime = runtime.clone();
        let second = tokio::spawn(async move {
            second_runtime
                .embed(
                    "two",
                    vec!["b".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let overloaded = runtime
            .embed(
                "three",
                vec!["c".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("expected overload")?;
        assert_eq!(overloaded.public_error().code, ErrorCode::QueueFull);
        assert_eq!(
            overloaded.public_error().retryability,
            impossible_embedding_core::Retryability::Retryable
        );
        assert!(first.await??.vectors.len() == 1);
        assert!(second.await??.vectors.len() == 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn alternating_compatibility_classes_do_not_starve_under_load()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.queue_depth = 64;
        limits.blocking_concurrency = 4;
        let runtime = ApplicationRuntime::new(limits)?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..48 {
            let runtime = runtime.clone();
            tasks.spawn(async move {
                let task = if index % 2 == 0 {
                    EmbeddingTask::Query
                } else {
                    EmbeddingTask::Document
                };
                let output = runtime
                    .embed(
                        "fake",
                        vec![format!("item-{index}")],
                        EmbedOptions {
                            task,
                            ..EmbedOptions::default()
                        },
                        CancellationToken::default(),
                        None,
                    )
                    .await?;
                Ok::<_, EngineFailure>((task, output.vectors[0][0]))
            });
        }
        let mut completed = 0;
        while let Some(result) = tasks.join_next().await {
            let (task, marker) = result??;
            let expected = if task == EmbeddingTask::Query {
                1.0
            } else {
                2.0
            };
            assert!((marker - expected).abs() < f32::EPSILON);
            completed += 1;
        }
        assert_eq!(completed, 48);
        let metrics = runtime.metrics().render();
        assert!(metrics.contains("operation=\"embed\",outcome=\"ok\"} 48"));
        assert!(!metrics.contains("item-"));
        assert!(!metrics.contains("model="));
        Ok(())
    }
}

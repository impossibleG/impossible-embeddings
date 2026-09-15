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
    HealthRegistry, LifecycleState, ModelKey, ModelState, ReadinessReason, ShutdownCoordinator,
    config::Limits,
    metrics::{Metrics, Operation, Outcome},
    shutdown::WorkPermit,
};
use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    fmt,
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, Once, RwLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{SyncSender, TrySendError, sync_channel},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::{
    sync::{Notify, mpsc, oneshot, watch},
    task::JoinHandle,
};

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
    /// Inputs allowed in one caller request.
    pub max_items: usize,
    /// Estimated tokens allowed in one caller request.
    pub max_tokens: usize,
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
            max_items: 128,
            max_tokens: 32_768,
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
            || self.max_items == 0
            || self.max_tokens == 0
            || self.max_batch_items == 0
            || self.max_batch_tokens == 0
            || self.blocking_concurrency == 0
            || self.request_timeout.is_zero()
            || self.max_items > self.max_batch_items
            || self.max_tokens > self.max_batch_tokens
        {
            return Err(LifecycleError::InvalidPolicy);
        }
        Ok(self)
    }
}

impl BatchPolicy {
    /// Build scheduler policy explicitly from the validated server resource limits.
    /// Request bounds and cross-request native batch bounds remain independent.
    #[must_use]
    pub fn from_limits(limits: &Limits, max_batch_wait: Duration) -> Self {
        Self {
            queue_depth: limits.max_queue_depth,
            max_items: limits.max_items,
            max_tokens: limits.max_tokens,
            max_batch_items: limits.max_batch_items,
            max_batch_tokens: limits.max_batch_tokens,
            max_batch_wait,
            blocking_concurrency: limits.max_concurrency,
            request_timeout: limits.request_timeout,
        }
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
                            Some(job) => {
                                // A defective native adapter must fail its own job, never remove a
                                // worker and silently reduce service capacity.
                                let _ = catch_unwind(AssertUnwindSafe(job));
                            }
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
    queue: Option<QueuePermit>,
}

struct ModelSlot {
    key: ModelKey,
    sender: mpsc::Sender<Request>,
    leases: Arc<AtomicUsize>,
    accepting: Arc<AtomicBool>,
    cancellation: CancellationToken,
    queued: Arc<AtomicUsize>,
}

struct SchedulerTask {
    stop: watch::Sender<bool>,
    handle: JoinHandle<()>,
}

// `catch_unwind` does not suppress Rust's default panic hook: the hook runs first and can print
// adapter-owned payloads and source paths. Installing this once gives every native worker a
// process-wide, payload-free policy. Rust has no stable thread-local panic hook, so embedders that
// need a different global policy must install their own equally privacy-safe hook before creating
// other application components and accept that this runtime deliberately does not chain it.
static SANITIZED_PANIC_HOOK: Once = Once::new();
static SANITIZED_PANIC_COUNT: AtomicUsize = AtomicUsize::new(0);

fn install_sanitized_panic_hook() {
    SANITIZED_PANIC_HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|_| {
            SANITIZED_PANIC_COUNT.fetch_add(1, Ordering::Relaxed);
            eprintln!("impossible runtime: an isolated worker failed");
        }));
    });
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
    installs: Arc<Mutex<HashMap<usize, InstallCancelToken>>>,
    health: HealthRegistry,
    shutdown: ShutdownCoordinator,
    pool: BlockingPool,
    next_key: AtomicUsize,
    draining: AtomicBool,
    admin: Mutex<()>,
    schedulers: Mutex<HashMap<ModelKey, SchedulerTask>>,
    retired_schedulers: Mutex<Vec<JoinHandle<()>>>,
    force_stop: watch::Sender<bool>,
    metrics: Metrics,
    active: Arc<AtomicUsize>,
    active_zero: Arc<Notify>,
    queued: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum GaugeKind {
    Active,
    Queue,
}

struct GaugeGuard {
    counter: Arc<AtomicUsize>,
    metrics: Metrics,
    kind: GaugeKind,
    zero_latch: Option<Arc<Notify>>,
}

impl GaugeGuard {
    fn increment(
        counter: Arc<AtomicUsize>,
        metrics: Metrics,
        kind: GaugeKind,
        zero_latch: Option<Arc<Notify>>,
    ) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        match kind {
            GaugeKind::Active => metrics.increment_active(),
            GaugeKind::Queue => metrics.increment_queue_depth(),
        }
        Self {
            counter,
            metrics,
            kind,
            zero_latch,
        }
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        let previous = self.counter.fetch_sub(1, Ordering::AcqRel);
        match self.kind {
            GaugeKind::Active => self.metrics.decrement_active(),
            GaugeKind::Queue => self.metrics.decrement_queue_depth(),
        }
        if previous == 1 {
            if let Some(latch) = &self.zero_latch {
                // `notify_one` stores a permit when the shutdown waiter is between creating and
                // polling its notification future, so the final active completion cannot be lost.
                latch.notify_one();
            }
        }
    }
}

async fn wait_for_active_zero(active: &AtomicUsize, latch: &Notify) {
    while active.load(Ordering::Acquire) != 0 {
        // Register before rechecking the predicate. This closes the interval in which the final
        // decrement could otherwise notify an unregistered waiter and leave shutdown asleep.
        let notified = latch.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if active.load(Ordering::Acquire) == 0 {
            break;
        }
        notified.await;
    }
}

struct QueuePermit {
    local: Arc<AtomicUsize>,
    _global: GaugeGuard,
}

impl QueuePermit {
    fn acquire(
        local: Arc<AtomicUsize>,
        global: Arc<AtomicUsize>,
        metrics: Metrics,
        limit: usize,
    ) -> Option<Self> {
        local
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                (value < limit).then_some(value + 1)
            })
            .ok()?;
        Some(Self {
            local,
            _global: GaugeGuard::increment(global, metrics, GaugeKind::Queue, None),
        })
    }
}

impl Drop for QueuePermit {
    fn drop(&mut self) {
        self.local.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Cloneable application service shared by every transport adapter.
#[derive(Clone)]
pub struct ApplicationRuntime(Arc<Inner>);

struct LoadReservation {
    runtime: ApplicationRuntime,
    model_id: String,
    key: ModelKey,
    armed: bool,
}

struct InstallRegistration {
    installs: Arc<Mutex<HashMap<usize, InstallCancelToken>>>,
    operation: usize,
}

struct ActiveInstall {
    _registration: InstallRegistration,
    _permit: WorkPermit,
    cancellation: InstallCancelToken,
}

struct CancelOnDrop {
    abandonment: CancellationToken,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abandonment.cancel();
        }
    }
}

impl Drop for InstallRegistration {
    fn drop(&mut self) {
        if let Ok(mut installs) = self.installs.lock() {
            installs.remove(&self.operation);
        }
    }
}

impl LoadReservation {
    fn publish(mut self, engine: Arc<dyn RuntimeEngine>) -> Result<(), LifecycleError> {
        let result = self
            .runtime
            .register_reserved_engine(self.model_id.clone(), self.key, engine);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for LoadReservation {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Ok(_admin) = self.runtime.0.admin.lock() {
            if let Ok(mut loading) = self.runtime.0.loading.lock() {
                if loading.get(&self.model_id) == Some(&self.key) {
                    loading.remove(&self.model_id);
                }
            }
            let _ = self.runtime.0.health.remove_model(self.key);
        }
    }
}

impl ApplicationRuntime {
    /// Construct a runtime with validated finite resource bounds.
    ///
    /// # Errors
    /// Returns [`LifecycleError::InvalidPolicy`] when any required bound is zero.
    pub fn new(policy: BatchPolicy) -> Result<Self, LifecycleError> {
        let policy = policy.validate()?;
        install_sanitized_panic_hook();
        let health = HealthRegistry::default();
        let _ = health.transition(LifecycleState::Ready, None);
        let (force_stop, _) = watch::channel(false);
        Ok(Self(Arc::new(Inner {
            policy,
            models: RwLock::new(HashMap::new()),
            loading: Mutex::new(HashMap::new()),
            installs: Arc::new(Mutex::new(HashMap::new())),
            health,
            shutdown: ShutdownCoordinator::default(),
            pool: BlockingPool::new(policy.blocking_concurrency, policy.queue_depth),
            next_key: AtomicUsize::new(1),
            draining: AtomicBool::new(false),
            admin: Mutex::new(()),
            schedulers: Mutex::new(HashMap::new()),
            retired_schedulers: Mutex::new(Vec::new()),
            force_stop,
            metrics: Metrics::default(),
            active: Arc::new(AtomicUsize::new(0)),
            active_zero: Arc::new(Notify::new()),
            queued: Arc::new(AtomicUsize::new(0)),
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
        let model_id = model_id.into();
        if model_id.trim().is_empty() {
            return Err(LifecycleError::LoadFailed);
        }
        let _admin = self
            .0
            .admin
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let mut models = self
            .0
            .models
            .write()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if models.contains_key(&model_id)
            || self
                .0
                .loading
                .lock()
                .is_ok_and(|loading| loading.contains_key(&model_id))
        {
            return Err(LifecycleError::AlreadyLoaded);
        }
        let key_value = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let key = ModelKey(u64::try_from(key_value).unwrap_or(u64::MAX));
        // Channel capacity is only an implementation detail. QueuePermit below is the single
        // admission authority across channel, scheduler pending state, and native-pool waiting.
        let (sender, receiver) = mpsc::channel(self.0.policy.queue_depth);
        let leases = Arc::new(AtomicUsize::new(0));
        let accepting = Arc::new(AtomicBool::new(true));
        let cancellation = CancellationToken::default();
        let queued = Arc::new(AtomicUsize::new(0));
        let slot = Arc::new(ModelSlot {
            key,
            sender,
            leases,
            accepting: Arc::clone(&accepting),
            cancellation,
            queued,
        });
        models.insert(model_id.clone(), Arc::clone(&slot));
        drop(models);
        self.0.health.set_model(key, ModelState::Ready);
        let policy = self.0.policy;
        let pool = self.0.pool.clone();
        let (stop, stop_rx) = watch::channel(false);
        let handle = tokio::spawn(run_scheduler(
            receiver, model_id, engine, policy, pool, accepting, stop_rx,
        ));
        self.0
            .schedulers
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?
            .insert(key, SchedulerTask { stop, handle });
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
        let model_id = manifest.canonical_id.clone();
        self.load_with(model_id, move || {
            store
                .verified_model(&manifest)
                .map_err(|_| LifecycleError::LoadFailed)
                .and_then(|verified| {
                    let engine = OnnxEmbeddingEngine::new();
                    engine
                        .load(&verified)
                        .map_err(|_| LifecycleError::LoadFailed)?;
                    engine.warm().map_err(|_| LifecycleError::LoadFailed)?;
                    Ok(Arc::new(engine) as Arc<dyn RuntimeEngine>)
                })
        })
        .await
    }

    async fn load_with<F>(&self, model_id: String, load: F) -> Result<(), LifecycleError>
    where
        F: FnOnce() -> Result<Arc<dyn RuntimeEngine>, LifecycleError> + Send + 'static,
    {
        // The permit and reservation deliberately remain owned by this public future. Native
        // initialization is not interruptible, but forced shutdown can terminate this future,
        // release lifecycle accounting, and drop the sole capability that can publish its result.
        let _permit = self
            .0
            .shutdown
            .admit()
            .ok_or(LifecycleError::ShuttingDown)?;
        let key_value = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let key = ModelKey(u64::try_from(key_value).unwrap_or(u64::MAX));
        // Construct cleanup ownership before publishing the reservation. Cancellation at every
        // later await or early return then has an owner that can remove both registry entries.
        let reservation = LoadReservation {
            runtime: self.clone(),
            model_id: model_id.clone(),
            key,
            armed: true,
        };
        {
            let _admin = self
                .0
                .admin
                .lock()
                .map_err(|_| LifecycleError::LoadFailed)?;
            if self.0.draining.load(Ordering::Acquire) {
                return Err(LifecycleError::ShuttingDown);
            }
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
            self.0.health.set_model(key, ModelState::Loading);
        }
        let (tx, rx) = oneshot::channel();
        let job = Box::new(move || {
            let _ = tx.send(load());
        }) as BlockingJob;
        if self.0.pool.try_execute(job).is_err() {
            return Err(LifecycleError::InUse);
        }
        let mut force_stop = self.0.force_stop.subscribe();
        if *force_stop.borrow() {
            return Err(LifecycleError::ShuttingDown);
        }
        let result = tokio::select! {
            biased;
            changed = force_stop.changed() => {
                let _ = changed;
                return Err(LifecycleError::ShuttingDown);
            }
            result = rx => result.map_err(|_| LifecycleError::LoadFailed)?,
        };
        let engine = match result {
            Ok(engine) => engine,
            Err(error) => return Err(error),
        };
        reservation.publish(engine)
    }

    fn register_reserved_engine(
        &self,
        model_id: String,
        key: ModelKey,
        engine: Arc<dyn RuntimeEngine>,
    ) -> Result<(), LifecycleError> {
        let _admin = self
            .0
            .admin
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let mut models = self
            .0
            .models
            .write()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if models.contains_key(&model_id) {
            return Err(LifecycleError::AlreadyLoaded);
        }
        {
            let mut loading = self
                .0
                .loading
                .lock()
                .map_err(|_| LifecycleError::LoadFailed)?;
            if loading.get(&model_id) != Some(&key) {
                return Err(LifecycleError::LoadFailed);
            }
            loading.remove(&model_id);
        }
        let (sender, receiver) = mpsc::channel(self.0.policy.queue_depth);
        let leases = Arc::new(AtomicUsize::new(0));
        let accepting = Arc::new(AtomicBool::new(true));
        let cancellation = CancellationToken::default();
        let queued = Arc::new(AtomicUsize::new(0));
        models.insert(
            model_id.clone(),
            Arc::new(ModelSlot {
                key,
                sender,
                leases,
                accepting: Arc::clone(&accepting),
                cancellation,
                queued,
            }),
        );
        drop(models);
        self.0.health.set_model(key, ModelState::Ready);
        let (stop, stop_rx) = watch::channel(false);
        let handle = tokio::spawn(run_scheduler(
            receiver,
            model_id,
            engine,
            self.0.policy,
            self.0.pool.clone(),
            accepting,
            stop_rx,
        ));
        self.0
            .schedulers
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?
            .insert(key, SchedulerTask { stop, handle });
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
        let mut force_stop = self.0.force_stop.subscribe();
        if *force_stop.borrow() {
            return Err(LifecycleError::ShuttingDown);
        }
        tokio::pin!(operation);
        tokio::select! {
            biased;
            changed = force_stop.changed() => {
                let _ = changed;
                Err(LifecycleError::ShuttingDown)
            }
            result = &mut operation => result,
        }
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
        let active = self.begin_install()?;
        let mut force_stop = self.0.force_stop.subscribe();
        if *force_stop.borrow() {
            active.cancellation.cancel();
            return Err(LifecycleError::ShuttingDown);
        }
        let install = installer.install(manifest, &active.cancellation);
        tokio::pin!(install);
        let result = tokio::select! {
            biased;
            changed = force_stop.changed() => {
                let _ = changed;
                active.cancellation.cancel();
                return Err(LifecycleError::ShuttingDown);
            }
            result = &mut install => result,
        };
        result.map_err(|_| LifecycleError::InstallFailed)
    }

    fn begin_install(&self) -> Result<ActiveInstall, LifecycleError> {
        // The admin lock makes shutdown admission and cancellation-token publication one atomic
        // lifecycle transition: shutdown cannot publish Draining/Stopped between the two.
        let _admin = self
            .0
            .admin
            .lock()
            .map_err(|_| LifecycleError::InstallFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let permit = self
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
        Ok(ActiveInstall {
            _registration: InstallRegistration {
                installs: Arc::clone(&self.0.installs),
                operation,
            },
            _permit: permit,
            cancellation,
        })
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
        let _active = GaugeGuard::increment(
            Arc::clone(&self.0.active),
            self.0.metrics.clone(),
            GaugeKind::Active,
            Some(Arc::clone(&self.0.active_zero)),
        );
        let result = self
            .embed_inner(model_id, inputs, options, cancellation, deadline)
            .await;
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
        // Start the server cap before admission, validation, and registry locks so it is genuinely
        // end-to-end rather than only an inference-response timeout.
        let server_deadline = Instant::now()
            .checked_add(self.0.policy.request_timeout)
            .ok_or_else(|| EngineFailure::public(ErrorCode::Internal))?;
        let _permit = self
            .0
            .shutdown
            .admit()
            .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        if inputs.is_empty() || inputs.len() > self.0.policy.max_items {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        let token_estimate = estimate_tokens(&inputs);
        if token_estimate > self.0.policy.max_tokens {
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
            deadline.map_or(server_deadline, |caller| caller.min(server_deadline));
        let abandonment = CancellationToken::default();
        let request_cancellation = CancellationToken::any(vec![
            cancellation,
            abandonment.clone(),
            slot.cancellation.clone(),
        ]);
        let control = ExecutionControl::new(request_cancellation, Some(effective_deadline));
        control.ensure_active()?;
        let queue = QueuePermit::acquire(
            Arc::clone(&slot.queued),
            Arc::clone(&self.0.queued),
            self.0.metrics.clone(),
            self.0.policy.queue_depth,
        )
        .ok_or_else(|| EngineFailure::public(ErrorCode::QueueFull))?;
        let (response, receive) = oneshot::channel();
        let request = Request {
            inputs,
            token_estimate,
            options,
            control: control.clone(),
            response,
            _lease: lease,
            queue: Some(queue),
        };
        slot.sender.try_send(request).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => EngineFailure::public(ErrorCode::QueueFull),
            mpsc::error::TrySendError::Closed(_) => {
                EngineFailure::public(ErrorCode::ModelUnavailable)
            }
        })?;
        await_response(
            control,
            abandonment,
            receive,
            self.0.force_stop.subscribe(),
            effective_deadline,
        )
        .await
    }

    /// Unregister an idle model. Existing leases make the operation fail without partial mutation.
    ///
    /// # Errors
    /// Returns not-found or in-use without changing the registry.
    pub fn unload(&self, model_id: &str) -> Result<(), LifecycleError> {
        let _admin = self.0.admin.lock().map_err(|_| LifecycleError::NotFound)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
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
        if let Ok(mut schedulers) = self.0.schedulers.lock() {
            if let Some(task) = schedulers.remove(&slot.key) {
                let _ = task.stop.send(true);
                if let Ok(mut retired) = self.0.retired_schedulers.lock() {
                    retired.retain(|handle| !handle.is_finished());
                    retired.push(task.handle);
                }
            }
        }
        let _ = self.0.health.remove_model(slot.key);
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
        if model_id != manifest.canonical_id {
            return Err(LifecycleError::DeleteFailed);
        }
        let _admin = self
            .0
            .admin
            .lock()
            .map_err(|_| LifecycleError::DeleteFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        if self
            .0
            .models
            .read()
            .map_err(|_| LifecycleError::DeleteFailed)?
            .contains_key(model_id)
            || self
                .0
                .loading
                .lock()
                .map_err(|_| LifecycleError::DeleteFailed)?
                .contains_key(model_id)
        {
            return Err(LifecycleError::InUse);
        }
        store
            .delete(manifest)
            .map_err(|_| LifecycleError::DeleteFailed)
    }

    /// Reject new work, allow accepted work to drain, then cancel and discard remaining work.
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        let Ok(admin) = self.0.admin.lock() else {
            return false;
        };
        if self.0.draining.swap(true, Ordering::AcqRel) {
            return self.0.shutdown.wait(Duration::ZERO);
        }
        let _ = self
            .0
            .health
            .transition(LifecycleState::Draining, Some(ReadinessReason::Draining));
        self.0.shutdown.begin();
        drop(admin);
        let started = Instant::now();
        while started.elapsed() < timeout {
            if self.0.shutdown.wait(Duration::ZERO) {
                break;
            }
            tokio::task::yield_now().await;
        }
        let drained = self.0.shutdown.wait(Duration::ZERO);
        if let Ok(models) = self.0.models.read() {
            for slot in models.values() {
                slot.accepting.store(false, Ordering::Release);
                slot.cancellation.cancel();
            }
        }
        if !drained {
            if let Ok(installs) = self.0.installs.lock() {
                for cancellation in installs.values() {
                    cancellation.cancel();
                }
            }
            let _ = self.0.force_stop.send(true);
        }
        let mut tasks = self
            .0
            .schedulers
            .lock()
            .map(|mut schedulers| {
                schedulers
                    .drain()
                    .map(|(_, task)| {
                        let _ = task.stop.send(true);
                        task.handle
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Ok(mut retired) = self.0.retired_schedulers.lock() {
            tasks.extend(retired.drain(..));
        }
        for mut task in tasks {
            if !task.is_finished() {
                task.abort();
            }
            // An aborted scheduler is cheap to acknowledge and must never be detached. Native
            // jobs are separately fenced by `accepting` and may finish internally.
            let _ = (&mut task).await;
        }
        if !drained {
            // The force-stop watch makes every admitted public request future terminal without
            // waiting for a non-cooperative native call. Do not publish Stopped before they have
            // observed that terminal state and released their shutdown permits.
            wait_for_active_zero(&self.0.active, &self.0.active_zero).await;
            while !self.0.shutdown.wait(Duration::ZERO) {
                tokio::task::yield_now().await;
            }
        }
        let _ = self
            .0
            .health
            .transition(LifecycleState::Stopped, Some(ReadinessReason::Stopped));
        drained
    }
}

async fn await_response(
    control: ExecutionControl,
    abandonment: CancellationToken,
    mut receive: oneshot::Receiver<Result<EmbeddingOutput, EngineFailure>>,
    mut force_stop: watch::Receiver<bool>,
    deadline: Instant,
) -> Result<EmbeddingOutput, EngineFailure> {
    let mut cancel_on_drop = CancelOnDrop {
        abandonment,
        armed: true,
    };
    let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
    tokio::pin!(sleep);
    let mut cancellation_poll = tokio::time::interval(Duration::from_millis(1));
    let result = loop {
        if *force_stop.borrow() {
            break Err(EngineFailure::public(ErrorCode::ModelUnavailable));
        }
        if let Err(error) = control.ensure_active() {
            break Err(error);
        }
        tokio::select! {
            biased;
            _ = cancellation_poll.tick() => continue,
            changed = force_stop.changed() => {
                if changed.is_err() || *force_stop.borrow() {
                    break Err(EngineFailure::public(ErrorCode::ModelUnavailable));
                }
            }
            () = &mut sleep => {
                break control.ensure_active().and_then(|()| {
                    Err(EngineFailure::public(ErrorCode::DeadlineExceeded))
                });
            }
            response = &mut receive => {
                if let Err(error) = control.ensure_active() {
                    break Err(error);
                }
                break response.unwrap_or_else(|_| {
                    Err(EngineFailure::public(ErrorCode::ModelUnavailable))
                });
            }
        }
    };
    // Reaching this point is a normal completion even when the operation returned an error.
    // Cancellation belongs exclusively to actual abandonment of this response future.
    cancel_on_drop.armed = false;
    result
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
    mut stop: watch::Receiver<bool>,
) {
    let mut pending = VecDeque::new();
    loop {
        if *stop.borrow() {
            receiver.close();
            fail_requests(pending.drain(..).collect(), ErrorCode::ModelUnavailable);
            while let Ok(request) = receiver.try_recv() {
                fail_requests(vec![request], ErrorCode::ModelUnavailable);
            }
            break;
        }
        if pending.is_empty() {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        continue;
                    }
                }
                request = receiver.recv() => match request {
                    Some(request) => pending.push_back(request),
                    None => break,
                }
            }
        }
        let wait = tokio::time::sleep(policy.max_batch_wait);
        tokio::pin!(wait);
        loop {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        break;
                    }
                }
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
        if *stop.borrow() {
            continue;
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
    for mut request in requests {
        // Admission queue accounting ends immediately before native execution starts.
        drop(request.queue.take());
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
    let batch_control = ExecutionControl::composite(
        requests
            .iter()
            .map(|request| request.control.clone())
            .collect::<Vec<_>>(),
    );
    // A panic is isolated to this batch and translated to the same stable failure as other native
    // adapter faults. The outer worker boundary provides a second line of defense.
    let result = catch_unwind(AssertUnwindSafe(|| {
        engine.embed(&model, &batch, &batch_control, requests[0].options)
    }));
    match result {
        Ok(Ok(output)) if output.vectors.len() == sizes.iter().sum::<usize>() => {
            publish_results(requests, sizes, output.vectors, &output.model, accepting);
        }
        Ok(Ok(_)) | Err(_) => fail_requests(requests, ErrorCode::InferenceFailed),
        Ok(Err(error)) => fail_requests(requests, error.public_error().code),
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
    use impossible_models::curated_manifests;
    use std::sync::atomic::AtomicUsize;

    struct FakeEngine {
        calls: AtomicUsize,
        finished: AtomicBool,
        block: Duration,
        fail: bool,
    }

    struct PanicOnceEngine(AtomicUsize);

    struct OrdinaryPanicOnceEngine(AtomicUsize);

    impl RuntimeEngine for OrdinaryPanicOnceEngine {
        fn embed(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            assert_ne!(
                self.0.fetch_add(1, Ordering::AcqRel),
                0,
                "adapter-owned panic payload"
            );
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: ResolvedModelIdentity::new("fake", "rev", "fake@1", "artifact", "semantic")?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    impl RuntimeEngine for PanicOnceEngine {
        fn embed(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                std::panic::resume_unwind(Box::new("native fault"));
            }
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: ResolvedModelIdentity::new("fake", "rev", "fake@1", "artifact", "semantic")?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    struct CooperativeEngine;

    impl RuntimeEngine for CooperativeEngine {
        fn embed(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            for _ in 0..30 {
                control.ensure_active()?;
                thread::sleep(Duration::from_millis(1));
            }
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: ResolvedModelIdentity::new("fake", "rev", "fake@1", "artifact", "semantic")?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    impl FakeEngine {
        fn new(block: Duration) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                finished: AtomicBool::new(false),
                block,
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                finished: AtomicBool::new(false),
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
            self.finished.store(true, Ordering::Release);
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
            max_items: 8,
            max_tokens: 32,
            max_batch_items: 8,
            max_batch_tokens: 32,
            max_batch_wait: Duration::from_millis(10),
            blocking_concurrency: 1,
            request_timeout: Duration::from_secs(2),
        }
    }

    #[test]
    fn server_limits_convert_without_conflating_request_and_batch_bounds() {
        let limits = Limits {
            max_items: 3,
            max_tokens: 30,
            max_batch_items: 7,
            max_batch_tokens: 70,
            ..Limits::default()
        };
        let policy = BatchPolicy::from_limits(&limits, Duration::from_millis(9));
        assert_eq!(policy.max_items, 3);
        assert_eq!(policy.max_tokens, 30);
        assert_eq!(policy.max_batch_items, 7);
        assert_eq!(policy.max_batch_tokens, 70);
        assert_eq!(policy.queue_depth, limits.max_queue_depth);
        assert_eq!(policy.blocking_concurrency, limits.max_concurrency);
        assert_eq!(policy.request_timeout, limits.request_timeout);
        let mut invalid = policy;
        invalid.max_batch_items = 2;
        assert!(matches!(
            ApplicationRuntime::new(invalid),
            Err(LifecycleError::InvalidPolicy)
        ));
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
    async fn configured_timeout_caps_a_longer_caller_deadline_and_cancel_wins()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.request_timeout = Duration::from_millis(10);
        let runtime = ApplicationRuntime::new(limits)?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::from_millis(80))))?;
        let started = Instant::now();
        let error = runtime
            .embed(
                "fake",
                vec!["x".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                Some(Instant::now() + Duration::from_secs(5)),
            )
            .await
            .err()
            .ok_or("expected configured timeout")?;
        assert_eq!(error.public_error().code, ErrorCode::DeadlineExceeded);
        assert!(started.elapsed() < Duration::from_millis(60));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let error = runtime
            .embed(
                "fake",
                vec!["x".into()],
                EmbedOptions::default(),
                cancelled,
                Some(Instant::now()),
            )
            .await
            .err()
            .ok_or("expected cancellation")?;
        assert_eq!(error.public_error().code, ErrorCode::Cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn native_batch_control_does_not_abort_a_surviving_constituent()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(CooperativeEngine))?;
        let cancelled = CancellationToken::default();
        let cancel_copy = cancelled.clone();
        let first = runtime.embed(
            "fake",
            vec!["first".into()],
            EmbedOptions::default(),
            cancelled,
            None,
        );
        let second = runtime.embed(
            "fake",
            vec!["second".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(15)).await;
            cancel_copy.cancel();
        });
        let (first, second) = tokio::join!(first, second);
        assert_eq!(
            first.err().ok_or("first must cancel")?.public_error().code,
            ErrorCode::Cancelled
        );
        assert_eq!(second?.vectors.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn native_panic_fails_one_batch_and_worker_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::ZERO;
        let runtime = ApplicationRuntime::new(limits)?;
        runtime.register_engine("fake", Arc::new(PanicOnceEngine(AtomicUsize::new(0))))?;
        let first = runtime
            .embed(
                "fake",
                vec!["first".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("panic must fail")?;
        assert_eq!(first.public_error().code, ErrorCode::InferenceFailed);
        let second = runtime
            .embed(
                "fake",
                vec!["second".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await?;
        assert_eq!(second.vectors.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn ordinary_panic_uses_process_wide_sanitized_hook_and_worker_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        let before = SANITIZED_PANIC_COUNT.load(Ordering::Acquire);
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine(
            "fake",
            Arc::new(OrdinaryPanicOnceEngine(AtomicUsize::new(0))),
        )?;
        let _ = runtime
            .embed(
                "fake",
                vec!["first".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await;
        assert!(SANITIZED_PANIC_COUNT.load(Ordering::Acquire) > before);
        assert_eq!(
            runtime
                .embed(
                    "fake",
                    vec!["second".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await?
                .vectors
                .len(),
            1
        );
        Ok(())
    }

    #[tokio::test]
    async fn normal_error_completion_does_not_cancel_a_shared_token()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::ZERO;
        let runtime = ApplicationRuntime::new(limits)?;
        runtime.register_engine("fake", Arc::new(PanicOnceEngine(AtomicUsize::new(0))))?;
        let shared = CancellationToken::default();
        let failure = runtime
            .embed(
                "fake",
                vec!["first".into()],
                EmbedOptions::default(),
                shared.clone(),
                None,
            )
            .await;
        assert_eq!(
            failure
                .err()
                .ok_or("first request must fail")?
                .public_error()
                .code,
            ErrorCode::InferenceFailed
        );
        assert!(!shared.is_cancelled());
        let survivor = runtime
            .embed(
                "fake",
                vec!["second".into()],
                EmbedOptions::default(),
                shared.clone(),
                None,
            )
            .await?;
        assert_eq!(survivor.vectors.len(), 1);
        assert!(!shared.is_cancelled());
        Ok(())
    }

    #[tokio::test]
    async fn abandoning_one_request_does_not_cancel_a_shared_caller_token()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::ZERO;
        let runtime = ApplicationRuntime::new(limits)?;
        let slow = Arc::new(FakeEngine::new(Duration::from_millis(40)));
        runtime.register_engine("fake", slow.clone())?;
        let shared = CancellationToken::default();

        let first_runtime = runtime.clone();
        let first_shared = shared.clone();
        let first = tokio::spawn(async move {
            first_runtime
                .embed(
                    "fake",
                    vec!["abandoned".into()],
                    EmbedOptions::default(),
                    first_shared,
                    None,
                )
                .await
        });
        while slow.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }

        let survivor_runtime = runtime.clone();
        let survivor_shared = shared.clone();
        let survivor = tokio::spawn(async move {
            survivor_runtime
                .embed(
                    "fake",
                    vec!["survivor".into()],
                    EmbedOptions::default(),
                    survivor_shared,
                    None,
                )
                .await
        });
        while !runtime
            .metrics()
            .render()
            .contains("impossible_queue_depth 1")
        {
            tokio::task::yield_now().await;
        }
        first.abort();
        let _ = first.await;

        let output = tokio::time::timeout(Duration::from_millis(250), survivor).await???;
        assert_eq!(output.vectors.len(), 1);
        assert!(!shared.is_cancelled());
        Ok(())
    }

    #[tokio::test]
    async fn exact_queue_capacity_and_abort_safe_gauges() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut limits = policy();
        limits.queue_depth = 1;
        limits.max_batch_wait = Duration::ZERO;
        let runtime = ApplicationRuntime::new(limits)?;
        let slow = Arc::new(FakeEngine::new(Duration::from_millis(100)));
        runtime.register_engine("fake", slow.clone())?;
        let first_runtime = runtime.clone();
        let first = tokio::spawn(async move {
            first_runtime
                .embed(
                    "fake",
                    vec!["first".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        while slow.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let queued_runtime = runtime.clone();
        let queued = tokio::spawn(async move {
            queued_runtime
                .embed(
                    "fake",
                    vec!["queued".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        while !runtime
            .metrics()
            .render()
            .contains("impossible_queue_depth 1")
        {
            tokio::task::yield_now().await;
        }
        let full = runtime
            .embed(
                "fake",
                vec!["full".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("queue depth one must reject the second waiter")?;
        assert_eq!(full.public_error().code, ErrorCode::QueueFull);
        queued.abort();
        let _ = queued.await;
        first.abort();
        let _ = first.await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let metrics = runtime.metrics().render();
        assert!(metrics.contains("impossible_active_requests 0"));
        assert!(metrics.contains("impossible_queue_depth 0"));
        Ok(())
    }

    #[tokio::test]
    async fn forced_shutdown_wakes_native_request_and_prevents_late_success()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let slow = Arc::new(FakeEngine::new(Duration::from_millis(100)));
        runtime.register_engine("fake", slow.clone())?;
        let request_runtime = runtime.clone();
        let request = tokio::spawn(async move {
            request_runtime
                .embed(
                    "fake",
                    vec!["x".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        while slow.calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        assert!(!runtime.shutdown(Duration::ZERO).await);
        assert!(request.is_finished());
        assert!(!slow.finished.load(Ordering::Acquire));
        assert!(
            runtime
                .0
                .schedulers
                .lock()
                .is_ok_and(|schedulers| schedulers.is_empty())
        );
        let error = tokio::time::timeout(Duration::from_millis(20), request)
            .await??
            .err()
            .ok_or("forced shutdown must fail accepted request")?;
        assert_eq!(error.public_error().code, ErrorCode::ModelUnavailable);
        assert!(!runtime.health().is_live());
        Ok(())
    }

    #[tokio::test]
    async fn forced_shutdown_detaches_hung_load_and_fences_late_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let (release, blocked) = std::sync::mpsc::sync_channel::<()>(0);
        let started = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let load_started = Arc::clone(&started);
        let load_finished = Arc::clone(&finished);
        let load_runtime = runtime.clone();
        let load = tokio::spawn(async move {
            load_runtime
                .load_with("hung".into(), move || {
                    load_started.store(true, Ordering::Release);
                    let _ = blocked.recv();
                    load_finished.store(true, Ordering::Release);
                    Ok(Arc::new(FakeEngine::new(Duration::ZERO)) as Arc<dyn RuntimeEngine>)
                })
                .await
        });
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }

        let drained =
            tokio::time::timeout(Duration::from_millis(250), runtime.shutdown(Duration::ZERO))
                .await?;
        assert!(!drained);
        assert!(!runtime.health().is_live());
        assert_eq!(load.await?, Err(LifecycleError::ShuttingDown));

        release.send(())?;
        while !finished.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(runtime.snapshot().registered_models, 0);
        assert_eq!(runtime.health().model_counts(), (0, 0));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn active_zero_latch_cannot_miss_final_completion()
    -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..500 {
            let active = Arc::new(AtomicUsize::new(1));
            let latch = Arc::new(Notify::new());
            let wait_active = Arc::clone(&active);
            let wait_latch = Arc::clone(&latch);
            let waiter = tokio::spawn(async move {
                wait_for_active_zero(&wait_active, &wait_latch).await;
            });
            tokio::task::yield_now().await;
            let previous = active.fetch_sub(1, Ordering::AcqRel);
            assert_eq!(previous, 1);
            latch.notify_one();
            tokio::time::timeout(Duration::from_millis(50), waiter).await??;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn install_admission_is_atomic_with_shutdown_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..20 {
            let runtime = ApplicationRuntime::new(policy())?;
            let candidate = runtime.clone();
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let candidate_barrier = Arc::clone(&barrier);
            let admission = tokio::task::spawn_blocking(move || {
                candidate_barrier.wait();
                let result = candidate.begin_install();
                let admitted = result.is_ok();
                drop(result);
                admitted
            });
            barrier.wait();
            let _ = runtime.shutdown(Duration::ZERO).await;
            let _was_admitted = admission.await?;
            assert!(matches!(
                runtime.begin_install(),
                Err(LifecycleError::ShuttingDown)
            ));
            assert!(
                runtime
                    .0
                    .installs
                    .lock()
                    .is_ok_and(|installs| installs.is_empty())
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_requires_canonical_identity_and_serializes_registration()
    -> Result<(), Box<dyn std::error::Error>> {
        static STORE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let sequence = STORE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "impossible-runtime-delete-{}-{sequence}",
            std::process::id()
        ));
        let store = ModelStore::new(&root)?;
        let manifest = curated_manifests()?
            .into_iter()
            .next()
            .ok_or("curated manifest required")?;
        let runtime = ApplicationRuntime::new(policy())?;
        assert_eq!(
            runtime.delete("different/model", &store, &manifest),
            Err(LifecycleError::DeleteFailed)
        );
        runtime
            .0
            .loading
            .lock()
            .map_err(|_| "loading registry poisoned")?
            .insert(manifest.canonical_id.clone(), ModelKey(42));
        assert_eq!(
            runtime.delete(&manifest.canonical_id, &store, &manifest),
            Err(LifecycleError::InUse)
        );
        runtime
            .0
            .loading
            .lock()
            .map_err(|_| "loading registry poisoned")?
            .remove(&manifest.canonical_id);

        let canonical_id = manifest.canonical_id.clone();
        let registration_runtime = runtime.clone();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let registration_barrier = Arc::clone(&barrier);
        let registration = tokio::task::spawn_blocking(move || {
            registration_barrier.wait();
            registration_runtime
                .register_engine(canonical_id, Arc::new(FakeEngine::new(Duration::ZERO)))
        });
        barrier.wait();
        let deletion = runtime.delete(&manifest.canonical_id, &store, &manifest);
        let registration = registration.await?;
        match (registration, deletion) {
            (Ok(()), Err(LifecycleError::InUse) | Ok(false)) => {}
            (registration, deletion) => {
                return Err(format!(
                    "unexpected registration/deletion outcome: {registration:?}, {deletion:?}"
                )
                .into());
            }
        }
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn repeated_register_and_unload_does_not_accumulate_health_entries()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        for _ in 0..20 {
            runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
            runtime.unload("fake")?;
            assert_eq!(runtime.health().model_counts(), (0, 0));
        }
        assert_eq!(runtime.snapshot().registered_models, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn simultaneous_registration_and_shutdown_has_no_post_stop_publication()
    -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..20 {
            let runtime = ApplicationRuntime::new(policy())?;
            let register_runtime = runtime.clone();
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let register_barrier = Arc::clone(&barrier);
            let registration = tokio::task::spawn_blocking(move || {
                register_barrier.wait();
                register_runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))
            });
            barrier.wait();
            let _ = runtime.shutdown(Duration::ZERO).await;
            let result = registration.await?;
            assert!(matches!(result, Ok(()) | Err(LifecycleError::ShuttingDown)));
            assert!(!runtime.health().is_live());
            assert!(!runtime.health().readiness().is_ready());
            assert_eq!(
                runtime.register_engine("late", Arc::new(FakeEngine::new(Duration::ZERO))),
                Err(LifecycleError::ShuttingDown)
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn aborted_tracked_operation_releases_shutdown_permit()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let operation_runtime = runtime.clone();
        let operation = tokio::spawn(async move {
            operation_runtime
                .install_with(std::future::pending::<Result<(), LifecycleError>>())
                .await
        });
        tokio::task::yield_now().await;
        operation.abort();
        let _ = operation.await;
        assert!(runtime.shutdown(Duration::from_millis(20)).await);
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
        assert!(metrics.contains("impossible_active_requests 0"));
        assert!(metrics.contains("impossible_queue_depth 0"));
        assert!(!metrics.contains("item-"));
        assert!(!metrics.contains("model="));
        Ok(())
    }
}

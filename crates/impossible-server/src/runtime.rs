//! Bounded, fair application scheduling independent of HTTP, gRPC, and MCP types.

use impossible_embedding_core::{
    CancellationToken, EmbedOptions, EmbeddingBatch, EmbeddingBatchCost, EmbeddingOutput,
    EngineFailure, ErrorCode, ExecutionControl, RequestedModel, ResolvedModelIdentity,
};
use impossible_embedding_onnx::OnnxEmbeddingEngine;
use impossible_models::{
    CancelToken as InstallCancelToken, CommitDecision, InstallCommitGate, Installer, Manifest,
    ModelStatus, ModelStore,
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
        Arc, Mutex, Once, RwLock, Weak,
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
    /// Return the immutable identity and native output width of the loaded model.
    ///
    /// The runtime uses this independently of adapter output so a defective or hostile adapter
    /// cannot substitute model identity or silently change vector width.
    ///
    /// # Errors
    /// Returns a stable model, adapter, or internal failure when no complete contract is loaded.
    fn model_contract(&self, model: &RequestedModel)
    -> Result<RuntimeModelContract, EngineFailure>;

    /// Validate one caller request and return its exact post-tokenization native cost.
    ///
    /// The default is conservative for simple custom engines. Production adapters should
    /// override it with tokenizer-exact accounting.
    ///
    /// # Errors
    /// Returns a stable invalid-request failure if cost arithmetic overflows.
    fn preflight(
        &self,
        _model: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        _options: EmbedOptions,
    ) -> Result<EmbeddingBatchCost, EngineFailure> {
        let mut tokens = 0_usize;
        let mut max_sequence_length = 0_usize;
        for input in batch.inputs() {
            let length = input
                .len()
                .checked_add(2)
                .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?
                .max(1);
            tokens = tokens
                .checked_add(length)
                .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?;
            max_sequence_length = max_sequence_length.max(length);
        }
        EmbeddingBatchCost::new(batch.inputs().len(), tokens, max_sequence_length)
    }

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
    fn model_contract(
        &self,
        model: &RequestedModel,
    ) -> Result<RuntimeModelContract, EngineFailure> {
        let (identity, native_dimensions) = self.resolved_model_contract(model)?;
        RuntimeModelContract::new(identity, native_dimensions)
    }

    fn preflight(
        &self,
        model: &RequestedModel,
        batch: &EmbeddingBatch<'_>,
        options: EmbedOptions,
    ) -> Result<EmbeddingBatchCost, EngineFailure> {
        self.preflight_with_options(model, batch, options)
    }

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

/// Immutable output contract captured before a request is admitted for native inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeModelContract {
    identity: ResolvedModelIdentity,
    native_dimensions: usize,
}

impl RuntimeModelContract {
    /// Construct a complete loaded-model contract.
    ///
    /// # Errors
    /// Returns an internal failure for an empty identity field or zero native width.
    pub fn new(
        identity: ResolvedModelIdentity,
        native_dimensions: usize,
    ) -> Result<Self, EngineFailure> {
        if native_dimensions == 0 || !identity_is_complete(&identity) {
            return Err(EngineFailure::public(ErrorCode::Internal));
        }
        Ok(Self {
            identity,
            native_dimensions,
        })
    }
}

/// Scheduler resource limits. All bounds are enforced before native inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchPolicy {
    /// Waiting requests accepted per loaded model.
    pub queue_depth: usize,
    /// Inputs allowed in one caller request.
    pub max_items: usize,
    /// UTF-8 bytes allowed in one decoded input before tokenization.
    pub max_input_bytes: usize,
    /// Aggregate UTF-8 bytes allowed in one decoded request before tokenization.
    pub max_request_bytes: usize,
    /// Post-tokenization, non-padding tokens allowed in one caller request.
    pub max_tokens: usize,
    /// Inputs in one native call.
    pub max_batch_items: usize,
    /// Padded token cells (`items * longest sequence`) allowed in one native call.
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
            max_input_bytes: 256 * 1024,
            max_request_bytes: 1024 * 1024,
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
            || self.max_input_bytes == 0
            || self.max_request_bytes == 0
            || self.max_tokens == 0
            || self.max_batch_items == 0
            || self.max_batch_tokens == 0
            || self.blocking_concurrency == 0
            || self.request_timeout.is_zero()
            || self.max_items > self.max_batch_items
            || self.max_input_bytes > self.max_request_bytes
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
            max_input_bytes: limits.max_input_bytes,
            max_request_bytes: limits.max_request_bytes,
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
    /// Dedicated runtime workers could not be initialized.
    InitializationFailed,
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
            Self::InitializationFailed => "runtime initialization failed",
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
    fn new(size: usize, queue_depth: usize) -> Result<Self, LifecycleError> {
        Self::new_with_spawner(size, queue_depth, |index, receiver| {
            thread::Builder::new()
                .name(format!("impossible-inference-{index}"))
                .spawn(move || run_blocking_worker(&receiver))
                .map(|_| ())
        })
    }

    fn new_with_spawner<F>(
        size: usize,
        queue_depth: usize,
        mut spawn_worker: F,
    ) -> Result<Self, LifecycleError>
    where
        F: FnMut(usize, Arc<Mutex<std::sync::mpsc::Receiver<BlockingJob>>>) -> std::io::Result<()>,
    {
        let (sender, receiver) = sync_channel::<BlockingJob>(queue_depth);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..size {
            spawn_worker(index, Arc::clone(&receiver))
                .map_err(|_| LifecycleError::InitializationFailed)?;
        }
        Ok(Self { sender })
    }

    fn try_execute(&self, job: BlockingJob) -> Result<(), BlockingJob> {
        self.sender.try_send(job).map_err(|error| match error {
            TrySendError::Full(job) | TrySendError::Disconnected(job) => job,
        })
    }
}

fn run_blocking_worker(receiver: &Mutex<std::sync::mpsc::Receiver<BlockingJob>>) {
    loop {
        let job = receiver.lock().ok().and_then(|guard| guard.recv().ok());
        match job {
            Some(job) => {
                // A defective native adapter must fail its own job, never remove a worker and
                // silently reduce service capacity.
                let _ = catch_unwind(AssertUnwindSafe(job));
            }
            None => break,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CompatibilityKey(EmbedOptions);

struct Request {
    sequence: usize,
    inputs: Vec<String>,
    cost: EmbeddingBatchCost,
    contract: RuntimeModelContract,
    options: EmbedOptions,
    control: ExecutionControl,
    response: oneshot::Sender<Result<EmbeddingOutput, EngineFailure>>,
    _lease: RuntimeLease,
    queue: Option<QueuePermit>,
}

struct PreflightSuccess {
    sequence: usize,
    inputs: Vec<String>,
    cost: EmbeddingBatchCost,
    contract: RuntimeModelContract,
    lease: RuntimeLease,
    queue: QueuePermit,
}

struct PreflightPlan {
    sequence: usize,
    inputs: Vec<String>,
    options: EmbedOptions,
    control: ExecutionControl,
    model: RequestedModel,
    engine: Arc<dyn RuntimeEngine>,
    lease: RuntimeLease,
    queue: QueuePermit,
    max_tokens: usize,
    max_batch_tokens: usize,
}

struct ModelSlot {
    key: ModelKey,
    sender: mpsc::Sender<Request>,
    leases: Arc<AtomicUsize>,
    accepting: Arc<AtomicBool>,
    cancellation: CancellationToken,
    queued: Arc<AtomicUsize>,
    engine: Arc<dyn RuntimeEngine>,
}

struct SchedulerTask {
    stop: watch::Sender<bool>,
    handle: JoinHandle<()>,
}

struct SchedulerCandidate {
    stop: watch::Sender<bool>,
    handle: Option<JoinHandle<()>>,
}

impl SchedulerCandidate {
    fn into_task(mut self) -> Result<SchedulerTask, LifecycleError> {
        let handle = self.handle.take().ok_or(LifecycleError::LoadFailed)?;
        Ok(SchedulerTask {
            stop: self.stop.clone(),
            handle,
        })
    }
}

impl Drop for SchedulerCandidate {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.stop.send(true);
            handle.abort();
        }
    }
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
    installs: Arc<Mutex<HashMap<usize, InstallControl>>>,
    health: HealthRegistry,
    shutdown: ShutdownCoordinator,
    pool: BlockingPool,
    next_key: AtomicUsize,
    next_request: AtomicUsize,
    draining: AtomicBool,
    admin: Mutex<()>,
    schedulers: Mutex<HashMap<ModelKey, SchedulerTask>>,
    retired_schedulers: Mutex<Vec<JoinHandle<()>>>,
    force_stop: watch::Sender<bool>,
    metrics: Metrics,
    active: Arc<AtomicUsize>,
    admitted: Arc<AtomicUsize>,
    admitted_zero: Arc<Notify>,
    queued: Arc<AtomicUsize>,
    shutdown_flight: Mutex<Option<ShutdownFlight>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // The final public owner is also a terminal lifecycle boundary. This path cannot await,
        // but it can fence publication, abort async schedulers, and release registry-owned engine
        // and verified-artifact references. Every operation is best-effort and poison-safe because
        // unwinding through `Drop` must never introduce a second panic.
        self.draining.store(true, Ordering::Release);
        self.shutdown.begin();
        self.shutdown.force_stop();
        let _ = self.force_stop.send(true);
        if let Ok(models) = self.models.read() {
            for slot in models.values() {
                slot.accepting.store(false, Ordering::Release);
                slot.cancellation.cancel();
            }
        }
        if let Ok(installs) = self.installs.lock() {
            for install in installs.values() {
                install.cancel_before_commit();
            }
        }
        if let Ok(mut schedulers) = self.schedulers.lock() {
            for (_, task) in schedulers.drain() {
                let _ = task.stop.send(true);
                task.handle.abort();
            }
        }
        if let Ok(mut retired) = self.retired_schedulers.lock() {
            for task in retired.drain(..) {
                task.abort();
            }
        }
        if let Ok(mut flight) = self.shutdown_flight.lock() {
            if let Some(flight) = flight.take() {
                flight.task.abort();
            }
        }
        if let Ok(mut models) = self.models.write() {
            models.clear();
        }
        if let Ok(mut loading) = self.loading.lock() {
            loading.clear();
        }
        self.health.clear_models();
        let _ = self
            .health
            .transition(LifecycleState::Stopped, Some(ReadinessReason::Stopped));
    }
}

struct ShutdownFlight {
    completion: watch::Receiver<Option<bool>>,
    deadline: watch::Sender<Option<Instant>>,
    task: JoinHandle<()>,
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
}

impl GaugeGuard {
    fn increment(counter: Arc<AtomicUsize>, metrics: Metrics, kind: GaugeKind) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        match kind {
            GaugeKind::Active => metrics.increment_active(),
            GaugeKind::Queue => metrics.increment_queue_depth(),
        }
        Self {
            counter,
            metrics,
            kind,
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
        debug_assert!(previous != 0, "gauge guard underflow");
    }
}

struct RuntimeWorkPermit {
    _shutdown: WorkPermit,
    admitted: Arc<AtomicUsize>,
    zero_latch: Arc<Notify>,
}

impl Drop for RuntimeWorkPermit {
    fn drop(&mut self) {
        let previous = self.admitted.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous != 0, "runtime work permit underflow");
        if previous == 1 {
            // `notify_one` retains a permit when no waiter is currently registered.
            self.zero_latch.notify_one();
        }
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
            _global: GaugeGuard::increment(global, metrics, GaugeKind::Queue),
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
    installs: Arc<Mutex<HashMap<usize, InstallControl>>>,
    operation: usize,
}

struct ActiveInstall {
    _registration: InstallRegistration,
    _permit: RuntimeWorkPermit,
    cancellation: InstallCancelToken,
    commit: InstallCommitGate,
}

#[derive(Clone)]
struct InstallControl {
    cancellation: InstallCancelToken,
    commit: InstallCommitGate,
}

impl InstallControl {
    fn cancel_before_commit(&self) -> CommitDecision {
        let decision = self.commit.request_cancel();
        if decision == CommitDecision::CancelledBeforeCommit {
            self.cancellation.cancel();
        }
        decision
    }
}

struct InstallCallerGuard {
    control: InstallControl,
    task: tokio::task::AbortHandle,
    armed: bool,
}

impl InstallCallerGuard {
    fn cancel_before_commit(&self) -> CommitDecision {
        let decision = self.control.cancel_before_commit();
        if decision == CommitDecision::CancelledBeforeCommit {
            self.task.abort();
        }
        decision
    }
}

impl Drop for InstallCallerGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancel_before_commit();
        }
    }
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
    async fn publish(
        mut self,
        executor: tokio::runtime::Handle,
        engine: Arc<dyn RuntimeEngine>,
    ) -> Result<(), LifecycleError> {
        let result = self
            .runtime
            .register_reserved_engine(self.model_id.clone(), self.key, executor, engine)
            .await;
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
            pool: BlockingPool::new(policy.blocking_concurrency, policy.queue_depth)?,
            next_key: AtomicUsize::new(1),
            next_request: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
            admin: Mutex::new(()),
            schedulers: Mutex::new(HashMap::new()),
            retired_schedulers: Mutex::new(Vec::new()),
            force_stop,
            metrics: Metrics::default(),
            active: Arc::new(AtomicUsize::new(0)),
            admitted: Arc::new(AtomicUsize::new(0)),
            admitted_zero: Arc::new(Notify::new()),
            queued: Arc::new(AtomicUsize::new(0)),
            shutdown_flight: Mutex::new(None),
        })))
    }

    fn admit_work(&self) -> Option<RuntimeWorkPermit> {
        let _admin = self.0.admin.lock().ok()?;
        if self.0.draining.load(Ordering::Acquire) {
            return None;
        }
        self.admit_work_locked()
    }

    // Callers must hold `admin`, which serializes the coordinator admission and async counter
    // increment with the shutdown boundary. Consequently shutdown can never observe zero between
    // those two halves of an accepted operation.
    fn admit_work_locked(&self) -> Option<RuntimeWorkPermit> {
        let shutdown = self.0.shutdown.admit()?;
        self.0.admitted.fetch_add(1, Ordering::AcqRel);
        Some(RuntimeWorkPermit {
            _shutdown: shutdown,
            admitted: Arc::clone(&self.0.admitted),
            zero_latch: Arc::clone(&self.0.admitted_zero),
        })
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
        // Registration publishes a scheduler-backed model. Resolve the executor before acquiring
        // lifecycle locks or mutating any registry so a synchronous caller outside Tokio gets a
        // stable failure rather than a panic after partial publication.
        let executor =
            tokio::runtime::Handle::try_current().map_err(|_| LifecycleError::LoadFailed)?;
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
        let loading = self
            .0
            .loading
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if models.contains_key(&model_id) || loading.contains_key(&model_id) {
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
            engine: Arc::clone(&engine),
        });
        let policy = self.0.policy;
        let pool = self.0.pool.clone();
        let (stop, stop_rx) = watch::channel(false);
        let scheduler_model_id = model_id.clone();
        let handle = executor.spawn(run_scheduler(
            receiver,
            scheduler_model_id,
            engine,
            policy,
            pool,
            accepting,
            stop_rx,
        ));
        let candidate = SchedulerCandidate {
            stop,
            handle: Some(handle),
        };
        if candidate
            .handle
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
        {
            return Err(LifecycleError::LoadFailed);
        }
        let mut schedulers = self
            .0
            .schedulers
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        let scheduler = candidate.into_task()?;
        schedulers.insert(key, scheduler);
        models.insert(model_id, Arc::clone(&slot));
        self.0.health.set_model(key, ModelState::Ready);
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
        // Capture the executor on the first poll, before admission or any lifecycle publication.
        // Completion is allowed to be polled from another thread without an ambient Tokio context,
        // while a future first polled outside Tokio fails without leaving a loading reservation.
        let executor =
            tokio::runtime::Handle::try_current().map_err(|_| LifecycleError::LoadFailed)?;
        // The permit and reservation deliberately remain owned by this public future. Native
        // initialization is not interruptible, but forced shutdown can terminate this future,
        // release lifecycle accounting, and drop the sole capability that can publish its result.
        let _permit = self.admit_work().ok_or(LifecycleError::ShuttingDown)?;
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
        reservation.publish(executor, engine).await
    }

    async fn register_reserved_engine(
        &self,
        model_id: String,
        key: ModelKey,
        executor: tokio::runtime::Handle,
        engine: Arc<dyn RuntimeEngine>,
    ) -> Result<(), LifecycleError> {
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
            engine: Arc::clone(&engine),
        });
        let (stop, stop_rx) = watch::channel(false);
        let (started, startup) = oneshot::channel();
        let policy = self.0.policy;
        let pool = self.0.pool.clone();
        let scheduler_model_id = model_id.clone();
        let handle = executor.spawn(async move {
            if started.send(()).is_err() {
                return;
            }
            run_scheduler(
                receiver,
                scheduler_model_id,
                engine,
                policy,
                pool,
                accepting,
                stop_rx,
            )
            .await;
        });
        let candidate = SchedulerCandidate {
            stop,
            handle: Some(handle),
        };
        // A handle captured from a runtime that has since been torn down accepts `spawn`, but the
        // task is cancelled without running. The startup handshake turns that into a stable error
        // before any model, scheduler, or ready-health publication.
        startup.await.map_err(|_| LifecycleError::LoadFailed)?;
        if candidate
            .handle
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
        {
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
        if models.contains_key(&model_id) {
            return Err(LifecycleError::AlreadyLoaded);
        }
        let mut loading = self
            .0
            .loading
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if loading.get(&model_id) != Some(&key) {
            return Err(LifecycleError::LoadFailed);
        }
        let mut schedulers = self
            .0
            .schedulers
            .lock()
            .map_err(|_| LifecycleError::LoadFailed)?;
        if self.0.draining.load(Ordering::Acquire) {
            return Err(LifecycleError::ShuttingDown);
        }
        if candidate
            .handle
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
        {
            return Err(LifecycleError::LoadFailed);
        }

        // Every fallible lock and shutdown check precedes this publication block. Candidate owns
        // scheduler rollback until it is transferred into the scheduler registry.
        let scheduler = candidate.into_task()?;
        schedulers.insert(key, scheduler);
        models.insert(model_id.clone(), slot);
        loading.remove(&model_id);
        self.0.health.set_model(key, ModelState::Ready);
        Ok(())
    }

    /// Run an owned model installation transaction with an explicit commit gate.
    ///
    /// # Errors
    /// Returns shutdown rejection or the operation's sanitized lifecycle failure.
    pub async fn install_with<F, Fut, T>(&self, operation: F) -> Result<T, LifecycleError>
    where
        F: FnOnce(InstallCancelToken, InstallCommitGate) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, LifecycleError>> + Send + 'static,
        T: Send + 'static,
    {
        let active = self.begin_install()?;
        self.run_install_transaction(active, operation).await
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
        let installer = installer.clone();
        let manifest = manifest.clone();
        self.run_install_transaction(active, move |cancellation, commit| async move {
            installer
                .install_transaction(&manifest, &cancellation, &commit, || {})
                .await
                .map_err(|_| LifecycleError::InstallFailed)
        })
        .await
    }

    async fn run_install_transaction<F, Fut, T>(
        &self,
        active: ActiveInstall,
        operation: F,
    ) -> Result<T, LifecycleError>
    where
        F: FnOnce(InstallCancelToken, InstallCommitGate) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, LifecycleError>> + Send + 'static,
        T: Send + 'static,
    {
        let control = InstallControl {
            cancellation: active.cancellation.clone(),
            commit: active.commit.clone(),
        };
        let mut force_stop = self.0.force_stop.subscribe();
        if *force_stop.borrow() {
            control.cancel_before_commit();
            return Err(LifecycleError::ShuttingDown);
        }
        let (send, mut receive) = oneshot::channel();
        let caller_control = control.clone();
        let task = tokio::spawn(async move {
            let result = operation(control.cancellation.clone(), control.commit.clone()).await;
            // Publish the truthful terminal result before releasing lifecycle registration and
            // admission. A shutdown unblocked by that release may publish force-stop immediately;
            // the caller's biased receive must already be ready in that race, including the
            // installer's already-installed fast path which never crosses the commit gate.
            let _ = send.send(result);
            drop(active);
        });
        let mut caller = InstallCallerGuard {
            control: caller_control,
            task: task.abort_handle(),
            armed: true,
        };
        let result = receive_install_result(&mut receive, &mut force_stop, &caller).await;
        caller.armed = false;
        result
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
            .admit_work_locked()
            .ok_or(LifecycleError::ShuttingDown)?;
        let operation = self.0.next_key.fetch_add(1, Ordering::Relaxed);
        let cancellation = InstallCancelToken::new();
        let commit = InstallCommitGate::new();
        self.0
            .installs
            .lock()
            .map_err(|_| LifecycleError::InstallFailed)?
            .insert(
                operation,
                InstallControl {
                    cancellation: cancellation.clone(),
                    commit: commit.clone(),
                },
            );
        Ok(ActiveInstall {
            _registration: InstallRegistration {
                installs: Arc::clone(&self.0.installs),
                operation,
            },
            _permit: permit,
            cancellation,
            commit,
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
            .admit_work()
            .ok_or_else(|| EngineFailure::public(ErrorCode::ModelUnavailable))?;
        if inputs.is_empty() || inputs.len() > self.0.policy.max_items {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        validate_raw_inputs(
            &inputs,
            self.0.policy.max_input_bytes,
            self.0.policy.max_request_bytes,
        )?;
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
        let prepared = execute_preflight(
            &self.0.pool,
            PreflightPlan {
                sequence: self.0.next_request.fetch_add(1, Ordering::Relaxed),
                inputs,
                options,
                control: control.clone(),
                model: RequestedModel::new(model_id.to_owned())?,
                engine: Arc::clone(&slot.engine),
                lease,
                queue,
                max_tokens: self.0.policy.max_tokens,
                max_batch_tokens: self.0.policy.max_batch_tokens,
            },
            abandonment.clone(),
            self.0.force_stop.subscribe(),
            effective_deadline,
        )
        .await?;
        let (response, receive) = oneshot::channel();
        let request = Request {
            sequence: prepared.sequence,
            inputs: prepared.inputs,
            cost: prepared.cost,
            contract: prepared.contract,
            options,
            control: control.clone(),
            response,
            _lease: prepared.lease,
            queue: Some(prepared.queue),
        };
        slot.sender.try_send(request).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => EngineFailure::public(ErrorCode::QueueFull),
            mpsc::error::TrySendError::Closed(_) => {
                EngineFailure::public(ErrorCode::ModelUnavailable)
            }
        })?;
        await_controlled(
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
        // An async function may be created under Tokio and first polled elsewhere. Resolve the
        // executor before acquiring lifecycle locks or publishing Draining so that unsupported
        // polling contexts fail atomically instead of panicking in `spawn`.
        let Ok(executor) = tokio::runtime::Handle::try_current() else {
            return false;
        };
        let requested_deadline = Instant::now().checked_add(timeout);
        let mut completion = {
            let Ok(admin) = self.0.admin.lock() else {
                return false;
            };
            let Ok(mut flight) = self.0.shutdown_flight.lock() else {
                return false;
            };
            if let Some(existing) = flight.as_ref() {
                existing.deadline.send_if_modified(|current| {
                    let tightened = match (*current, requested_deadline) {
                        (None, Some(requested)) => Some(requested),
                        (Some(current), Some(requested)) => Some(current.min(requested)),
                        (current, None) => current,
                    };
                    if tightened == *current {
                        false
                    } else {
                        *current = tightened;
                        true
                    }
                });
                existing.completion.clone()
            } else {
                self.0.draining.store(true, Ordering::Release);
                let _ = self
                    .0
                    .health
                    .transition(LifecycleState::Draining, Some(ReadinessReason::Draining));
                self.0.shutdown.begin();
                let initially_drained = self.0.admitted.load(Ordering::Acquire) == 0;
                let (finished, completion) = watch::channel(None);
                let (deadline, deadline_updates) = watch::channel(requested_deadline);
                let runtime = Arc::downgrade(&self.0);
                let task = executor.spawn(async move {
                    let result =
                        Self::finish_shutdown(runtime, deadline_updates, initially_drained).await;
                    let _ = finished.send(Some(result));
                });
                *flight = Some(ShutdownFlight {
                    completion: completion.clone(),
                    deadline,
                    task,
                });
                drop(admin);
                completion
            }
        };

        loop {
            if let Some(result) = *completion.borrow() {
                return result;
            }
            if completion.changed().await.is_err() {
                // The tracked task retains its sender until it records a terminal result. A
                // closed channel here means the task failed unexpectedly; keep the public API
                // bounded and conservative.
                return false;
            }
        }
    }

    async fn finish_shutdown(
        runtime: Weak<Inner>,
        mut deadline: watch::Receiver<Option<Instant>>,
        initially_drained: bool,
    ) -> bool {
        let Some(inner) = runtime.upgrade() else {
            return false;
        };
        let admitted = Arc::clone(&inner.admitted);
        let admitted_zero = Arc::clone(&inner.admitted_zero);
        drop(inner);
        let drained = if initially_drained {
            true
        } else {
            wait_for_drain(&admitted, &admitted_zero, &mut deadline).await
        };
        let Some(inner) = runtime.upgrade() else {
            return false;
        };
        if !drained {
            if let Ok(installs) = inner.installs.lock() {
                for install in installs.values() {
                    install.cancel_before_commit();
                }
            }
            // Publish the forced-stop classification before model cancellation wakes request
            // futures. This keeps terminal shutdown distinct from caller-requested cancellation.
            let _ = inner.force_stop.send(true);
        }
        if let Ok(models) = inner.models.read() {
            for slot in models.values() {
                slot.accepting.store(false, Ordering::Release);
                slot.cancellation.cancel();
            }
        }
        // Forced shutdown is a lifecycle fence, not a demand that arbitrary retained futures be
        // dropped. Existing RAII guards remain valid and clean up normally whenever their owners
        // resume or disappear, but they cannot hold the public service in Draining indefinitely.
        inner.shutdown.force_stop();
        let mut tasks = inner
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
        if let Ok(mut retired) = inner.retired_schedulers.lock() {
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
        // Scheduler fencing precedes registry release so no scheduler can retain a model session
        // or verified-store lease after terminal shutdown. Draining rejects all concurrent admin
        // work, and the admin lock serializes this final publication with any operation admitted
        // immediately before the boundary.
        if let Ok(_admin) = inner.admin.lock() {
            if let Ok(mut models) = inner.models.write() {
                models.clear();
            }
            if let Ok(mut loading) = inner.loading.lock() {
                loading.clear();
            }
        }
        inner.health.clear_models();
        let _ = inner
            .health
            .transition(LifecycleState::Stopped, Some(ReadinessReason::Stopped));
        drained
    }
}

async fn receive_install_result<T>(
    receive: &mut oneshot::Receiver<Result<T, LifecycleError>>,
    force_stop: &mut watch::Receiver<bool>,
    caller: &InstallCallerGuard,
) -> Result<T, LifecycleError> {
    tokio::select! {
        biased;
        result = &mut *receive => result.unwrap_or(Err(LifecycleError::InstallFailed)),
        changed = force_stop.changed() => {
            let _ = changed;
            match caller.cancel_before_commit() {
                CommitDecision::CancelledBeforeCommit => Err(LifecycleError::ShuttingDown),
                CommitDecision::AlreadyCommitted => {
                    (&mut *receive).await.unwrap_or(Err(LifecycleError::InstallFailed))
                }
            }
        }
    }
}

async fn wait_for_drain(
    admitted: &AtomicUsize,
    zero_latch: &Notify,
    deadline: &mut watch::Receiver<Option<Instant>>,
) -> bool {
    loop {
        // Register first, then check the predicate. The stored `notify_one` permit closes the
        // final-completion race without periodically waking or consuming a CPU core.
        let notified = zero_latch.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let current_deadline = *deadline.borrow();
        if current_deadline.is_some_and(|at| Instant::now() >= at) {
            return false;
        }
        if admitted.load(Ordering::Acquire) == 0 {
            return true;
        }
        if let Some(at) = current_deadline {
            let timer = tokio::time::sleep_until(tokio::time::Instant::from_std(at));
            tokio::pin!(timer);
            tokio::select! {
                biased;
                changed = deadline.changed() => {
                    if changed.is_err() {
                        return false;
                    }
                }
                () = &mut timer => {
                    return false;
                }
                () = &mut notified => {}
            }
        } else {
            tokio::select! {
                biased;
                changed = deadline.changed() => {
                    if changed.is_err() {
                        return false;
                    }
                }
                () = &mut notified => {}
            }
        }
    }
}

fn validate_raw_inputs(
    inputs: &[String],
    max_input_bytes: usize,
    max_request_bytes: usize,
) -> Result<(), EngineFailure> {
    let mut total = 0_usize;
    for input in inputs {
        let bytes = input.len();
        if bytes > max_input_bytes {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
        total = total
            .checked_add(bytes)
            .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?;
        if total > max_request_bytes {
            return Err(EngineFailure::public(ErrorCode::InvalidRequest));
        }
    }
    Ok(())
}

fn identity_is_complete(identity: &ResolvedModelIdentity) -> bool {
    [
        identity.canonical_id.as_str(),
        identity.revision.as_str(),
        identity.runtime.as_str(),
        identity.artifact_fingerprint.as_str(),
        identity.semantic_fingerprint.as_str(),
    ]
    .into_iter()
    .all(|value| !value.trim().is_empty())
}

fn validate_runtime_contract(
    requested: &RequestedModel,
    contract: &RuntimeModelContract,
) -> Result<(), EngineFailure> {
    if contract.native_dimensions == 0
        || contract.identity.canonical_id != requested.as_str()
        || !identity_is_complete(&contract.identity)
    {
        return Err(EngineFailure::public(ErrorCode::InferenceFailed));
    }
    Ok(())
}

async fn execute_preflight(
    pool: &BlockingPool,
    plan: PreflightPlan,
    abandonment: CancellationToken,
    force_stop: watch::Receiver<bool>,
    deadline: Instant,
) -> Result<PreflightSuccess, EngineFailure> {
    let control = plan.control.clone();
    let (send, receive) = oneshot::channel();
    let job = Box::new(move || {
        let result = catch_unwind(AssertUnwindSafe(|| {
            plan.control.ensure_active()?;
            let contract = plan.engine.model_contract(&plan.model)?;
            validate_runtime_contract(&plan.model, &contract)?;
            let batch = EmbeddingBatch::new(
                plan.inputs
                    .iter()
                    .map(|input| Cow::Borrowed(input.as_str())),
            )?;
            let cost = plan.engine.preflight(&plan.model, &batch, plan.options)?;
            plan.control.ensure_active()?;
            if cost.items() != plan.inputs.len()
                || cost.tokens() > plan.max_tokens
                || cost.padded_tokens()? > plan.max_batch_tokens
            {
                return Err(EngineFailure::public(ErrorCode::InvalidRequest));
            }
            Ok(PreflightSuccess {
                sequence: plan.sequence,
                inputs: plan.inputs,
                cost,
                contract,
                lease: plan.lease,
                queue: plan.queue,
            })
        }))
        .unwrap_or_else(|_| Err(EngineFailure::public(ErrorCode::InferenceFailed)));
        let _ = send.send(result);
    }) as BlockingJob;
    if pool.try_execute(job).is_err() {
        return Err(EngineFailure::public(ErrorCode::QueueFull));
    }
    await_controlled(control, abandonment, receive, force_stop, deadline).await
}

async fn await_controlled<T>(
    control: ExecutionControl,
    abandonment: CancellationToken,
    mut receive: oneshot::Receiver<Result<T, EngineFailure>>,
    mut force_stop: watch::Receiver<bool>,
    deadline: Instant,
) -> Result<T, EngineFailure> {
    let mut cancel_on_drop = CancelOnDrop {
        abandonment,
        armed: true,
    };
    let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
    tokio::pin!(sleep);
    let cancellation = control.cancellation().clone();
    let cancellation_event = cancellation.cancelled();
    tokio::pin!(cancellation_event);
    let result = loop {
        if *force_stop.borrow() {
            break Err(EngineFailure::public(ErrorCode::ModelUnavailable));
        }
        if let Err(error) = control.ensure_active() {
            break Err(error);
        }
        tokio::select! {
            biased;
            changed = force_stop.changed() => {
                if changed.is_err() || *force_stop.borrow() {
                    break Err(EngineFailure::public(ErrorCode::ModelUnavailable));
                }
            }
            () = &mut cancellation_event => {
                if *force_stop.borrow() {
                    break Err(EngineFailure::public(ErrorCode::ModelUnavailable));
                }
                break control.ensure_active().and_then(|()| {
                    Err(EngineFailure::public(ErrorCode::Cancelled))
                });
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
    'scheduler: loop {
        if *stop.borrow() {
            break 'scheduler;
        }
        if pending.is_empty() {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        break 'scheduler;
                    }
                }
                request = receiver.recv() => match request {
                    Some(request) => pending.push_back(request),
                    None => break 'scheduler,
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
                        break 'scheduler;
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
        pending
            .make_contiguous()
            .sort_by_key(|request| request.sequence);
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
            let mut max_sequence_length = requests[0].cost.max_sequence_length();
            let scan = pending.len();
            for _ in 0..scan {
                let Some(candidate) = pending.pop_front() else {
                    break;
                };
                let Some(combined_items) = items.checked_add(candidate.inputs.len()) else {
                    pending.push_back(candidate);
                    continue;
                };
                let combined_sequence =
                    max_sequence_length.max(candidate.cost.max_sequence_length());
                let combined_work = combined_items.checked_mul(combined_sequence);
                let compatible = CompatibilityKey(candidate.options) == key
                    && combined_items <= policy.max_batch_items
                    && combined_work.is_some_and(|work| work <= policy.max_batch_tokens);
                if compatible {
                    items = combined_items;
                    max_sequence_length = combined_sequence;
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
    receiver.close();
    fail_requests(pending.drain(..).collect(), ErrorCode::ModelUnavailable);
    while let Ok(request) = receiver.try_recv() {
        fail_requests(vec![request], ErrorCode::ModelUnavailable);
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
        Ok(Ok(output)) if valid_engine_output(&output, &requests, &sizes) => {
            publish_results(requests, sizes, output.vectors, &output.model, accepting);
        }
        Ok(Ok(_)) | Err(_) => fail_requests(requests, ErrorCode::InferenceFailed),
        Ok(Err(error)) => fail_requests(requests, error.public_error().code),
    }
}

fn valid_engine_output(output: &EmbeddingOutput, requests: &[Request], sizes: &[usize]) -> bool {
    let Some(total) = sizes
        .iter()
        .try_fold(0_usize, |sum, size| sum.checked_add(*size))
    else {
        return false;
    };
    if output.vectors.len() != total || !identity_is_complete(&output.model) {
        return false;
    }
    let mut offset = 0_usize;
    for (request, size) in requests.iter().zip(sizes) {
        if request.contract.identity != output.model {
            return false;
        }
        let expected_dimensions = request
            .options
            .dimensions
            .unwrap_or(request.contract.native_dimensions);
        let Some(end) = offset.checked_add(*size) else {
            return false;
        };
        let Some(vectors) = output.vectors.get(offset..end) else {
            return false;
        };
        if vectors.iter().any(|vector| {
            vector.len() != expected_dimensions || !vector.iter().all(|value| value.is_finite())
        }) {
            return false;
        }
        offset = end;
    }
    offset == total
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
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    struct CountingWake(AtomicUsize);

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn test_contract(
        model: &RequestedModel,
        native_dimensions: usize,
    ) -> Result<RuntimeModelContract, EngineFailure> {
        RuntimeModelContract::new(test_identity(model)?, native_dimensions)
    }

    fn test_identity(model: &RequestedModel) -> Result<ResolvedModelIdentity, EngineFailure> {
        ResolvedModelIdentity::new(model.as_str(), "rev", "fake@1", "artifact", "semantic")
    }

    struct FakeEngine {
        calls: AtomicUsize,
        finished: AtomicBool,
        block: Duration,
        fail: bool,
    }

    struct DropProbeEngine;

    impl RuntimeEngine for DropProbeEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn embed(
            &self,
            model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    struct CostAwareEngine {
        calls: AtomicUsize,
        observed_work: Mutex<Vec<usize>>,
        reject: Option<String>,
    }

    impl CostAwareEngine {
        fn new(reject: Option<&str>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                observed_work: Mutex::new(Vec::new()),
                reject: reject.map(str::to_owned),
            }
        }

        fn sequence_length(input: &str) -> Result<usize, EngineFailure> {
            input
                .strip_prefix('s')
                .and_then(|length| length.parse::<usize>().ok())
                .filter(|length| *length > 0)
                .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))
        }
    }

    impl RuntimeEngine for CostAwareEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn preflight(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _options: EmbedOptions,
        ) -> Result<EmbeddingBatchCost, EngineFailure> {
            let mut tokens = 0_usize;
            let mut longest = 0_usize;
            for input in batch.inputs() {
                if self.reject.as_deref() == Some(input.as_ref()) {
                    return Err(EngineFailure::public(ErrorCode::InvalidRequest));
                }
                let length = Self::sequence_length(input)?;
                tokens = tokens
                    .checked_add(length)
                    .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?;
                longest = longest.max(length);
            }
            EmbeddingBatchCost::new(batch.inputs().len(), tokens, longest)
        }

        fn embed(
            &self,
            model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            let longest = batch
                .inputs()
                .iter()
                .map(|input| Self::sequence_length(input))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(1);
            let work = batch
                .inputs()
                .len()
                .checked_mul(longest)
                .ok_or_else(|| EngineFailure::public(ErrorCode::InvalidRequest))?;
            self.observed_work
                .lock()
                .map_err(|_| EngineFailure::public(ErrorCode::Internal))?
                .push(work);
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    struct PanicOnceEngine(AtomicUsize);

    struct OrdinaryPanicOnceEngine(AtomicUsize);

    impl RuntimeEngine for OrdinaryPanicOnceEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn embed(
            &self,
            model: &RequestedModel,
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
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    impl RuntimeEngine for PanicOnceEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn embed(
            &self,
            model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            if self.0.fetch_add(1, Ordering::AcqRel) == 0 {
                std::panic::resume_unwind(Box::new("native fault"));
            }
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    struct CooperativeEngine;

    impl RuntimeEngine for CooperativeEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn embed(
            &self,
            model: &RequestedModel,
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
                model: test_identity(model)?,
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
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 2)
        }

        fn embed(
            &self,
            model: &RequestedModel,
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
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    struct PreflightProbe {
        preflights: AtomicUsize,
        embeds: AtomicUsize,
        delay: Duration,
        panic_once: bool,
    }

    impl RuntimeEngine for PreflightProbe {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 1)
        }

        fn preflight(
            &self,
            _model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _options: EmbedOptions,
        ) -> Result<EmbeddingBatchCost, EngineFailure> {
            let call = self.preflights.fetch_add(1, Ordering::AcqRel);
            if self.panic_once && call == 0 {
                std::panic::resume_unwind(Box::new("private tokenizer panic"));
            }
            thread::sleep(self.delay);
            EmbeddingBatchCost::new(batch.inputs().len(), batch.inputs().len(), 1)
        }

        fn embed(
            &self,
            model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            self.embeds.fetch_add(1, Ordering::AcqRel);
            Ok(EmbeddingOutput {
                vectors: batch.inputs().iter().map(|_| vec![1.0]).collect(),
                model: test_identity(model)?,
            })
        }

        fn warm(&self) -> Result<(), EngineFailure> {
            Ok(())
        }
    }

    #[derive(Clone, Copy)]
    enum HostileOutput {
        WrongIdentity,
        IncompleteIdentity,
        WrongCount,
        RaggedWidth,
        NonFinite,
        IgnoresRequestedWidth,
    }

    struct HostileEngine(HostileOutput);

    impl RuntimeEngine for HostileEngine {
        fn model_contract(
            &self,
            model: &RequestedModel,
        ) -> Result<RuntimeModelContract, EngineFailure> {
            test_contract(model, 2)
        }

        fn embed(
            &self,
            model: &RequestedModel,
            batch: &EmbeddingBatch<'_>,
            _control: &ExecutionControl,
            _options: EmbedOptions,
        ) -> Result<EmbeddingOutput, EngineFailure> {
            let mut identity = test_identity(model)?;
            let mut vectors = batch
                .inputs()
                .iter()
                .map(|_| vec![1.0, 2.0])
                .collect::<Vec<_>>();
            match self.0 {
                HostileOutput::WrongIdentity => identity.revision = "substituted".into(),
                HostileOutput::IncompleteIdentity => identity.runtime.clear(),
                HostileOutput::WrongCount => vectors.push(vec![1.0, 2.0]),
                HostileOutput::RaggedWidth => {
                    vectors[0].pop();
                }
                HostileOutput::NonFinite => vectors[0][0] = f32::NAN,
                HostileOutput::IgnoresRequestedWidth => {}
            }
            Ok(EmbeddingOutput {
                vectors,
                model: identity,
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
            max_input_bytes: 1024,
            max_request_bytes: 4096,
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
            max_input_bytes: 300,
            max_request_bytes: 600,
            max_tokens: 30,
            max_batch_items: 7,
            max_batch_tokens: 70,
            ..Limits::default()
        };
        let policy = BatchPolicy::from_limits(&limits, Duration::from_millis(9));
        assert_eq!(policy.max_items, 3);
        assert_eq!(policy.max_input_bytes, 300);
        assert_eq!(policy.max_request_bytes, 600);
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
        let mut invalid = policy;
        invalid.max_input_bytes = invalid.max_request_bytes + 1;
        assert!(matches!(
            ApplicationRuntime::new(invalid),
            Err(LifecycleError::InvalidPolicy)
        ));
    }

    #[test]
    fn blocking_pool_propagates_worker_spawn_failure() {
        let result = BlockingPool::new_with_spawner(1, 1, |_index, _receiver| {
            Err(std::io::Error::other("deliberate test failure"))
        });

        assert!(matches!(result, Err(LifecycleError::InitializationFailed)));
    }

    #[tokio::test]
    async fn polling_shutdown_without_tokio_fails_without_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let outside_runtime = runtime.clone();
        let polling = thread::spawn(move || {
            catch_unwind(AssertUnwindSafe(|| {
                let mut shutdown = Box::pin(outside_runtime.shutdown(Duration::ZERO));
                let waker = Waker::from(Arc::new(NoopWake));
                let mut context = Context::from_waker(&waker);
                Future::poll(shutdown.as_mut(), &mut context)
            }))
        })
        .join()
        .map_err(|_| "polling thread panicked")?;
        let poll = polling.map_err(|_| "shutdown panicked outside Tokio")?;

        assert!(matches!(poll, Poll::Ready(false)));
        assert!(!runtime.snapshot().draining);
        assert!(runtime.health().is_live());
        assert_eq!(runtime.health().model_counts(), (0, 0));
        assert!(runtime.shutdown(Duration::ZERO).await);
        assert!(!runtime.health().is_live());
        Ok(())
    }

    #[test]
    fn registration_without_tokio_is_a_stable_atomic_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))
        }));

        let Ok(registration) = result else {
            return Err("registration panicked without Tokio".into());
        };
        assert_eq!(registration, Err(LifecycleError::LoadFailed));
        assert_eq!(runtime.snapshot().registered_models, 0);
        assert_eq!(runtime.health().model_counts(), (0, 0));
        assert!(
            runtime
                .0
                .schedulers
                .lock()
                .is_ok_and(|schedulers| schedulers.is_empty())
        );
        Ok(())
    }

    #[test]
    fn completed_load_can_be_polled_outside_its_tokio_context()
    -> Result<(), Box<dyn std::error::Error>> {
        let executor = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let runtime = ApplicationRuntime::new(policy())?;
        let observer = runtime.clone();
        let (started, wait_started) = std::sync::mpsc::sync_channel(0);
        let (release, wait_release) = std::sync::mpsc::sync_channel(0);
        let mut load = Box::pin(async move {
            runtime
                .load_with("portable-poll".into(), move || {
                    started.send(()).map_err(|_| LifecycleError::LoadFailed)?;
                    wait_release
                        .recv()
                        .map_err(|_| LifecycleError::LoadFailed)?;
                    Ok(Arc::new(FakeEngine::new(Duration::ZERO)) as Arc<dyn RuntimeEngine>)
                })
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        {
            let _entered = executor.enter();
            assert!(matches!(load.as_mut().poll(&mut context), Poll::Pending));
        }
        wait_started.recv_timeout(Duration::from_secs(1))?;
        release.send(())?;

        let deadline = Instant::now() + Duration::from_secs(1);
        let result = loop {
            match load.as_mut().poll(&mut context) {
                Poll::Ready(result) => break result,
                Poll::Pending if Instant::now() < deadline => thread::yield_now(),
                Poll::Pending => return Err("load did not complete outside Tokio context".into()),
            }
        };
        assert_eq!(result, Ok(()));
        assert_eq!(observer.snapshot().registered_models, 1);
        assert_eq!(observer.health().model_counts(), (1, 1));
        assert!(
            observer
                .0
                .schedulers
                .lock()
                .is_ok_and(|schedulers| schedulers.len() == 1)
        );
        executor.block_on(async { assert!(observer.shutdown(Duration::from_secs(1)).await) });
        Ok(())
    }

    #[test]
    fn load_first_polled_without_tokio_fails_before_lifecycle_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let observer = runtime.clone();
        let load_ran = Arc::new(AtomicBool::new(false));
        let load_ran_in_job = Arc::clone(&load_ran);
        let mut load = Box::pin(async move {
            runtime
                .load_with("no-executor".into(), move || {
                    load_ran_in_job.store(true, Ordering::Release);
                    Ok(Arc::new(FakeEngine::new(Duration::ZERO)) as Arc<dyn RuntimeEngine>)
                })
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let result = catch_unwind(AssertUnwindSafe(|| load.as_mut().poll(&mut context)))
            .map_err(|_| "load panicked without Tokio")?;

        assert_eq!(result, Poll::Ready(Err(LifecycleError::LoadFailed)));
        assert!(!load_ran.load(Ordering::Acquire));
        assert_eq!(observer.snapshot().registered_models, 0);
        assert_eq!(observer.health().model_counts(), (0, 0));
        assert!(
            observer
                .0
                .loading
                .lock()
                .is_ok_and(|loading| loading.is_empty())
        );
        assert!(
            observer
                .0
                .schedulers
                .lock()
                .is_ok_and(|schedulers| schedulers.is_empty())
        );
        Ok(())
    }

    #[test]
    fn runtime_teardown_before_load_publication_is_atomic() -> Result<(), Box<dyn std::error::Error>>
    {
        let executor = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let runtime = ApplicationRuntime::new(policy())?;
        let observer = runtime.clone();
        let (started, wait_started) = std::sync::mpsc::sync_channel(0);
        let (release, wait_release) = std::sync::mpsc::sync_channel(0);
        let (finished, wait_finished) = std::sync::mpsc::sync_channel(0);
        let mut load = Box::pin(async move {
            runtime
                .load_with("torn-down-runtime".into(), move || {
                    started.send(()).map_err(|_| LifecycleError::LoadFailed)?;
                    wait_release
                        .recv()
                        .map_err(|_| LifecycleError::LoadFailed)?;
                    finished.send(()).map_err(|_| LifecycleError::LoadFailed)?;
                    Ok(Arc::new(FakeEngine::new(Duration::ZERO)) as Arc<dyn RuntimeEngine>)
                })
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        {
            let _entered = executor.enter();
            assert!(matches!(load.as_mut().poll(&mut context), Poll::Pending));
        }
        wait_started.recv_timeout(Duration::from_secs(1))?;
        drop(executor);
        release.send(())?;
        wait_finished.recv_timeout(Duration::from_secs(1))?;

        let deadline = Instant::now() + Duration::from_secs(1);
        let result = loop {
            match load.as_mut().poll(&mut context) {
                Poll::Ready(result) => break result,
                Poll::Pending if Instant::now() < deadline => thread::yield_now(),
                Poll::Pending => return Err("load remained pending after runtime teardown".into()),
            }
        };
        assert_eq!(result, Err(LifecycleError::LoadFailed));
        assert_eq!(observer.snapshot().registered_models, 0);
        assert_eq!(observer.health().model_counts(), (0, 0));
        assert!(
            observer
                .0
                .loading
                .lock()
                .is_ok_and(|loading| loading.is_empty())
        );
        assert!(
            observer
                .0
                .schedulers
                .lock()
                .is_ok_and(|schedulers| schedulers.is_empty())
        );
        Ok(())
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
    async fn invalid_preflight_is_isolated_from_concurrent_valid_callers()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::from_millis(25);
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(CostAwareEngine::new(Some("invalid")));
        runtime.register_engine("fake", engine.clone())?;

        let valid_a = runtime.embed(
            "fake",
            vec!["s2".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let invalid = runtime.embed(
            "fake",
            vec!["invalid".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let valid_b = runtime.embed(
            "fake",
            vec!["s3".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let (valid_a, invalid, valid_b) = tokio::join!(valid_a, invalid, valid_b);

        assert_eq!(valid_a?.vectors.len(), 1);
        assert_eq!(valid_b?.vectors.len(), 1);
        assert_eq!(
            invalid
                .err()
                .ok_or("invalid request must fail")?
                .public_error()
                .code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(engine.calls.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn heterogeneous_padding_never_exceeds_native_work_bound()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_items = 4;
        limits.max_tokens = 16;
        limits.max_batch_items = 4;
        limits.max_batch_tokens = 16;
        limits.max_batch_wait = Duration::from_millis(20);
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(CostAwareEngine::new(None));
        runtime.register_engine("fake", engine.clone())?;

        // Each request fits independently (3x1 and 1x10), and their non-padding sum is 13. A
        // sum-based scheduler would construct a 4x10 tensor under this 16-token cap; the
        // padded-work scheduler must split them at the configured 16-cell bound.
        let short = runtime.embed(
            "fake",
            vec!["s1".into(), "s1".into(), "s1".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let long = runtime.embed(
            "fake",
            vec!["s10".into()],
            EmbedOptions::default(),
            CancellationToken::default(),
            None,
        );
        let (short, long) = tokio::join!(short, long);
        assert_eq!(short?.vectors.len(), 3);
        assert_eq!(long?.vectors.len(), 1);
        assert_eq!(engine.calls.load(Ordering::Acquire), 2);
        assert!(
            engine
                .observed_work
                .lock()
                .is_ok_and(|work| work.iter().all(|value| *value <= 16))
        );
        Ok(())
    }

    #[tokio::test]
    async fn padding_bound_holds_across_adversarial_concurrent_lengths()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.queue_depth = 64;
        limits.max_items = 8;
        limits.max_tokens = 32;
        limits.max_batch_items = 8;
        limits.max_batch_tokens = 32;
        limits.max_batch_wait = Duration::from_millis(5);
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(CostAwareEngine::new(None));
        runtime.register_engine("fake", engine.clone())?;

        for round in 0..20_usize {
            let requests = (0..16_usize)
                .map(|index| {
                    let runtime = runtime.clone();
                    let length = 1 + ((index * 17 + round * 11) % 31);
                    tokio::spawn(async move {
                        runtime
                            .embed(
                                "fake",
                                vec![format!("s{length}")],
                                EmbedOptions::default(),
                                CancellationToken::default(),
                                None,
                            )
                            .await
                    })
                })
                .collect::<Vec<_>>();
            for request in requests {
                assert_eq!(request.await??.vectors.len(), 1);
            }
        }
        let work = engine
            .observed_work
            .lock()
            .map_err(|_| "poisoned work log")?;
        assert!(!work.is_empty());
        assert!(work.iter().all(|value| *value <= 32));
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
        if runtime.shutdown(Duration::ZERO).await {
            return Err("forced shutdown unexpectedly drained".into());
        }
        if !request.is_finished() {
            return Err("accepted request was not woken before shutdown returned".into());
        }
        if slow.finished.load(Ordering::Acquire) {
            return Err("native request unexpectedly finished before shutdown returned".into());
        }
        if !runtime
            .0
            .schedulers
            .lock()
            .is_ok_and(|schedulers| schedulers.is_empty())
        {
            return Err("scheduler registry was not cleared".into());
        }
        let error = tokio::time::timeout(Duration::from_millis(20), request)
            .await??
            .err()
            .ok_or("forced shutdown must fail accepted request")?;
        if error.public_error().code != ErrorCode::ModelUnavailable {
            return Err(format!("unexpected forced-shutdown error: {error:?}").into());
        }
        if runtime.health().is_live() {
            return Err("runtime remained live after shutdown".into());
        }
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

    #[tokio::test]
    async fn forced_shutdown_is_bounded_with_retained_embed_future()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine(
            "retained-embed",
            Arc::new(FakeEngine::new(Duration::from_millis(100))),
        )?;
        let request_runtime = runtime.clone();
        let mut request = Box::pin(async move {
            request_runtime
                .embed(
                    "retained-embed",
                    vec!["x".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(request.as_mut().poll(&mut context), Poll::Pending));

        let drained =
            tokio::time::timeout(Duration::from_millis(250), runtime.shutdown(Duration::ZERO))
                .await?;
        assert!(!drained);
        assert!(!runtime.health().is_live());
        // The future is intentionally retained and never repolled until after terminal shutdown.
        drop(request);
        Ok(())
    }

    #[tokio::test]
    async fn forced_shutdown_is_bounded_with_retained_load_future()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let observer = runtime.clone();
        let (started, wait_started) = std::sync::mpsc::sync_channel(0);
        let (release, wait_release) = std::sync::mpsc::sync_channel(0);
        let mut load = Box::pin(async move {
            runtime
                .load_with("retained-load".into(), move || {
                    started.send(()).map_err(|_| LifecycleError::LoadFailed)?;
                    wait_release
                        .recv()
                        .map_err(|_| LifecycleError::LoadFailed)?;
                    Ok(Arc::new(FakeEngine::new(Duration::ZERO)) as Arc<dyn RuntimeEngine>)
                })
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(load.as_mut().poll(&mut context), Poll::Pending));
        wait_started.recv_timeout(Duration::from_secs(1))?;

        let drained = tokio::time::timeout(
            Duration::from_millis(250),
            observer.shutdown(Duration::ZERO),
        )
        .await?;
        assert!(!drained);
        assert!(!observer.health().is_live());
        assert_eq!(observer.snapshot().registered_models, 0);

        release.send(())?;
        drop(load);
        Ok(())
    }

    #[tokio::test]
    async fn forced_shutdown_is_bounded_with_retained_install_future()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let install_runtime = runtime.clone();
        let mut install = Box::pin(async move {
            install_runtime
                .install_with(|_, _| std::future::pending::<Result<(), LifecycleError>>())
                .await
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(install.as_mut().poll(&mut context), Poll::Pending));

        let drained =
            tokio::time::timeout(Duration::from_millis(250), runtime.shutdown(Duration::ZERO))
                .await?;
        assert!(!drained);
        assert!(!runtime.health().is_live());
        drop(install);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forced_shutdown_cancels_owned_install_before_commit()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let install_runtime = runtime.clone();
        let (started, wait_started) = oneshot::channel();
        let install = tokio::spawn(async move {
            install_runtime
                .install_with(move |_, _| async move {
                    let _ = started.send(());
                    std::future::pending::<Result<(), LifecycleError>>().await
                })
                .await
        });
        wait_started.await?;

        if runtime.shutdown(Duration::ZERO).await {
            return Err("forced shutdown unexpectedly drained".into());
        }
        let install_result = install.await?;
        if install_result != Err(LifecycleError::ShuttingDown) {
            return Err(format!("unexpected pre-commit result: {install_result:?}").into());
        }
        if runtime.health().is_live() {
            return Err("runtime remained live after shutdown".into());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn forced_shutdown_preserves_truthful_result_after_install_commit()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let install_runtime = runtime.clone();
        let (committed, wait_committed) = oneshot::channel();
        let (release, wait_release) = oneshot::channel();
        let install = tokio::spawn(async move {
            install_runtime
                .install_with(move |_, commit| async move {
                    assert!(commit.begin_commit());
                    let _ = committed.send(());
                    let _ = wait_release.await;
                    Ok::<_, LifecycleError>(17_u8)
                })
                .await
        });
        wait_committed.await?;

        assert!(
            !tokio::time::timeout(Duration::from_millis(250), runtime.shutdown(Duration::ZERO))
                .await?
        );
        assert!(!runtime.health().is_live());
        release
            .send(())
            .map_err(|()| "install receiver disappeared")?;
        assert_eq!(install.await??, 17);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn already_installed_fast_path_completes_truthfully_at_shutdown_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let install_runtime = runtime.clone();
        let (started, wait_started) = oneshot::channel();
        let (release, wait_release) = oneshot::channel();
        let install = tokio::spawn(async move {
            install_runtime
                .install_with(move |_, _| async move {
                    let _ = started.send(());
                    let _ = wait_release.await;
                    // The real Installer returns this without crossing its commit gate when the
                    // exact identity is already installed and verified.
                    Ok::<_, LifecycleError>(ModelStatus::IntegrityVerified)
                })
                .await
        });
        wait_started.await?;

        let shutdown_runtime = runtime.clone();
        let shutdown =
            tokio::spawn(async move { shutdown_runtime.shutdown(Duration::from_secs(1)).await });
        tokio::task::yield_now().await;
        release
            .send(())
            .map_err(|()| "install operation disappeared")?;

        assert_eq!(install.await??, ModelStatus::IntegrityVerified);
        assert!(shutdown.await?);
        assert!(!runtime.health().is_live());
        Ok(())
    }

    #[tokio::test]
    async fn already_installed_result_wins_deterministic_force_stop_tie()
    -> Result<(), Box<dyn std::error::Error>> {
        let control = InstallControl {
            cancellation: InstallCancelToken::new(),
            commit: InstallCommitGate::new(),
        };
        let pending = tokio::spawn(std::future::pending::<()>());
        let mut caller = InstallCallerGuard {
            control,
            task: pending.abort_handle(),
            armed: true,
        };
        let (result_send, mut result_receive) = oneshot::channel();
        let (force_send, mut force_receive) = watch::channel(false);

        // Model the already-installed fast path: it has a terminal verified status without ever
        // crossing the durable commit gate. Both branches are ready before the selector is polled,
        // making this an exact deterministic check of result-vs-forced-shutdown precedence.
        result_send
            .send(Ok(ModelStatus::IntegrityVerified))
            .map_err(|_| "install result receiver disappeared")?;
        force_send
            .send(true)
            .map_err(|_| "force-stop receiver disappeared")?;
        let result =
            receive_install_result(&mut result_receive, &mut force_receive, &caller).await?;
        caller.armed = false;
        pending.abort();

        assert_eq!(result, ModelStatus::IntegrityVerified);
        assert!(!caller.control.cancellation.is_cancelled());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_latch_cannot_miss_final_completion() -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..500 {
            let admitted = Arc::new(AtomicUsize::new(1));
            let latch = Arc::new(Notify::new());
            let wait_admitted = Arc::clone(&admitted);
            let wait_latch = Arc::clone(&latch);
            let (_deadline_tx, mut deadline_rx) = watch::channel(None);
            let waiter = tokio::spawn(async move {
                wait_for_drain(&wait_admitted, &wait_latch, &mut deadline_rx).await
            });
            tokio::task::yield_now().await;
            let previous = admitted.fetch_sub(1, Ordering::AcqRel);
            assert_eq!(previous, 1);
            latch.notify_one();
            assert!(tokio::time::timeout(Duration::from_millis(50), waiter).await??);
        }
        Ok(())
    }

    #[tokio::test]
    async fn drain_wait_does_not_self_wake_or_busy_poll() -> Result<(), Box<dyn std::error::Error>>
    {
        let admitted = AtomicUsize::new(1);
        let latch = Notify::new();
        let (_deadline_tx, mut deadline_rx) = watch::channel(None);
        let mut drain = Box::pin(wait_for_drain(&admitted, &latch, &mut deadline_rx));
        let wake_counter = Arc::new(CountingWake(AtomicUsize::new(0)));
        let task_waker = Waker::from(Arc::clone(&wake_counter));
        let mut context = Context::from_waker(&task_waker);

        assert!(matches!(drain.as_mut().poll(&mut context), Poll::Pending));
        thread::sleep(Duration::from_millis(20));
        assert_eq!(wake_counter.0.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[tokio::test]
    async fn request_cancellation_is_event_driven_without_wake_amplification()
    -> Result<(), Box<dyn std::error::Error>> {
        let caller = CancellationToken::default();
        let abandonment = CancellationToken::default();
        let model = CancellationToken::default();
        let combined =
            CancellationToken::any(vec![caller.clone(), abandonment.clone(), model.clone()]);
        let control = ExecutionControl::new(combined, None);
        let (_response, receive) = oneshot::channel::<Result<(), EngineFailure>>();
        let (_force_stop, force_stop) = watch::channel(false);
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut wait = Box::pin(await_controlled(
            control,
            abandonment,
            receive,
            force_stop,
            deadline,
        ));
        let wake_counter = Arc::new(CountingWake(AtomicUsize::new(0)));
        let task_waker = Waker::from(Arc::clone(&wake_counter));
        let mut context = Context::from_waker(&task_waker);

        assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
        thread::sleep(Duration::from_millis(20));
        assert_eq!(wake_counter.0.load(Ordering::Acquire), 0);
        caller.cancel();
        assert_eq!(wake_counter.0.load(Ordering::Acquire), 1);
        let Poll::Ready(result) = wait.as_mut().poll(&mut context) else {
            return Err("cancellation event did not complete the request".into());
        };
        assert_eq!(
            result
                .err()
                .ok_or("cancelled request succeeded")?
                .public_error()
                .code,
            ErrorCode::Cancelled
        );
        assert!(!model.is_cancelled());
        Ok(())
    }

    #[tokio::test]
    async fn closed_scheduler_stop_channel_is_terminal() -> Result<(), Box<dyn std::error::Error>> {
        let limits = policy();
        let pool = BlockingPool::new(1, limits.queue_depth)?;
        let (request_sender, request_receiver) = mpsc::channel(limits.queue_depth);
        let accepting = Arc::new(AtomicBool::new(true));
        let (stop, stop_receiver) = watch::channel(false);
        let scheduler = tokio::spawn(run_scheduler(
            request_receiver,
            "closed-stop".into(),
            Arc::new(DropProbeEngine),
            limits,
            pool,
            accepting,
            stop_receiver,
        ));
        drop(stop);
        tokio::time::timeout(Duration::from_millis(100), scheduler).await??;
        assert!(request_sender.is_closed());
        Ok(())
    }

    #[tokio::test]
    async fn final_runtime_owner_releases_scheduler_and_engine()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let engine: Arc<dyn RuntimeEngine> = Arc::new(DropProbeEngine);
        let engine_weak = Arc::downgrade(&engine);
        runtime.register_engine("drop-probe", Arc::clone(&engine))?;
        drop(engine);

        let last_owner = runtime.clone();
        drop(runtime);
        assert!(engine_weak.upgrade().is_some());
        drop(last_owner);
        tokio::time::timeout(Duration::from_millis(250), async {
            while engine_weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn terminal_shutdown_clears_registry_and_releases_engine()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let engine: Arc<dyn RuntimeEngine> = Arc::new(DropProbeEngine);
        let engine_weak = Arc::downgrade(&engine);
        runtime.register_engine("shutdown-probe", Arc::clone(&engine))?;
        drop(engine);

        assert!(runtime.shutdown(Duration::from_millis(100)).await);
        assert_eq!(runtime.snapshot().registered_models, 0);
        assert_eq!(runtime.snapshot().ready_models, 0);
        assert!(
            runtime
                .0
                .loading
                .lock()
                .is_ok_and(|loading| loading.is_empty())
        );
        tokio::time::timeout(Duration::from_millis(250), async {
            while engine_weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await?;
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
                .install_with(|_, _| std::future::pending::<Result<(), LifecycleError>>())
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
        assert_eq!(runtime.health().model_counts(), (0, 0));
        assert_eq!(runtime.snapshot().ready_models, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn later_shutdown_caller_tightens_the_shared_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine("fake", Arc::new(FakeEngine::new(Duration::ZERO)))?;
        let permit = runtime.admit_work().ok_or("test work was not admitted")?;
        let mut forced = runtime.0.force_stop.subscribe();

        let long_runtime = runtime.clone();
        let long = tokio::spawn(async move { long_runtime.shutdown(Duration::from_secs(5)).await });
        while !runtime.snapshot().draining {
            tokio::task::yield_now().await;
        }
        let zero_runtime = runtime.clone();
        let zero = tokio::spawn(async move { zero_runtime.shutdown(Duration::ZERO).await });

        tokio::time::timeout(Duration::from_millis(250), async {
            while !*forced.borrow() {
                if forced.changed().await.is_err() {
                    break;
                }
            }
        })
        .await?;
        assert!(*forced.borrow());
        drop(permit);

        let (long_result, zero_result) = tokio::time::timeout(Duration::from_millis(250), async {
            tokio::join!(long, zero)
        })
        .await?;
        assert!(!long_result?);
        assert!(!zero_result?);
        assert_eq!(runtime.health().model_counts(), (0, 0));
        assert_eq!(runtime.snapshot().ready_models, 0);
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_survives_first_caller_abort_and_retry()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let permit = runtime.admit_work().ok_or("test work was not admitted")?;

        let first_runtime = runtime.clone();
        let first =
            tokio::spawn(async move { first_runtime.shutdown(Duration::from_millis(20)).await });
        while !runtime.snapshot().draining {
            tokio::task::yield_now().await;
        }
        first.abort();
        let _ = first.await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(permit);

        let result = tokio::time::timeout(
            Duration::from_millis(250),
            runtime.shutdown(Duration::from_secs(5)),
        )
        .await?;
        if result {
            return Err("retry changed forced shutdown into graceful shutdown".into());
        }
        if runtime.health().is_live() {
            return Err("runtime remained live after shutdown retry".into());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_shutdown_callers_share_one_terminal_result()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = ApplicationRuntime::new(policy())?;
        let permit = runtime.admit_work().ok_or("test work was not admitted")?;

        let first_runtime = runtime.clone();
        let second_runtime = runtime.clone();
        let first =
            tokio::spawn(async move { first_runtime.shutdown(Duration::from_millis(20)).await });
        let second =
            tokio::spawn(async move { second_runtime.shutdown(Duration::from_millis(20)).await });
        while !runtime.snapshot().draining {
            tokio::task::yield_now().await;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(permit);
        let (first, second) = tokio::join!(first, second);

        if first? || second? {
            return Err("concurrent forced shutdown reported a graceful drain".into());
        }
        if runtime.health().is_live() {
            return Err("runtime remained live after concurrent shutdown".into());
        }
        if runtime.shutdown(Duration::ZERO).await {
            return Err("repeated shutdown changed the terminal result".into());
        }
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

    #[tokio::test(flavor = "current_thread")]
    async fn preflight_never_blocks_the_tokio_executor_and_panics_are_isolated()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_batch_wait = Duration::ZERO;
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(PreflightProbe {
            preflights: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
            delay: Duration::from_millis(200),
            panic_once: false,
        });
        runtime.register_engine("fake", engine.clone())?;
        let request_runtime = runtime.clone();
        let request = tokio::spawn(async move {
            request_runtime
                .embed(
                    "fake",
                    vec!["small".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
        });
        while engine.preflights.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        let started = Instant::now();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(started.elapsed() < Duration::from_millis(50));
        assert_eq!(request.await??.vectors.len(), 1);

        let panic_runtime = ApplicationRuntime::new(limits)?;
        let panic_engine = Arc::new(PreflightProbe {
            preflights: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
            delay: Duration::ZERO,
            panic_once: true,
        });
        panic_runtime.register_engine("fake", panic_engine.clone())?;
        let first = panic_runtime
            .embed(
                "fake",
                vec!["small".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("preflight panic must fail")?;
        assert_eq!(first.public_error().code, ErrorCode::InferenceFailed);
        assert_eq!(
            panic_runtime
                .embed(
                    "fake",
                    vec!["small".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await?
                .vectors
                .len(),
            1
        );
        assert_eq!(panic_engine.embeds.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn raw_utf8_limits_reject_before_tokenization_even_when_truncating()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.max_input_bytes = 4;
        limits.max_request_bytes = 6;
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(PreflightProbe {
            preflights: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
            delay: Duration::ZERO,
            panic_once: false,
        });
        runtime.register_engine("fake", engine.clone())?;
        for inputs in [
            vec!["ééé".into()],
            vec!["abcd".into(), "abc".into()],
            vec!["x".repeat(1_000_000)],
        ] {
            let error = runtime
                .embed(
                    "fake",
                    inputs,
                    EmbedOptions {
                        truncation: impossible_embedding_core::Truncation::Truncate,
                        ..EmbedOptions::default()
                    },
                    CancellationToken::default(),
                    None,
                )
                .await
                .err()
                .ok_or("oversized request must fail")?;
            assert_eq!(error.public_error().code, ErrorCode::InvalidRequest);
        }
        assert_eq!(engine.preflights.load(Ordering::Acquire), 0);
        assert_eq!(
            runtime
                .embed(
                    "fake",
                    vec!["abc".into(), "def".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await?
                .vectors
                .len(),
            2
        );
        assert_eq!(engine.preflights.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn deadline_during_preflight_releases_accounting_and_never_reaches_inference()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut limits = policy();
        limits.request_timeout = Duration::from_millis(15);
        let runtime = ApplicationRuntime::new(limits)?;
        let engine = Arc::new(PreflightProbe {
            preflights: AtomicUsize::new(0),
            embeds: AtomicUsize::new(0),
            delay: Duration::from_millis(200),
            panic_once: false,
        });
        runtime.register_engine("fake", engine.clone())?;
        let started = Instant::now();
        let error = runtime
            .embed(
                "fake",
                vec!["small".into()],
                EmbedOptions::default(),
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("deadline must fail preflight")?;
        assert_eq!(error.public_error().code, ErrorCode::DeadlineExceeded);
        assert!(started.elapsed() < Duration::from_millis(100));
        tokio::time::sleep(Duration::from_millis(220)).await;
        assert_eq!(engine.embeds.load(Ordering::Acquire), 0);
        assert!(
            runtime
                .metrics()
                .render()
                .contains("impossible_queue_depth 0")
        );
        assert_eq!(runtime.0.admitted.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[tokio::test]
    async fn hostile_success_outputs_are_rejected_at_the_runtime_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        for mode in [
            HostileOutput::WrongIdentity,
            HostileOutput::IncompleteIdentity,
            HostileOutput::WrongCount,
            HostileOutput::RaggedWidth,
            HostileOutput::NonFinite,
        ] {
            let runtime = ApplicationRuntime::new(policy())?;
            runtime.register_engine("fake", Arc::new(HostileEngine(mode)))?;
            let error = runtime
                .embed(
                    "fake",
                    vec!["a".into()],
                    EmbedOptions::default(),
                    CancellationToken::default(),
                    None,
                )
                .await
                .err()
                .ok_or("hostile adapter output must fail")?;
            assert_eq!(error.public_error().code, ErrorCode::InferenceFailed);
        }

        let runtime = ApplicationRuntime::new(policy())?;
        runtime.register_engine(
            "fake",
            Arc::new(HostileEngine(HostileOutput::IgnoresRequestedWidth)),
        )?;
        let error = runtime
            .embed(
                "fake",
                vec!["a".into()],
                EmbedOptions {
                    dimensions: Some(1),
                    ..EmbedOptions::default()
                },
                CancellationToken::default(),
                None,
            )
            .await
            .err()
            .ok_or("wrong requested width must fail")?;
        assert_eq!(error.public_error().code, ErrorCode::InferenceFailed);
        Ok(())
    }
}

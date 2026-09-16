//! Process-level ownership for the HTTP and gRPC listeners.

use std::{fmt, net::SocketAddr, time::Duration};

use tokio::{net::TcpListener, task::JoinHandle};

use crate::{
    AppState, AppStateError, ApplicationError, GrpcServerError, GrpcServerHandle, StartupReport,
    grpc::spawn_grpc_server, http,
};
use impossible_server_core::ServerConfig;

/// A sanitized process-assembly failure.
#[derive(Debug)]
pub enum ProcessError {
    /// Shared application state or credentials could not be initialized.
    ApplicationState(AppStateError),
    /// The HTTP address could not be reserved.
    HttpBind,
    /// The gRPC address could not be reserved.
    GrpcBind,
    /// An explicitly configured strict preload failed.
    Preload(ApplicationError),
    /// The HTTP transport failed.
    HttpTransport,
    /// The HTTP task terminated abnormally.
    HttpJoin,
    /// The gRPC transport failed.
    Grpc(GrpcServerError),
    /// The operating-system shutdown signal could not be observed.
    Signal,
    /// A transport stopped before process shutdown was requested.
    UnexpectedTransportStop,
    /// Graceful shutdown exceeded the configured bound.
    ShutdownTimeout,
}

impl fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ApplicationState(_) => "server initialization failed",
            Self::HttpBind => "HTTP listener could not be bound",
            Self::GrpcBind => "gRPC listener could not be bound",
            Self::Preload(_) => "configured model preload failed",
            Self::HttpTransport => "HTTP transport failed",
            Self::HttpJoin => "HTTP transport task failed",
            Self::Grpc(_) => "gRPC transport failed",
            Self::Signal => "shutdown signal handling failed",
            Self::UnexpectedTransportStop => "a server transport stopped unexpectedly",
            Self::ShutdownTimeout => "server shutdown exceeded its deadline",
        })
    }
}

impl std::error::Error for ProcessError {}

impl From<AppStateError> for ProcessError {
    fn from(error: AppStateError) -> Self {
        Self::ApplicationState(error)
    }
}

impl From<GrpcServerError> for ProcessError {
    fn from(error: GrpcServerError) -> Self {
        Self::Grpc(error)
    }
}

/// Running HTTP and gRPC transports with one shared application and shutdown owner.
pub struct ServerHost {
    state: AppState,
    http_addr: SocketAddr,
    grpc_addr: SocketAddr,
    shutdown_timeout: Duration,
    http: Option<JoinHandle<Result<(), ProcessError>>>,
    grpc: Option<GrpcServerHandle>,
    preload: StartupReport,
}

impl fmt::Debug for ServerHost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerHost")
            .field("http_addr", &"[REDACTED]")
            .field("grpc_addr", &"[REDACTED]")
            .field("running", &self.http.is_some())
            .finish_non_exhaustive()
    }
}

impl ServerHost {
    /// Initialize shared state, reserve both listeners, preload only installed models, and start
    /// both transports. No background transport is published unless both binds succeed.
    ///
    /// # Errors
    /// Returns a sanitized initialization, bind, preload, or transport setup failure.
    pub async fn start(config: ServerConfig) -> Result<Self, ProcessError> {
        let state = AppState::new(&config)?;
        let http_listener = TcpListener::bind(config.http_bind)
            .await
            .map_err(|_| ProcessError::HttpBind)?;
        let http_addr = http_listener
            .local_addr()
            .map_err(|_| ProcessError::HttpBind)?;
        let grpc_listener = TcpListener::bind(config.grpc_bind)
            .await
            .map_err(|_| ProcessError::GrpcBind)?;
        let grpc_addr = grpc_listener
            .local_addr()
            .map_err(|_| ProcessError::GrpcBind)?;

        let preload = match state.application().preload().await {
            Ok(report) => report,
            Err(error) => {
                let _ = state.application().shutdown().await;
                return Err(ProcessError::Preload(error));
            }
        };

        // Reflection is the only fallible gRPC setup step. Complete it before the infallible HTTP
        // task spawn so a startup failure cannot leave a detached listener behind.
        let grpc = spawn_grpc_server(grpc_listener, state.clone()).await?;
        let shutdown = state.shutdown_trigger().clone();
        let router = http::router(state.clone());
        let http = tokio::spawn(async move {
            let shutdown_signal = async move { shutdown.cancelled().await };
            axum::serve(http_listener, router)
                .with_graceful_shutdown(shutdown_signal)
                .await
                .map_err(|_| ProcessError::HttpTransport)
        });

        Ok(Self {
            state,
            http_addr,
            grpc_addr,
            shutdown_timeout: config.limits.shutdown_timeout,
            http: Some(http),
            grpc: Some(grpc),
            preload,
        })
    }

    /// Bound HTTP address, including an operating-system-selected ephemeral port.
    #[must_use]
    pub const fn http_addr(&self) -> SocketAddr {
        self.http_addr
    }

    /// Bound gRPC address, including an operating-system-selected ephemeral port.
    #[must_use]
    pub const fn grpc_addr(&self) -> SocketAddr {
        self.grpc_addr
    }

    /// Shared process state used by both listeners.
    #[must_use]
    pub const fn state(&self) -> &AppState {
        &self.state
    }

    /// Privacy-safe preload outcomes produced before either listener became ready.
    #[must_use]
    pub const fn preload_report(&self) -> &StartupReport {
        &self.preload
    }

    /// Wait for Ctrl-C, SIGTERM where supported, or an unexpected transport termination, then
    /// run the single bounded shutdown path.
    ///
    /// # Errors
    /// Returns a sanitized signal, transport, or shutdown failure.
    pub async fn run_until_signal(self) -> Result<(), ProcessError> {
        let mut signal = Box::pin(shutdown_signal());
        let unexpected = loop {
            tokio::select! {
                result = &mut signal => {
                    result?;
                    break false;
                }
                () = tokio::time::sleep(Duration::from_millis(50)) => {
                    let http_finished = self.http.as_ref().is_none_or(JoinHandle::is_finished);
                    let grpc_finished = self.grpc.as_ref().is_none_or(GrpcServerHandle::is_finished);
                    if http_finished || grpc_finished {
                        break true;
                    }
                }
            }
        };
        let shutdown = self.shutdown().await;
        if unexpected && shutdown.is_ok() {
            Err(ProcessError::UnexpectedTransportStop)
        } else {
            shutdown
        }
    }

    /// Trigger shutdown exactly once and concurrently drain the application and both transports.
    ///
    /// # Errors
    /// Returns a sanitized transport failure or deadline overrun.
    pub async fn shutdown(mut self) -> Result<(), ProcessError> {
        self.state.shutdown_trigger().trigger();
        let timeout = self.shutdown_timeout;
        let application = self.state.application().shutdown();
        let http = finish_http(self.http.take(), timeout);
        let grpc = finish_grpc(self.grpc.take());
        let (application_drained, http_result, grpc_result) = tokio::join!(application, http, grpc);
        http_result?;
        grpc_result?;
        if application_drained {
            Ok(())
        } else {
            Err(ProcessError::ShutdownTimeout)
        }
    }
}

impl Drop for ServerHost {
    fn drop(&mut self) {
        self.state.shutdown_trigger().trigger();
        if let Some(http) = self.http.take() {
            http.abort();
        }
        // GrpcServerHandle::drop requests shutdown and aborts its task.
        let _ = self.grpc.take();
    }
}

async fn finish_http(
    task: Option<JoinHandle<Result<(), ProcessError>>>,
    timeout: Duration,
) -> Result<(), ProcessError> {
    let Some(mut task) = task else {
        return Ok(());
    };
    match tokio::time::timeout(timeout, &mut task).await {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => Err(ProcessError::HttpJoin),
        Err(_) => {
            task.abort();
            let _ = task.await;
            Err(ProcessError::ShutdownTimeout)
        }
    }
}

async fn finish_grpc(server: Option<GrpcServerHandle>) -> Result<(), ProcessError> {
    match server {
        Some(server) => server.shutdown().await.map_err(Into::into),
        None => Ok(()),
    }
}

async fn shutdown_signal() -> Result<(), ProcessError> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).map_err(|_| ProcessError::Signal)?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.map_err(|_| ProcessError::Signal),
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .map_err(|_| ProcessError::Signal)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    static FIXTURE: AtomicUsize = AtomicUsize::new(0);

    fn fixture(label: &str) -> std::path::PathBuf {
        let id = FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "impossible-process-{label}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("fixture directory");
        path
    }

    fn available_address() -> SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve address");
        listener.local_addr().expect("local address")
    }

    fn config(label: &str) -> ServerConfig {
        ServerConfig {
            http_bind: available_address(),
            grpc_bind: available_address(),
            cache_directory: fixture(label),
            ..ServerConfig::default()
        }
    }

    #[tokio::test]
    async fn partial_bind_failure_releases_the_first_listener() {
        let config = config("partial-bind");
        let occupied = TcpListener::bind(config.grpc_bind)
            .await
            .expect("occupy gRPC address");
        let http_address = config.http_bind;
        let result = ServerHost::start(config.clone()).await;
        assert!(matches!(result, Err(ProcessError::GrpcBind)));
        let rebound = TcpListener::bind(http_address)
            .await
            .expect("HTTP reservation released");
        drop(rebound);
        drop(occupied);
        let _ = fs::remove_dir_all(&config.cache_directory);
    }

    #[tokio::test]
    async fn listeners_share_readiness_and_repeated_shutdown_trigger_is_safe() {
        let config = config("readiness");
        let directory = config.cache_directory.clone();
        let host = ServerHost::start(config).await.expect("start host");
        assert!(!host.state().application().readiness().is_ready());
        assert_eq!(
            host.grpc_addr(),
            host.grpc.as_ref().expect("gRPC").local_addr()
        );

        let mut stream = tokio::net::TcpStream::connect(host.http_addr())
            .await
            .expect("connect HTTP");
        stream
            .write_all(
                b"GET /health/ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            )
            .await
            .expect("write request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read response");
        assert!(response.starts_with(b"HTTP/1.1 503"));

        host.state().shutdown_trigger().trigger();
        host.state().shutdown_trigger().trigger();
        host.shutdown().await.expect("bounded shutdown");
        let _ = fs::remove_dir_all(directory);
    }
}

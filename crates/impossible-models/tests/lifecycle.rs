//! End-to-end lifecycle tests using only loopback fixture servers and synthetic bytes.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use impossible_models::{
    Artifact, CancelToken, Dimensions, InstallOptions, Installer, License, Manifest, ModelStatus,
    ModelStore, Pooling, Prefixes, SemanticVerification, TensorMetadata, TokenizerMetadata,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use url::Url;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn fixture(url: String, bytes: &[u8], hash: Option<String>) -> Manifest {
    Manifest {
        schema_version: 1,
        canonical_id: "tests/tiny-model".into(),
        revision: "0123456789abcdef0123456789abcdef01234567".into(),
        license: License {
            spdx: "MIT".into(),
            source_url: "https://example.invalid/license".into(),
        },
        semantic_verification: SemanticVerification::Verified {
            evidence: "deterministic test fixture".into(),
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
        artifacts: vec![Artifact {
            path: "weights/model.bin".into(),
            url,
            sha256: hash.unwrap_or_else(|| format!("{:x}", Sha256::digest(bytes))),
            size: u64::try_from(bytes.len()).unwrap_or_default(),
        }],
    }
}

async fn server(
    body: Vec<u8>,
    pause_after: Option<usize>,
    requests: Arc<AtomicUsize>,
) -> TestResult<(Url, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            requests.fetch_add(1, Ordering::SeqCst);
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await;
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            if stream.write_all(header.as_bytes()).await.is_err() {
                continue;
            }
            if let Some(split) = pause_after {
                if stream.write_all(&body[..split]).await.is_err() {
                    continue;
                }
                let _ = stream.flush().await;
                tokio::time::sleep(Duration::from_millis(150)).await;
                let _ = stream.write_all(&body[split..]).await;
            } else {
                let _ = stream.write_all(&body).await;
            }
        }
    });
    Ok((Url::parse(&format!("http://{address}/artifact"))?, handle))
}

fn installer(temp: &TempDir, origin: Url, offline: bool) -> TestResult<Installer> {
    let store = ModelStore::new(temp.path())?;
    let options = InstallOptions {
        offline,
        allowed_origins: vec![origin],
        max_redirects: 1,
        max_artifact_bytes: 1024 * 1024,
    };
    Ok(Installer::new(store, options)?)
}

#[tokio::test]
async fn installs_and_promotes_only_after_hash_verification() -> TestResult {
    let temp = TempDir::new()?;
    let body = b"verified fixture".to_vec();
    let requests = Arc::new(AtomicUsize::new(0));
    let (url, task) = server(body.clone(), None, Arc::clone(&requests)).await?;
    let manifest = fixture(url.to_string(), &body, None);
    let status = installer(&temp, url, false)?
        .install(&manifest, &CancelToken::new())
        .await;
    assert!(matches!(status, Ok(ModelStatus::Loadable)), "{status:?}");
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    task.abort();
    Ok(())
}

#[tokio::test]
async fn hash_mismatch_never_promotes() -> TestResult {
    let temp = TempDir::new()?;
    let body = b"wrong content".to_vec();
    let requests = Arc::new(AtomicUsize::new(0));
    let (url, task) = server(body.clone(), None, requests).await?;
    let manifest = fixture(url.to_string(), &body, Some("a".repeat(64)));
    let result = installer(&temp, url, false)?
        .install(&manifest, &CancelToken::new())
        .await;
    assert!(matches!(
        result,
        Err(impossible_models::Error::HashMismatch { .. })
    ));
    let store = ModelStore::new(temp.path())?;
    assert_eq!(store.status(&manifest).ok(), Some(ModelStatus::Missing));
    task.abort();
    Ok(())
}

#[tokio::test]
async fn cancellation_leaves_only_partial_state_and_retry_recovers() -> TestResult {
    let temp = TempDir::new()?;
    let body = vec![42_u8; 64 * 1024];
    let requests = Arc::new(AtomicUsize::new(0));
    let (slow_url, slow_task) = server(body.clone(), Some(1024), Arc::clone(&requests)).await?;
    let manifest = fixture(slow_url.to_string(), &body, None);
    let cancel = CancelToken::new();
    let pending = {
        let installer = installer(&temp, slow_url, false)?;
        let manifest = manifest.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move { installer.install(&manifest, &cancel).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    cancel.cancel();
    assert!(matches!(
        pending.await,
        Ok(Err(impossible_models::Error::Cancelled))
    ));
    assert_eq!(
        ModelStore::new(temp.path())
            .and_then(|store| store.status(&manifest))
            .ok(),
        Some(ModelStatus::Missing)
    );
    assert!(
        temp.path()
            .join("staging")
            .read_dir()
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false)
    );
    slow_task.abort();

    let retry_requests = Arc::new(AtomicUsize::new(0));
    let (retry_url, retry_task) = server(body.clone(), None, retry_requests).await?;
    let retry_manifest = fixture(retry_url.to_string(), &body, None);
    assert!(
        matches!(
            installer(&temp, retry_url, false)?
                .install(&retry_manifest, &CancelToken::new())
                .await,
            Ok(ModelStatus::Loadable)
        ),
        "retry did not become loadable"
    );
    retry_task.abort();
    Ok(())
}

#[tokio::test]
async fn concurrent_install_is_coalesced_by_cross_process_lock() -> TestResult {
    let temp = TempDir::new()?;
    let body = vec![7_u8; 4096];
    let requests = Arc::new(AtomicUsize::new(0));
    let (url, task) = server(body.clone(), Some(128), Arc::clone(&requests)).await?;
    let manifest = fixture(url.to_string(), &body, None);
    let installer = installer(&temp, url, false)?;
    let left_cancel = CancelToken::new();
    let right_cancel = CancelToken::new();
    let (left, right) = tokio::join!(
        installer.install(&manifest, &left_cancel),
        installer.install(&manifest, &right_cancel)
    );
    assert!(matches!(left, Ok(ModelStatus::Loadable)), "{left:?}");
    assert!(matches!(right, Ok(ModelStatus::Loadable)), "{right:?}");
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    task.abort();
    Ok(())
}

#[tokio::test]
async fn offline_mode_performs_no_request() -> TestResult {
    let temp = TempDir::new()?;
    let url = Url::parse("http://127.0.0.1:9/artifact")?;
    let manifest = fixture(url.to_string(), b"unused", None);
    let result = installer(&temp, url, true)?
        .install(&manifest, &CancelToken::new())
        .await;
    assert!(matches!(result, Err(impossible_models::Error::Offline)));
    Ok(())
}

#[test]
fn explicit_import_verifies_and_exact_delete_requires_identity() -> TestResult {
    let temp = TempDir::new()?;
    let source = TempDir::new()?;
    let body = b"local model";
    let manifest = fixture("https://example.invalid/model".into(), body, None);
    let weights = source.path().join("weights");
    std::fs::create_dir_all(&weights)?;
    std::fs::write(weights.join("model.bin"), body)?;
    let store = ModelStore::new(temp.path())?;
    assert_eq!(
        store.import(&manifest, source.path()).ok(),
        Some(ModelStatus::Loadable)
    );
    let mut wrong = manifest.clone();
    wrong.revision = "abcdef0123456789abcdef0123456789abcdef01".into();
    assert_eq!(store.delete(&wrong).ok(), Some(false));
    assert_eq!(store.delete(&manifest).ok(), Some(true));
    Ok(())
}

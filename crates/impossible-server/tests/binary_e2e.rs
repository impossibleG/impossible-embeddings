//! Cross-platform, offline system test for the shipped process and every public transport.
//!
//! The `e2e-fixture` feature is intentionally required: it admits one generated manifest into the
//! child process catalog so CI can exercise real ONNX inference without downloading model weights.

#![cfg(feature = "e2e-fixture")]
#![allow(clippy::too_many_lines)]

use std::{fs, net::TcpListener, path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, Result, anyhow, ensure};
use impossible_models::{
    Artifact, Dimensions, License, Manifest, ModelStatus, ModelStore, OnnxInputNames, Pooling,
    Prefixes, RuntimeMetadata, SemanticTrustRoot, SemanticVerification, TensorMetadata,
    TokenizerMetadata, TrustedSemanticEvidence,
};
use impossible_protocol::v1::{
    EmbedRequest, EmbeddingTask, Truncation, embedding_service_client::EmbeddingServiceClient,
};
use prost::Message;
use reqwest::{Client, Response, StatusCode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokenizers::{
    Tokenizer, models::wordpiece::WordPiece, pre_tokenizers::whitespace::Whitespace,
    processors::template::TemplateProcessing,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
    time::{Instant, sleep, timeout},
};
use tonic::{Request, metadata::MetadataValue, transport::Endpoint};
use tonic_health::pb::{
    HealthCheckRequest, health_check_response::ServingStatus, health_client::HealthClient,
};
use tonic_reflection::pb::v1::{
    ServerReflectionRequest, server_reflection_client::ServerReflectionClient,
    server_reflection_request::MessageRequest, server_reflection_response::MessageResponse,
};

const MODEL_ID: &str = "fixture/cast-embedding";
const PUBLIC_TOKEN: &str = "fixture-public-credential";
const ADMIN_TOKEN: &str = "fixture-admin-credential";
const AUTHORIZATION: &str = "authorization";
const FIXTURE_SEMANTIC_FINGERPRINT: &str =
    "sha256:35880bb9026dc54cf7540cc5ee23e7ccba794b68830ba5330e03b53773b20bb2";

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
}

#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(int64, tag = "2")]
    version: i64,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "4")]
    op_type: String,
    #[prost(message, repeated, tag = "5")]
    attribute: Vec<AttributeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct AttributeProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(int64, tag = "3")]
    i: i64,
    #[prost(int32, tag = "20")]
    r#type: i32,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<TensorTypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    elem_type: i32,
    #[prost(message, optional, tag = "2")]
    shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
struct Dimension {
    #[prost(int64, tag = "1")]
    dim_value: i64,
    #[prost(string, tag = "2")]
    dim_param: String,
}

struct Fixture {
    _directory: TempDir,
    cache: PathBuf,
    manifest_path: PathBuf,
    model_path: PathBuf,
    private_marker: String,
}

struct RunningChild {
    child: Option<Child>,
}

impl RunningChild {
    async fn stop(mut self) -> Result<std::process::Output> {
        let mut child = self.child.take().context("child already stopped")?;
        child.start_kill().context("request child termination")?;
        timeout(Duration::from_secs(10), child.wait_with_output())
            .await
            .context("child shutdown exceeded bound")?
            .context("wait for child")
    }
}

impl Drop for RunningChild {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }
}

fn tensor_value(name: &str, element_type: i32) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: element_type,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        Dimension {
                            dim_value: 0,
                            dim_param: "batch".into(),
                        },
                        Dimension {
                            dim_value: 0,
                            dim_param: "sequence".into(),
                        },
                    ],
                }),
            }),
        }),
    }
}

fn output_value(name: &str, dimensions: i64) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        Dimension {
                            dim_value: 0,
                            dim_param: "batch".into(),
                        },
                        Dimension {
                            dim_value: dimensions,
                            dim_param: String::new(),
                        },
                    ],
                }),
            }),
        }),
    }
}

fn cast_model() -> ModelProto {
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: vec![
                NodeProto {
                    input: vec!["input_ids".into()],
                    output: vec!["cast_tokens".into()],
                    op_type: "Cast".into(),
                    attribute: vec![AttributeProto {
                        name: "to".into(),
                        i: 1,
                        r#type: 2,
                        ints: vec![],
                    }],
                },
                NodeProto {
                    input: vec!["cast_tokens".into()],
                    output: vec!["pooled".into()],
                    op_type: "ReduceMean".into(),
                    attribute: vec![
                        AttributeProto {
                            name: "axes".into(),
                            i: 0,
                            r#type: 7,
                            ints: vec![1],
                        },
                        AttributeProto {
                            name: "keepdims".into(),
                            i: 1,
                            r#type: 2,
                            ints: vec![],
                        },
                    ],
                },
                NodeProto {
                    input: vec!["pooled".into(); 4],
                    output: vec!["sentence_embedding".into()],
                    op_type: "Concat".into(),
                    attribute: vec![AttributeProto {
                        name: "axis".into(),
                        i: 1,
                        r#type: 2,
                        ints: vec![],
                    }],
                },
            ],
            name: "synthetic_embedding_fixture".into(),
            input: vec![
                tensor_value("input_ids", 7),
                tensor_value("attention_mask", 7),
            ],
            output: vec![output_value("sentence_embedding", 4)],
        }),
        opset_import: vec![OperatorSetIdProto { version: 13 }],
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_fixture() -> Result<Fixture> {
    let directory = tempfile::tempdir().context("fixture root")?;
    let source = directory.path().join("source");
    let cache = directory.path().join("cache");
    fs::create_dir_all(&source).context("fixture source")?;

    let model_path = source.join("model.onnx");
    let mut model = Vec::new();
    cast_model().encode(&mut model)?;
    fs::write(&model_path, &model).context("write ONNX fixture")?;

    let vocabulary = [
        ("[UNK]".to_owned(), 0),
        ("[CLS]".to_owned(), 1),
        ("[SEP]".to_owned(), 2),
        ("hello".to_owned(), 3),
        ("world".to_owned(), 4),
    ];
    let wordpiece = WordPiece::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .map_err(|error| anyhow!(error.to_string()))?;
    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .map_err(anyhow::Error::msg)?
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .map_err(anyhow::Error::msg)?,
    ));
    let tokenizer_path = source.join("tokenizer.json");
    tokenizer
        .save(&tokenizer_path, false)
        .map_err(|error| anyhow!(error.to_string()))?;
    let tokenizer_bytes = fs::read(&tokenizer_path).context("read tokenizer fixture")?;

    let manifest = Manifest {
        schema_version: 1,
        canonical_id: MODEL_ID.into(),
        revision: "1111111111111111111111111111111111111111".into(),
        license: License {
            spdx: "MIT".into(),
            source_url: "https://example.invalid/fixture".into(),
        },
        semantic_verification: SemanticVerification::Verified {
            evidence: "repository-authored-e2e-fixture".into(),
        },
        tokenizer: TokenizerMetadata {
            kind: "wordpiece".into(),
            max_tokens: 8,
            lowercase: false,
        },
        pooling: Pooling::Mean,
        prefixes: Prefixes {
            query: String::new(),
            document: String::new(),
        },
        dimensions: Dimensions {
            native: 4,
            matryoshka: vec![2],
        },
        tensors: TensorMetadata {
            format: "onnx".into(),
            dtype: "float32".into(),
            architecture: "synthetic-cast".into(),
        },
        runtime: RuntimeMetadata::Onnx {
            model_file: "model.onnx".into(),
            tokenizer_file: "tokenizer.json".into(),
            inputs: OnnxInputNames {
                input_ids: "input_ids".into(),
                attention_mask: Some("attention_mask".into()),
                token_type_ids: None,
            },
            output: "sentence_embedding".into(),
            pad_token_id: 0,
            normalize: false,
        },
        artifacts: vec![
            Artifact {
                path: "model.onnx".into(),
                url: "https://example.invalid/model.onnx".into(),
                sha256: sha256(&model),
                size: u64::try_from(model.len())?,
            },
            Artifact {
                path: "tokenizer.json".into(),
                url: "https://example.invalid/tokenizer.json".into(),
                sha256: sha256(&tokenizer_bytes),
                size: u64::try_from(tokenizer_bytes.len())?,
            },
        ],
    };
    let manifest_bytes = manifest.to_json()?;
    ensure!(manifest.semantic_fingerprint()? == FIXTURE_SEMANTIC_FINGERPRINT);
    let manifest_path = directory.path().join("fixture-manifest.json");
    fs::write(&manifest_path, manifest_bytes).context("write fixture manifest")?;

    let trust = SemanticTrustRoot::from_evidence([TrustedSemanticEvidence {
        manifest_fingerprint: manifest.semantic_fingerprint()?,
        evidence: "binary system test".into(),
    }])?;
    let store = ModelStore::with_trust_root(&cache, trust)?;
    ensure!(store.import(&manifest, &source)? == ModelStatus::Loadable);

    Ok(Fixture {
        _directory: directory,
        cache,
        manifest_path,
        model_path,
        private_marker: "private-input-must-not-be-logged".into(),
    })
}

fn ephemeral_address() -> Result<String> {
    let listener = TcpListener::bind("127.0.0.1:0").context("reserve ephemeral address")?;
    let address = listener.local_addr().context("read ephemeral address")?;
    drop(listener);
    Ok(address.to_string())
}

fn ephemeral_remote_http_address() -> Result<(String, String)> {
    let listener = TcpListener::bind("127.0.0.1:0").context("reserve remote HTTP port")?;
    let port = listener
        .local_addr()
        .context("read remote HTTP port")?
        .port();
    drop(listener);
    Ok((
        format!("0.0.0.0:{port}"),
        format!("http://127.0.0.1:{port}"),
    ))
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_impossible-embedding")
}

fn base_command(fixture: &Fixture) -> Command {
    let mut command = Command::new(binary());
    command
        .env("IE_E2E_MANIFEST", &fixture.manifest_path)
        .env("IE_E2E_PUBLIC_TOKEN", PUBLIC_TOKEN)
        .env("IE_E2E_ADMIN_TOKEN", ADMIN_TOKEN)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn common_config(command: &mut Command, fixture: &Fixture) {
    command
        .arg("--cache-directory")
        .arg(&fixture.cache)
        .arg("--offline")
        .arg("--auth-env")
        .arg("IE_E2E_PUBLIC_TOKEN")
        .arg("--admin-auth-env")
        .arg("IE_E2E_ADMIN_TOKEN")
        .arg("--admin-api-enabled")
        .arg("true")
        .arg("--shutdown-timeout-ms")
        .arg("3000");
}

async fn start_server(fixture: &Fixture) -> Result<(RunningChild, String, String)> {
    let http_address = ephemeral_address()?;
    let grpc_address = ephemeral_address()?;
    let mut command = base_command(fixture);
    command
        .arg("serve")
        .arg("--http-bind")
        .arg(&http_address)
        .arg("--grpc-bind")
        .arg(&grpc_address);
    common_config(&mut command, fixture);
    let child = command.spawn().context("spawn server binary")?;
    let mut running = RunningChild { child: Some(child) };
    let base = format!("http://{http_address}");
    if let Err(error) =
        wait_for_http(&mut running, &format!("{base}/health/live"), StatusCode::OK).await
    {
        let output = running.stop().await?;
        return Err(error.context(format!(
            "server output: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok((running, base, grpc_address))
}

async fn start_metrics_server(
    fixture: &Fixture,
    http_bind: &str,
    base: String,
    extra_arguments: &[&str],
) -> Result<RunningChild> {
    let grpc_address = ephemeral_address()?;
    let mut command = base_command(fixture);
    command
        .arg("serve")
        .arg("--http-bind")
        .arg(http_bind)
        .arg("--grpc-bind")
        .arg(grpc_address)
        .arg("--cache-directory")
        .arg(&fixture.cache)
        .arg("--offline")
        .arg("--admin-api-enabled")
        .arg("false")
        .arg("--shutdown-timeout-ms")
        .arg("3000")
        .args(extra_arguments);
    let child = command.spawn().context("spawn metrics policy server")?;
    let mut running = RunningChild { child: Some(child) };
    if let Err(error) =
        wait_for_http(&mut running, &format!("{base}/health/live"), StatusCode::OK).await
    {
        let output = running.stop().await?;
        return Err(error.context(format!(
            "metrics policy server output: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(running)
}

async fn wait_for_http(child: &mut RunningChild, url: &str, expected: StatusCode) -> Result<()> {
    let client = Client::new();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(process) = child.child.as_mut() {
            if let Some(status) = process.try_wait()? {
                return Err(anyhow!("server exited during startup with {status}"));
            }
        }
        if let Ok(response) = client.get(url).send().await {
            if response.status() == expected {
                return Ok(());
            }
        }
        ensure!(Instant::now() < deadline, "server startup exceeded bound");
        sleep(Duration::from_millis(25)).await;
    }
}

async fn body_json(response: Response) -> Result<Value> {
    let status = response.status();
    let bytes = response.bytes().await.context("read response")?;
    ensure!(status.is_success(), "unexpected HTTP status {status}");
    serde_json::from_slice(&bytes).context("decode JSON response")
}

fn public_post(client: &Client, url: &str) -> reqwest::RequestBuilder {
    client.post(url).bearer_auth(PUBLIC_TOKEN)
}

fn admin_post(client: &Client, url: &str) -> reqwest::RequestBuilder {
    client.post(url).bearer_auth(ADMIN_TOKEN)
}

fn mcp_embedding(response: &Value) -> Result<Value> {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .context("MCP text content")?;
    serde_json::from_str(text).context("decode MCP tool content")
}

fn assert_full_result_equal(native: &Value, other: &Value) {
    assert_eq!(other["embeddings"], native["embeddings"]);
    assert_eq!(other["model"], native["model"]);
    assert_eq!(other["usage"], native["usage"]);
}

async fn grpc_channel(address: &str) -> Result<tonic::transport::Channel> {
    Endpoint::from_shared(format!("http://{address}"))?
        .connect_timeout(Duration::from_secs(5))
        .connect()
        .await
        .context("connect gRPC")
}

async fn wait_for_grpc_health(
    health: &mut HealthClient<tonic::transport::Channel>,
    expected: ServingStatus,
) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let response = health
            .check(HealthCheckRequest {
                service: String::new(),
            })
            .await?
            .into_inner();
        if response.status == expected as i32 {
            return Ok(());
        }
        ensure!(
            Instant::now() < deadline,
            "gRPC readiness refresh exceeded bound"
        );
        sleep(Duration::from_millis(25)).await;
    }
}

fn grpc_request() -> EmbedRequest {
    EmbedRequest {
        model: MODEL_ID.into(),
        input: vec!["hello".into(), "world hello".into()],
        task: EmbeddingTask::Document as i32,
        truncation: Truncation::Reject as i32,
        dimensions: None,
        normalize: None,
    }
}

fn grpc_as_json(response: impossible_protocol::v1::EmbedResponse) -> Result<Value> {
    let model = response.model.context("gRPC identity")?;
    let usage = response.usage.context("gRPC usage")?;
    Ok(json!({
        "embeddings": response.embeddings.into_iter().map(|vector| vector.values).collect::<Vec<_>>(),
        "model": {
            "canonical_id": model.canonical_id,
            "revision": model.revision,
            "runtime": model.runtime,
            "artifact_fingerprint": model.artifact_fingerprint,
            "semantic_fingerprint": model.semantic_fingerprint
        },
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "input_tokens": usage.input_tokens
        }
    }))
}

async fn exercise_stdio(fixture: &Fixture, expected: &Value) -> Result<std::process::Output> {
    let mut command = base_command(fixture);
    command.arg("mcp").arg("--stdio");
    common_config(&mut command, fixture);
    command
        .arg("--preload-models")
        .arg(MODEL_ID)
        .arg("--startup-policy")
        .arg("strict")
        .stdin(Stdio::piped());
    let mut child = command.spawn().context("spawn MCP stdio binary")?;
    let mut input = child.stdin.take().context("MCP stdin")?;
    let output = child.stdout.take().context("MCP stdout")?;
    let mut lines = BufReader::new(output).lines();

    input
        .write_all(
            concat!(
                "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{},\"clientInfo\":{\"name\":\"e2e\",\"version\":\"1\"}}}\n",
                "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
                "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"embed\",\"arguments\":{\"model\":\"fixture/cast-embedding\",\"input\":[\"hello\",\"world hello\"]}}}\n"
            )
            .as_bytes(),
        )
        .await?;
    input.shutdown().await?;
    drop(input);

    let initialize: Value = serde_json::from_str(
        &timeout(Duration::from_secs(10), lines.next_line())
            .await??
            .context("initialize response")?,
    )?;
    assert_eq!(initialize["id"], 1);
    let embed: Value = serde_json::from_str(
        &timeout(Duration::from_secs(10), lines.next_line())
            .await??
            .context("embed response")?,
    )?;
    assert_eq!(embed["id"], 2);
    assert_full_result_equal(expected, &mcp_embedding(&embed)?);
    ensure!(
        timeout(Duration::from_secs(10), lines.next_line())
            .await
            .context("MCP stdio EOF exceeded bound")??
            .is_none(),
        "unexpected MCP stdout frame"
    );
    drop(lines);

    timeout(Duration::from_secs(10), child.wait_with_output())
        .await
        .context("MCP stdio shutdown exceeded bound")?
        .context("wait for MCP stdio")
}

fn assert_private_data_absent(output: &std::process::Output, fixture: &Fixture) -> Result<()> {
    let logs = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    for private in [
        PUBLIC_TOKEN,
        ADMIN_TOKEN,
        fixture.private_marker.as_str(),
        fixture.cache.to_string_lossy().as_ref(),
        fixture.manifest_path.to_string_lossy().as_ref(),
    ] {
        ensure!(
            !logs.contains(private),
            "private fixture data appeared in logs"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_metrics_auth_uses_admin_then_public_then_accepted_no_auth() -> Result<()> {
    let fixture = create_fixture()?;
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;

    let (remote_bind, remote_base) = ephemeral_remote_http_address()?;
    let remote_public = start_metrics_server(
        &fixture,
        &remote_bind,
        remote_base.clone(),
        &["--auth-env", "IE_E2E_PUBLIC_TOKEN"],
    )
    .await?;
    assert_eq!(
        client
            .get(format!("{remote_base}/metrics"))
            .send()
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .get(format!("{remote_base}/metrics"))
            .bearer_auth(ADMIN_TOKEN)
            .send()
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .get(format!("{remote_base}/metrics"))
            .bearer_auth(PUBLIC_TOKEN)
            .send()
            .await?
            .status(),
        StatusCode::OK
    );
    let output = remote_public.stop().await?;
    assert_private_data_absent(&output, &fixture)?;

    let local_address = ephemeral_address()?;
    let local_base = format!("http://{local_address}");
    let local = start_metrics_server(&fixture, &local_address, local_base.clone(), &[]).await?;
    assert_eq!(
        client
            .get(format!("{local_base}/metrics"))
            .send()
            .await?
            .status(),
        StatusCode::OK
    );
    let output = local.stop().await?;
    assert_private_data_absent(&output, &fixture)?;

    let (insecure_bind, insecure_base) = ephemeral_remote_http_address()?;
    let insecure = start_metrics_server(
        &fixture,
        &insecure_bind,
        insecure_base.clone(),
        &["--allow-insecure-remote"],
    )
    .await?;
    assert_eq!(
        client
            .get(format!("{insecure_base}/metrics"))
            .send()
            .await?
            .status(),
        StatusCode::OK
    );
    let output = insecure.stop().await?;
    assert_private_data_absent(&output, &fixture)?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn actual_binary_is_consistent_across_every_transport() -> Result<()> {
    let fixture = create_fixture()?;
    let (mut server, base, grpc_address) = start_server(&fixture).await?;
    eprintln!("phase: server-started");
    let client = Client::builder().timeout(Duration::from_secs(10)).build()?;

    // Operational surfaces are public, bounded, and initially truthful: the process is live but
    // no model has been loaded yet.
    assert_eq!(
        client
            .get(format!("{base}/health/live"))
            .send()
            .await?
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        client
            .get(format!("{base}/health/ready"))
            .send()
            .await?
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    let status = client.get(format!("{base}/")).send().await?;
    assert_eq!(status.status(), StatusCode::OK);
    let status_text = status.text().await?;
    ensure!(status_text.contains("Impossible Embedding"));
    ensure!(!status_text.contains(&fixture.cache.to_string_lossy().into_owned()));
    let openapi = body_json(client.get(format!("{base}/openapi.json")).send().await?).await?;
    assert_eq!(openapi["openapi"], "3.1.0");
    ensure!(openapi["paths"]["/v1/embed"]["post"].is_object());
    let channel = grpc_channel(&grpc_address).await?;
    let mut health = HealthClient::new(channel.clone());
    wait_for_grpc_health(&mut health, ServingStatus::NotServing).await?;

    // Public/admin credentials are independent and the admin API controls readiness.
    assert_eq!(
        client
            .get(format!("{base}/v1/models"))
            .send()
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        client
            .get(format!("{base}/metrics"))
            .bearer_auth(PUBLIC_TOKEN)
            .send()
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let models = body_json(
        client
            .get(format!("{base}/v1/models"))
            .bearer_auth(PUBLIC_TOKEN)
            .send()
            .await?,
    )
    .await?;
    ensure!(
        models["data"]
            .as_array()
            .is_some_and(|models| models.iter().any(|model| model["model"] == MODEL_ID))
    );
    let load_url = format!("{base}/v1/admin/models/load");
    assert_eq!(
        client
            .post(&load_url)
            .json(&json!({"model": MODEL_ID}))
            .send()
            .await?
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let loaded = admin_post(&client, &load_url)
        .json(&json!({"model": MODEL_ID}))
        .send()
        .await?;
    assert_eq!(loaded.status(), StatusCode::OK);
    eprintln!("phase: model-loaded");
    wait_for_http(&mut server, &format!("{base}/health/ready"), StatusCode::OK).await?;

    wait_for_grpc_health(&mut health, ServingStatus::Serving).await?;
    let mut reflection = ServerReflectionClient::new(channel.clone());
    let reflection_request = ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::ListServices(String::new())),
    };
    let mut reflection_stream = reflection
        .server_reflection_info(tokio_stream::iter([reflection_request]))
        .await?
        .into_inner();
    let reflection_response = reflection_stream
        .message()
        .await?
        .context("reflection response")?;
    let Some(MessageResponse::ListServicesResponse(services)) =
        reflection_response.message_response
    else {
        return Err(anyhow!("unexpected reflection response"));
    };
    ensure!(
        services
            .service
            .iter()
            .any(|service| service.name == "impossible.embedding.v1.EmbeddingService")
    );
    eprintln!("phase: grpc-operational");

    let request_body = json!({
        "model": MODEL_ID,
        "input": ["hello", "world hello"],
        "task": "document"
    });
    let native = body_json(
        public_post(&client, &format!("{base}/v1/embed"))
            .json(&request_body)
            .send()
            .await?,
    )
    .await?;
    assert_eq!(native["usage"]["input_tokens"], json!([3, 4]));
    assert_eq!(native["embeddings"].as_array().map(Vec::len), Some(2));
    assert_ne!(native["embeddings"][0], native["embeddings"][1]);
    eprintln!("phase: native-embedded");

    let compatibility = body_json(
        public_post(&client, &format!("{base}/v1/embeddings"))
            .json(&json!({
                "model": MODEL_ID,
                "input": ["hello", "world hello"]
            }))
            .send()
            .await?,
    )
    .await?;
    assert_eq!(compatibility["model"], native["model"]["canonical_id"]);
    assert_eq!(
        compatibility["usage"]["total_tokens"],
        native["usage"]["total_tokens"]
    );
    assert_eq!(compatibility["data"][0]["index"], 0);
    assert_eq!(compatibility["data"][1]["index"], 1);
    assert_eq!(
        compatibility["data"][0]["embedding"],
        native["embeddings"][0]
    );
    assert_eq!(
        compatibility["data"][1]["embedding"],
        native["embeddings"][1]
    );

    let mut grpc_client = EmbeddingServiceClient::new(channel);
    let unauthenticated = match grpc_client.embed(grpc_request()).await {
        Ok(_) => return Err(anyhow!("gRPC request without auth unexpectedly succeeded")),
        Err(status) => status,
    };
    assert_eq!(unauthenticated.code(), tonic::Code::Unauthenticated);
    let mut request = Request::new(grpc_request());
    request.metadata_mut().insert(
        AUTHORIZATION,
        MetadataValue::try_from(format!("Bearer {PUBLIC_TOKEN}"))?,
    );
    let grpc = grpc_as_json(grpc_client.embed(request).await?.into_inner())?;
    assert_full_result_equal(&native, &grpc);
    let mut expired = Request::new(grpc_request());
    expired.metadata_mut().insert(
        AUTHORIZATION,
        MetadataValue::try_from(format!("Bearer {PUBLIC_TOKEN}"))?,
    );
    expired
        .metadata_mut()
        .insert("grpc-timeout", MetadataValue::from_static("1n"));
    let expired = match grpc_client.embed(expired).await {
        Ok(_) => return Err(anyhow!("expired gRPC request unexpectedly succeeded")),
        Err(status) => status,
    };
    // Tonic may enforce the wire deadline client-side before the server's equivalent deadline
    // response wins the race. Both outcomes prove bounded cancellation across the real channel.
    ensure!(
        matches!(
            expired.code(),
            tonic::Code::DeadlineExceeded | tonic::Code::Cancelled
        ),
        "expired gRPC request returned an unrelated status"
    );
    eprintln!("phase: grpc-embedded");

    let mcp = body_json(
        public_post(&client, &format!("{base}/mcp"))
            .header("accept", "application/json, text/event-stream")
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "embed", "arguments": {
                    "model": MODEL_ID,
                    "input": ["hello", "world hello"]
                }}
            }))
            .send()
            .await?,
    )
    .await?;
    assert_full_result_equal(&native, &mcp_embedding(&mcp)?);
    eprintln!("phase: mcp-http-embedded");

    // Unload/load transitions are reflected by both HTTP readiness and gRPC health.
    let unload = admin_post(&client, &format!("{base}/v1/admin/models/unload"))
        .json(&json!({"model": MODEL_ID}))
        .send()
        .await?;
    assert_eq!(unload.status(), StatusCode::OK);
    wait_for_http(
        &mut server,
        &format!("{base}/health/ready"),
        StatusCode::SERVICE_UNAVAILABLE,
    )
    .await?;
    wait_for_grpc_health(&mut health, ServingStatus::NotServing).await?;
    let reloaded = admin_post(&client, &load_url)
        .json(&json!({"model": MODEL_ID}))
        .send()
        .await?;
    assert_eq!(reloaded.status(), StatusCode::OK);
    wait_for_http(&mut server, &format!("{base}/health/ready"), StatusCode::OK).await?;
    wait_for_grpc_health(&mut health, ServingStatus::Serving).await?;

    // Request-size rejection is stable and does not echo input. Explicit offline installation of a
    // missing curated identity also fails without attempting external network access.
    let oversized = public_post(&client, &format!("{base}/v1/embed"))
        .json(&json!({"model": MODEL_ID, "input": ["x".repeat(300_000)]}))
        .send()
        .await?;
    assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let offline = admin_post(&client, &format!("{base}/v1/admin/models/install"))
        .json(&json!({"model": "multilingual-e5-small"}))
        .send()
        .await?;
    assert!(!offline.status().is_success());
    eprintln!("phase: lifecycle-complete");

    let server_output = server.stop().await?;
    eprintln!("phase: server-stopped");
    assert_private_data_absent(&server_output, &fixture)?;
    let stdio_output = exercise_stdio(&fixture, &native).await?;
    eprintln!("phase: stdio-complete");
    ensure!(stdio_output.status.success(), "MCP stdio process failed");
    assert_private_data_absent(&stdio_output, &fixture)?;

    // A corrupted installed artifact cannot pass strict preload. The failure is bounded and its
    // output remains sanitized.
    fs::write(&fixture.model_path, b"corrupt-source-only")?;
    let installed_model = {
        let bytes = fs::read(&fixture.manifest_path)?;
        let manifest = Manifest::from_json(&bytes)?;
        ModelStore::new(&fixture.cache)?
            .layout()
            .model_dir(&manifest)?
            .join("model.onnx")
    };
    fs::write(installed_model, b"corrupt-installed-artifact")?;
    let http_address = ephemeral_address()?;
    let grpc_address = ephemeral_address()?;
    let mut corrupt = base_command(&fixture);
    corrupt
        .arg("serve")
        .arg("--http-bind")
        .arg(http_address)
        .arg("--grpc-bind")
        .arg(grpc_address);
    common_config(&mut corrupt, &fixture);
    corrupt
        .arg("--preload-models")
        .arg(MODEL_ID)
        .arg("--startup-policy")
        .arg("strict");
    let output = timeout(Duration::from_secs(15), corrupt.output())
        .await
        .context("corrupt preload exceeded bound")??;
    ensure!(
        !output.status.success(),
        "corrupt preload unexpectedly succeeded"
    );
    assert_private_data_absent(&output, &fixture)?;
    Ok(())
}

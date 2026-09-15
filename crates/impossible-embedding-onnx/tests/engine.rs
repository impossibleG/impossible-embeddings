//! End-to-end tests using a repository-authored synthetic Cast graph and tokenizer.

use std::{borrow::Cow, fs, path::Path};

use anyhow::{Context, Result};
use impossible_embedding_core::{
    CancellationToken, EmbeddingBatch, ErrorCode, ExecutionControl, RequestedModel,
};
use impossible_embedding_onnx::{EmbedOptions, EmbeddingTask, OnnxEmbeddingEngine, Truncation};
use impossible_models::{
    Artifact, Dimensions, License, Manifest, ModelStatus, ModelStore, OnnxInputNames, Pooling,
    Prefixes, RuntimeMetadata, SemanticTrustRoot, SemanticVerification, TensorMetadata,
    TokenizerMetadata, TrustedSemanticEvidence,
};
use impossible_server_core::{HealthRegistry, LifecycleState, ModelKey};
use prost::Message;
use sha2::{Digest, Sha256};
use tokenizers::{
    Tokenizer, models::wordpiece::WordPiece, pre_tokenizers::whitespace::Whitespace,
    processors::template::TemplateProcessing,
};

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

fn embedding_output(name: &str, dimensions: i64) -> ValueInfoProto {
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
                        i: 1,      // TensorProto::FLOAT
                        r#type: 2, // AttributeProto::INT
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
                            r#type: 7, // AttributeProto::INTS
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
                    input: vec![
                        "pooled".into(),
                        "pooled".into(),
                        "pooled".into(),
                        "pooled".into(),
                    ],
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
            output: vec![embedding_output("sentence_embedding", 4)],
        }),
        opset_import: vec![OperatorSetIdProto { version: 13 }],
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_fixture(root: &Path) -> Result<Manifest> {
    let mut model = Vec::new();
    cast_model().encode(&mut model)?;
    fs::write(root.join("model.onnx"), &model)?;

    let vocab = [
        ("[UNK]".to_owned(), 0),
        ("[CLS]".to_owned(), 1),
        ("[SEP]".to_owned(), 2),
        ("hello".to_owned(), 3),
        ("world".to_owned(), 4),
        ("query".to_owned(), 5),
        (":".to_owned(), 6),
    ];
    let wordpiece = WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
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
    tokenizer
        .save(root.join("tokenizer.json"), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let tokenizer_bytes = fs::read(root.join("tokenizer.json"))?;
    Ok(Manifest {
        schema_version: 1,
        canonical_id: "fixture/cast-embedding".into(),
        revision: "1111111111111111111111111111111111111111".into(),
        license: License {
            spdx: "MIT".into(),
            source_url: "https://example.invalid/model".into(),
        },
        semantic_verification: SemanticVerification::Verified {
            evidence: "synthetic-fixture-v1".into(),
        },
        tokenizer: TokenizerMetadata {
            kind: "wordpiece".into(),
            max_tokens: 5,
            lowercase: false,
        },
        pooling: Pooling::Mean,
        prefixes: Prefixes {
            query: "query: ".into(),
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
            normalize: true,
        },
        artifacts: vec![
            Artifact {
                path: "model.onnx".into(),
                url: "https://example.invalid/model.onnx".into(),
                sha256: digest(&model),
                size: u64::try_from(model.len())?,
            },
            Artifact {
                path: "tokenizer.json".into(),
                url: "https://example.invalid/tokenizer.json".into(),
                sha256: digest(&tokenizer_bytes),
                size: u64::try_from(tokenizer_bytes.len())?,
            },
        ],
    })
}

fn setup() -> Result<(tempfile::TempDir, OnnxEmbeddingEngine)> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("source");
    fs::create_dir(&source)?;
    let manifest = write_fixture(&source)?;
    let store = trusted_store(directory.path().join("cache"), &manifest)?;
    anyhow::ensure!(store.import(&manifest, &source)? == ModelStatus::Loadable);
    let verified = store.verified_model(&manifest)?;
    let engine = OnnxEmbeddingEngine::new();
    let identity = engine.load(&verified)?;
    anyhow::ensure!(identity.canonical_id == "fixture/cast-embedding");
    anyhow::ensure!(identity.artifact_fingerprint.starts_with("sha256:"));
    Ok((directory, engine))
}

fn trusted_store(path: impl AsRef<Path>, manifest: &Manifest) -> Result<ModelStore> {
    let trust_root = SemanticTrustRoot::from_evidence([TrustedSemanticEvidence {
        manifest_fingerprint: manifest.semantic_fingerprint()?,
        evidence: "repository-authored synthetic fixture".into(),
    }])?;
    Ok(ModelStore::with_trust_root(path, trust_root)?)
}

fn embed(
    engine: &OnnxEmbeddingEngine,
    inputs: &[&str],
    options: EmbedOptions,
) -> Result<Vec<Vec<f32>>> {
    let requested = RequestedModel::new("fixture/cast-embedding")?;
    let batch = EmbeddingBatch::new(inputs.iter().map(|value| Cow::Borrowed(*value)))?;
    let control = ExecutionControl::new(CancellationToken::default(), None);
    Ok(engine
        .embed_with_options(&requested, &batch, &control, options)?
        .vectors)
}

#[test]
fn lifecycle_and_real_in_process_ort_execution() -> Result<()> {
    let (_directory, engine) = setup()?;
    anyhow::ensure!(engine.is_loaded());
    engine.warm()?;
    let vectors = embed(&engine, &["hello", "hello world"], EmbedOptions::default())?;
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), 4);
    for vector in vectors {
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
    engine.unload()?;
    anyhow::ensure!(!engine.is_loaded());
    engine.unload()?;
    Ok(())
}

#[test]
fn verified_store_model_loads_and_drives_infrastructure_readiness() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("source");
    fs::create_dir(&source)?;
    let manifest = write_fixture(&source)?;
    let store = trusted_store(directory.path().join("cache"), &manifest)?;
    let status = store.import(&manifest, &source)?;
    assert_eq!(status, ModelStatus::Loadable);

    let health = HealthRegistry::default();
    health.set_model_verification(ModelKey(7), status);
    assert_eq!(health.model_counts(), (0, 1));

    let verified = store.verified_model(&manifest)?;
    let engine = OnnxEmbeddingEngine::new();
    engine.load(&verified)?;
    engine.warm()?;
    health.set_model(ModelKey(7), impossible_server_core::ModelState::Ready);
    assert_eq!(health.model_counts(), (1, 1));
    assert!(health.transition(LifecycleState::Ready, None));
    assert!(health.readiness().is_ready());
    Ok(())
}

#[test]
fn handles_empty_unicode_prefixes_limits_and_dimensions() -> Result<()> {
    let (_directory, engine) = setup()?;
    let defaults = EmbedOptions::default();
    assert_eq!(embed(&engine, &[""], defaults)?.len(), 1);
    assert_eq!(embed(&engine, &["Olá 世界"], defaults)?.len(), 1);

    let query = EmbedOptions {
        task: EmbeddingTask::Query,
        ..defaults
    };
    assert_eq!(
        embed(&engine, &["hello"], query)?,
        embed(&engine, &["query: hello"], query)?
    );

    let Err(error) = embed(&engine, &["hello world hello world"], defaults) else {
        anyhow::bail!("over-limit input must be rejected");
    };
    assert_eq!(
        error
            .downcast_ref::<impossible_embedding_core::EngineFailure>()
            .context("expected engine failure")?
            .public_error()
            .code,
        ErrorCode::InvalidRequest
    );
    let truncated = embed(
        &engine,
        &["hello world hello world"],
        EmbedOptions {
            truncation: Truncation::Truncate,
            dimensions: Some(2),
            ..defaults
        },
    )?;
    assert_eq!(truncated[0].len(), 2);
    let norm = truncated[0]
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    assert_eq!(
        embed(
            &engine,
            &["hello"],
            EmbedOptions {
                dimensions: Some(4),
                ..defaults
            }
        )?[0]
            .len(),
        4
    );
    assert!(
        embed(
            &engine,
            &["hello"],
            EmbedOptions {
                dimensions: Some(3),
                ..defaults
            }
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn rejects_declared_native_width_that_disagrees_with_2d_graph() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("source");
    fs::create_dir(&source)?;
    let mut manifest = write_fixture(&source)?;
    manifest.dimensions.native = 3;
    manifest.dimensions.matryoshka.clear();
    let store = trusted_store(directory.path().join("cache"), &manifest)?;
    assert_eq!(store.import(&manifest, &source)?, ModelStatus::Loadable);
    let verified = store.verified_model(&manifest)?;
    let Err(error) = OnnxEmbeddingEngine::new().load(&verified) else {
        anyhow::bail!("static graph width mismatch must fail");
    };
    assert_eq!(error.public_error().code, ErrorCode::InvalidRequest);
    Ok(())
}

#[test]
fn load_rehashes_verified_artifacts_and_loaded_engine_retains_delete_lease() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("source");
    fs::create_dir(&source)?;
    let manifest = write_fixture(&source)?;
    let store = trusted_store(directory.path().join("cache"), &manifest)?;
    store.import(&manifest, &source)?;
    let verified = store.verified_model(&manifest)?;
    fs::write(
        store.layout().model_dir(&manifest).join("tokenizer.json"),
        b"tampered",
    )?;
    assert!(OnnxEmbeddingEngine::new().load(&verified).is_err());
    drop(verified);

    store.import(&manifest, &source)?;
    let verified = store.verified_model(&manifest)?;
    let engine = OnnxEmbeddingEngine::new();
    engine.load(&verified)?;
    drop(verified);
    assert!(matches!(
        store.delete(&manifest),
        Err(impossible_models::Error::InUse)
    ));
    engine.unload()?;
    assert!(store.delete(&manifest)?);
    Ok(())
}

#[test]
fn errors_are_privacy_safe_and_artifacts_cannot_escape_manifest_root() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("source");
    fs::create_dir(&source)?;
    let mut manifest = write_fixture(&source)?;
    manifest.runtime = RuntimeMetadata::Onnx {
        model_file: "../secret.onnx".into(),
        tokenizer_file: "tokenizer.json".into(),
        inputs: OnnxInputNames {
            input_ids: "input_ids".into(),
            attention_mask: None,
            token_type_ids: None,
        },
        output: "output".into(),
        pad_token_id: 0,
        normalize: false,
    };
    let Err(error) = manifest.validate() else {
        anyhow::bail!("traversal must fail");
    };
    assert!(
        !error
            .to_string()
            .contains(directory.path().to_string_lossy().as_ref())
    );
    Ok(())
}

//! End-to-end tests using a repository-authored synthetic Cast graph and tokenizer.

use std::{borrow::Cow, fs, path::Path};

use anyhow::{Context, Result};
use impossible_embedding_core::{
    CancellationToken, EmbeddingBatch, ErrorCode, ExecutionControl, RequestedModel,
};
use impossible_embedding_onnx::{EmbedOptions, EmbeddingTask, OnnxEmbeddingEngine, Truncation};
use prost::Message;
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
                            dim_param: "batch".into(),
                        },
                        Dimension {
                            dim_param: "sequence".into(),
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
            node: vec![NodeProto {
                input: vec!["input_ids".into()],
                output: vec!["sentence_embedding".into()],
                op_type: "Cast".into(),
                attribute: vec![AttributeProto {
                    name: "to".into(),
                    i: 1,      // TensorProto::FLOAT
                    r#type: 2, // AttributeProto::INT
                }],
            }],
            name: "synthetic_embedding_fixture".into(),
            input: vec![
                tensor_value("input_ids", 7),
                tensor_value("attention_mask", 7),
            ],
            output: vec![tensor_value("sentence_embedding", 1)],
        }),
        opset_import: vec![OperatorSetIdProto { version: 13 }],
    }
}

fn write_fixture(root: &Path) -> Result<()> {
    let mut model = Vec::new();
    cast_model().encode(&mut model)?;
    fs::write(root.join("model.onnx"), model)?;

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

    fs::write(
        root.join("manifest.json"),
        r#"{
  "schema_version": 1,
  "canonical_id": "fixture/cast-embedding",
  "revision": "synthetic-v1",
  "model_file": "model.onnx",
  "tokenizer_file": "tokenizer.json",
  "inputs": {"input_ids": "input_ids", "attention_mask": "attention_mask", "token_type_ids": null},
  "output": "sentence_embedding",
  "pooling": "mean",
  "max_tokens": 5,
  "pad_token_id": 0,
  "normalize": true,
  "matryoshka_dimensions": [2],
  "prefixes": {"query": "query: ", "document": null}
}"#,
    )?;
    Ok(())
}

fn setup() -> Result<(tempfile::TempDir, OnnxEmbeddingEngine)> {
    let directory = tempfile::tempdir()?;
    write_fixture(directory.path())?;
    let engine = OnnxEmbeddingEngine::new();
    let identity = engine.load(&directory.path().join("manifest.json"))?;
    anyhow::ensure!(identity.canonical_id == "fixture/cast-embedding");
    anyhow::ensure!(identity.artifact_fingerprint.starts_with("sha256:"));
    Ok((directory, engine))
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
    assert_eq!(vectors[0].len(), 4); // padded to the longest input
    assert!(vectors[0][3].abs() < f32::EPSILON);
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
fn errors_are_privacy_safe_and_artifacts_cannot_escape_manifest_root() -> Result<()> {
    let directory = tempfile::tempdir()?;
    fs::write(
        directory.path().join("manifest.json"),
        r#"{
      "schema_version":1,"canonical_id":"fixture/model","revision":"v1",
      "model_file":"../secret.onnx","tokenizer_file":"tokenizer.json",
      "inputs":{"input_ids":"input_ids","attention_mask":"attention_mask","token_type_ids":null},
      "output":"output","pooling":"mean","max_tokens":8,"normalize":false
    }"#,
    )?;
    let Err(error) = OnnxEmbeddingEngine::new().load(&directory.path().join("manifest.json"))
    else {
        anyhow::bail!("traversal must fail");
    };
    assert_eq!(error.public_error().code, ErrorCode::InvalidRequest);
    assert!(
        !error
            .to_string()
            .contains(directory.path().to_string_lossy().as_ref())
    );
    assert!(!format!("{error:?}").contains(directory.path().to_string_lossy().as_ref()));
    Ok(())
}

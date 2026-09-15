//! In-process runtime feasibility proof using a legal synthetic graph.

use std::fs;

use anyhow::Result;
use ort::{session::Session, value::Tensor};
use prost::Message;

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
}

fn tensor_value(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: vec![Dimension { dim_value: 1 }, Dimension { dim_value: 3 }],
                }),
            }),
        }),
    }
}

fn identity_model() -> ModelProto {
    ModelProto {
        ir_version: 8,
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["input".into()],
                output: vec!["output".into()],
                op_type: "Identity".into(),
            }],
            name: "identity_fixture".into(),
            input: vec![tensor_value("input")],
            output: vec![tensor_value("output")],
        }),
        opset_import: vec![OperatorSetIdProto { version: 13 }],
    }
}

#[test]
fn loads_and_executes_synthetic_identity_model_on_cpu() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("identity.onnx");
    let mut model_bytes = Vec::new();
    identity_model().encode(&mut model_bytes)?;
    fs::write(&model_path, model_bytes)?;

    let mut session = Session::builder()?.commit_from_file(&model_path)?;
    let input = Tensor::from_array(([1_usize, 3], vec![1.25_f32, -2.5, 4.0]))?;
    let outputs = session.run(ort::inputs!["input" => input])?;
    let output = outputs["output"].try_extract_array::<f32>()?;

    assert_eq!(output.shape(), &[1, 3]);
    assert_eq!(output.as_slice(), Some(&[1.25, -2.5, 4.0][..]));
    Ok(())
}

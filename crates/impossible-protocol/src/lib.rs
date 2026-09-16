//! Versioned protobuf contracts.

use impossible_embedding_core::{
    ErrorCode as CoreErrorCode, PublicError, Retryability as CoreRetryability,
};

/// Embedding protocol version 1.
#[allow(
    missing_docs,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::similar_names
)]
pub mod v1 {
    tonic::include_proto!("impossible.embedding.v1");

    /// Encoded v1 descriptor set used by gRPC server reflection.
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("impossible_embedding_v1_descriptor");
}

/// Maps a core public failure to the stable v1 protobuf detail.
///
/// Only privacy-reviewed public fields are copied. Private engine diagnostics are not accepted by
/// this boundary and therefore cannot enter the protobuf message.
#[must_use]
pub fn public_error_detail(error: &PublicError) -> v1::PublicErrorDetail {
    v1::PublicErrorDetail {
        code: proto_error_code(error.code) as i32,
        retryability: proto_retryability(error.retryability) as i32,
        message: error.message.to_owned(),
    }
}

/// Returns the normative gRPC status code for a core application error.
#[must_use]
pub const fn grpc_status_code(code: CoreErrorCode) -> tonic::Code {
    match code {
        CoreErrorCode::InvalidRequest => tonic::Code::InvalidArgument,
        CoreErrorCode::ModelUnavailable => tonic::Code::Unavailable,
        CoreErrorCode::QueueFull => tonic::Code::ResourceExhausted,
        CoreErrorCode::Cancelled => tonic::Code::Cancelled,
        CoreErrorCode::DeadlineExceeded => tonic::Code::DeadlineExceeded,
        CoreErrorCode::InferenceFailed | CoreErrorCode::Internal => tonic::Code::Internal,
        _ => tonic::Code::Unknown,
    }
}

const fn proto_error_code(code: CoreErrorCode) -> v1::ErrorCode {
    match code {
        CoreErrorCode::InvalidRequest => v1::ErrorCode::InvalidRequest,
        CoreErrorCode::ModelUnavailable => v1::ErrorCode::ModelUnavailable,
        CoreErrorCode::QueueFull => v1::ErrorCode::QueueFull,
        CoreErrorCode::Cancelled => v1::ErrorCode::Cancelled,
        CoreErrorCode::DeadlineExceeded => v1::ErrorCode::DeadlineExceeded,
        CoreErrorCode::InferenceFailed => v1::ErrorCode::InferenceFailed,
        CoreErrorCode::Internal => v1::ErrorCode::Internal,
        _ => v1::ErrorCode::Unspecified,
    }
}

const fn proto_retryability(retryability: CoreRetryability) -> v1::Retryability {
    match retryability {
        CoreRetryability::Never => v1::Retryability::Never,
        CoreRetryability::Retryable => v1::Retryability::Retryable,
        CoreRetryability::Unknown => v1::Retryability::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use impossible_embedding_core::{
        ErrorCode as CoreErrorCode, PublicError, Retryability as CoreRetryability,
    };
    use prost::Message;

    use super::{
        grpc_status_code, public_error_detail,
        v1::{
            EmbedRequest, EmbedResponse, EmbeddingTask, ErrorCode, ResolvedModelIdentity,
            Retryability, TokenUsage, Truncation,
        },
    };

    #[test]
    fn generated_contract_is_available_without_system_protoc() -> Result<(), &'static str> {
        let request = EmbedRequest {
            model: "fixture".into(),
            input: vec!["hello".into()],
            task: EmbeddingTask::Query as i32,
            truncation: Truncation::Reject as i32,
            dimensions: Some(384),
            normalize: Some(true),
        };
        assert_eq!(request.input, ["hello"]);

        let response = EmbedResponse {
            embeddings: Vec::new(),
            model: Some(ResolvedModelIdentity {
                canonical_id: "registry.example/model".into(),
                revision: "immutable-revision".into(),
                runtime: "onnx-runtime@1".into(),
                artifact_fingerprint: "sha256:artifact".into(),
                semantic_fingerprint: "sha256:semantics".into(),
            }),
            usage: Some(TokenUsage {
                prompt_tokens: 1,
                total_tokens: 1,
                input_tokens: vec![1],
            }),
        };
        let Some(identity) = response.model else {
            return Err("fixture response must include identity");
        };
        assert_ne!(request.model, identity.canonical_id);
        assert!(!identity.artifact_fingerprint.is_empty());
        assert!(!identity.semantic_fingerprint.is_empty());
        Ok(())
    }

    #[test]
    fn descriptor_set_is_exported_for_reflection() -> Result<(), &'static str> {
        let descriptor = prost_types::FileDescriptorSet::decode(super::v1::FILE_DESCRIPTOR_SET)
            .map_err(|_| "descriptor set must decode")?;
        let Some(file) = descriptor.file.first() else {
            return Err("descriptor set must contain the v1 file");
        };
        assert_eq!(file.package.as_deref(), Some("impossible.embedding.v1"));
        assert!(
            file.service
                .iter()
                .any(|service| service.name.as_deref() == Some("EmbeddingService"))
        );

        let request = file
            .message_type
            .iter()
            .find(|message| message.name.as_deref() == Some("EmbedRequest"))
            .ok_or("descriptor must contain EmbedRequest")?;
        let response = file
            .message_type
            .iter()
            .find(|message| message.name.as_deref() == Some("EmbedResponse"))
            .ok_or("descriptor must contain EmbedResponse")?;
        let request_fields = request
            .field
            .iter()
            .map(|field| (field.name.as_deref().unwrap_or_default(), field.number))
            .collect::<Vec<_>>();
        assert_eq!(
            request_fields,
            [
                ("model", Some(1)),
                ("input", Some(2)),
                ("task", Some(3)),
                ("truncation", Some(4)),
                ("dimensions", Some(5)),
                ("normalize", Some(6)),
            ]
        );
        let response_fields = response
            .field
            .iter()
            .map(|field| (field.name.as_deref().unwrap_or_default(), field.number))
            .collect::<Vec<_>>();
        assert_eq!(
            response_fields,
            [
                ("embeddings", Some(1)),
                ("model", Some(2)),
                ("usage", Some(3)),
            ]
        );
        Ok(())
    }

    #[test]
    fn embedding_contract_field_numbers_are_stable() {
        assert_eq!(EmbeddingTask::Unspecified as i32, 0);
        assert_eq!(EmbeddingTask::Query as i32, 1);
        assert_eq!(EmbeddingTask::Document as i32, 2);
        assert_eq!(Truncation::Unspecified as i32, 0);
        assert_eq!(Truncation::Reject as i32, 1);
        assert_eq!(Truncation::Truncate as i32, 2);
    }

    #[test]
    fn checked_in_openapi_contract_is_valid_and_has_frozen_routes() -> Result<(), &'static str> {
        fn validate_references(
            value: &serde_json::Value,
            root: &serde_json::Value,
        ) -> Result<(), &'static str> {
            match value {
                serde_json::Value::Object(object) => {
                    if let Some(reference) = object.get("$ref").and_then(serde_json::Value::as_str)
                    {
                        let pointer = reference
                            .strip_prefix('#')
                            .ok_or("OpenAPI references must be internal")?;
                        if root.pointer(pointer).is_none() {
                            return Err("OpenAPI reference must resolve");
                        }
                    }
                    for nested in object.values() {
                        validate_references(nested, root)?;
                    }
                }
                serde_json::Value::Array(array) => {
                    for nested in array {
                        validate_references(nested, root)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }

        let document: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/openapi-v1.json"))
                .map_err(|_| "OpenAPI document must be valid JSON")?;
        assert_eq!(document["openapi"], "3.1.0");
        let paths = document["paths"]
            .as_object()
            .ok_or("OpenAPI paths must be an object")?;
        let frozen = [
            ("/", "get"),
            ("/v1/embeddings", "post"),
            ("/v1/embed", "post"),
            ("/v1/models", "get"),
            ("/v1/admin/models/install", "post"),
            ("/v1/admin/models/load", "post"),
            ("/v1/admin/models/unload", "post"),
            ("/v1/admin/models/delete", "post"),
            ("/health/live", "get"),
            ("/health/ready", "get"),
            ("/metrics", "get"),
            ("/openapi.json", "get"),
            ("/mcp", "post"),
        ];
        for (path, method) in frozen {
            assert!(
                paths.get(path).and_then(|item| item.get(method)).is_some(),
                "missing frozen route {method} {path}"
            );
        }
        assert_eq!(paths.len(), frozen.len());
        let mut operation_ids = std::collections::HashSet::new();
        for path_item in paths.values() {
            let Some(path_item) = path_item.as_object() else {
                return Err("OpenAPI path item must be an object");
            };
            for operation in path_item.values() {
                let Some(operation_id) = operation
                    .get("operationId")
                    .and_then(serde_json::Value::as_str)
                else {
                    return Err("every frozen operation must have an operationId");
                };
                if !operation_ids.insert(operation_id) {
                    return Err("OpenAPI operationId values must be unique");
                }
            }
        }
        validate_references(&document, &document)?;
        assert_eq!(
            document["components"]["schemas"]["OpenAiEmbeddingRequest"]["additionalProperties"],
            false
        );
        assert_eq!(
            document["components"]["schemas"]["NativeEmbeddingRequest"]["additionalProperties"],
            false
        );
        assert_eq!(
            document["components"]["schemas"]["ErrorEnvelope"]["properties"]["error"]["required"],
            serde_json::json!(["code", "message", "retryable"])
        );
        assert_eq!(
            document["components"]["schemas"]["OpenAiEmbeddingRequest"]["properties"]["encoding_format"]
                ["enum"],
            serde_json::json!(["float"])
        );
        Ok(())
    }

    #[test]
    fn every_core_error_has_the_normative_grpc_mapping() {
        let cases = [
            (
                CoreErrorCode::InvalidRequest,
                ErrorCode::InvalidRequest,
                CoreRetryability::Never,
                Retryability::Never,
                tonic::Code::InvalidArgument,
            ),
            (
                CoreErrorCode::ModelUnavailable,
                ErrorCode::ModelUnavailable,
                CoreRetryability::Retryable,
                Retryability::Retryable,
                tonic::Code::Unavailable,
            ),
            (
                CoreErrorCode::QueueFull,
                ErrorCode::QueueFull,
                CoreRetryability::Retryable,
                Retryability::Retryable,
                tonic::Code::ResourceExhausted,
            ),
            (
                CoreErrorCode::Cancelled,
                ErrorCode::Cancelled,
                CoreRetryability::Never,
                Retryability::Never,
                tonic::Code::Cancelled,
            ),
            (
                CoreErrorCode::DeadlineExceeded,
                ErrorCode::DeadlineExceeded,
                CoreRetryability::Never,
                Retryability::Never,
                tonic::Code::DeadlineExceeded,
            ),
            (
                CoreErrorCode::InferenceFailed,
                ErrorCode::InferenceFailed,
                CoreRetryability::Unknown,
                Retryability::Unknown,
                tonic::Code::Internal,
            ),
            (
                CoreErrorCode::Internal,
                ErrorCode::Internal,
                CoreRetryability::Unknown,
                Retryability::Unknown,
                tonic::Code::Internal,
            ),
        ];

        for (core_code, proto_code, core_retryability, proto_retryability, grpc_code) in cases {
            let public = PublicError::for_code(core_code);
            let detail = public_error_detail(&public);
            assert_eq!(public.retryability, core_retryability);
            assert_eq!(detail.code, proto_code as i32);
            assert_eq!(detail.retryability, proto_retryability as i32);
            assert_eq!(detail.message, public.message);
            assert_eq!(grpc_status_code(core_code), grpc_code);
        }
    }

    #[test]
    fn protobuf_error_enum_numbers_are_stable() {
        assert_eq!(ErrorCode::Unspecified as i32, 0);
        assert_eq!(ErrorCode::InvalidRequest as i32, 1);
        assert_eq!(ErrorCode::ModelUnavailable as i32, 2);
        assert_eq!(ErrorCode::QueueFull as i32, 3);
        assert_eq!(ErrorCode::Cancelled as i32, 4);
        assert_eq!(ErrorCode::DeadlineExceeded as i32, 5);
        assert_eq!(ErrorCode::InferenceFailed as i32, 6);
        assert_eq!(ErrorCode::Internal as i32, 7);
        assert_eq!(Retryability::Unspecified as i32, 0);
        assert_eq!(Retryability::Never as i32, 1);
        assert_eq!(Retryability::Retryable as i32, 2);
        assert_eq!(Retryability::Unknown as i32, 3);
    }
}

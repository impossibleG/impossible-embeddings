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

    use super::{
        grpc_status_code, public_error_detail,
        v1::{EmbedRequest, EmbedResponse, ErrorCode, ResolvedModelIdentity, Retryability},
    };

    #[test]
    fn generated_contract_is_available_without_system_protoc() -> Result<(), &'static str> {
        let request = EmbedRequest {
            model: "fixture".into(),
            input: vec!["hello".into()],
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

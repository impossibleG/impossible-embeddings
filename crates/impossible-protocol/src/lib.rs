//! Versioned protobuf contracts.

/// Embedding protocol version 1.
#[allow(
    missing_docs,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::similar_names
)]
pub mod v1 {
    tonic::include_proto!("impossible.embedding.v1");
}

#[cfg(test)]
mod tests {
    use super::v1::{EmbedRequest, EmbedResponse, ResolvedModelIdentity};

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
            }),
        };
        let Some(identity) = response.model else {
            return Err("fixture response must include identity");
        };
        assert_ne!(request.model, identity.canonical_id);
        assert!(!identity.artifact_fingerprint.is_empty());
        Ok(())
    }
}

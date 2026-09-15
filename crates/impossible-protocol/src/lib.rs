//! Versioned protobuf contracts.

/// Embedding protocol version 1.
#[allow(
    missing_docs,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::missing_errors_doc
)]
pub mod v1 {
    tonic::include_proto!("impossible.embedding.v1");
}

#[cfg(test)]
mod tests {
    use super::v1::EmbedRequest;

    #[test]
    fn generated_contract_is_available_without_system_protoc() {
        let request = EmbedRequest {
            model: "fixture".into(),
            input: vec!["hello".into()],
        };
        assert_eq!(request.input, ["hello"]);
    }
}

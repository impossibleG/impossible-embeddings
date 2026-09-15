//! Engine-neutral embedding domain contracts.

use std::borrow::Cow;

use thiserror::Error;

/// A validated batch of non-empty text inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingBatch<'a> {
    inputs: Vec<Cow<'a, str>>,
}

impl<'a> EmbeddingBatch<'a> {
    /// Validates and constructs a batch.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::EmptyBatch`] when no inputs are provided.
    pub fn new(inputs: impl IntoIterator<Item = Cow<'a, str>>) -> Result<Self, EmbeddingError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        if inputs.is_empty() {
            return Err(EmbeddingError::EmptyBatch);
        }
        Ok(Self { inputs })
    }

    /// Returns the ordered input texts.
    #[must_use]
    pub fn inputs(&self) -> &[Cow<'a, str>] {
        &self.inputs
    }
}

/// Output vectors in the same order as their inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutput {
    /// Dense vectors, ordered by request input.
    pub vectors: Vec<Vec<f32>>,
}

/// Stable engine failures that transports can map without engine-specific types.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    /// An embedding request contained no inputs.
    #[error("the embedding batch must contain at least one input")]
    EmptyBatch,
    /// The selected model is unavailable.
    #[error("the requested model is unavailable")]
    ModelUnavailable,
    /// The engine could not complete inference.
    #[error("embedding inference failed")]
    Inference,
}

/// Engine boundary implemented by in-process and delegated runtimes.
pub trait EmbeddingEngine: Send + Sync {
    /// Produces one dense vector for each input, preserving order.
    ///
    /// # Errors
    ///
    /// Returns a stable [`EmbeddingError`] rather than an engine-private error.
    fn embed(&self, batch: &EmbeddingBatch<'_>) -> Result<EmbeddingOutput, EmbeddingError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_batches() {
        let result = EmbeddingBatch::new(Vec::<Cow<'_, str>>::new());
        assert_eq!(result, Err(EmbeddingError::EmptyBatch));
    }

    #[test]
    fn preserves_input_order() -> Result<(), EmbeddingError> {
        let batch = EmbeddingBatch::new([Cow::Borrowed("first"), Cow::Borrowed("second")])?;
        assert_eq!(batch.inputs(), ["first", "second"]);
        Ok(())
    }
}

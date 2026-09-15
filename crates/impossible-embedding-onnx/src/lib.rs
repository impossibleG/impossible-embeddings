//! ONNX Runtime adapter boundary.
//!
//! The foundation milestone contains only a feasibility test. Production model loading and
//! embedding semantics will be added after the adapter contract is validated.

/// Identifies the first candidate runtime without exposing runtime-private types.
pub const ENGINE_NAME: &str = "onnx-runtime";

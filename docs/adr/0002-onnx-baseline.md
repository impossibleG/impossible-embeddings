# ADR 0002: ONNX Runtime as the first in-process CPU baseline

- Status: Provisional
- Date: 2026-09-15

## Context

A first engine must run locally on CPU across major platforms and support a controlled model
catalog. The foundation also needs a small proof independent of external model services.

## Decision

Evaluate ONNX Runtime as the first in-process engine. The repository includes a generated identity
graph fixture to prove loading and execution. This is a feasibility gate, not evidence that a real
embedding model, tokenizer, pooling implementation, or accelerator is complete.

## Consequences

The adapter stays behind the embedding engine boundary. Production adoption still requires model
conformance, packaging, licensing, memory, cancellation, and performance validation.

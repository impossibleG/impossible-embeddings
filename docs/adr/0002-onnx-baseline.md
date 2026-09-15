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

The loader treats the manifest as an exact graph contract. Every declared input name must be
unique; the session must expose exactly that supported input set; and every input must be a rank-2
Int64 tensor. The selected output must be a unique rank-2 or rank-3 Float32 tensor whose static
hidden width matches the manifest. Contract failures reject the load before warmup or inference.

## Consequences

The adapter stays behind the embedding engine boundary. Production adoption still requires model
conformance, packaging, licensing, memory, cancellation, and performance validation.

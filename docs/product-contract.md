# Product Contract

Impossible Embedding will provide a reliable, local-first service for dense text embeddings.

## Stable promises

- OpenAI-compatible embedding requests plus a typed native API.
- CPU operation as the universal baseline, with explicit optional accelerators.
- Curated, verified model installation and explicit experimental model import.
- Bounded queues, batching, cancellation, graceful shutdown, and actionable readiness.
- Structured operational telemetry that excludes request text and embedding values by default.
- HTTP/OpenAPI, gRPC, and a thin MCP surface where each transport is useful.
- Native release archives and container images, with no cloud service dependency.
- Offline inference after the selected runtime and model artifacts are installed.

## Non-promises

- Universal compatibility with every model repository or architecture.
- Silent model downloads, full-disk scanning, or automatic network exposure.
- A graphical administration product.
- Identical performance or accelerator support on every platform.
- WebSockets when request/response or gRPC streaming is sufficient.

## Foundation boundary

The first milestone establishes workspace policy, domain boundaries, reproducible protocol
generation, and in-process ONNX feasibility. It intentionally does not expose a network port or
claim production readiness.

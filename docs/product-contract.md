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

## Execution semantics

Each request carries a monotonic deadline and a shared cancellation signal through admission,
queueing, inference, and response publication. Cancelled or expired queued work never starts.
Some native runtimes, including the initial ONNX adapter, cannot guarantee interruption after a
native inference call begins. In that case computation may finish, but the server checks the signal
again and discards the late result rather than publishing it. Cancellation is therefore a response
and queue-resource guarantee, not a claim that every runtime can immediately reclaim compute.

## Model identity

The model string in a request is an alias or selection expression. Successful responses carry a
separate resolved identity containing a canonical model id, immutable revision, runtime/version,
and fingerprint over every inference-affecting artifact. Aliases are never presented as proof of
which model produced an embedding.

## Public failures

Every transport maps failures to the stable codes `invalid_request`, `model_unavailable`,
`queue_full`, `cancelled`, `deadline_exceeded`, `inference_failed`, and `internal`, together with
explicit retryability. Public messages are privacy-reviewed static text. Runtime errors, paths, and
other diagnostic sources remain separate and may only enter explicitly access-controlled debug
telemetry; they are never serialized to clients or ordinary logs.

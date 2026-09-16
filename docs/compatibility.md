# Compatibility and known limitations

## Release targets

The release workflow builds and smoke-tests these CPU targets before publishing a tag:

| Target | Archive | Status |
| --- | --- | --- |
| Windows x86-64 MSVC | ZIP | Release target |
| Linux x86-64 GNU/glibc | tar.gz | Release target |

Linux ARM64, macOS, musl, GPU execution providers, and mobile platforms are not release targets in
v0.1. Source builds may work elsewhere, but they are not claimed as supported until a native archive
and inference qualification run are added. The container recipe is Linux x86-64 only.

Native archives include the executable, any runtime shared libraries emitted by the build, example
configuration, protocol schemas, project licenses, third-party notices, and ONNX Runtime's license.
The current baseline may link ONNX Runtime into the executable instead of emitting a separate file.

## Model compatibility

The server is manifest-driven, not a universal model loader. Only curated models whose tokenizer,
ONNX inputs/outputs, pooling, prefixes, dimensions, maximum sequence length, and hashes are fully
described can load. Remote repository code is never executed. A model catalog entry reports its
semantic verification status; pinned artifacts without passing golden-vector evidence remain visibly
unverified.

Do not assume two aliases, revisions, or configurations are interchangeable. Use the resolved model
identity and semantic fingerprint returned by native HTTP and gRPC.

## Protocol compatibility

- HTTP v1 is defined by `docs/openapi-v1.json`. The OpenAI-compatible surface is intentionally
  narrow: dense float embeddings only.
- gRPC v1 is defined by `crates/impossible-protocol/proto/embedding.proto`; reflection and standard
  health are enabled.
- MCP implements revision 2025-03-26 with `embed` and `list_models` only. HTTP is stateless; stdio is
  connection-stateful.
- WebSockets are not provided because embedding is bounded request/response work.

## Known limitations

- CPU ONNX inference is the only execution backend in v0.1. There is no CUDA, ROCm, DirectML, Metal,
  CoreML, or OpenVINO provider selection.
- The process does not terminate TLS. Remote deployments require an external TLS boundary.
- There is no built-in multi-tenant quota, user database, distributed scheduler, model registry,
  autoscaler, or dashboard.
- Model installation needs network access unless the exact verified artifact is already installed.
  Inference can run fully offline afterward.
- Cancellation after a native ONNX call begins suppresses publication but cannot guarantee immediate
  interruption of native compute.
- Metrics are process-local. Logs and metrics deliberately omit input text, vectors, host paths, and
  hardware details.
- Benchmarks are not portable between models or hosts. This project does not publish host-specific
  benchmark results in the repository.
- Voice, speech, audio, image, and OCR inputs are outside this repository; sibling servers can reuse
  the operational shell without pretending those modalities share an inference contract.

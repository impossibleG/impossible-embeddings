# Impossible Embedding

Impossible Embedding is a local-first server for dense text embeddings with open models. It owns
the operational pieces that are easy to get wrong: verified model installation, bounded dynamic
batching, predictable cancellation, readiness, metrics, authentication, and graceful shutdown.

Inference runs locally through ONNX Runtime. Nothing calls OpenAI or any other hosted inference
service. `POST /v1/embeddings` is only a wire-compatible adapter so existing OpenAI-compatible
clients can use a local base URL.

## What is included

- Native and OpenAI-compatible HTTP APIs, with a checked-in OpenAPI v1 contract.
- gRPC embeddings, standard gRPC health, and reflection.
- MCP 2025-03-26 over stateless HTTP and stateful standard input/output.
- Curated, revision-pinned model manifests with verified downloads and explicit semantic status.
- Atomic model installation, offline reuse, load/unload/delete operations, and concurrent-use
  protection.
- Bounded request sizes, queueing, tokenization, batching, concurrency, and shutdown.
- Prometheus metrics, structured privacy-safe logs, liveness, readiness, and a small status page.
- Native Windows and Linux x86-64 release archives and a non-root container recipe.

## Quick start from source

Rust 1.85.1 is selected by `rust-toolchain.toml`. Start the server on explicit loopback addresses:

```shell
cargo run --locked --release -- serve \
  --http-bind 127.0.0.1:8080 \
  --grpc-bind 127.0.0.1:50051
```

In a second terminal, install and load a curated model through the local administration API:

```shell
cargo run --locked --release -- models install bge-small-en --server http://127.0.0.1:8080
cargo run --locked --release -- models load bge-small-en --server http://127.0.0.1:8080
```

Installation is the only model operation that may use the network. It is always explicit, pins the
revision and hashes from the curated manifest, and never executes repository code. Once installed,
start with `--offline` to prohibit model network access.

Embed locally:

```shell
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H "content-type: application/json" \
  -d '{"model":"bge-small-en","input":["local embeddings"]}'
```

The native route at `/v1/embed` additionally exposes query/document task selection, explicit
truncation, supported dimensions, normalization, exact token counts, and immutable resolved model
identity. See [API examples](docs/api-examples.md).

## Configuration and security

Configuration precedence is command line, `IMPOSSIBLE_*` environment variables, TOML, then safe
defaults. The example file is [`config/impossible-embedding.example.toml`](config/impossible-embedding.example.toml).
The default listeners are loopback-only. A non-loopback listener requires public authentication,
separate administration authentication when administration is enabled, or the explicit
`allow_insecure_remote` acknowledgement. Secret values are accepted from environment variables or
regular files, never command-line values.

TLS is intentionally not built into the process. For remote access, place the server behind a
TLS-terminating reverse proxy or service mesh, retain application authentication, and restrict the
admin and metrics routes at the network boundary. See the [operations runbook](docs/operations.md)
and [security policy](SECURITY.md).

## Container

The image runs as an unprivileged user and keeps model data in the explicit
`/var/lib/impossible-embedding` volume. It supports a read-only root filesystem.

```shell
docker build -t impossible-embedding .
docker volume create impossible-models
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --mount type=volume,source=impossible-models,target=/var/lib/impossible-embedding \
  impossible-embedding --version
```

The container keeps the safe loopback listener defaults. Publishing ports requires an explicit
non-loopback bind and the authentication policy described in the runbook.

## Development

```shell
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
pwsh -File scripts/generate-third-party-notices.ps1 -Check
pwsh -File scripts/privacy-scan.ps1
```

The first build may fetch Rust dependencies and the pinned ONNX Runtime distribution. Tests never
download model weights or send inference text over the network. Release details, supported
platforms, and honest limitations are documented in [compatibility](docs/compatibility.md) and
[releasing](docs/releasing.md).

## License

Licensed under either Apache License 2.0 or the MIT license, at your option. Distributed third-party
components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with their complete
deduplicated copyright, notice, and permission texts in
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

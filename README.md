# Impossible Embedding

Impossible Embedding is a local-first, production-oriented embedding server. Its goal is a
stable API and dependable model lifecycle while keeping inference and data on infrastructure
you control.

The server runs open embedding models locally through ONNX Runtime. It exposes native HTTP,
OpenAI-compatible HTTP, gRPC, MCP over HTTP, and MCP over stdio. Compatibility describes only the
wire format: no OpenAI account, API key, model, or service is used.

## Running

```shell
cargo run --release -- serve --preload-models bge-small-en
```

Startup only loads explicitly configured models that are already installed; it never downloads a
model. Installation is an explicit authenticated administration operation:

```shell
cargo run --release -- models install bge-small-en
cargo run --release -- models list
```

Use `--auth-env NAME` or `--auth-file PATH` for client authentication. Secret values are never
accepted as command-line arguments. The server has corresponding public and separate admin
credential-source options. Configuration precedence is command line, `IMPOSSIBLE_*` environment,
TOML, then safe loopback defaults. Run `impossible-embedding <command> --help` for the complete
bounded resource and lifecycle configuration.

For MCP stdio, stdout is reserved exclusively for JSON-RPC frames:

```shell
cargo run --release -- mcp --stdio --preload-models bge-small-en
```

## Development

Install the Rust toolchain selected by `rust-toolchain.toml`, then run:

```shell
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The ONNX smoke test may download a platform runtime during dependency setup. Once dependencies
are cached, tests and builds do not download models or send inference input over the network.

Before committing, run the repository privacy check:

```powershell
pwsh -File scripts/privacy-scan.ps1
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

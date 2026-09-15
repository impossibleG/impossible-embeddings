# Impossible Embedding

Impossible Embedding is a local-first, production-oriented embedding server. Its goal is a
stable API and dependable model lifecycle while keeping inference and data on infrastructure
you control.

This repository is in its foundation phase. The current workspace proves two critical choices:

- Protocol code generation works without a system `protoc` installation.
- A small, redistributable synthetic ONNX graph can execute in-process on CPU.

No network server is shipped yet. See [the product contract](docs/product-contract.md) for the
intended boundary and [the backlog](docs/backlog.md) for the staged delivery plan.

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

# ADR 0001: Rust server nucleus with replaceable engines

- Status: Accepted
- Date: 2026-09-15

## Context

The product needs predictable lifecycle behavior, bounded concurrency, native distribution, and
the ability to adopt inference runtimes without changing its public API.

## Decision

Use a Rust 2024 workspace for the server nucleus. Keep domain types, operational lifecycle,
protocol contracts, and inference engines in separate crates. Runtime adapters may wrap native
libraries or supervised processes; the public contract must not expose an engine's private types.

## Consequences

Rust owns reliability and packaging, while model compatibility remains replaceable. An engine can
use another implementation language later when that materially improves model support.

The workspace pins Rust 1.85.1 and exact versions of the Tonic code-generation family that support
that compiler. Exact pins prevent compatible-looking patch upgrades from silently raising the
minimum supported Rust version; dependency upgrades must pass the pinned-toolchain checks before
the manifest and lockfile move together.

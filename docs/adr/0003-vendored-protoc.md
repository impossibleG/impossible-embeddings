# ADR 0003: Vendored protocol compiler

- Status: Accepted
- Date: 2026-09-15

## Context

Requiring a system `protoc` makes fresh builds platform-dependent and difficult to reproduce.

## Decision

Build protocol definitions with `protoc-bin-vendored` and set the compiler path only inside the
protocol crate build script. Generated files remain build artifacts and are not committed.

## Consequences

Developers and CI do not install `protoc` separately. Vendored compiler packages increase the
dependency graph and are covered by dependency and license review.

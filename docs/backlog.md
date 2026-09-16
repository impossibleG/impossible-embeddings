# Delivery status and backlog

## Implemented for v0.1

- [x] Rust workspace policy, locked CI, dependency/license policy, and privacy guardrail
- [x] Dense embedding contracts, exact resolved identity, and stable public errors
- [x] Manifest-driven tokenizer, ONNX inference, pooling, normalization, and dimension handling
- [x] Verified atomic model install/cache, offline reuse, recovery, locking, and safe deletion
- [x] Bounded tokenization, queueing, dynamic batching, cancellation, deadlines, and shutdown
- [x] Native and OpenAI-compatible HTTP with checked OpenAPI, health, metrics, and status page
- [x] gRPC embedding service, error details, health, reflection, and authentication
- [x] MCP 2025-03-26 over stateless HTTP and stateful standard input/output
- [x] CLI process assembly, doctor report, model administration client, and preload policies
- [x] Non-root, read-only-root-compatible multi-stage container recipe
- [x] Windows and Linux x86-64 release archives, package smoke tests, checksums, SPDX SBOM, and
  tag-driven provenance workflow
- [x] Operational, API, compatibility, security, and release documentation

## Qualification before the first tag

- [ ] Confirm at least one curated model is marked verified only after an independently checked
  golden-vector inference run on its exact pinned ONNX artifacts
- [ ] Execute the tag workflow in the public repository and verify both native archives and their
  GitHub attestations from a fresh environment
- [ ] Run the container smoke on the published image candidate; no image publication is claimed by
  the repository workflow

## Later, not promised by v0.1

- [ ] Additional semantically qualified curated models
- [ ] Linux ARM64 and macOS native qualification
- [ ] Optional GPU execution providers behind explicit feature and compatibility boundaries
- [ ] Performance methodology without committing host-identifying benchmark data
- [ ] External TLS integration recipes for selected reverse proxies
- [ ] Reusable operational skeleton extraction for voice and OCR sibling repositories

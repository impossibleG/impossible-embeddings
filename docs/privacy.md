# Privacy and Data Handling

Inference inputs, output vectors, model paths, tokens, and credentials are sensitive data.

The server design follows these rules:

- Request content and embedding values are not logged by default.
- Telemetry uses request identifiers, sizes, timings, status codes, and aggregate counters.
- Local paths are normalized or redacted before diagnostic output.
- Model discovery is limited to documented caches and explicitly configured directories.
- No telemetry leaves the host unless an operator explicitly configures an exporter.
- Tests use synthetic fixtures and never depend on contributor-specific files or hardware facts.

Repository automation scans tracked files for common local-path and attribution leaks. This is a
guardrail, not a substitute for review.

## Panic output

Rust invokes the process panic hook before `catch_unwind`. The default hook prints the panic
payload and source location, which may contain request data or local paths supplied by a runtime
adapter. Creating the application runtime therefore installs one process-wide sanitized hook. It
emits only a fixed failure sentence; payloads and locations are discarded, while native jobs are
still caught and converted to stable public errors so workers survive.

Rust does not provide a stable thread-local panic hook. This policy consequently applies to the
whole process, including code outside native workers. Embedders must not replace it with a hook
that prints arbitrary payloads or locations. Panics remain bugs rather than an observability
channel; use structured, explicitly redacted diagnostics for operational detail.

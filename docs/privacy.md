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

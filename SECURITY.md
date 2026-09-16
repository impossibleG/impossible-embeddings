# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub private vulnerability
reporting for `impossibleG/impossible-embedding` and include affected versions, impact, and minimal
reproduction steps. Do not include real credentials, private model data, or sensitive request text.

This community project does not promise a response time. A supported-version table will be added
after the first stable release; until then, security fixes target the latest tagged release and the
default branch.

## Security boundary

Impossible Embedding performs local inference and does not use a hosted inference service. Explicit
model installation downloads only revision-pinned curated artifacts and verifies size and SHA-256
before atomic activation. Inference never silently installs a model. Offline mode disables network
installation.

The process defaults to loopback HTTP and gRPC listeners. Non-loopback configuration fails closed
unless public authentication is configured and, when enabled, administration has a separate
credential. An unauthenticated remote bind requires an explicit insecure acknowledgement. Browser
origins are exact allow-list entries; wildcards are rejected.

Secrets are loaded from named environment variables or regular files. They are not accepted as
literal command-line values and are redacted from configuration diagnostics. Standard logs,
metrics, status pages, health responses, and public errors exclude input text, embedding vectors,
credentials, host paths, and machine details.

## Deployment responsibilities

The server does not terminate TLS. Operators exposing it beyond one machine must provide TLS and
network policy with a reverse proxy, ingress, or service mesh; restrict administrative and metrics
routes; rotate both credentials; set filesystem permissions on the model cache; and run the process
as an unprivileged identity. The supplied container uses an unprivileged user and supports a
read-only root filesystem with a dedicated writable model volume.

Model artifacts and embedding outputs are untrusted data at application boundaries. Back up only
configuration and intentionally installed artifacts, protect those backups, and validate restored
cache contents through the normal startup checks.

## Dependency and release integrity

CI runs locked builds, lint, tests, dependency license/advisory policy, and a repository privacy
scan. Tagged releases re-run those gates, produce checksummed native archives and an SPDX SBOM, and
request keyless Sigstore-backed GitHub artifact attestations. Attestations depend on repository and
GitHub plan support; verify that an attestation exists rather than treating its absence as success.
Verification commands are in `docs/releasing.md`.

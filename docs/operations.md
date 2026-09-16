# Operations runbook

## Process model

`impossible-embedding serve` starts HTTP and gRPC together. Startup validates the complete merged
configuration, opens the model cache, and attempts only explicitly listed preloads. It never
downloads weights during startup. `startup_policy = "strict"` fails startup when any preload fails;
`best_effort` starts and exposes the failure through readiness and the model catalog.

Use `impossible-embedding doctor` with the same configuration before rollout. Its JSON is deliberately
aggregate and does not reveal filesystem paths or hardware details.

## Configuration

The precedence order is CLI > environment > TOML > defaults. Every environment key begins with
`IMPOSSIBLE_`; the CLI help is the authoritative option list. Unknown, duplicate, malformed, or
conflicting configuration fails closed.

Keep secrets out of TOML and process arguments. Configure `auth_env`/`admin_auth_env` or
`auth_file`/`admin_auth_file`; the referenced environment variable or regular file contains the
secret. Public and administration credentials must be distinct for remote operation. Rotate by
restarting with updated sources.

Loopback is the default and recommended local deployment. For remote access:

1. Terminate TLS at a reverse proxy, ingress, or service mesh.
2. Configure public and separate admin credentials.
3. Bind explicit non-loopback addresses and restrict both ports with network policy.
4. Restrict `/v1/admin/*` and `/metrics` at the proxy as a second control.
5. Set exact `allowed_origins` only for browser clients that require cross-origin access.

The application has no built-in TLS. `allow_insecure_remote` acknowledges an unauthenticated remote
listener; it is not encryption and is unsuitable for an untrusted network.

## Model lifecycle

Run administration through the CLI or authenticated HTTP endpoints:

```shell
impossible-embedding models list --server http://127.0.0.1:8080
impossible-embedding models install bge-small-en --server http://127.0.0.1:8080
impossible-embedding models load bge-small-en --server http://127.0.0.1:8080
impossible-embedding models unload bge-small-en --server http://127.0.0.1:8080
impossible-embedding models delete bge-small-en --server http://127.0.0.1:8080
```

Installation uses bounded streaming, exact sizes, hashes, per-model locks, staging directories, and
atomic promotion. Interrupted or corrupt staging data is repaired or quarantined on the next
operation. A loaded or leased model cannot be deleted. `offline = true` allows installed models but
rejects operations that require a download.

Persist the complete resolved identity returned with embeddings when reproducibility matters. An
alias is a selector, not proof of the model revision or semantics.

## Health, metrics, and logs

- `/health/live` indicates the process can serve control traffic.
- `/health/ready` is successful only when lifecycle and model readiness permit inference.
- Standard gRPC health reports the same readiness and changes to not-serving during drain.
- `/metrics` is Prometheus text and follows the public authentication policy.
- `/` is a host-detail-free aggregate status page.

Logs are structured JSON on standard error. Configure filtering with `RUST_LOG`; keep production at
`info` unless diagnosing a bounded incident. Inputs, vectors, secret values, runtime error chains,
host paths, and machine specifications are intentionally excluded.

Alert on sustained queue saturation, inference failures, readiness loss, and shutdown timeouts.
Tune limits from observed request/token distributions and latency objectives. Do not copy benchmark
numbers between different models or machines.

## Shutdown and restart

On Ctrl+C or the platform termination signal, listeners stop accepting work, health becomes
not-ready, and admitted work drains within `shutdown_timeout_ms`. The process exits with failure if
the bound is exceeded. Set an orchestrator termination grace period above that bound. Do not send a
hard kill until the grace period expires; model installs use durable activation but hard termination
can still leave quarantine or staging cleanup for the next start.

For rolling replacement, remove the instance from traffic after readiness changes, then wait for
normal process exit. The model cache may persist across replacements but must not be shared over an
unsafe or semantics-changing filesystem.

## Container operation

The image runs as UID/GID 65532, writes logs to standard error, and needs only the explicit
`/var/lib/impossible-embedding` model volume. A read-only root filesystem plus a small `/tmp` tmpfs is
supported. Do not bake credentials or downloaded model weights into the image. Mount a configuration
file read-only when used.

The image intentionally keeps loopback defaults. To publish a port, explicitly select non-loopback
HTTP/gRPC binds and satisfy the authentication checks. Prefer a private container network behind a
TLS proxy rather than publishing both ports directly.

## Backup, restore, and upgrades

Configuration, credential sources, and the model cache are the only persistent state. Stop or drain
the process before a filesystem-level backup. Restore into a private directory owned by the runtime
identity and let startup revalidate installed artifacts. Never restore staging, lock, or quarantine
content as an active model.

Before an upgrade, verify archive checksums and available provenance attestations, read release
notes, run `doctor`, and canary one instance. v1 protocol fields and error codes are frozen, while
operational defaults may become stricter in a minor release. Rollback uses the previous executable
against the same cache only when its release notes declare manifest compatibility.

## Troubleshooting

- Not ready with no models: install and load a curated model, or configure an installed preload.
- Preload failure: list model states, confirm the exact revision is installed, and compare strict vs
  best-effort startup policy.
- Authentication failure: confirm the intended public/admin source exists and contains no newline.
- Queue full: reduce client concurrency or increase bounded capacity only after checking memory and
  latency.
- Offline install failure: expected when the selected exact model is absent; temporarily use an
  explicit online install or provision the verified cache out of band.
- Container permission failure: ensure the model volume is writable by UID/GID 65532.

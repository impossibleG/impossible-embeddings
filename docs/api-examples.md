# API examples

Examples use loopback without authentication. When authentication is configured, add
`Authorization: Bearer <token>` over a TLS-protected connection. Never place a real token in shell
history or committed files.

## OpenAI-compatible HTTP

This endpoint is format compatibility only; it never contacts OpenAI.

```shell
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H "content-type: application/json" \
  -d '{"model":"bge-small-en","input":["one","two"],"encoding_format":"float"}'
```

Set an existing SDK's base URL to `http://127.0.0.1:8080/v1` and use any non-empty placeholder API
key only when that SDK requires one syntactically. The server ignores no credentials: if server
authentication is enabled, the SDK key must be the configured public credential.

## Native HTTP

```shell
curl -sS http://127.0.0.1:8080/v1/embed \
  -H "content-type: application/json" \
  -d '{"model":"bge-small-en","input":["search phrase"],"task":"query","truncation":"reject","normalize":true}'
```

The response reports vectors, exact token usage, and the resolved canonical model id, immutable
revision, runtime, artifact fingerprint, and semantic fingerprint. See `docs/openapi-v1.json` for
the normative HTTP contract.

## Administration

```shell
curl -sS http://127.0.0.1:8080/v1/admin/models/install \
  -H "content-type: application/json" \
  -d '{"model":"bge-small-en"}'
curl -sS http://127.0.0.1:8080/v1/admin/models/load \
  -H "content-type: application/json" \
  -d '{"model":"bge-small-en"}'
curl -sS http://127.0.0.1:8080/v1/models
```

Install is explicit and may download. Load, unload, delete, and inference do not silently install.

## gRPC

The descriptor is reflected and the source contract is `api/embedding.proto` in native archives.
With `grpcurl`:

```shell
grpcurl -plaintext -d '{"model":"bge-small-en","input":["local text"],"task":"EMBEDDING_TASK_DOCUMENT"}' \
  127.0.0.1:50051 impossible.embedding.v1.EmbeddingService/Embed
grpcurl -plaintext -d '{"service":"impossible.embedding.v1.EmbeddingService"}' \
  127.0.0.1:50051 grpc.health.v1.Health/Check
```

Use `-H 'authorization: Bearer ...'` when public authentication is configured. Application failures
include `impossible.embedding.v1.PublicErrorDetail` in gRPC status details.

## MCP over HTTP

The HTTP adapter is stateless and implements MCP revision 2025-03-26. It exposes only `embed` and
`list_models`; administrative tools are deliberately absent.

```shell
curl -sS http://127.0.0.1:8080/mcp \
  -H "content-type: application/json" \
  -H "accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"local-client","version":"1"}}}'
```

## MCP over standard input/output

```shell
impossible-embedding mcp --stdio --preload-models bge-small-en --offline
```

Standard output is reserved for newline-delimited JSON-RPC frames. Structured logs remain on
standard error. Stdio enforces initialize/initialized sequencing for its connection.

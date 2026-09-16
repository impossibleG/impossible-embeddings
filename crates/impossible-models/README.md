# impossible-models

`impossible-models` owns the immutable model catalog and local artifact lifecycle. It intentionally
does not execute model repository code, search an entire machine, or accept arbitrary download
origins.

## Cache contract

The caller chooses the cache root. Exact model identities are mapped by the canonical semantic
SHA-256 fingerprint so registry names can never become filesystem paths and the same revision and
bytes with different inference settings cannot share cache state:

```text
<root>/
  locks/<identity-key>.lock
  staging/<identity-key>.partial/
  transactions/<identity-key>.repair
  transactions/<identity-key>.previous/
  quarantine/<identity-key>.<process>.<sequence>.invalid/
  models/<identity-key>/
    manifest.json
    <declared artifacts>
```

An installation is visible under `models` only after its size and SHA-256 checks pass. A cancelled
or interrupted transfer can leave only a `.partial` staging directory; the next installation under
the same per-model lock clears it and starts from a known state.
If the exact final identity is corrupt or its stored manifest is missing, a successful reinstall
moves that state to `quarantine` and atomically promotes the verified replacement. Quarantine names
contain no host or user information. Repair uses a durable transaction marker and deterministic
backup path; after interruption, the next status/install/import operation completes the prepared
promotion or restores the displaced state, then clears the transaction.
Promotion and recovery flush the marker file and affected parent directories around every rename
and marker removal. Unix uses directory `fsync`; Windows opens directory handles with backup
semantics and calls `FlushFileBuffers` through `sync_all`. Actual guarantees still depend on the
filesystem, storage device, and their write-cache configuration.

Discovery searches only directories supplied as `DiscoveryRoot` values. Import reads only a
caller-supplied directory and rejects symbolic-link artifacts. Delete derives one content-addressed
target from the requested identity, requires the complete stored manifest to match, verifies
containment, and rejects symbolic links and Windows reparse points before removal. A verified model
capability holds a shared in-use lease; deletion and replacement cannot proceed while a runtime
retains that capability.

Artifact paths follow a deliberately conservative portable subset: slash-separated ASCII
alphanumeric, dash, underscore, and dot components. Empty components, traversal, backslashes,
colons/alternate data streams, trailing dots or spaces, Unicode filesystem aliases, DOS device
names, and case-insensitive collisions are rejected before filesystem access. `manifest.json` is
reserved for the installed identity record.

## Trust states

Artifact integrity and semantic compatibility are separate. `IntegrityVerified` means every local
byte matches the immutable manifest. `Loadable` additionally requires both the manifest claim and
an independently configured trust-root fingerprint covering its complete inference contract. A
custom manifest cannot make itself loadable by claiming `Verified`. The default trust root is built
only from repository-curated manifests; operators may supply separately authenticated evidence
records explicitly. The initial three catalog entries deliberately remain
`Unverified`: their upstream revisions, sizes, and hashes are pinned, but this repository has not
yet committed runtime golden-vector evidence. Server integrations must not advertise them as ready
until that evidence exists. BGE Small English v1.5 is the first `Verified` entry; its independent
PyTorch oracle, tokenizer IDs, vector digests, tolerances, and lifecycle results are recorded in
`docs/model-qualification/bge-small-en-v1.5.json`. The E5 and Nomic entries remain unverified.

## Network policy

Downloads require an explicit origin allowlist. HTTPS is mandatory outside loopback-only test
fixtures, redirects are checked hop by hop, response and streamed byte counts are bounded, and
offline mode returns before issuing a request. The default policy contains the exact Hub, Xet, and
CDN origins in Hugging Face's published download allowlist; it does not trust an `hf.co` wildcard.
Operators with stricter egress policy can replace that list.

Curated manifests contain no weights. Their licenses are upstream assertions, not legal advice.

## ONNX artifact limitation

The v0.1 ONNX adapter accepts one self-contained `.onnx` artifact whose tensor weights are embedded
in the protobuf. ONNX graphs using `external_data` or `data_location = EXTERNAL` are rejected before
ONNX Runtime is initialized. Sidecar weight files are intentionally outside the v0.1 artifact
contract, even if they are separately declared in a manifest.

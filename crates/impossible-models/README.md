# impossible-models

`impossible-models` owns the immutable model catalog and local artifact lifecycle. It intentionally
does not execute model repository code, search an entire machine, or accept arbitrary download
origins.

## Cache contract

The caller chooses the cache root. Exact model identities are mapped to a SHA-256 key so registry
names can never become filesystem paths:

```text
<root>/
  locks/<identity-key>.lock
  staging/<identity-key>.partial/
  models/<identity-key>/
    manifest.json
    <declared artifacts>
```

An installation is visible under `models` only after its size and SHA-256 checks pass. A cancelled
or interrupted transfer can leave only a `.partial` staging directory; the next installation under
the same per-model lock clears it and starts from a known state.

Discovery searches only directories supplied as `DiscoveryRoot` values. Import reads only a
caller-supplied directory and rejects symbolic-link artifacts. Delete derives one content-addressed
target from the requested identity, checks the stored identity, verifies containment, and rejects
symbolic links before removal.

## Trust states

Artifact integrity and semantic compatibility are separate. `IntegrityVerified` means every local
byte matches the immutable manifest. `Loadable` additionally requires a curated manifest carrying
independent semantic verification evidence. The initial three catalog entries deliberately remain
`Unverified`: their upstream revisions, sizes, and hashes are pinned, but this repository has not
yet committed runtime golden-vector evidence. Server integrations must not advertise them as ready
until that evidence exists.

## Network policy

Downloads require an explicit origin allowlist. HTTPS is mandatory outside loopback-only test
fixtures, redirects are checked hop by hop, response and streamed byte counts are bounded, and
offline mode returns before issuing a request. Default policy permits only `https://huggingface.co`.

Curated manifests contain no weights. Their licenses are upstream assertions, not legal advice.

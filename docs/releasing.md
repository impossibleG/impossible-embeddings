# Releasing

## Maintainer procedure

1. Update the single workspace version in `Cargo.toml` and regenerate `Cargo.lock` if required.
2. Regenerate `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES.txt`, then run the generator again with `-Check`.
3. Run the complete commands in the README from a clean checkout.
4. Commit the version change, then create and push an annotated `vMAJOR.MINOR.PATCH` tag.
5. Let `.github/workflows/release.yml` publish the release. Do not upload hand-built replacements.

The workflow rejects a tag that differs from the workspace version. It runs locked quality,
dependency, advisory, privacy, and notice gates; builds the hardened container; creates Windows and
Linux x86-64 native archives; executes each packaged binary and its HTTP process; generates an SPDX
SBOM for each target and SHA-256 checksums; and requests keyless Sigstore-backed GitHub attestations before creating
the GitHub release. An existing release is never overwritten.

The repository does not claim that a workflow feature succeeded until the tag's run is green. In
particular, artifact attestation availability depends on GitHub repository visibility and plan.

## Consumer verification

Download an archive and `SHA256SUMS` from the same release, then verify the archive:

```shell
sha256sum --check SHA256SUMS --ignore-missing
```

On Windows PowerShell, compare:

```powershell
(Get-FileHash .\impossible-embedding-0.1.0-x86_64-pc-windows-msvc.zip -Algorithm SHA256).Hash.ToLowerInvariant()
```

When the release displays a GitHub attestation, verify its keyless signature and workflow identity:

```shell
gh attestation verify impossible-embedding-0.1.0-x86_64-unknown-linux-gnu.tar.gz \
  --repo impossibleG/impossible-embedding
```

A checksum detects corruption only when its source is trusted. The attestation additionally binds
the artifact digest to the repository's tag workflow. Treat a missing or unverifiable attestation as
missing provenance, not as a successful verification.

## Archive contract

Each native archive contains one versioned top-level directory and no model weights. It contains the
executable, any runtime libraries emitted by the target build, configuration example, HTTP and
protobuf schemas, project licenses, ONNX Runtime license, third-party inventory, deduplicated full
copyright/license/notice texts, a target-specific SPDX SBOM, README, and security policy. The SBOM
records the checksum-pinned ONNX Runtime 1.22.0 distribution selected by `ort-sys` and whether the
packaged target statically or dynamically links it. The adjacent
`.sha256` file is convenient per-archive verification; `SHA256SUMS` covers all primary assets and the
target-specific SBOMs.

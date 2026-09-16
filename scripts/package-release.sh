#!/usr/bin/env bash
set -euo pipefail

target="${1:-x86_64-unknown-linux-gnu}"
output_directory="${2:-dist}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$repo_root/Cargo.toml" | head -n 1)"
if [[ -z "$version" ]]; then
  version="$(cargo metadata --locked --no-deps --format-version 1 | python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "impossible-server"))')"
fi

target_root="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$target_root" != /* ]]; then target_root="$repo_root/$target_root"; fi
profile_root="$target_root/$target/release"
export CARGO_INCREMENTAL=0
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$repo_root=. --remap-path-prefix=${HOME:-/nonexistent}=~"
cargo build --locked --release --target "$target" --bin impossible-embedding

binary="$profile_root/impossible-embedding"
[[ -x "$binary" ]] || { echo "release executable is missing" >&2; exit 1; }
mapfile -t runtime_libraries < <(find "$profile_root" -maxdepth 1 \( -type f -o -type l \) -name 'libonnxruntime.so*' -print | sort)

mkdir -p "$repo_root/$output_directory"
package_name="impossible-embedding-$version-$target"
temporary="$(mktemp -d)"
trap 'rm -rf -- "$temporary"' EXIT
stage="$temporary/$package_name"
mkdir -p "$stage/config" "$stage/api"
cp -P "$binary" "$stage/"
if (( ${#runtime_libraries[@]} > 0 )); then cp -P "${runtime_libraries[@]}" "$stage/"; fi
cp "$repo_root/README.md" "$repo_root/SECURITY.md" "$repo_root/LICENSE-MIT" "$repo_root/LICENSE-APACHE" "$repo_root/THIRD_PARTY_NOTICES.md" "$repo_root/THIRD_PARTY_LICENSES.txt" "$stage/"
cp -R "$repo_root/licenses" "$stage/licenses"
cp "$repo_root/config/impossible-embedding.example.toml" "$stage/config/"
cp "$repo_root/docs/openapi-v1.json" "$repo_root/crates/impossible-protocol/proto/embedding.proto" "$stage/api/"

command -v pwsh >/dev/null || { echo "PowerShell is required to generate the release SBOM" >&2; exit 1; }
linkage=static
if (( ${#runtime_libraries[@]} > 0 )); then linkage=dynamic; fi
sbom="$repo_root/$output_directory/$package_name.spdx.json"
pwsh -NoProfile -File "$repo_root/scripts/generate-sbom.ps1" -Target "$target" -OnnxRuntimeLinkage "$linkage" -OutputPath "$sbom"
cp "$sbom" "$stage/sbom.spdx.json"

archive="$repo_root/$output_directory/$package_name.tar.gz"
epoch="${SOURCE_DATE_EPOCH:-$(git -C "$repo_root" log -1 --format=%ct)}"
tar --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner -C "$temporary" -cf - "$package_name" | gzip -n > "$archive"
(cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
printf '%s\n' "$archive"

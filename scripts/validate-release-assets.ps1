$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Require-Text {
    param(
        [Parameter(Mandatory = $true)][string]$Content,
        [Parameter(Mandatory = $true)][string]$Needle,
        [Parameter(Mandatory = $true)][string]$Description
    )

    if (-not $Content.Contains($Needle, [StringComparison]::Ordinal)) {
        throw "release validation failed: $Description"
    }
}

$dockerfile = [IO.File]::ReadAllText((Join-Path $repoRoot "Dockerfile"))
$dockerignore = [IO.File]::ReadAllText((Join-Path $repoRoot ".dockerignore"))
$release = [IO.File]::ReadAllText((Join-Path $repoRoot ".github/workflows/release.yml"))
$workflows = Get-ChildItem -LiteralPath (Join-Path $repoRoot ".github/workflows") -File -Filter "*.yml"

foreach ($required in @(".git", "target", "dist", ".env", ".env.*", "*.key", "*.pem", "*.token", "credentials*", "secrets*")) {
    if ($dockerignore -notmatch "(?m)^$([regex]::Escape($required))$") {
        throw "release validation failed: Docker context does not exclude $required"
    }
}

Require-Text $dockerfile "USER 65532:65532" "container must run as the dedicated unprivileged identity"
Require-Text $dockerfile 'ENTRYPOINT ["/usr/local/bin/impossible-embedding"]' "container must use the binary as its exec-form entrypoint"
Require-Text $dockerfile 'VOLUME ["/var/lib/impossible-embedding"]' "container must declare its writable model cache"

Require-Text $release 'cargo test --locked --package impossible-server --features e2e-fixture --test binary_e2e -- --test-threads=1' "tag workflow must run the real-binary transport test"
foreach ($required in @("--read-only", "--tmpfs /tmp:rw,noexec,nosuid,size=16m", '--mount type=volume,source="$volume",target=/var/lib/impossible-embedding', "/health/live", "/openapi.json", "{{.Config.User}}")) {
    Require-Text $release $required "container release smoke is incomplete"
}
Require-Text $release "Require tag and package version agreement" "tag/version agreement gate is missing"
Require-Text $release "permissions:`n  contents: read" "workflow default permissions must remain read-only"

foreach ($workflow in $workflows) {
    $content = [IO.File]::ReadAllText($workflow.FullName)
    foreach ($match in [regex]::Matches($content, '(?m)^\s*-\s+uses:\s*([^@\s]+)@([^\s#]+)')) {
        if ($match.Groups[2].Value -notmatch '^[0-9a-f]{40}$') {
            throw "release validation failed: action reference is not pinned to a full commit"
        }
    }
}

Write-Output "Release asset validation passed."

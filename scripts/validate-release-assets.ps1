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
Require-Text $dockerfile "/usr/share/licenses/impossible-embedding/" "container must install project and dependency notices"
Require-Text $dockerfile "THIRD_PARTY_LICENSES.txt" "container must install full third-party license texts"

Require-Text $release 'cargo test --locked --package impossible-server --features e2e-fixture --test binary_e2e -- --test-threads=1' "tag workflow must run the real-binary transport test"
foreach ($required in @("--read-only", "--tmpfs /tmp:rw,noexec,nosuid,size=16m", '--mount type=volume,source="$volume",target=/var/lib/impossible-embedding', "/health/live", "/openapi.json", "{{.Config.User}}", "/usr/share/licenses/impossible-embedding/THIRD_PARTY_LICENSES.txt")) {
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

$licenseBundle = [IO.File]::ReadAllText((Join-Path $repoRoot "THIRD_PARTY_LICENSES.txt"))
foreach ($required in @("PACKAGE: tokio ", "Permission is hereby granted", "Apache License", "Redistribution and use in source and binary forms")) {
    Require-Text $licenseBundle $required "third-party bundle is missing representative MIT, Apache, or BSD content"
}

$temporary = Join-Path ([IO.Path]::GetTempPath()) ("impossible-sbom-validation-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $temporary | Out-Null
try {
    foreach ($case in @(
        @{ Target = "x86_64-pc-windows-msvc"; Linkage = "static"; Relationship = "STATIC_LINK" },
        @{ Target = "x86_64-unknown-linux-gnu"; Linkage = "dynamic"; Relationship = "DYNAMIC_LINK" }
    )) {
        $path = Join-Path $temporary "$($case.Target).json"
        & (Join-Path $repoRoot "scripts/generate-sbom.ps1") -OutputPath $path -Target $case.Target -OnnxRuntimeLinkage $case.Linkage | Out-Null
        $sbom = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        $onnx = @($sbom.packages | Where-Object name -eq "onnxruntime")
        if ($onnx.Count -ne 1 -or $onnx[0].versionInfo -ne "1.22.0" -or $onnx[0].licenseDeclared -ne "MIT") {
            throw "release validation failed: ONNX Runtime package metadata is incomplete"
        }
        if ($onnx[0].downloadLocation -notmatch [regex]::Escape($case.Target) -or
            $onnx[0].checksums.checksumValue -notmatch '^[0-9a-f]{64}$' -or
            $onnx[0].sourceInfo -notmatch 'github.com/microsoft/onnxruntime/tree/v1.22.0') {
            throw "release validation failed: ONNX Runtime source/download/checksum is incomplete"
        }
        $relations = @($sbom.relationships | Where-Object relatedSpdxElement -eq $onnx[0].SPDXID)
        if (-not ($relations.relationshipType -contains "DEPENDS_ON") -or -not ($relations.relationshipType -contains $case.Relationship)) {
            throw "release validation failed: ONNX Runtime linkage/dependency relationship is incomplete"
        }
    }
} finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Recurse -Force }
}

Write-Output "Release asset validation passed."

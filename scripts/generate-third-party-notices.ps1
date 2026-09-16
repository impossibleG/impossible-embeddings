param(
    [switch]$Check,
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputPath) {
    $OutputPath = Join-Path $repoRoot "THIRD_PARTY_NOTICES.md"
}

Push-Location $repoRoot
try {
    $metadataJson = (& cargo metadata --locked --format-version 1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed"
    }
} finally {
    Pop-Location
}

$metadata = $metadataJson | ConvertFrom-Json
$workspace = [System.Collections.Generic.HashSet[string]]::new([string[]]$metadata.workspace_members)
$dependencies = $metadata.packages |
    Where-Object { -not $workspace.Contains([string]$_.id) } |
    Sort-Object name, version, source

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Third-party notices")
$lines.Add("")
$lines.Add("This file is generated from the locked Rust dependency graph by ``scripts/generate-third-party-notices.ps1``. Do not edit it by hand.")
$lines.Add("")
$lines.Add("Impossible Embedding is distributed under MIT OR Apache-2.0. Native packages also redistribute the ONNX Runtime shared library under the MIT license; its license text is included at ``licenses/ONNXRUNTIME-LICENSE``.")
$lines.Add("")
$lines.Add("The inventory below records declared package licenses. The complete corresponding license text is included in each source package and remains available from its registry or repository. ``cargo deny`` validates this graph against the repository license policy.")
$lines.Add("")
$lines.Add("| Package | Version | Declared license | Source |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($package in $dependencies) {
    $license = if ([string]::IsNullOrWhiteSpace([string]$package.license)) { "NOASSERTION" } else { [string]$package.license }
    $source = if ([string]$package.source -like "registry+*") { "crates.io" } elseif ([string]$package.source -like "git+*") { "Git" } else { "other" }
    $lines.Add("| ``$($package.name)`` | $($package.version) | ``$license`` | $source |")
}
$content = ($lines -join "`n") + "`n"

if ($Check) {
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "third-party notice output is missing"
    }
    $existing = [IO.File]::ReadAllText($OutputPath).Replace("`r`n", "`n")
    if ($existing -cne $content) {
        throw "THIRD_PARTY_NOTICES.md is stale; regenerate it with scripts/generate-third-party-notices.ps1"
    }
    Write-Output "Third-party notices are current for $($dependencies.Count) dependencies."
    exit 0
}

[IO.File]::WriteAllText($OutputPath, $content, [Text.UTF8Encoding]::new($false))
Write-Output "Generated notices for $($dependencies.Count) dependencies."

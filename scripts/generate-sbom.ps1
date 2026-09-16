param(
    [string]$OutputPath = "dist/impossible-embedding.spdx.json",
    [ValidateSet("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu")]
    [string]$Target = "x86_64-unknown-linux-gnu",
    [ValidateSet("static", "dynamic")]
    [string]$OnnxRuntimeLinkage = "dynamic"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$absoluteOutput = if ([IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $repoRoot $OutputPath }

Push-Location $repoRoot
try {
    $metadataJson = (& cargo metadata --locked --format-version 1 --filter-platform $Target | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "cargo metadata failed" }
    $lockDigest = (Get-FileHash -LiteralPath "Cargo.lock" -Algorithm SHA256).Hash.ToLowerInvariant()
} finally {
    Pop-Location
}

$metadata = $metadataJson | ConvertFrom-Json
$workspace = [System.Collections.Generic.HashSet[string]]::new([string[]]$metadata.workspace_members)
$server = $metadata.packages | Where-Object { $_.name -eq "impossible-server" -and $workspace.Contains([string]$_.id) } | Select-Object -First 1
if (-not $server) { throw "unable to find impossible-server package" }

$nodeById = @{}
foreach ($node in $metadata.resolve.nodes) { $nodeById[[string]$node.id] = $node }
$includedIds = [System.Collections.Generic.HashSet[string]]::new()
$pending = [System.Collections.Generic.Queue[string]]::new()
$pending.Enqueue([string]$server.id)
while ($pending.Count -gt 0) {
    $id = $pending.Dequeue()
    if (-not $includedIds.Add($id)) { continue }
    $node = $nodeById[$id]
    if (-not $node) { continue }
    foreach ($dependency in $node.deps) {
        $isRuntime = @($dependency.dep_kinds | Where-Object { $null -eq $_.kind }).Count -gt 0
        if ($isRuntime) { $pending.Enqueue([string]$dependency.pkg) }
    }
}

$selected = @($metadata.packages | Where-Object { $includedIds.Contains([string]$_.id) } | Sort-Object name, version, id)
$idByCargoId = @{}
$usedIds = @{}
$packages = [System.Collections.Generic.List[object]]::new()

function New-UniqueSpdxId {
    param([Parameter(Mandatory = $true)][string]$Stem)
    $candidate = "SPDXRef-Package-$Stem"
    $suffix = 1
    while ($usedIds.ContainsKey($candidate)) {
        $suffix++
        $candidate = "SPDXRef-Package-$Stem-$suffix"
    }
    $usedIds[$candidate] = $true
    return $candidate
}

$artifactId = New-UniqueSpdxId (("impossible-embedding-native-$Target") -replace '[^A-Za-z0-9.-]', '-')
$packages.Add([ordered]@{
    SPDXID = $artifactId
    name = "impossible-embedding-native-$Target"
    versionInfo = [string]$server.version
    downloadLocation = "NOASSERTION"
    filesAnalyzed = $false
    primaryPackagePurpose = "APPLICATION"
    licenseConcluded = "NOASSERTION"
    licenseDeclared = "MIT OR Apache-2.0"
    copyrightText = "Copyright (c) 2026 Impossible Embedding contributors"
    sourceInfo = "Native release artifact for target $Target"
})

foreach ($package in $selected) {
    $stem = ([string]$package.name -replace '[^A-Za-z0-9.-]', '-') + "-" + ([string]$package.version -replace '[^A-Za-z0-9.-]', '-')
    $candidate = New-UniqueSpdxId $stem
    $idByCargoId[[string]$package.id] = $candidate
    $license = if ([string]::IsNullOrWhiteSpace([string]$package.license)) {
        "NOASSERTION"
    } else {
        ([string]$package.license).Replace(" / ", " OR ").Replace("/", " OR ")
    }
    $download = if ([string]$package.source -like "registry+*") {
        "https://crates.io/api/v1/crates/$($package.name)/$($package.version)/download"
    } elseif ([string]::IsNullOrWhiteSpace([string]$package.source)) {
        "NOASSERTION"
    } else {
        [string]$package.source
    }
    $entry = [ordered]@{
        SPDXID = $candidate
        name = [string]$package.name
        versionInfo = [string]$package.version
        downloadLocation = $download
        filesAnalyzed = $false
        licenseConcluded = "NOASSERTION"
        licenseDeclared = $license
        copyrightText = "NOASSERTION"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$package.checksum)) {
        $entry.checksums = @([ordered]@{ algorithm = "SHA256"; checksumValue = ([string]$package.checksum).ToLowerInvariant() })
    }
    $packages.Add($entry)
}

$ortSys = $selected | Where-Object name -eq "ort-sys" | Select-Object -First 1
if (-not $ortSys) { throw "target runtime graph does not contain ort-sys" }
$ortRoot = Split-Path -Parent ([string]$ortSys.manifest_path)
$ortBuild = [IO.File]::ReadAllText((Join-Path $ortRoot "build.rs"))
$versionMatch = [regex]::Match($ortBuild, 'const ONNXRUNTIME_VERSION: &str = "([0-9]+\.[0-9]+\.[0-9]+)";')
if (-not $versionMatch.Success) { throw "unable to determine ONNX Runtime version from ort-sys" }
$onnxVersion = $versionMatch.Groups[1].Value
$distribution = Get-Content -LiteralPath (Join-Path $ortRoot "dist.txt") |
    Where-Object { $_ -and -not $_.StartsWith("#") } |
    ForEach-Object { ,($_ -split "`t") } |
    Where-Object { $_[0] -eq "none" -and $_[1] -eq $Target } |
    Select-Object -First 1
if (-not $distribution -or $distribution.Count -ne 4) { throw "ort-sys has no CPU distribution for target $Target" }
$onnxUrl = [string]$distribution[2]
$onnxChecksum = ([string]$distribution[3]).ToLowerInvariant()
if ($onnxUrl -notmatch [regex]::Escape("ms@$onnxVersion")) { throw "ONNX Runtime distribution URL/version mismatch" }
if ($onnxChecksum -notmatch '^[0-9a-f]{64}$') { throw "ONNX Runtime distribution checksum is invalid" }

$onnxId = New-UniqueSpdxId "onnxruntime-$onnxVersion-$Target"
$os = if ($Target -like "*-windows-*") { "windows" } else { "linux" }
$packages.Add([ordered]@{
    SPDXID = $onnxId
    name = "onnxruntime"
    versionInfo = $onnxVersion
    downloadLocation = $onnxUrl
    filesAnalyzed = $false
    primaryPackagePurpose = "LIBRARY"
    checksums = @([ordered]@{ algorithm = "SHA256"; checksumValue = $onnxChecksum })
    licenseConcluded = "MIT"
    licenseDeclared = "MIT"
    copyrightText = "Copyright (c) Microsoft Corporation. All rights reserved."
    sourceInfo = "Source: https://github.com/microsoft/onnxruntime/tree/v$onnxVersion; CPU distribution selected and checksum-verified by ort-sys $($ortSys.version); packaged linkage: $OnnxRuntimeLinkage."
    externalRefs = @([ordered]@{
        referenceCategory = "PACKAGE-MANAGER"
        referenceType = "purl"
        referenceLocator = "pkg:generic/onnxruntime@$onnxVersion?arch=x86_64&os=$os"
    })
})

$relationships = [System.Collections.Generic.List[object]]::new()
$relationships.Add([ordered]@{ spdxElementId = "SPDXRef-DOCUMENT"; relationshipType = "DESCRIBES"; relatedSpdxElement = $artifactId })
$relationships.Add([ordered]@{ spdxElementId = $artifactId; relationshipType = "CONTAINS"; relatedSpdxElement = $idByCargoId[[string]$server.id] })
$linkRelationship = if ($OnnxRuntimeLinkage -eq "static") { "STATIC_LINK" } else { "DYNAMIC_LINK" }
$relationships.Add([ordered]@{ spdxElementId = $artifactId; relationshipType = $linkRelationship; relatedSpdxElement = $onnxId })
$relationships.Add([ordered]@{ spdxElementId = $idByCargoId[[string]$ortSys.id]; relationshipType = "DEPENDS_ON"; relatedSpdxElement = $onnxId })
foreach ($nodeId in @($includedIds | Sort-Object)) {
    $node = $nodeById[$nodeId]
    foreach ($dependency in $node.deps) {
        $isRuntime = @($dependency.dep_kinds | Where-Object { $null -eq $_.kind }).Count -gt 0
        if ($isRuntime -and $includedIds.Contains([string]$dependency.pkg)) {
            $relationships.Add([ordered]@{
                spdxElementId = $idByCargoId[[string]$node.id]
                relationshipType = "DEPENDS_ON"
                relatedSpdxElement = $idByCargoId[[string]$dependency.pkg]
            })
        }
    }
}

$created = if ($env:SOURCE_DATE_EPOCH) {
    [DateTimeOffset]::FromUnixTimeSeconds([long]$env:SOURCE_DATE_EPOCH).UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
} else {
    [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
}
$document = [ordered]@{
    spdxVersion = "SPDX-2.3"
    dataLicense = "CC0-1.0"
    SPDXID = "SPDXRef-DOCUMENT"
    name = "impossible-embedding-$($server.version)-$Target-$OnnxRuntimeLinkage"
    documentNamespace = "https://github.com/impossibleG/impossible-embedding/sbom/$($server.version)/$Target/$OnnxRuntimeLinkage/$lockDigest"
    creationInfo = [ordered]@{
        created = $created
        creators = @("Tool: impossible-embedding/scripts/generate-sbom.ps1")
        licenseListVersion = "3.25"
    }
    packages = $packages
    relationships = $relationships
}

$directory = Split-Path -Parent $absoluteOutput
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$json = ($document | ConvertTo-Json -Depth 12) + "`n"
[IO.File]::WriteAllText($absoluteOutput, $json, [Text.UTF8Encoding]::new($false))
Write-Output "Generated target-specific SPDX SBOM for $Target with $($packages.Count) packages ($OnnxRuntimeLinkage ONNX Runtime)."

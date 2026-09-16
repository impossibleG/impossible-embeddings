param(
    [string]$OutputPath = "dist/impossible-embedding.spdx.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$absoluteOutput = if ([IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $repoRoot $OutputPath }

Push-Location $repoRoot
try {
    $metadataJson = (& cargo metadata --locked --format-version 1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed"
    }
    $lockDigest = (Get-FileHash -LiteralPath "Cargo.lock" -Algorithm SHA256).Hash.ToLowerInvariant()
} finally {
    Pop-Location
}

$metadata = $metadataJson | ConvertFrom-Json
$workspace = [System.Collections.Generic.HashSet[string]]::new([string[]]$metadata.workspace_members)
$ordered = $metadata.packages | Sort-Object name, version, id
$idByCargoId = @{}
$usedIds = @{}
$packages = [System.Collections.Generic.List[object]]::new()

foreach ($package in $ordered) {
    $stem = ([string]$package.name -replace '[^A-Za-z0-9.-]', '-') + "-" + ([string]$package.version -replace '[^A-Za-z0-9.-]', '-')
    $candidate = "SPDXRef-Package-$stem"
    $suffix = 1
    while ($usedIds.ContainsKey($candidate)) {
        $suffix++
        $candidate = "SPDXRef-Package-$stem-$suffix"
    }
    $usedIds[$candidate] = $true
    $idByCargoId[[string]$package.id] = $candidate
    $license = if ([string]::IsNullOrWhiteSpace([string]$package.license)) {
        "NOASSERTION"
    } else {
        ([string]$package.license).Replace(" / ", " OR ").Replace("/", " OR ")
    }
    $download = if ([string]::IsNullOrWhiteSpace([string]$package.source)) { "NOASSERTION" } else { [string]$package.source }
    $packages.Add([ordered]@{
        SPDXID = $candidate
        name = [string]$package.name
        versionInfo = [string]$package.version
        downloadLocation = $download
        filesAnalyzed = $false
        licenseConcluded = "NOASSERTION"
        licenseDeclared = $license
        copyrightText = "NOASSERTION"
    })
}

$relationships = [System.Collections.Generic.List[object]]::new()
foreach ($member in ($metadata.workspace_members | Sort-Object)) {
    $relationships.Add([ordered]@{
        spdxElementId = "SPDXRef-DOCUMENT"
        relationshipType = "DESCRIBES"
        relatedSpdxElement = $idByCargoId[[string]$member]
    })
}
foreach ($node in ($metadata.resolve.nodes | Sort-Object id)) {
    foreach ($dependency in ($node.dependencies | Sort-Object)) {
        $relationships.Add([ordered]@{
            spdxElementId = $idByCargoId[[string]$node.id]
            relationshipType = "DEPENDS_ON"
            relatedSpdxElement = $idByCargoId[[string]$dependency]
        })
    }
}

$created = if ($env:SOURCE_DATE_EPOCH) {
    [DateTimeOffset]::FromUnixTimeSeconds([long]$env:SOURCE_DATE_EPOCH).UtcDateTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
} else {
    [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
}
$version = ($ordered | Where-Object { $_.name -eq "impossible-server" -and $workspace.Contains([string]$_.id) } | Select-Object -First 1).version
$document = [ordered]@{
    spdxVersion = "SPDX-2.3"
    dataLicense = "CC0-1.0"
    SPDXID = "SPDXRef-DOCUMENT"
    name = "impossible-embedding-$version"
    documentNamespace = "https://github.com/impossibleG/impossible-embedding/sbom/$version/$lockDigest"
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
Write-Output "Generated SPDX SBOM with $($packages.Count) packages."

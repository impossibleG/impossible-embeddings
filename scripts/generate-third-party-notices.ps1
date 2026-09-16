param(
    [switch]$Check,
    [string]$OutputPath,
    [string]$LicenseOutputPath
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputPath) {
    $OutputPath = Join-Path $repoRoot "THIRD_PARTY_NOTICES.md"
}
if (-not $LicenseOutputPath) {
    $LicenseOutputPath = Join-Path $repoRoot "THIRD_PARTY_LICENSES.txt"
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
$lines.Add("Impossible Embedding is distributed under MIT OR Apache-2.0. Native packages also redistribute ONNX Runtime 1.22.0 under the MIT license, either statically or as a shared library as recorded in each target SBOM; its license text is included at ``licenses/ONNXRUNTIME-LICENSE``.")
$lines.Add("")
$lines.Add("The inventory below records declared package licenses. The corresponding copyright, notice, and license/permission texts are mapped and deduplicated in ``THIRD_PARTY_LICENSES.txt``. ``cargo deny`` validates this graph against the repository license policy.")
$lines.Add("")
$lines.Add("| Package | Version | Declared license | Source |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($package in $dependencies) {
    $license = if ([string]::IsNullOrWhiteSpace([string]$package.license)) { "NOASSERTION" } else { [string]$package.license }
    $source = if ([string]$package.source -like "registry+*") { "crates.io" } elseif ([string]$package.source -like "git+*") { "Git" } else { "other" }
    $lines.Add("| ``$($package.name)`` | $($package.version) | ``$license`` | $source |")
}
$content = ($lines -join "`n") + "`n"

function Normalize-Text {
    param([Parameter(Mandatory = $true)][string]$Text)
    $normalizedLines = $Text.Replace("`r`n", "`n").Replace("`r", "`n").Split("`n") |
        ForEach-Object { $_.TrimEnd() }
    return (($normalizedLines -join "`n").TrimEnd() + "`n")
}

function Get-TextDigest {
    param([Parameter(Mandatory = $true)][string]$Text)
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return [Convert]::ToHexString($hash).ToLowerInvariant()
}

$packageRecords = [System.Collections.Generic.List[object]]::new()
$texts = @{}
$repositoryTexts = @{}
$filePattern = '^(?i)(AUTHORS|COPYING|COPYRIGHT|LICENCE|LICENSE|NOTICE|UNLICENSE)(?:[._-].*)?$'

foreach ($package in $dependencies) {
    $crateRoot = Split-Path -Parent ([string]$package.manifest_path)
    $files = @(Get-ChildItem -LiteralPath $crateRoot -File | Where-Object { $_.Name -match $filePattern })
    if (-not [string]::IsNullOrWhiteSpace([string]$package.license_file)) {
        $declaredFile = Join-Path $crateRoot ([string]$package.license_file)
        if (Test-Path -LiteralPath $declaredFile -PathType Leaf) {
            $files += Get-Item -LiteralPath $declaredFile
        }
    }
    $files = @($files | Sort-Object Name, FullName -Unique)
    $references = [System.Collections.Generic.List[object]]::new()
    foreach ($file in $files) {
        $text = Normalize-Text ([IO.File]::ReadAllText($file.FullName))
        $digest = Get-TextDigest $text
        $texts[$digest] = $text
        $references.Add([ordered]@{ name = $file.Name; digest = $digest; provenance = "crate" })
    }

    $repository = [string]$package.repository
    if ($references.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($repository)) {
        if (-not $repositoryTexts.ContainsKey($repository)) {
            $repositoryTexts[$repository] = @($references)
        }
    }

    $packageRecords.Add([ordered]@{
        name = [string]$package.name
        version = [string]$package.version
        license = if ([string]::IsNullOrWhiteSpace([string]$package.license)) { "NOASSERTION" } else { [string]$package.license }
        repository = if ([string]::IsNullOrWhiteSpace($repository)) { "NOASSERTION" } else { $repository }
        authors = @($package.authors | ForEach-Object { [string]$_ })
        references = $references
    })
}

# Some crates.io archives are workspace leaf/meta packages and omit the repository-level license
# file. Reuse a byte-identical text from another locked crate from that same upstream repository.
foreach ($record in $packageRecords) {
    if ($record.references.Count -eq 0 -and $repositoryTexts.ContainsKey([string]$record.repository)) {
        foreach ($reference in $repositoryTexts[[string]$record.repository]) {
            $record.references.Add([ordered]@{
                name = [string]$reference.name
                digest = [string]$reference.digest
                provenance = "same-upstream-repository"
            })
        }
    }
}

$mitPermission = @'
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'@

# Last-resort handling is explicit instead of silently pretending a crate archive contained a
# file it did not. All currently affected packages permit MIT redistribution.
foreach ($record in $packageRecords) {
    if ($record.references.Count -eq 0) {
        if ([string]$record.license -notmatch '(?i)(^|\W)MIT(\W|$)') {
            throw "dependency $($record.name) $($record.version) has no redistributable license text"
        }
        $attribution = if ($record.authors.Count -gt 0) {
            "Package metadata authors: " + ($record.authors -join "; ")
        } else {
            "Copyright retained by the upstream copyright holders identified at $($record.repository)."
        }
        $fallback = Normalize-Text ("MIT License`n`n$attribution`n`n$mitPermission")
        $digest = Get-TextDigest $fallback
        $texts[$digest] = $fallback
        $record.references.Add([ordered]@{
            name = "MIT-permission-and-metadata-attribution"
            digest = $digest
            provenance = "declared-license-fallback"
        })
    }
}

$bundle = [System.Collections.Generic.List[string]]::new()
$bundle.Add("THIRD-PARTY COPYRIGHT, NOTICE, AND LICENSE TEXTS")
$bundle.Add("")
$bundle.Add("Generated from the locked Cargo dependency graph by scripts/generate-third-party-notices.ps1.")
$bundle.Add("Each package maps to one or more SHA-256-addressed texts. Identical texts are stored once.")
$bundle.Add("")
$bundle.Add("PACKAGE INDEX")
$bundle.Add("=============")
foreach ($record in $packageRecords) {
    $bundle.Add("")
    $bundle.Add("PACKAGE: $($record.name) $($record.version)")
    $bundle.Add("DECLARED-LICENSE: $($record.license)")
    $bundle.Add("UPSTREAM: $($record.repository)")
    if ($record.authors.Count -gt 0) { $bundle.Add("METADATA-AUTHORS: $($record.authors -join '; ')") }
    foreach ($reference in $record.references) {
        $bundle.Add("TEXT: $($reference.digest) [$($reference.provenance)] $($reference.name)")
    }
}
$bundle.Add("")
$bundle.Add("DEDUPLICATED TEXTS")
$bundle.Add("==================")
foreach ($digest in @($texts.Keys | Sort-Object)) {
    $bundle.Add("")
    $bundle.Add("--------------------------------------------------------------------------------")
    $bundle.Add("TEXT-SHA256: $digest")
    $bundle.Add("--------------------------------------------------------------------------------")
    $bundle.Add($texts[$digest].TrimEnd())
}
$licenseContent = ($bundle -join "`n") + "`n"

if ($Check) {
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "third-party notice output is missing"
    }
    $existing = [IO.File]::ReadAllText($OutputPath).Replace("`r`n", "`n")
    if ($existing -cne $content) {
        throw "THIRD_PARTY_NOTICES.md is stale; regenerate it with scripts/generate-third-party-notices.ps1"
    }
    if (-not (Test-Path -LiteralPath $LicenseOutputPath -PathType Leaf)) {
        throw "third-party license text output is missing"
    }
    $existingLicenses = [IO.File]::ReadAllText($LicenseOutputPath).Replace("`r`n", "`n")
    if ($existingLicenses -cne $licenseContent) {
        throw "THIRD_PARTY_LICENSES.txt is stale; regenerate it with scripts/generate-third-party-notices.ps1"
    }
    Write-Output "Third-party notices and texts are current for $($dependencies.Count) dependencies."
    exit 0
}

[IO.File]::WriteAllText($OutputPath, $content, [Text.UTF8Encoding]::new($false))
[IO.File]::WriteAllText($LicenseOutputPath, $licenseContent, [Text.UTF8Encoding]::new($false))
Write-Output "Generated notices and $($texts.Count) deduplicated texts for $($dependencies.Count) dependencies."

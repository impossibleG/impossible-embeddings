param(
    [string]$Target = "x86_64-pc-windows-msvc",
    [string]$OutputDirectory = "dist",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$metadata = (& cargo metadata --locked --no-deps --format-version 1 | Out-String) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "cargo metadata failed" }
$version = [string](($metadata.packages | Where-Object name -eq "impossible-server" | Select-Object -First 1).version)
if ([string]::IsNullOrWhiteSpace($version)) { throw "unable to determine package version" }

$targetRoot = if ($env:CARGO_TARGET_DIR) {
    if ([IO.Path]::IsPathRooted($env:CARGO_TARGET_DIR)) { $env:CARGO_TARGET_DIR } else { Join-Path $repoRoot $env:CARGO_TARGET_DIR }
} else { Join-Path $repoRoot "target" }
$profileRoot = Join-Path (Join-Path $targetRoot $Target) "release"
$binaryName = if ($Target -like "*-windows-*") { "impossible-embedding.exe" } else { "impossible-embedding" }
$binaryPath = Join-Path $profileRoot $binaryName

if (-not $SkipBuild) {
    $previousRustFlags = $env:RUSTFLAGS
    $remaps = @("--remap-path-prefix=$repoRoot=.")
    if ($env:USERPROFILE) { $remaps += "--remap-path-prefix=$env:USERPROFILE=~" }
    $remap = $remaps -join " "
    $env:RUSTFLAGS = if ($previousRustFlags) { "$previousRustFlags $remap" } else { $remap }
    try {
        & cargo build --locked --release --target $Target --bin impossible-embedding
        if ($LASTEXITCODE -ne 0) { throw "release build failed" }
    } finally {
        $env:RUSTFLAGS = $previousRustFlags
    }
}
if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) { throw "release executable is missing" }

$runtimePattern = if ($Target -like "*-windows-*") { "onnxruntime.dll" } elseif ($Target -like "*-linux-*") { "libonnxruntime.so*" } else { "libonnxruntime.dylib*" }
$runtimeLibraries = @(Get-ChildItem -LiteralPath $profileRoot -File -Filter $runtimePattern -ErrorAction SilentlyContinue)

$outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$packageName = "impossible-embedding-$version-$Target"
$temporary = Join-Path ([IO.Path]::GetTempPath()) ("impossible-package-" + [Guid]::NewGuid().ToString("N"))
$stage = Join-Path $temporary $packageName

try {
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item -LiteralPath $binaryPath -Destination $stage
    foreach ($library in $runtimeLibraries) { Copy-Item -LiteralPath $library.FullName -Destination $stage }
    foreach ($file in @("README.md", "SECURITY.md", "LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md")) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $file) -Destination $stage
    }
    Copy-Item -LiteralPath (Join-Path $repoRoot "THIRD_PARTY_LICENSES.txt") -Destination $stage
    Copy-Item -LiteralPath (Join-Path $repoRoot "licenses") -Destination $stage -Recurse
    New-Item -ItemType Directory -Force -Path (Join-Path $stage "config") | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot "config/impossible-embedding.example.toml") -Destination (Join-Path $stage "config")
    New-Item -ItemType Directory -Force -Path (Join-Path $stage "api") | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot "docs/openapi-v1.json") -Destination (Join-Path $stage "api")
    Copy-Item -LiteralPath (Join-Path $repoRoot "crates/impossible-protocol/proto/embedding.proto") -Destination (Join-Path $stage "api")

    $linkage = if ($runtimeLibraries.Count -gt 0) { "dynamic" } else { "static" }
    $sbomName = "$packageName.spdx.json"
    $sbomPath = Join-Path $outputRoot $sbomName
    & (Join-Path $repoRoot "scripts/generate-sbom.ps1") -Target $Target -OnnxRuntimeLinkage $linkage -OutputPath $sbomPath
    if ($LASTEXITCODE -ne 0) { throw "SBOM generation failed" }
    Copy-Item -LiteralPath $sbomPath -Destination (Join-Path $stage "sbom.spdx.json")

    $archive = Join-Path $outputRoot "$packageName.zip"
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive }
    Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
    $digest = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    [IO.File]::WriteAllText("$archive.sha256", "$digest *$([IO.Path]::GetFileName($archive))`n", [Text.UTF8Encoding]::new($false))
    Write-Output $archive
} finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Recurse -Force }
}

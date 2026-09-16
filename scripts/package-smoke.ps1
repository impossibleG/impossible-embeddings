param(
    [Parameter(Mandatory = $true)][string]$Archive
)

$ErrorActionPreference = "Stop"
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$temporary = Join-Path ([IO.Path]::GetTempPath()) ("impossible-smoke-" + [Guid]::NewGuid().ToString("N"))
$process = $null
$previousLibraryPath = $env:LD_LIBRARY_PATH

function Remove-TemporaryDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    for ($attempt = 0; $attempt -lt 50; $attempt++) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -eq 49) { throw }
            Start-Sleep -Milliseconds 100
        }
    }
}

try {
    New-Item -ItemType Directory -Force -Path $temporary | Out-Null
    $checksumPath = "$archivePath.sha256"
    if (Test-Path -LiteralPath $checksumPath -PathType Leaf) {
        $declared = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        $actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($declared -cne $actual) { throw "archive checksum does not match" }
    }

    if ($archivePath.EndsWith(".zip", [StringComparison]::OrdinalIgnoreCase)) {
        Expand-Archive -LiteralPath $archivePath -DestinationPath $temporary
    } elseif ($archivePath.EndsWith(".tar.gz", [StringComparison]::OrdinalIgnoreCase)) {
        & tar -xzf $archivePath -C $temporary
        if ($LASTEXITCODE -ne 0) { throw "unable to extract release archive" }
    } else {
        throw "unsupported release archive"
    }

    $roots = @(Get-ChildItem -LiteralPath $temporary -Directory)
    if ($roots.Count -ne 1) { throw "release archive must contain one top-level directory" }
    $root = $roots[0].FullName
    foreach ($required in @(
        "README.md", "SECURITY.md", "LICENSE-MIT", "LICENSE-APACHE", "THIRD_PARTY_NOTICES.md",
        "config/impossible-embedding.example.toml", "api/openapi-v1.json", "api/embedding.proto",
        "licenses/ONNXRUNTIME-LICENSE"
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) {
            throw "release archive is missing required content"
        }
    }

    $binaryName = if ($IsWindows) { "impossible-embedding.exe" } else { "impossible-embedding" }
    $binary = Join-Path $root $binaryName
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) { throw "release executable is missing" }
    if (-not $IsWindows) { $env:LD_LIBRARY_PATH = $root }
    $versionOutput = (& $binary --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch '^impossible-embedding [0-9]+\.[0-9]+\.[0-9]+') {
        throw "release executable version smoke failed"
    }
    & $binary --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "release executable help smoke failed" }

    $escapedBackslash = [regex]::Escape([string][char]92)
    $separator = "/"
    $privatePatterns = @(
        "(?i)[a-z]:${escapedBackslash}users${escapedBackslash}[^${escapedBackslash}]+${escapedBackslash}",
        "${separator}home${separator}[^/]+${separator}",
        "${separator}Users${separator}[^/]+${separator}"
    )
    foreach ($file in (Get-ChildItem -LiteralPath $root -File -Recurse)) {
        if ($file.Name.EndsWith(".sha256", [StringComparison]::OrdinalIgnoreCase)) { continue }
        $content = [Text.Encoding]::Latin1.GetString([IO.File]::ReadAllBytes($file.FullName))
        foreach ($pattern in $privatePatterns) {
            if ([regex]::IsMatch($content, $pattern)) {
                throw "release archive contains a private build path"
            }
        }
    }

    $cache = Join-Path $temporary "cache"
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $stdout = Join-Path $temporary "server.stdout.log"
    $stderr = Join-Path $temporary "server.stderr.log"
    $started = $false
    for ($attempt = 0; $attempt -lt 5 -and -not $started; $attempt++) {
        $httpPort = Get-Random -Minimum 20000 -Maximum 44000
        $grpcPort = $httpPort + 1
        $arguments = @(
            "serve", "--http-bind", "127.0.0.1:$httpPort", "--grpc-bind", "127.0.0.1:$grpcPort",
            "--cache-directory", $cache, "--admin-api-enabled", "false", "--offline"
        )
        $process = Start-Process -FilePath $binary -ArgumentList $arguments -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $client = [Net.Http.HttpClient]::new()
        try {
            for ($poll = 0; $poll -lt 80; $poll++) {
                if ($process.HasExited) { break }
                try {
                    $response = $client.GetAsync("http://127.0.0.1:$httpPort/health/live").GetAwaiter().GetResult()
                    if ([int]$response.StatusCode -eq 200) { $started = $true; break }
                } catch { }
                Start-Sleep -Milliseconds 100
            }
            if ($started) {
                foreach ($path in @("/openapi.json", "/v1/models")) {
                    $response = $client.GetAsync("http://127.0.0.1:$httpPort$path").GetAwaiter().GetResult()
                    if ([int]$response.StatusCode -ne 200) { throw "HTTP transport smoke failed" }
                }
                $readiness = $client.GetAsync("http://127.0.0.1:$httpPort/health/ready").GetAwaiter().GetResult()
                if ([int]$readiness.StatusCode -notin @(200, 503)) { throw "readiness endpoint returned an invalid status" }
            }
        } finally {
            $client.Dispose()
            if ($process -and -not $process.HasExited) {
                Stop-Process -Id $process.Id -Force
                if (-not $process.WaitForExit(5000)) { throw "packaged server did not stop" }
            }
            if ($process) { $process.Dispose() }
            $process = $null
        }
    }
    if (-not $started) { throw "packaged server did not become live" }
    Write-Output "Release package smoke passed: $versionOutput"
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit(5000) | Out-Null
    }
    if ($process) { $process.Dispose() }
    $env:LD_LIBRARY_PATH = $previousLibraryPath
    if (Test-Path -LiteralPath $temporary) { Remove-TemporaryDirectory -Path $temporary }
}

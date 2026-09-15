$ErrorActionPreference = "Stop"

$patterns = @(
    'C:\\Users\\',
    '/Users/',
    '/home/',
    '(?i)co-authored-by:',
    '(?i)(cpu|gpu|processor|graphics card):\s+[^<]'
)

$files = git ls-files | Where-Object { $_ -ne 'scripts/privacy-scan.ps1' }
if ($LASTEXITCODE -ne 0) {
    throw "Unable to enumerate tracked files."
}

$findings = New-Object System.Collections.Generic.List[string]
foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        continue
    }

    $content = Get-Content -LiteralPath $file -Raw
    foreach ($pattern in $patterns) {
        if ($content -match $pattern) {
            $findings.Add("$file matches prohibited repository pattern")
            break
        }
    }
}

if ($findings.Count -gt 0) {
    $findings | Write-Error
    exit 1
}

Write-Output "Privacy scan passed for $($files.Count) tracked files."

$ErrorActionPreference = "Stop"

$patterns = @(
    '(?i)c:\\users\\',
    '/Users/',
    '/home/',
    '(?i)co-authored-by:',
    '(?i)(cpu|gpu|processor|graphics card):\s+[^<]'
)

function Test-ProhibitedContent {
    param([Parameter(Mandatory = $true)][string]$Content)

    foreach ($pattern in $patterns) {
        if ($Content -match $pattern) {
            return $true
        }
    }
    return $false
}

# Keep this synthetic: it proves lowercase Windows home paths are rejected without embedding a
# contributor's real account name or machine path in the repository.
$lowercaseWindowsFixture = @('c:', 'users', 'example-account', 'private.txt') -join '\'
if (-not (Test-ProhibitedContent -Content $lowercaseWindowsFixture)) {
    throw "Privacy scan self-test failed: lowercase Windows home path was not detected."
}

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
    if (Test-ProhibitedContent -Content $content) {
        $findings.Add("$file matches prohibited repository pattern")
    }
}

if ($findings.Count -gt 0) {
    $findings | Write-Error
    exit 1
}

Write-Output "Privacy scan passed for $($files.Count) tracked files."

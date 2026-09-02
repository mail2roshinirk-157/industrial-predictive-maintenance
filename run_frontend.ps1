# Self-contained runner script for the React frontend using a portable Node.js environment

$ProjectDir = $PSScriptRoot
if (-not $ProjectDir) { $ProjectDir = Get-Location }
$NodeDir = Join-Path $ProjectDir "node_portable"
$NodeZip = Join-Path $ProjectDir "node.zip"
$NodeExePath = Join-Path $NodeDir "node-v22.11.0-win-x64"

# If the old 20.11.1 version folder is there, delete it to clean up
$OldNodePath = Join-Path $NodeDir "node-v20.11.1-win-x64"
if (Test-Path $OldNodePath) {
    Write-Host "Removing old Node.js v20.11.1 version..." -ForegroundColor Yellow
    Remove-Item $OldNodePath -Recurse -Force
}

if (-not (Test-Path $NodeExePath)) {
    Write-Host "Portable Node.js not found. Downloading Node.js v22.11.0..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri "https://nodejs.org/dist/v22.11.0/node-v22.11.0-win-x64.zip" -OutFile $NodeZip
    
    Write-Host "Extracting Node.js..." -ForegroundColor Yellow
    if (-not (Test-Path $NodeDir)) {
        New-Item -ItemType Directory -Path $NodeDir | Out-Null
    }
    Expand-Archive -Path $NodeZip -DestinationPath $NodeDir -Force
    
    Write-Host "Cleaning up zip file..." -ForegroundColor Yellow
    Remove-Item $NodeZip -Force
}

# Add Node.js to path for this session
$env:PATH = "$NodeExePath;$env:PATH"

Write-Host "Using Node.js from $NodeExePath" -ForegroundColor Green
node -v
npm -v

# Navigate to frontend and run
cd (Join-Path $ProjectDir "frontend")

# Perform clean installation of dependencies if we haven't already succeeded
if (-not (Test-Path "node_modules")) {
    Write-Host "Performing clean installation of frontend dependencies..." -ForegroundColor Yellow
    npm install
}

# Force install the native binding due to npm optional dependency issues on some Windows systems
Write-Host "Ensuring rolldown native binding is installed..." -ForegroundColor Yellow
npm install @rolldown/binding-win32-x64-msvc --no-save

Write-Host "Starting React frontend dev server..." -ForegroundColor Green
npm run dev

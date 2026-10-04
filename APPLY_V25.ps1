$ErrorActionPreference = "Stop"

Write-Host "Applying V2.5.1 performance hotfix..." -ForegroundColor Cyan

node tools/apply-v25.mjs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Running audit..." -ForegroundColor Cyan
npm run site:audit
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Building..." -ForegroundColor Cyan
npm run docs:build
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "V2.5.1 build passed." -ForegroundColor Green
Write-Host "Next: npm run docs:preview"

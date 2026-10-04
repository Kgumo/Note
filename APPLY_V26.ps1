$ErrorActionPreference = "Stop"

Write-Host "Applying V2.6.1 media pipeline hotfix..." -ForegroundColor Cyan

node tools/apply-v26.mjs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Installing/syncing media dependency..." -ForegroundColor Cyan
npm install --legacy-peer-deps
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Auditing referenced media..." -ForegroundColor Cyan
npm run media:audit
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Preparing responsive media and building..." -ForegroundColor Cyan
npm run docs:build
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "V2.6.1 media pipeline build passed." -ForegroundColor Green
Write-Host "Next: npm run docs:preview"

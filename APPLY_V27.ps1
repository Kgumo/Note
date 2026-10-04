$ErrorActionPreference = "Stop"

Write-Host "Applying V2.7 production/readability layer..." -ForegroundColor Cyan

node tools/apply-v27.mjs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Running production-equivalent build..." -ForegroundColor Cyan

npm run docs:build
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "V2.7 build + media artifact verification passed." -ForegroundColor Green
Write-Host "Next: npm run docs:preview"

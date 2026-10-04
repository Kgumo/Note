$ErrorActionPreference = "Stop"

Write-Host "Applying V2.8 build-directory + knowledge-graph update..." -ForegroundColor Cyan

node tools/apply-v28.mjs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Generating docs/build knowledge branch..." -ForegroundColor Cyan
npm run knowledge:prepare
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Building the complete VitePress site..." -ForegroundColor Cyan
npm run docs:build
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "V2.8 build passed." -ForegroundColor Green
Write-Host "Next: npm run docs:preview"

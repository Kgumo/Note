$ErrorActionPreference = "Stop"

Write-Host "Applying V2.9 Obsidian compatibility + mobile layout..." -ForegroundColor Cyan

node tools/apply-v29.mjs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Auditing Obsidian wiki syntax..." -ForegroundColor Cyan
npm run obsidian:audit
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Running media audit..." -ForegroundColor Cyan
npm run media:audit
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Running full production build..." -ForegroundColor Cyan
npm run docs:build
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "V2.9 build passed." -ForegroundColor Green
Write-Host "Next: npm run docs:preview"
Write-Host "Test desktop and a mobile viewport (390x844 / 430x932)."

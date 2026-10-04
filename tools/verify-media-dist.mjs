import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const manifestFile =
  path.join(root, '.vitepress', 'cache', 'media-manifest.json')
const distDir =
  path.join(root, 'dist')

if (!fs.existsSync(distDir)) {
  console.error('Media verification failed: dist/ does not exist.')
  process.exit(1)
}

if (!fs.existsSync(manifestFile)) {
  console.warn(
    'Media manifest does not exist. No responsive-media verification was possible.'
  )
  process.exit(0)
}

const manifest =
  JSON.parse(
    fs.readFileSync(manifestFile, 'utf8')
  )

let sources = 0
let variants = 0
let missing = 0
let bytes = 0

for (
  const entry
  of Object.values(manifest.entries || {})
) {
  if (!entry?.variants?.length) continue

  sources += 1

  for (const variant of entry.variants) {
    variants += 1

    const relative =
      String(variant.url || '')
        .replace(/^\/+/, '')

    const target =
      path.join(distDir, relative)

    if (!fs.existsSync(target)) {
      missing += 1
      console.error(
        `Missing generated media in dist: ${relative}`
      )
      continue
    }

    bytes += fs.statSync(target).size
  }
}

console.log('\n=== Production media verification ===\n')
console.log(`Optimized source images: ${sources}`)
console.log(`Responsive variants in manifest: ${variants}`)
console.log(`Responsive variants missing from dist: ${missing}`)
console.log(
  `Generated media present in dist: ${(bytes / 1024 / 1024).toFixed(2)} MB`
)

if (missing > 0) {
  console.error(
    '\nBuild rejected: responsive media would be missing from GitHub Pages.'
  )
  process.exit(1)
}

console.log('\nProduction media verification passed.\n')

import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()

const ignored = new Set([
  '.git',
  'node_modules',
  'dist',
  '.vitepress',
  '.v25-backup',
  '.typography-backup-v3.1',
  '.layout-backup-v3.2'
])

const imageExt = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.avif',
  '.svg'
])

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (!ignored.has(entry.name)) {
        walk(full, files)
      }
    } else {
      files.push(full)
    }
  }

  return files
}

function human(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`
}

function rel(file) {
  return path.relative(root, file).replaceAll('\\', '/')
}

const files = walk(root)
const markdown = files.filter((file) => file.endsWith('.md'))

const externalDomains = new Map()
let legacyCount = 0
let markdownImages = 0

const externalImage =
  /!\[[^\]]*]\((https?:\/\/[^)\s]+)[^)]*\)/g

for (const file of markdown) {
  const source = fs.readFileSync(file, 'utf8')

  legacyCount += (
    source.match(
      /https:\/\/obsidiannote\.netlify\.app\//g
    ) || []
  ).length

  markdownImages += (
    source.match(/!\[[^\]]*]\([^)]+\)/g) || []
  ).length

  for (const match of source.matchAll(externalImage)) {
    try {
      const host = new URL(match[1]).hostname
      externalDomains.set(
        host,
        (externalDomains.get(host) || 0) + 1
      )
    } catch {
      // Ignore invalid URL.
    }
  }
}

const assets = files
  .filter((file) =>
    imageExt.has(path.extname(file).toLowerCase())
  )
  .map((file) => ({
    file,
    size: fs.statSync(file).size
  }))
  .sort((a, b) => b.size - a.size)

console.log('\n=== Note V2.5 performance audit ===\n')
console.log(`Markdown files: ${markdown.length}`)
console.log(`Markdown image references: ${markdownImages}`)
console.log(`Legacy Netlify refs: ${legacyCount}`)

console.log('\nExternal image domains:')
if (!externalDomains.size) {
  console.log('  none')
} else {
  for (
    const [host, count]
    of [...externalDomains.entries()]
      .sort((a, b) => b[1] - a[1])
  ) {
    console.log(`  ${host}: ${count}`)
  }
}

console.log('\nLargest local images:')
for (const item of assets.slice(0, 30)) {
  console.log(
    `  ${human(item.size).padStart(9)}  ${rel(item.file)}`
  )
}

const over1Mb = assets.filter(
  (item) => item.size >= 1024 ** 2
).length

const over500Kb = assets.filter(
  (item) => item.size >= 500 * 1024
).length

console.log('\nImage thresholds:')
console.log(`  >= 1 MB:   ${over1Mb}`)
console.log(`  >= 500 KB: ${over500Kb}`)

console.log('\n=== end audit ===\n')

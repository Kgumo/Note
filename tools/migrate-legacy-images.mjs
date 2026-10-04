import fs from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'

const root = process.cwd()
const docsDir = path.join(root, 'docs')
const outputDir = path.join(docsDir, 'assets', 'legacy')
const manifestFile =
  path.join(root, 'tools', 'legacy-image-manifest.json')

const apply = process.argv.includes('--apply')

const LEGACY_RE =
  /https:\/\/obsidiannote\.netlify\.app\/[^\s)"'<>\]]+/g

const ignored = new Set([
  '.git',
  'node_modules',
  'dist',
  '.vitepress'
])

async function walkMarkdown(dir, output = []) {
  const entries = await fs.readdir(dir, {
    withFileTypes: true
  })

  for (const entry of entries) {
    const full = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (!ignored.has(entry.name)) {
        await walkMarkdown(full, output)
      }
    } else if (entry.name.endsWith('.md')) {
      output.push(full)
    }
  }

  return output
}

function extensionFromType(contentType = '') {
  const type = contentType
    .split(';', 1)[0]
    .trim()
    .toLowerCase()

  return {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/avif': '.avif',
    'image/svg+xml': '.svg'
  }[type] || ''
}

function extensionFromUrl(url) {
  try {
    const ext = path
      .extname(new URL(url).pathname)
      .toLowerCase()

    if (
      [
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.webp',
        '.avif',
        '.svg'
      ].includes(ext)
    ) {
      return ext === '.jpeg' ? '.jpg' : ext
    }
  } catch {
    // Ignore.
  }

  return ''
}

function targetFor(url, extension) {
  const hash = crypto
    .createHash('sha256')
    .update(url)
    .digest('hex')
    .slice(0, 18)

  return path.join(
    outputDir,
    `legacy-${hash}${extension}`
  )
}

function markdownRelative(markdownFile, assetFile) {
  let value = path
    .relative(
      path.dirname(markdownFile),
      assetFile
    )
    .replaceAll('\\', '/')

  if (!value.startsWith('.')) {
    value = `./${value}`
  }

  return value
}

async function download(url) {
  const response = await fetch(url, {
    redirect: 'follow',
    headers: {
      'user-agent': 'Kgumo-Note-V2.5-migration/1.0'
    }
  })

  if (!response.ok) {
    throw new Error(
      `${response.status} ${response.statusText}`
    )
  }

  const type =
    response.headers.get('content-type') || ''

  const extension =
    extensionFromType(type) ||
    extensionFromUrl(url)

  if (!extension) {
    throw new Error(
      `unsupported content type: ${type || 'unknown'}`
    )
  }

  const target = targetFor(url, extension)

  try {
    await fs.access(target)
    return target
  } catch {
    // Download.
  }

  const buffer = Buffer.from(
    await response.arrayBuffer()
  )

  await fs.mkdir(
    path.dirname(target),
    { recursive: true }
  )

  await fs.writeFile(target, buffer)

  return target
}

async function pool(items, worker, concurrency = 6) {
  const results = new Map()
  let cursor = 0

  async function run() {
    while (cursor < items.length) {
      const item = items[cursor++]

      try {
        results.set(item, {
          ok: true,
          value: await worker(item)
        })
      } catch (error) {
        results.set(item, {
          ok: false,
          error:
            error instanceof Error
              ? error.message
              : String(error)
        })
      }
    }
  }

  await Promise.all(
    Array.from(
      {
        length: Math.min(
          concurrency,
          Math.max(items.length, 1)
        )
      },
      run
    )
  )

  return results
}

const markdownFiles = await walkMarkdown(docsDir)
const sources = new Map()
const urls = new Set()

for (const file of markdownFiles) {
  const source = await fs.readFile(file, 'utf8')
  sources.set(file, source)

  for (const match of source.matchAll(LEGACY_RE)) {
    urls.add(match[0])
  }
}

console.log(
  `Found ${urls.size} unique legacy Netlify image URLs.`
)

if (!urls.size) {
  process.exit(0)
}

if (!apply) {
  console.log('\nDry run only; nothing changed.')
  console.log(
    'Apply with:\n' +
    '  npm run migrate:legacy-images -- --apply\n'
  )
  process.exit(0)
}

await fs.mkdir(outputDir, { recursive: true })

const results = await pool([...urls], download, 6)

const migrated = new Map()
const failed = []

for (const [url, result] of results) {
  if (result.ok) {
    migrated.set(url, result.value)
  } else {
    failed.push({
      url,
      error: result.error
    })
  }
}

let changedFiles = 0
let replaced = 0

for (const [file, original] of sources) {
  let source = original

  for (const [url, asset] of migrated) {
    if (!source.includes(url)) continue

    const target = markdownRelative(file, asset)
    const count = source.split(url).length - 1

    source = source.replaceAll(url, target)
    replaced += count
  }

  if (source !== original) {
    await fs.writeFile(file, source, 'utf8')
    changedFiles += 1
  }
}

const manifest = {
  generatedAt: new Date().toISOString(),
  migrated: Object.fromEntries(
    [...migrated.entries()].map(
      ([url, file]) => [
        url,
        path
          .relative(root, file)
          .replaceAll('\\', '/')
      ]
    )
  ),
  failed
}

await fs.writeFile(
  manifestFile,
  JSON.stringify(manifest, null, 2) + '\n',
  'utf8'
)

console.log('\nMigration complete.')
console.log(`Images migrated: ${migrated.size}`)
console.log(`References replaced: ${replaced}`)
console.log(`Markdown files changed: ${changedFiles}`)
console.log(`Failed: ${failed.length}`)

if (failed.length) {
  console.log(
    '\nFailed URLs were kept unchanged:'
  )

  for (const item of failed) {
    console.log(`  ${item.url}`)
    console.log(`    ${item.error}`)
  }
}

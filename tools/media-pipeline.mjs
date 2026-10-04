import fs from 'node:fs/promises'
import fsSync from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { pathToFileURL } from 'node:url'
import MarkdownIt from 'markdown-it'
import sharp from 'sharp'
import { installObsidianCompat } from '../.vitepress/utils/obsidian-compat.mjs'

const root = process.cwd()
const docsDir = path.join(root, 'docs')
const configFile = path.join(root, 'media.config.mjs')

const configModule = await import(
  `${pathToFileURL(configFile).href}?t=${Date.now()}`
)
const config = configModule.default

const generatedDir =
  path.join(root, config.generatedPublicDir)

const manifestFile =
  path.join(root, config.manifestFile)

const mirroredDir =
  path.join(root, config.mirroredAssetDir)

const md = new MarkdownIt({
  html: true
})

installObsidianCompat(md, {
  docsDir
})

const IMAGE_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.avif',
  '.svg'
])

const RASTER_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.avif'
])

const ignoredDirectories = new Set([
  '.git',
  'node_modules',
  'dist',
  '_media',
  '.vitepress'
])

function slash(value) {
  return value.replaceAll('\\', '/')
}

function repoRelative(file) {
  return slash(path.relative(root, file))
}

function humanBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) {
    return `${(bytes / 1024).toFixed(0)} KB`
  }
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`
}

function hashBuffer(buffer) {
  return crypto
    .createHash('sha256')
    .update(buffer)
    .digest('hex')
}

function isExternal(src) {
  return /^(?:https?:)?\/\//i.test(src)
}

function cleanSrc(src) {
  let value = String(src || '')
    .split('#', 1)[0]
    .split('?', 1)[0]

  try {
    value = decodeURIComponent(value)
  } catch {
    // Keep the source unchanged if URI decoding fails.
  }

  return slash(value)
}

function resolveLocalSource(src, markdownFile) {
  if (!src || isExternal(src) || src.startsWith('data:')) {
    return null
  }

  const clean = cleanSrc(src)
  const publicDir = path.join(docsDir, 'public')

  // Repository's existing Obsidian convention:
  // ![](images/foo.png) -> docs/public/images/foo.png
  if (
    clean.startsWith('images/') ||
    clean.startsWith('./images/')
  ) {
    return path.join(
      publicDir,
      clean.replace(/^\.\//, '')
    )
  }

  if (
    clean.startsWith('public/') ||
    clean.startsWith('./public/')
  ) {
    return path.join(
      publicDir,
      clean
        .replace(/^\.\//, '')
        .replace(/^public\//, '')
    )
  }

  // VitePress public root.
  if (clean.startsWith('/')) {
    return path.join(
      publicDir,
      clean.replace(/^\/+/, '')
    )
  }

  // Standard relative Markdown attachment.
  return path.resolve(
    path.dirname(markdownFile),
    clean
  )
}

async function walk(dir, output = []) {
  let entries

  try {
    entries = await fs.readdir(dir, {
      withFileTypes: true
    })
  } catch {
    return output
  }

  for (const entry of entries) {
    const full = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (!ignoredDirectories.has(entry.name)) {
        await walk(full, output)
      }
      continue
    }

    output.push(full)
  }

  return output
}

function collectImageTokens(tokens, output = []) {
  for (const token of tokens) {
    if (token.type === 'image') {
      output.push(token)
    }

    if (token.children?.length) {
      collectImageTokens(token.children, output)
    }
  }

  return output
}

async function scanMarkdown() {
  const files = (await walk(docsDir))
    .filter((file) => file.endsWith('.md'))

  const localRefs = new Map()
  const externalRefs = new Map()
  const obsidian = {
    resolvedImages: [],
    unresolvedImages: [],
    resolvedLinks: [],
    unresolvedLinks: [],
    unsupportedEmbeds: []
  }

  for (const file of files) {
    const source = await fs.readFile(file, 'utf8')
    const env = {
      path: file
    }

    const tokens = md.parse(source, env)
    const images = collectImageTokens(tokens)

    for (const token of images) {
      const src = token.attrGet('src')
      if (!src) continue

      if (isExternal(src)) {
        let refs = externalRefs.get(src)
        if (!refs) {
          refs = []
          externalRefs.set(src, refs)
        }

        refs.push({
          markdownFile: file,
          line: token.map?.[0] ?? null
        })
        continue
      }

      const resolved =
        resolveLocalSource(
          src,
          file
        )

      if (!resolved) continue

      const key =
        path.resolve(resolved)

      let item =
        localRefs.get(key)

      if (!item) {
        item = {
          file: key,
          refs: []
        }

        localRefs.set(
          key,
          item
        )
      }

      item.refs.push({
        markdownFile: file,
        src,
        line: token.map?.[0] ?? null
      })
    }

    const compat =
      env.__obsidianCompat || {}

    for (
      const key
      of Object.keys(obsidian)
    ) {
      for (
        const value
        of compat[key] || []
      ) {
        obsidian[key].push({
          markdownFile: file,
          value
        })
      }
    }
  }

  return {
    markdownFiles: files,
    localRefs,
    externalRefs,
    obsidian
  }
}

async function fileMetadata(file) {
  try {
    const stat = await fs.stat(file)
    if (!stat.isFile()) return null

    const ext = path.extname(file).toLowerCase()

    if (!IMAGE_EXTENSIONS.has(ext)) {
      return null
    }

    if (ext === '.svg') {
      return {
        bytes: stat.size,
        ext,
        width: null,
        height: null,
        animated: false,
        pages: 1
      }
    }

    const image = sharp(file, {
      animated: true,
      failOn: 'none'
    })

    const metadata = await image.metadata()

    return {
      bytes: stat.size,
      ext,
      width: metadata.width ?? null,
      height: metadata.pageHeight ?? metadata.height ?? null,
      animated: (metadata.pages ?? 1) > 1,
      pages: metadata.pages ?? 1
    }
  } catch {
    return null
  }
}

function desiredWidths(width) {
  if (!width) return []

  const max =
    Math.min(width, config.maxWidth)

  const widths = config.responsiveWidths
    .filter((candidate) => candidate < max)

  widths.push(max)

  return [...new Set(widths)]
    .filter((candidate) => candidate > 0)
    .sort((a, b) => a - b)
}

async function writeWebp({
  sourceFile,
  sourceHash,
  width,
  animated
}) {
  const suffix =
    animated
      ? `anim-w${width}`
      : `w${width}`

  const filename =
    `${sourceHash.slice(0, 18)}-${suffix}.webp`

  const target =
    path.join(generatedDir, filename)

  try {
    const stat = await fs.stat(target)

    return {
      file: target,
      bytes: stat.size
    }
  } catch {
    // Generate below.
  }

  await fs.mkdir(generatedDir, {
    recursive: true
  })

  let pipeline = sharp(sourceFile, {
    animated,
    failOn: 'none'
  })

  pipeline = pipeline.rotate()

  if (width) {
    pipeline = pipeline.resize({
      width,
      withoutEnlargement: true,
      fit: 'inside'
    })
  }

  pipeline = pipeline.webp({
    quality:
      animated
        ? config.animatedWebpQuality
        : config.webpQuality,
    effort: 4,
    smartSubsample: true
  })

  await pipeline.toFile(target)

  const stat = await fs.stat(target)

  return {
    file: target,
    bytes: stat.size
  }
}

async function optimizeReferencedImage(file) {
  const metadata = await fileMetadata(file)

  if (!metadata) {
    return {
      status: 'missing-or-unsupported',
      source: repoRelative(file)
    }
  }

  const entry = {
    source: repoRelative(file),
    sourceBytes: metadata.bytes,
    width: metadata.width,
    height: metadata.height,
    animated: metadata.animated,
    variants: []
  }

  if (
    metadata.ext === '.svg' ||
    !RASTER_EXTENSIONS.has(metadata.ext)
  ) {
    return {
      status: 'metadata-only',
      entry
    }
  }

  const shouldOptimize =
    metadata.bytes >= config.minBytes ||
    (
      config.alwaysOptimizeGif &&
      metadata.ext === '.gif'
    )

  if (!shouldOptimize || !metadata.width) {
    return {
      status: 'metadata-only',
      entry
    }
  }

  const buffer = await fs.readFile(file)
  const sourceHash = hashBuffer(buffer)

  const widths =
    metadata.animated
      ? [
          Math.min(
            metadata.width,
            config.animatedMaxWidth
          )
        ]
      : desiredWidths(metadata.width)

  for (const width of widths) {
    const output = await writeWebp({
      sourceFile: file,
      sourceHash,
      width,
      animated: metadata.animated
    })

    // Every responsive derivative should be smaller than the original.
    // Otherwise using the original is more sensible.
    if (
      output.bytes <
      metadata.bytes * config.maxOutputRatio
    ) {
      entry.variants.push({
        width,
        bytes: output.bytes,
        url:
          '/_media/' +
          path.basename(output.file)
      })
    }
  }

  return {
    status:
      entry.variants.length
        ? 'optimized'
        : 'metadata-only',
    entry
  }
}

async function prepare() {
  const scan = await scanMarkdown()

  await fs.mkdir(
    path.dirname(manifestFile),
    { recursive: true }
  )

  await fs.mkdir(
    generatedDir,
    { recursive: true }
  )

  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    entries: {}
  }

  const usedGeneratedFiles = new Set()
  let optimizedSources = 0
  let sourceBytes = 0
  let derivativeBytes = 0
  let missing = 0

  const items = [...scan.localRefs.values()]
  let cursor = 0
  const concurrency = Math.min(
    4,
    Math.max(1, items.length)
  )

  async function worker() {
    while (cursor < items.length) {
      const item = items[cursor++]
      const result =
        await optimizeReferencedImage(item.file)

      if (!result.entry) {
        missing += 1
        continue
      }

      manifest.entries[result.entry.source] =
        result.entry

      sourceBytes += result.entry.sourceBytes

      if (result.entry.variants.length) {
        optimizedSources += 1

        for (
          const variant
          of result.entry.variants
        ) {
          derivativeBytes += variant.bytes
          usedGeneratedFiles.add(
            path.basename(variant.url)
          )
        }
      }
    }
  }

  await Promise.all(
    Array.from(
      { length: concurrency },
      worker
    )
  )

  // Remove stale generated derivatives left by old/changed source images.
  for (
    const entry
    of await fs.readdir(generatedDir, {
      withFileTypes: true
    })
  ) {
    if (
      entry.isFile() &&
      entry.name.endsWith('.webp') &&
      !usedGeneratedFiles.has(entry.name)
    ) {
      await fs.unlink(
        path.join(generatedDir, entry.name)
      )
    }
  }

  await fs.writeFile(
    manifestFile,
    JSON.stringify(manifest, null, 2) + '\n',
    'utf8'
  )

  console.log('\n=== Media prepare ===\n')
  console.log(
    `Markdown files: ${scan.markdownFiles.length}`
  )
  console.log(
    `Referenced local images: ${scan.localRefs.size}`
  )
  console.log(
    `Images with web derivatives: ${optimizedSources}`
  )
  console.log(
    `Missing/unsupported local images: ${missing}`
  )
  console.log(
    `Referenced source bytes: ${humanBytes(sourceBytes)}`
  )
  console.log(
    `Generated derivative bytes: ${humanBytes(derivativeBytes)}`
  )
  console.log(
    `Manifest: ${repoRelative(manifestFile)}`
  )
  console.log('')
}

async function audit() {
  const scan = await scanMarkdown()

  const local = []
  const missingItems = []

  for (
    const item
    of scan.localRefs.values()
  ) {
    const metadata =
      await fileMetadata(item.file)

    if (!metadata) {
      missingItems.push(item)
      continue
    }

    local.push({
      file: item.file,
      bytes: metadata.bytes,
      width: metadata.width,
      height: metadata.height,
      animated: metadata.animated,
      refs: item.refs.length
    })
  }

  local.sort(
    (a, b) => b.bytes - a.bytes
  )

  const domains = new Map()

  for (
    const [url, refs]
    of scan.externalRefs
  ) {
    try {
      const host =
        new URL(url).hostname

      domains.set(
        host,
        (domains.get(host) || 0) +
        refs.length
      )
    } catch {
      // Ignore invalid URL.
    }
  }

  const over1Mb =
    local.filter(
      (item) =>
        item.bytes >= 1024 ** 2
    ).length

  const over500Kb =
    local.filter(
      (item) =>
        item.bytes >= 500 * 1024
    ).length

  console.log(
    '\n=== Media audit ===\n'
  )

  console.log(
    `Markdown files: ${scan.markdownFiles.length}`
  )
  console.log(
    `Referenced local images: ${scan.localRefs.size}`
  )
  console.log(
    `External image URLs: ${scan.externalRefs.size}`
  )
  console.log(
    `Missing/unsupported local images: ${missingItems.length}`
  )

  console.log('\nObsidian compatibility:')
  console.log(
    `  wiki images resolved: ${scan.obsidian.resolvedImages.length}`
  )
  console.log(
    `  wiki images unresolved: ${scan.obsidian.unresolvedImages.length}`
  )
  console.log(
    `  wiki links resolved: ${scan.obsidian.resolvedLinks.length}`
  )
  console.log(
    `  wiki links unresolved: ${scan.obsidian.unresolvedLinks.length}`
  )
  console.log(
    `  unsupported non-image embeds: ${scan.obsidian.unsupportedEmbeds.length}`
  )

  console.log(
    '\nExternal image domains:'
  )

  if (!domains.size) {
    console.log('  none')
  } else {
    for (
      const [host, count]
      of [...domains.entries()]
        .sort(
          (a, b) => b[1] - a[1]
        )
    ) {
      const mirrored =
        config.mirrorDomains.includes(host)
          ? ' [mirror allowlisted]'
          : ''

      console.log(
        `  ${host}: ${count}${mirrored}`
      )
    }
  }

  if (missingItems.length) {
    console.log(
      '\nMissing/unsupported local references:'
    )

    for (
      const item
      of missingItems.slice(0, 20)
    ) {
      const firstRef =
        item.refs?.[0]

      console.log(
        `  ${repoRelative(item.file)}`
      )

      if (firstRef) {
        console.log(
          `    from ${repoRelative(firstRef.markdownFile)} -> ${firstRef.src}`
        )
      }
    }
  }

  if (
    scan.obsidian.unresolvedImages.length
  ) {
    console.log(
      '\nUnresolved Obsidian image embeds:'
    )

    for (
      const item
      of scan.obsidian.unresolvedImages.slice(
        0,
        20
      )
    ) {
      console.log(
        `  ${repoRelative(item.markdownFile)} -> ![[${item.value}]]`
      )
    }
  }

  if (
    scan.obsidian.unresolvedLinks.length
  ) {
    console.log(
      '\nUnresolved Obsidian wiki links:'
    )

    for (
      const item
      of scan.obsidian.unresolvedLinks.slice(
        0,
        20
      )
    ) {
      console.log(
        `  ${repoRelative(item.markdownFile)} -> [[${item.value}]]`
      )
    }
  }

  console.log(
    '\nLargest REFERENCED local images:'
  )

  for (
    const item
    of local.slice(0, 30)
  ) {
    const dimensions =
      item.width && item.height
        ? `${item.width}x${item.height}`
        : 'unknown'

    console.log(
      `  ${humanBytes(item.bytes).padStart(9)}  ` +
      `${dimensions.padStart(12)}  ` +
      `${repoRelative(item.file)}`
    )
  }

  console.log(
    '\nReferenced image thresholds:'
  )
  console.log(
    `  >= 1 MB:   ${over1Mb}`
  )
  console.log(
    `  >= 500 KB: ${over500Kb}`
  )

  console.log(
    '\n=== end media audit ===\n'
  )
}

function extensionFromContentType(contentType = '') {
  const type =
    contentType.split(';', 1)[0]
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
    const ext =
      path.extname(
        new URL(url).pathname
      ).toLowerCase()

    if (IMAGE_EXTENSIONS.has(ext)) {
      return ext === '.jpeg' ? '.jpg' : ext
    }
  } catch {
    // Ignore.
  }

  return ''
}

async function downloadExternal(url) {
  const response = await fetch(url, {
    redirect: 'follow',
    headers: {
      'user-agent':
        'Kgumo-Note-Media-Pipeline/2.6'
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
    extensionFromContentType(type) ||
    extensionFromUrl(url)

  if (!extension) {
    throw new Error(
      `unsupported content type: ${type || 'unknown'}`
    )
  }

  const hash =
    crypto
      .createHash('sha256')
      .update(url)
      .digest('hex')
      .slice(0, 18)

  let host = 'external'

  try {
    host = new URL(url)
      .hostname
      .replace(/[^a-z0-9.-]+/gi, '_')
  } catch {
    // Keep "external".
  }

  const target =
    path.join(
      mirroredDir,
      host,
      `${hash}${extension}`
    )

  try {
    await fs.access(target)
    return target
  } catch {
    // Download below.
  }

  await fs.mkdir(
    path.dirname(target),
    { recursive: true }
  )

  const bytes =
    Buffer.from(
      await response.arrayBuffer()
    )

  await fs.writeFile(target, bytes)

  return target
}

function relativeMarkdownPath(
  markdownFile,
  target
) {
  let rel =
    slash(
      path.relative(
        path.dirname(markdownFile),
        target
      )
    )

  if (!rel.startsWith('.')) {
    rel = `./${rel}`
  }

  return rel
}

async function mirror({ apply }) {
  const scan = await scanMarkdown()

  const eligible = []

  for (const [url, refs] of scan.externalRefs) {
    try {
      const host = new URL(url).hostname

      if (
        config.mirrorDomains.includes(host)
      ) {
        eligible.push({
          url,
          host,
          refs
        })
      }
    } catch {
      // Ignore.
    }
  }

  console.log(
    `Found ${eligible.length} unique external image URLs ` +
    `on mirror-allowlisted domains.`
  )

  if (!eligible.length) return

  console.log(
    'Allowlisted domains: ' +
    config.mirrorDomains.join(', ')
  )

  if (!apply) {
    console.log(
      '\nDry run only. Nothing changed.'
    )
    console.log(
      'Apply with:\n' +
      '  npm run media:mirror:apply\n'
    )
    return
  }

  const replacementsByFile = new Map()
  let migrated = 0
  const failures = []

  let cursor = 0

  async function worker() {
    while (cursor < eligible.length) {
      const item = eligible[cursor++]

      try {
        const target =
          await downloadExternal(item.url)

        for (const ref of item.refs) {
          let replacements =
            replacementsByFile.get(
              ref.markdownFile
            )

          if (!replacements) {
            replacements = []
            replacementsByFile.set(
              ref.markdownFile,
              replacements
            )
          }

          replacements.push({
            from: item.url,
            to: relativeMarkdownPath(
              ref.markdownFile,
              target
            )
          })
        }

        migrated += 1
      } catch (error) {
        failures.push({
          url: item.url,
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
          6,
          Math.max(1, eligible.length)
        )
      },
      worker
    )
  )

  let changedFiles = 0
  let referencesChanged = 0

  for (
    const [file, replacements]
    of replacementsByFile
  ) {
    let source =
      await fs.readFile(file, 'utf8')

    const original = source

    for (
      const replacement
      of replacements
    ) {
      const count =
        source.split(replacement.from).length - 1

      source =
        source.replaceAll(
          replacement.from,
          replacement.to
        )

      referencesChanged += count
    }

    if (source !== original) {
      await fs.writeFile(file, source, 'utf8')
      changedFiles += 1
    }
  }

  console.log('\nMirror complete.')
  console.log(
    `Downloaded/reused images: ${migrated}`
  )
  console.log(
    `Markdown files changed: ${changedFiles}`
  )
  console.log(
    `References changed: ${referencesChanged}`
  )
  console.log(
    `Failures: ${failures.length}`
  )

  if (failures.length) {
    console.log('\nFailed URLs remain external:')

    for (const item of failures) {
      console.log(`  ${item.url}`)
      console.log(`    ${item.error}`)
    }
  }
}

async function clean() {
  await fs.rm(
    generatedDir,
    {
      recursive: true,
      force: true
    }
  )

  await fs.rm(
    manifestFile,
    {
      force: true
    }
  )

  console.log(
    'Generated media cache removed.'
  )
}

const command =
  process.argv[2] || 'audit'

if (command === 'prepare') {
  await prepare()
} else if (command === 'audit') {
  await audit()
} else if (command === 'mirror') {
  await mirror({
    apply: process.argv.includes('--apply')
  })
} else if (command === 'clean') {
  await clean()
} else {
  console.error(
    `Unknown media command: ${command}`
  )
  process.exit(1)
}

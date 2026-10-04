import fs from 'node:fs/promises'
import fsSync from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'

const root = process.cwd()
const buildDir = path.join(root, 'docs', 'build')
const outputFile =
  path.join(root, '.vitepress', 'cache', 'build-graph.generated.mjs')

const ignoredDirectories = new Set([
  '.git',
  '.obsidian',
  'node_modules',
  'dist',
  'assets',
  'public'
])

function slash(value) {
  return value.replaceAll('\\', '/')
}

function cleanName(value) {
  return value
    .replace(/\.md$/i, '')
    .replace(/^\d+[._、\-\s]*/, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function makeId(relativePath) {
  const hash = crypto
    .createHash('sha1')
    .update(relativePath)
    .digest('hex')
    .slice(0, 10)

  return `build-doc-${hash}`
}

async function markdownTitle(file, fallback) {
  try {
    const source = await fs.readFile(file, 'utf8')

    const frontmatter = source.match(/^---\s*\n([\s\S]*?)\n---/)
    if (frontmatter) {
      const titleMatch =
        frontmatter[1].match(/^title\s*:\s*["']?(.+?)["']?\s*$/m)

      if (titleMatch?.[1]?.trim()) {
        return titleMatch[1].trim()
      }
    }

    const h1 = source.match(/^#\s+(.+?)\s*$/m)

    if (h1?.[1]?.trim()) {
      return h1[1].trim()
    }
  } catch {
    // fall back to filename
  }

  return fallback
}

function routeFor(relativePath, isDirectory = false) {
  const normalized = slash(relativePath)
    .replace(/\.md$/i, '')
    .replace(/\/index$/i, '')

  if (!normalized || normalized === 'index') {
    return '/build/'
  }

  return `/build/${normalized}${isDirectory ? '/' : ''}`
}

function semanticTargets(text) {
  const value = text.toLowerCase()
  const targets = new Set()

  if (value.includes('onnx runtime')) targets.add('onnx-runtime')
  if (value.includes('onnx')) targets.add('onnx')
  if (value.includes('c++')) targets.add('c++')
  if (value.includes('pytorch')) targets.add('pytorch')
  if (value.includes('cmake')) targets.add('cmake')
  if (value.includes('cuda')) targets.add('cuda')
  if (value.includes('tensorrt')) targets.add('tensorrt')
  if (value.includes('qt')) targets.add('qt')
  if (value.includes('opencv')) targets.add('cv')
  if (value.includes('cnn') || value.includes('cifar')) targets.add('cnn')

  if (
    value.includes('推理') ||
    value.includes('inference')
  ) {
    targets.add('inference')
  }

  if (
    value.includes('部署') ||
    value.includes('deploy')
  ) {
    targets.add('deployment')
  }

  if (
    value.includes('转换') ||
    value.includes('export')
  ) {
    targets.add('model-export')
  }

  return [...targets]
}

async function walkDirectory(
  absoluteDir,
  relativeDir,
  parentId,
  depth,
  nodes,
  links
) {
  const entries =
    await fs.readdir(absoluteDir, {
      withFileTypes: true
    })

  const sorted = entries
    .filter((entry) =>
      !entry.name.startsWith('.') &&
      !entry.name.startsWith('_') &&
      !ignoredDirectories.has(entry.name)
    )
    .sort((a, b) =>
      a.name.localeCompare(
        b.name,
        'zh-CN',
        { numeric: true }
      )
    )

  for (const entry of sorted) {
    const absolute =
      path.join(absoluteDir, entry.name)

    const relative =
      relativeDir
        ? path.join(relativeDir, entry.name)
        : entry.name

    if (entry.isDirectory()) {
      const id = makeId(`dir:${slash(relative)}`)
      const label = cleanName(entry.name)

      const hasIndex =
        fsSync.existsSync(
          path.join(absolute, 'index.md')
        ) ||
        fsSync.existsSync(
          path.join(absolute, 'README.md')
        )

      nodes.push({
        id,
        name: label || entry.name,
        group: depth <= 1 ? 'stage' : 'folder',
        level: Math.min(depth + 2, 5),
        link: hasIndex
          ? routeFor(relative, true)
          : null,
        source: slash(relative)
      })

      links.push({
        source: parentId,
        target: id,
        value: depth <= 1 ? 9 : 7,
        kind: 'hierarchy'
      })

      for (const target of semanticTargets(label)) {
        links.push({
          source: id,
          target,
          value: 6,
          kind: 'semantic'
        })
      }

      await walkDirectory(
        absolute,
        relative,
        id,
        depth + 1,
        nodes,
        links
      )

      continue
    }

    if (!entry.isFile() || !entry.name.endsWith('.md')) {
      continue
    }

    if (
      entry.name.toLowerCase() === 'index.md' ||
      entry.name.toLowerCase() === 'readme.md'
    ) {
      continue
    }

    const id =
      makeId(`file:${slash(relative)}`)

    const fallback =
      cleanName(entry.name)

    const title =
      await markdownTitle(
        absolute,
        fallback || entry.name
      )

    nodes.push({
      id,
      name: title,
      group: 'note',
      level: Math.min(depth + 2, 5),
      link: routeFor(relative),
      source: slash(relative)
    })

    links.push({
      source: parentId,
      target: id,
      value: 7,
      kind: 'hierarchy'
    })

    for (
      const target
      of semanticTargets(
        `${title} ${slash(relative)}`
      )
    ) {
      links.push({
        source: id,
        target,
        value: 7,
        kind: 'semantic'
      })
    }
  }
}

async function main() {
  const nodes = [
    {
      id: 'build',
      name: '综合技术',
      group: 'integration',
      level: 1,
      link: '/build/',
      source: 'build'
    }
  ]

  const links = [
    { source: 'build', target: 'c++', value: 10, kind: 'bridge' },
    { source: 'build', target: 'ai', value: 10, kind: 'bridge' },
    { source: 'build', target: 'onnx', value: 10, kind: 'bridge' },
    { source: 'build', target: 'projects', value: 9, kind: 'bridge' }
  ]

  if (fsSync.existsSync(buildDir)) {
    await walkDirectory(
      buildDir,
      '',
      'build',
      1,
      nodes,
      links
    )
  }

  await fs.mkdir(
    path.dirname(outputFile),
    { recursive: true }
  )

  const source =
`// AUTO-GENERATED by tools/generate-build-graph.mjs
// Do not edit manually. It is rebuilt from docs/build.

export const buildNodes = ${JSON.stringify(nodes, null, 2)}

export const buildLinks = ${JSON.stringify(links, null, 2)}
`

  await fs.writeFile(
    outputFile,
    source,
    'utf8'
  )

  console.log('\n=== Knowledge graph prepare ===\n')
  console.log(`build nodes: ${nodes.length}`)
  console.log(`build links: ${links.length}`)
  console.log(
    `generated: ${slash(path.relative(root, outputFile))}`
  )
  console.log('')
}

await main()

import fs from 'node:fs/promises'
import path from 'node:path'

import {
  resolveObsidianImage,
  resolveObsidianNote
} from '../.vitepress/utils/obsidian-compat.mjs'

const root = process.cwd()
const docsDir =
  path.join(root, 'docs')

const ignored = new Set([
  '.git',
  'node_modules',
  'dist',
  '_media'
])

async function walk(dir, output = []) {
  let entries

  try {
    entries =
      await fs.readdir(
        dir,
        {
          withFileTypes: true
        }
      )
  } catch {
    return output
  }

  for (const entry of entries) {
    const full =
      path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (!ignored.has(entry.name)) {
        await walk(full, output)
      }
    } else if (
      entry.name.endsWith('.md')
    ) {
      output.push(full)
    }
  }

  return output
}

function lineOf(source, offset) {
  return (
    source
      .slice(0, offset)
      .split('\n')
      .length
  )
}

function imageExtension(target) {
  return path
    .extname(
      target
        .split('|', 1)[0]
        .split('#', 1)[0]
    )
    .toLowerCase()
}

const imageExtensions =
  new Set([
    '.png',
    '.jpg',
    '.jpeg',
    '.gif',
    '.webp',
    '.avif',
    '.svg'
  ])

const files =
  await walk(docsDir)

const results = {
  resolvedImages: [],
  unresolvedImages: [],
  resolvedLinks: [],
  unresolvedLinks: [],
  unsupportedEmbeds: []
}

for (const file of files) {
  const source =
    await fs.readFile(
      file,
      'utf8'
    )

  for (
    const match
    of source.matchAll(
      /!\[\[([^\]]+)\]\]/g
    )
  ) {
    const inner =
      match[1]

    const target =
      inner.split('|', 1)[0].trim()

    if (
      !imageExtensions.has(
        imageExtension(target)
      )
    ) {
      results.unsupportedEmbeds.push({
        file,
        line: lineOf(
          source,
          match.index
        ),
        target
      })
      continue
    }

    const resolved =
      resolveObsidianImage(
        target,
        file,
        docsDir
      )

    const entry = {
      file,
      line: lineOf(
        source,
        match.index
      ),
      target,
      resolved
    }

    if (resolved) {
      results.resolvedImages.push(entry)
    } else {
      results.unresolvedImages.push(entry)
    }
  }

  for (
    const match
    of source.matchAll(
      /(?<!!)\[\[([^\]]+)\]\]/g
    )
  ) {
    const inner =
      match[1]

    const target =
      inner.split('|', 1)[0].trim()

    const resolved =
      resolveObsidianNote(
        target,
        file,
        docsDir
      )

    const entry = {
      file,
      line: lineOf(
        source,
        match.index
      ),
      target,
      resolved
    }

    if (resolved) {
      results.resolvedLinks.push(entry)
    } else {
      results.unresolvedLinks.push(entry)
    }
  }
}

function rel(file) {
  return path
    .relative(root, file)
    .replaceAll('\\', '/')
}

console.log(
  '\n=== Obsidian compatibility audit ===\n'
)

console.log(
  `Markdown files: ${files.length}`
)
console.log(
  `Wiki images resolved: ${results.resolvedImages.length}`
)
console.log(
  `Wiki images unresolved: ${results.unresolvedImages.length}`
)
console.log(
  `Wiki links resolved: ${results.resolvedLinks.length}`
)
console.log(
  `Wiki links unresolved: ${results.unresolvedLinks.length}`
)
console.log(
  `Unsupported non-image embeds: ${results.unsupportedEmbeds.length}`
)

if (
  results.unresolvedImages.length
) {
  console.log(
    '\nUnresolved images:'
  )

  for (
    const item
    of results.unresolvedImages.slice(
      0,
      30
    )
  ) {
    console.log(
      `  ${rel(item.file)}:${item.line}  ![[${item.target}]]`
    )
  }
}

if (
  results.unresolvedLinks.length
) {
  console.log(
    '\nUnresolved wiki links:'
  )

  for (
    const item
    of results.unresolvedLinks.slice(
      0,
      30
    )
  ) {
    console.log(
      `  ${rel(item.file)}:${item.line}  [[${item.target}]]`
    )
  }
}

if (
  results.unsupportedEmbeds.length
) {
  console.log(
    '\nNon-image embeds left unchanged:'
  )

  for (
    const item
    of results.unsupportedEmbeds.slice(
      0,
      20
    )
  ) {
    console.log(
      `  ${rel(item.file)}:${item.line}  ![[${item.target}]]`
    )
  }
}

console.log(
  '\n=== end Obsidian audit ===\n'
)

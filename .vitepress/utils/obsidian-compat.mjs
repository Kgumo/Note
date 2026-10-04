import fs from 'node:fs'
import path from 'node:path'

const IMAGE_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.avif',
  '.svg'
])

const INDEX_CACHE = new Map()

function slash(value) {
  return value.replaceAll('\\', '/')
}

function normalizeLookup(value) {
  return slash(
    String(value || '')
      .trim()
      .replace(/^\.?\//, '')
  ).toLocaleLowerCase()
}

function safeDecode(value) {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

function walkSync(dir, output = []) {
  if (!fs.existsSync(dir)) return output

  for (
    const entry
    of fs.readdirSync(dir, {
      withFileTypes: true
    })
  ) {
    if (
      entry.name === '.git' ||
      entry.name === 'node_modules' ||
      entry.name === 'dist' ||
      entry.name === '_media'
    ) {
      continue
    }

    const full =
      path.join(dir, entry.name)

    if (entry.isDirectory()) {
      walkSync(full, output)
    } else {
      output.push(full)
    }
  }

  return output
}

function commonDistance(fromDir, targetFile) {
  const relative =
    path.relative(
      fromDir,
      path.dirname(targetFile)
    )

  if (!relative || relative === '.') {
    return 0
  }

  return relative
    .split(path.sep)
    .filter(Boolean)
    .length
}

function chooseNearest(files, markdownFile) {
  if (!files?.length) return null
  if (files.length === 1) return files[0]

  const currentDir =
    markdownFile
      ? path.dirname(markdownFile)
      : ''

  return [...files].sort(
    (a, b) =>
      commonDistance(currentDir, a) -
      commonDistance(currentDir, b)
  )[0]
}

function buildIndex(docsDir) {
  const absoluteDocs =
    path.resolve(docsDir)

  const cached =
    INDEX_CACHE.get(absoluteDocs)

  if (cached) return cached

  const publicDir =
    path.join(absoluteDocs, 'public')

  const imageByBase = new Map()
  const imageByRelative = new Map()
  const noteByBase = new Map()
  const noteByRelative = new Map()

  for (
    const file
    of walkSync(absoluteDocs)
  ) {
    const ext =
      path.extname(file).toLowerCase()

    const relative =
      slash(
        path.relative(
          absoluteDocs,
          file
        )
      )

    const normalizedRelative =
      normalizeLookup(relative)

    if (IMAGE_EXTENSIONS.has(ext)) {
      imageByRelative.set(
        normalizedRelative,
        file
      )

      const base =
        path.basename(file)
          .toLocaleLowerCase()

      const list =
        imageByBase.get(base) || []

      list.push(file)
      imageByBase.set(base, list)
      continue
    }

    if (ext === '.md') {
      noteByRelative.set(
        normalizedRelative
          .replace(/\.md$/i, ''),
        file
      )

      const stem =
        path.basename(
          file,
          '.md'
        ).toLocaleLowerCase()

      const list =
        noteByBase.get(stem) || []

      list.push(file)
      noteByBase.set(stem, list)
    }
  }

  const index = {
    docsDir: absoluteDocs,
    publicDir,
    imageByBase,
    imageByRelative,
    noteByBase,
    noteByRelative
  }

  INDEX_CACHE.set(
    absoluteDocs,
    index
  )

  return index
}

function pathExists(file) {
  try {
    return fs.statSync(file).isFile()
  } catch {
    return false
  }
}

function imageUrlFor(file, markdownFile, index) {
  const relativePublic =
    slash(
      path.relative(
        index.publicDir,
        file
      )
    )

  if (
    relativePublic &&
    !relativePublic.startsWith('../') &&
    relativePublic !== '..'
  ) {
    return `/${relativePublic}`
  }

  let relative =
    slash(
      path.relative(
        path.dirname(markdownFile),
        file
      )
    )

  if (!relative.startsWith('.')) {
    relative = `./${relative}`
  }

  return relative
}

export function resolveObsidianImage(
  target,
  markdownFile,
  docsDir
) {
  const index =
    buildIndex(docsDir)

  const decoded =
    safeDecode(target)
      .trim()
      .replace(/^\/+/, '')

  if (!decoded) return null

  const currentDir =
    path.dirname(markdownFile)

  const directCandidates = [
    path.resolve(currentDir, decoded),
    path.resolve(index.docsDir, decoded),
    path.resolve(index.publicDir, decoded),
    path.resolve(
      index.publicDir,
      'images',
      decoded
    )
  ]

  for (const candidate of directCandidates) {
    if (pathExists(candidate)) {
      return {
        file: candidate,
        src: imageUrlFor(
          candidate,
          markdownFile,
          index
        )
      }
    }
  }

  const normalized =
    normalizeLookup(decoded)

  const relativeMatches = [
    normalized,
    normalizeLookup(
      `public/${decoded}`
    ),
    normalizeLookup(
      `public/images/${decoded}`
    )
  ]

  for (
    const key
    of relativeMatches
  ) {
    const candidate =
      index.imageByRelative.get(key)

    if (candidate) {
      return {
        file: candidate,
        src: imageUrlFor(
          candidate,
          markdownFile,
          index
        )
      }
    }
  }

  const byBase =
    index.imageByBase.get(
      path.basename(decoded)
        .toLocaleLowerCase()
    )

  const nearest =
    chooseNearest(
      byBase,
      markdownFile
    )

  if (!nearest) return null

  return {
    file: nearest,
    src: imageUrlFor(
      nearest,
      markdownFile,
      index
    )
  }
}

function noteRouteFor(file, docsDir, heading = '') {
  let relative =
    slash(
      path.relative(
        docsDir,
        file
      )
    )
      .replace(/\.md$/i, '')
      .replace(/\/index$/i, '')

  if (relative === 'index') {
    relative = ''
  }

  let href =
    relative
      ? `/${relative}`
      : '/'

  if (heading) {
    const slug =
      heading
        .trim()
        .toLocaleLowerCase()
        .replace(/[“”"'`~!@#$%^&*()+=[\]{}|\\:;,.<>/?，。！？、；：]/g, '')
        .replace(/\s+/g, '-')

    if (slug) {
      href += `#${slug}`
    }
  }

  return href
}

export function resolveObsidianNote(
  target,
  markdownFile,
  docsDir
) {
  const index =
    buildIndex(docsDir)

  const decoded =
    safeDecode(target).trim()

  if (!decoded) return null

  const hashIndex =
    decoded.indexOf('#')

  const rawPath =
    (
      hashIndex >= 0
        ? decoded.slice(0, hashIndex)
        : decoded
    ).trim()

  const heading =
    hashIndex >= 0
      ? decoded.slice(hashIndex + 1).trim()
      : ''

  // [[#Heading]] means current note.
  if (!rawPath && heading) {
    return {
      file: markdownFile,
      href: noteRouteFor(
        markdownFile,
        index.docsDir,
        heading
      )
    }
  }

  const currentDir =
    path.dirname(markdownFile)

  const withMd =
    rawPath.toLowerCase().endsWith('.md')
      ? rawPath
      : `${rawPath}.md`

  const directCandidates = [
    path.resolve(currentDir, withMd),
    path.resolve(index.docsDir, withMd)
  ]

  for (const candidate of directCandidates) {
    if (pathExists(candidate)) {
      return {
        file: candidate,
        href: noteRouteFor(
          candidate,
          index.docsDir,
          heading
        )
      }
    }
  }

  const normalized =
    normalizeLookup(rawPath)
      .replace(/\.md$/i, '')

  const relative =
    index.noteByRelative.get(
      normalized
    )

  if (relative) {
    return {
      file: relative,
      href: noteRouteFor(
        relative,
        index.docsDir,
        heading
      )
    }
  }

  const stem =
    path.basename(
      rawPath,
      path.extname(rawPath)
    ).toLocaleLowerCase()

  const nearest =
    chooseNearest(
      index.noteByBase.get(stem),
      markdownFile
    )

  if (!nearest) return null

  return {
    file: nearest,
    href: noteRouteFor(
      nearest,
      index.docsDir,
      heading
    )
  }
}

function parseEmbedInner(raw) {
  const pieces =
    raw.split('|')

  const target =
    (pieces.shift() || '').trim()

  const modifier =
    pieces.join('|').trim()

  const result = {
    target,
    alt: path.basename(target),
    width: null,
    height: null
  }

  if (!modifier) return result

  const dimensions =
    modifier.match(
      /^(\d{1,5})(?:x(\d{1,5}))?$/
    )

  if (dimensions) {
    result.width =
      Number(dimensions[1])

    result.height =
      dimensions[2]
        ? Number(dimensions[2])
        : null

    return result
  }

  result.alt = modifier
  return result
}

function statsFor(env) {
  env.__obsidianCompat ||= {
    resolvedImages: [],
    unresolvedImages: [],
    resolvedLinks: [],
    unresolvedLinks: [],
    unsupportedEmbeds: []
  }

  return env.__obsidianCompat
}

export function installObsidianCompat(
  md,
  {
    docsDir
  }
) {
  const absoluteDocs =
    path.resolve(docsDir)

  // Build once so resolution during Markdown parsing is O(1)/small-list.
  buildIndex(absoluteDocs)

  md.inline.ruler.before(
    'image',
    'obsidian-wiki-image',
    (state, silent) => {
      const start =
        state.pos

      if (
        state.src.charCodeAt(start) !== 0x21 ||
        state.src.slice(
          start,
          start + 3
        ) !== '![['
      ) {
        return false
      }

      const close =
        state.src.indexOf(
          ']]',
          start + 3
        )

      if (close < 0) {
        return false
      }

      const raw =
        state.src.slice(
          start + 3,
          close
        )

      const parsed =
        parseEmbedInner(raw)

      const extension =
        path.extname(
          parsed.target
            .split('#', 1)[0]
        ).toLowerCase()

      // Leave note/PDF/audio embeds untouched for now.
      if (
        !IMAGE_EXTENSIONS.has(extension)
      ) {
        if (!silent) {
          statsFor(state.env)
            .unsupportedEmbeds
            .push(parsed.target)

          const token =
            state.push(
              'text',
              '',
              0
            )

          token.content =
            state.src.slice(
              start,
              close + 2
            )
        }

        state.pos =
          close + 2

        return true
      }

      const markdownFile =
        state.env?.path
          ? path.resolve(
              state.env.path
            )
          : null

      if (!markdownFile) {
        return false
      }

      const resolved =
        resolveObsidianImage(
          parsed.target,
          markdownFile,
          absoluteDocs
        )

      if (!silent) {
        const stats =
          statsFor(state.env)

        if (!resolved) {
          stats.unresolvedImages.push(
            parsed.target
          )

          const token =
            state.push(
              'text',
              '',
              0
            )

          token.content =
            state.src.slice(
              start,
              close + 2
            )
        } else {
          stats.resolvedImages.push({
            target: parsed.target,
            file: resolved.file
          })

          const token =
            state.push(
              'image',
              'img',
              0
            )

          token.attrs = [
            ['src', resolved.src],
            ['alt', parsed.alt]
          ]

          if (parsed.width) {
            token.attrSet(
              'width',
              String(parsed.width)
            )
          }

          if (parsed.height) {
            token.attrSet(
              'height',
              String(parsed.height)
            )
          }

          const child =
            new state.Token(
              'text',
              '',
              0
            )

          child.content =
            parsed.alt

          token.children = [child]
          token.content =
            parsed.alt
        }
      }

      state.pos =
        close + 2

      return true
    }
  )

  md.inline.ruler.before(
    'link',
    'obsidian-wikilink',
    (state, silent) => {
      const start =
        state.pos

      if (
        state.src.slice(
          start,
          start + 2
        ) !== '[['
      ) {
        return false
      }

      const close =
        state.src.indexOf(
          ']]',
          start + 2
        )

      if (close < 0) {
        return false
      }

      const raw =
        state.src.slice(
          start + 2,
          close
        )

      const pieces =
        raw.split('|')

      const target =
        (pieces.shift() || '').trim()

      const label =
        pieces.length
          ? pieces.join('|').trim()
          : (
              target
                .split('#', 1)[0]
                .split('/')
                .pop() ||
              target
            )

      const markdownFile =
        state.env?.path
          ? path.resolve(
              state.env.path
            )
          : null

      if (!markdownFile) {
        return false
      }

      const resolved =
        resolveObsidianNote(
          target,
          markdownFile,
          absoluteDocs
        )

      if (!silent) {
        const stats =
          statsFor(state.env)

        if (!resolved) {
          stats.unresolvedLinks.push(
            target
          )

          const token =
            state.push(
              'text',
              '',
              0
            )

          token.content = label
        } else {
          stats.resolvedLinks.push({
            target,
            file: resolved.file
          })

          const open =
            state.push(
              'link_open',
              'a',
              1
            )

          open.attrSet(
            'href',
            resolved.href
          )

          const text =
            state.push(
              'text',
              '',
              0
            )

          text.content = label

          state.push(
            'link_close',
            'a',
            -1
          )
        }
      }

      state.pos =
        close + 2

      return true
    }
  )
}

export function clearObsidianIndexCache() {
  INDEX_CACHE.clear()
}

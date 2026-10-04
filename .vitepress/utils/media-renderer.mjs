import fs from 'node:fs'
import path from 'node:path'

function slash(value) {
  return value.replaceAll('\\', '/')
}

function isExternal(src) {
  return /^(?:https?:)?\/\//i.test(src) ||
    src.startsWith('data:')
}

function cleanSrc(src) {
  let value = String(src || '')
    .split('#', 1)[0]
    .split('?', 1)[0]

  try {
    value = decodeURIComponent(value)
  } catch {
    // Keep original.
  }

  return slash(value)
}

function resolveSourceKey({
  src,
  markdownFile,
  rootDir,
  docsDir
}) {
  if (
    !src ||
    isExternal(src) ||
    src.startsWith('#')
  ) {
    return null
  }

  const clean = cleanSrc(src)
  const publicDir =
    path.join(docsDir, 'public')

  let file

  if (
    clean.startsWith('images/') ||
    clean.startsWith('./images/')
  ) {
    file = path.join(
      publicDir,
      clean.replace(/^\.\//, '')
    )
  } else if (
    clean.startsWith('public/') ||
    clean.startsWith('./public/')
  ) {
    file = path.join(
      publicDir,
      clean
        .replace(/^\.\//, '')
        .replace(/^public\//, '')
    )
  } else if (clean.startsWith('/')) {
    file = path.join(
      publicDir,
      clean.replace(/^\/+/, '')
    )
  } else if (markdownFile) {
    const absoluteMarkdown =
      path.isAbsolute(markdownFile)
        ? markdownFile
        : path.join(docsDir, markdownFile)

    file = path.resolve(
      path.dirname(absoluteMarkdown),
      clean
    )
  } else {
    return null
  }

  return slash(
    path.relative(rootDir, file)
  )
}

function withBase(base, url) {
  const cleanBase =
    base === '/'
      ? '/'
      : `/${base.replace(/^\/|\/$/g, '')}/`

  return (
    cleanBase +
    url.replace(/^\/+/, '')
  ).replace(/\/{2,}/g, '/')
}

function ensureAttr(html, name, value) {
  const attrPattern =
    new RegExp(
      `\\s${name}=(["'])[^"']*\\1`
    )

  if (attrPattern.test(html)) {
    return html.replace(
      attrPattern,
      ` ${name}="${value}"`
    )
  }

  return html.replace(
    /^<img\b/,
    `<img ${name}="${value}"`
  )
}

function removeAttr(html, name) {
  const attrPattern =
    new RegExp(
      `\\s${name}=(["'])[^"']*\\1`,
      'g'
    )

  return html.replace(attrPattern, '')
}

export function installMediaRenderer(
  md,
  {
    rootDir,
    docsDir,
    manifestFile,
    siteBase = '/',
    eagerImageMaxLine = 28
  }
) {
  const previous =
    md.renderer.rules.image

  if (!previous) {
    throw new Error(
      'installMediaRenderer must run after the V2 image renderer'
    )
  }

  let manifest = null

  function getManifest() {
    if (manifest) return manifest

    try {
      manifest = JSON.parse(
        fs.readFileSync(
          manifestFile,
          'utf8'
        )
      )
    } catch {
      manifest = {
        entries: {}
      }
    }

    return manifest
  }

  md.renderer.rules.image = (
    tokens,
    idx,
    options,
    env,
    self
  ) => {
    const token = tokens[idx]
    const src =
      token.attrGet('src') || ''

    const key =
      resolveSourceKey({
        src,
        markdownFile: env?.path,
        rootDir,
        docsDir
      })

    const entry =
      key
        ? getManifest().entries?.[key]
        : null

    let html =
      previous(
        tokens,
        idx,
        options,
        env,
        self
      )

    if (entry?.width && entry?.height) {
      html = ensureAttr(
        html,
        'width',
        String(entry.width)
      )

      html = ensureAttr(
        html,
        'height',
        String(entry.height)
      )
    }

    // Only promote an image when it is both the first Markdown image
    // AND near the start of the article. Images far down the page stay lazy.
    const imageIndex =
      env.__v26MarkdownImageIndex || 0

    env.__v26MarkdownImageIndex =
      imageIndex + 1

    const startLine =
      token.map?.[0] ?? Number.POSITIVE_INFINITY

    const shouldPrioritize =
      imageIndex === 0 &&
      startLine <= eagerImageMaxLine

    if (shouldPrioritize) {
      html = ensureAttr(
        html,
        'loading',
        'eager'
      )

      html = ensureAttr(
        html,
        'fetchpriority',
        'high'
      )
    } else {
      html = ensureAttr(
        html,
        'loading',
        'lazy'
      )

      html =
        removeAttr(
          html,
          'fetchpriority'
        )
    }

    if (!entry?.variants?.length) {
      return html
    }

    const variants =
      [...entry.variants]
        .sort(
          (a, b) => a.width - b.width
        )

    const srcset =
      variants
        .map(
          (variant) =>
            `${withBase(siteBase, variant.url)} ${variant.width}w`
        )
        .join(', ')

    const source =
      `<source type="image/webp" ` +
      `srcset="${md.utils.escapeHtml(srcset)}" ` +
      `sizes="(max-width: 768px) 92vw, (max-width: 1200px) 86vw, 860px">`

    return (
      '<picture class="note-picture">' +
      source +
      html +
      '</picture>'
    )
  }
}

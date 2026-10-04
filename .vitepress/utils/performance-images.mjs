import fs from 'node:fs'
import path from 'node:path'

const sizeCache = new Map()

function isExternal(src) {
  return /^(?:[a-z]+:)?\/\//i.test(src) ||
    src.startsWith('data:') ||
    src.startsWith('#')
}

function cleanSource(src) {
  let value = src.split('#', 1)[0].split('?', 1)[0]

  try {
    value = decodeURIComponent(value)
  } catch {
    // Keep original string if it is not valid URI encoding.
  }

  return value.replaceAll('\\', '/')
}

function resolveLocalImage(src, env, docsDir) {
  if (!src || isExternal(src)) return null

  const clean = cleanSource(src)
  const publicDir = path.join(docsDir, 'public')

  // Existing Obsidian-friendly convention in this repository.
  if (clean.startsWith('images/') || clean.startsWith('./images/')) {
    return path.join(
      publicDir,
      clean.replace(/^\.\//, '')
    )
  }

  if (clean.startsWith('public/') || clean.startsWith('./public/')) {
    return path.join(
      publicDir,
      clean
        .replace(/^\.\//, '')
        .replace(/^public\//, '')
    )
  }

  // VitePress public-root image.
  if (clean.startsWith('/')) {
    return path.join(publicDir, clean.replace(/^\/+/, ''))
  }

  // Standard relative Markdown asset.
  if (env?.path) {
    const markdownFile = path.isAbsolute(env.path)
      ? env.path
      : path.join(docsDir, env.path)

    return path.resolve(path.dirname(markdownFile), clean)
  }

  return null
}

function readUint24LE(buffer, offset) {
  return (
    buffer[offset] |
    (buffer[offset + 1] << 8) |
    (buffer[offset + 2] << 16)
  )
}

function jpegSize(buffer) {
  if (
    buffer.length < 4 ||
    buffer[0] !== 0xff ||
    buffer[1] !== 0xd8
  ) {
    return null
  }

  let offset = 2

  while (offset + 9 < buffer.length) {
    if (buffer[offset] !== 0xff) {
      offset += 1
      continue
    }

    let marker = buffer[offset + 1]

    while (marker === 0xff) {
      offset += 1
      marker = buffer[offset + 1]
    }

    const standalone =
      marker === 0x01 ||
      (marker >= 0xd0 && marker <= 0xd9)

    if (standalone) {
      offset += 2
      continue
    }

    if (offset + 4 > buffer.length) break

    const length = buffer.readUInt16BE(offset + 2)
    if (length < 2) break

    const isSof =
      (marker >= 0xc0 && marker <= 0xc3) ||
      (marker >= 0xc5 && marker <= 0xc7) ||
      (marker >= 0xc9 && marker <= 0xcb) ||
      (marker >= 0xcd && marker <= 0xcf)

    if (isSof && offset + 8 < buffer.length) {
      return {
        height: buffer.readUInt16BE(offset + 5),
        width: buffer.readUInt16BE(offset + 7)
      }
    }

    offset += 2 + length
  }

  return null
}

function svgSize(file) {
  try {
    const source = fs.readFileSync(file, 'utf8').slice(0, 64 * 1024)

    const width = source.match(/\bwidth=["']([\d.]+)(?:px)?["']/i)
    const height = source.match(/\bheight=["']([\d.]+)(?:px)?["']/i)

    if (width && height) {
      return {
        width: Math.round(Number(width[1])),
        height: Math.round(Number(height[1]))
      }
    }

    const viewBox = source.match(
      /\bviewBox=["'][\d.+-]+\s+[\d.+-]+\s+([\d.+-]+)\s+([\d.+-]+)["']/i
    )

    if (viewBox) {
      return {
        width: Math.round(Number(viewBox[1])),
        height: Math.round(Number(viewBox[2]))
      }
    }
  } catch {
    return null
  }

  return null
}

function getImageSize(file) {
  if (!file) return null

  if (sizeCache.has(file)) {
    return sizeCache.get(file)
  }

  let result = null

  try {
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      sizeCache.set(file, null)
      return null
    }

    if (path.extname(file).toLowerCase() === '.svg') {
      result = svgSize(file)
      sizeCache.set(file, result)
      return result
    }

    const buffer = fs.readFileSync(file)

    // PNG
    if (
      buffer.length >= 24 &&
      buffer[0] === 0x89 &&
      buffer.toString('ascii', 1, 4) === 'PNG'
    ) {
      result = {
        width: buffer.readUInt32BE(16),
        height: buffer.readUInt32BE(20)
      }
    }

    // GIF
    else if (
      buffer.length >= 10 &&
      ['GIF87a', 'GIF89a'].includes(
        buffer.toString('ascii', 0, 6)
      )
    ) {
      result = {
        width: buffer.readUInt16LE(6),
        height: buffer.readUInt16LE(8)
      }
    }

    // JPEG
    else if (
      buffer.length >= 4 &&
      buffer[0] === 0xff &&
      buffer[1] === 0xd8
    ) {
      result = jpegSize(buffer)
    }

    // WebP
    else if (
      buffer.length >= 30 &&
      buffer.toString('ascii', 0, 4) === 'RIFF' &&
      buffer.toString('ascii', 8, 12) === 'WEBP'
    ) {
      const chunk = buffer.toString('ascii', 12, 16)

      if (chunk === 'VP8X') {
        result = {
          width: readUint24LE(buffer, 24) + 1,
          height: readUint24LE(buffer, 27) + 1
        }
      } else if (chunk === 'VP8L' && buffer[20] === 0x2f) {
        const b1 = buffer[21]
        const b2 = buffer[22]
        const b3 = buffer[23]
        const b4 = buffer[24]

        result = {
          width: 1 + ((b1 | (b2 << 8)) & 0x3fff),
          height: 1 + (
            ((b2 >> 6) | (b3 << 2) | (b4 << 10)) & 0x3fff
          )
        }
      } else if (
        chunk === 'VP8 ' &&
        buffer[23] === 0x9d &&
        buffer[24] === 0x01 &&
        buffer[25] === 0x2a
      ) {
        result = {
          width: buffer.readUInt16LE(26) & 0x3fff,
          height: buffer.readUInt16LE(28) & 0x3fff
        }
      }
    }
  } catch {
    result = null
  }

  if (
    result &&
    (
      !Number.isFinite(result.width) ||
      !Number.isFinite(result.height) ||
      result.width <= 0 ||
      result.height <= 0
    )
  ) {
    result = null
  }

  sizeCache.set(file, result)
  return result
}

/**
 * Wrap the current V2 renderer rather than replacing it.
 *
 * V2 remains responsible for Obsidian path compatibility and lazy loading.
 * This layer adds:
 * - intrinsic width / height for local images;
 * - eager/high priority for the first Markdown image on each page;
 * - lazy/async behavior remains unchanged for all later images.
 */
export function enhanceMarkdownImages(md, { docsDir }) {
  const previous = md.renderer.rules.image

  if (!previous) {
    throw new Error(
      'enhanceMarkdownImages must run after the V2 image renderer is installed'
    )
  }

  md.renderer.rules.image = (
    tokens,
    idx,
    options,
    env,
    self
  ) => {
    const token = tokens[idx]
    const srcIndex = token.attrIndex('src')

    let originalSrc = ''

    if (srcIndex >= 0) {
      originalSrc = token.attrs[srcIndex][1] || ''

      const localFile = resolveLocalImage(
        originalSrc,
        env,
        docsDir
      )

      const size = getImageSize(localFile)

      if (size) {
        if (token.attrIndex('width') < 0) {
          token.attrSet('width', String(size.width))
        }

        if (token.attrIndex('height') < 0) {
          token.attrSet('height', String(size.height))
        }
      }
    }

    const firstImage = !env.__noteFirstMarkdownImage
    env.__noteFirstMarkdownImage = true

    let html = previous(
      tokens,
      idx,
      options,
      env,
      self
    )

    if (firstImage) {
      // The V2 renderer intentionally adds lazy loading to every image.
      // Override only the first article image to improve perceived loading.
      html = html.replace(
        /\sloading=(["'])lazy\1/,
        ' loading="eager"'
      )

      if (!/\sfetchpriority=/.test(html)) {
        html = html.replace(
          /^<img\b/,
          '<img fetchpriority="high"'
        )
      }
    }

    return html
  }
}

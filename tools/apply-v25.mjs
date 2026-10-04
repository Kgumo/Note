import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const configFile = path.join(root, '.vitepress', 'config.mjs')
const indexFile = path.join(root, 'docs', 'index.md')
const packageFile = path.join(root, 'package.json')

function backup(file) {
  const backupRoot = path.join(root, '.v25-backup')
  fs.mkdirSync(backupRoot, { recursive: true })

  const relative = path
    .relative(root, file)
    .replaceAll('\\', '__')
    .replaceAll('/', '__')

  const target = path.join(backupRoot, relative)

  if (!fs.existsSync(target)) {
    fs.copyFileSync(file, target)
  }
}

function toLf(value) {
  return value.replace(/\r\n/g, '\n')
}

function patchConfig() {
  backup(configFile)

  let source = fs.readFileSync(configFile, 'utf8')
  const hadCrLf = /\r\n/.test(source)

  // Normalize only in memory so Windows CRLF does not break anchors.
  source = toLf(source)

  if (!source.includes("enhanceMarkdownImages")) {
    const importPattern =
      /import\s+\{\s*createRequire\s*\}\s+from\s+['"]module['"]\s*\n/

    if (!importPattern.test(source)) {
      throw new Error(
        'createRequire import anchor not found in config.mjs'
      )
    }

    source = source.replace(
      importPattern,
      (match) =>
        match +
        "import { enhanceMarkdownImages } from './utils/performance-images.mjs'\n"
    )
  }

  if (!source.includes("qmake: 'makefile'")) {
    const markdownPattern =
      /markdown:\s*\{\s*\n\s*lineNumbers:\s*true,\s*\n/

    if (!markdownPattern.test(source)) {
      throw new Error(
        'markdown.lineNumbers anchor not found in config.mjs'
      )
    }

    source = source.replace(
      markdownPattern,
      (match) =>
        match +
        "    languageAlias: {\n" +
        "      qmake: 'makefile'\n" +
        "    },\n"
    )
  }

  if (!source.includes('V25_IMAGE_PERFORMANCE')) {
    // Current V2 ends its custom image renderer immediately before
    // markdown.config closes. Match structurally instead of relying on
    // exact whitespace or Windows line endings.
    const imageRendererEnd =
      /(return\s+defaultImageRule\(tokens,\s*idx,\s*options,\s*env,\s*self\)\s*\n\s*\}\s*\n)(\s*\}\s*\n\s*\},\s*\n\s*vite:\s*\{)/

    if (!imageRendererEnd.test(source)) {
      const excerptIndex = source.indexOf(
        'return defaultImageRule(tokens, idx, options, env, self)'
      )

      const excerpt =
        excerptIndex >= 0
          ? source.slice(
              Math.max(0, excerptIndex - 180),
              Math.min(source.length, excerptIndex + 320)
            )
          : '(defaultImageRule return not found)'

      throw new Error(
        'V2 image renderer end could not be identified.\n\n' +
        'Nearby config excerpt:\n' +
        excerpt
      )
    }

    source = source.replace(
      imageRendererEnd,
      (
        _full,
        rendererClose,
        configClose
      ) =>
        rendererClose +
        "\n" +
        "      // V25_IMAGE_PERFORMANCE\n" +
        "      enhanceMarkdownImages(md, {\n" +
        "        docsDir: path.resolve(__dirname, '../docs')\n" +
        "      })\n" +
        configClose
    )
  }

  if (!source.includes('V25_HERO_PRELOAD')) {
    const iconPattern =
      /(\['link',\s*\{\s*rel:\s*'icon',\s*href:\s*`\$\{siteBase\}head\.svg`\s*\}\],\s*\n)/

    if (iconPattern.test(source)) {
      source = source.replace(
        iconPattern,
        "$1" +
        "    // V25_HERO_PRELOAD\n" +
        "    ['link', { rel: 'preload', as: 'image', href: `${siteBase}logo.svg`, type: 'image/svg+xml' }],\n"
      )
    } else {
      console.warn(
        'Hero preload anchor not found; skipping this optional optimization.'
      )
    }
  }

  // Preserve the working-tree line-ending convention.
  if (hadCrLf) {
    source = source.replace(/\n/g, '\r\n')
  }

  fs.writeFileSync(configFile, source, 'utf8')
}

function patchHomepageRuntime() {
  backup(indexFile)

  let source = fs.readFileSync(indexFile, 'utf8')
  const hadCrLf = /\r\n/.test(source)
  source = toLf(source)

  if (source.includes('V25_HOME_RUNTIME_CLEANUP')) {
    return
  }

  const start = source.indexOf('<script setup>')
  const end = source.indexOf('</script>', start)

  if (start < 0 || end < 0) {
    console.warn(
      'docs/index.md script block not found; homepage runtime cleanup skipped.'
    )
    return
  }

  const replacement = `<script setup>
// V25_HOME_RUNTIME_CLEANUP
import { onBeforeUnmount, onMounted } from 'vue'

let intervalId = null
let timeoutId = null
const createdParticles = []

onMounted(() => {
  const taglines = [
    "代码是写给人看的，只是顺便让机器能运行",
    "Stay hungry, stay foolish",
    "求知若饥，虚心若愚",
    "技术是解决问题的艺术",
    "吾魂兮无求乎永生 竭尽兮人事之所能"
  ]

  let current = 0
  const el = document.querySelector('.VPHero .tagline')

  const changeTagline = () => {
    if (!el) return

    current = (current + 1) % taglines.length
    el.style.opacity = 0

    if (timeoutId) {
      window.clearTimeout(timeoutId)
    }

    timeoutId = window.setTimeout(() => {
      if (!el.isConnected) return
      el.textContent = taglines[current]
      el.style.opacity = 1
    }, 500)
  }

  changeTagline()
  intervalId = window.setInterval(changeTagline, 5000)

  const features = document.querySelectorAll('.VPFeature')

  features.forEach((feature) => {
    if (feature.querySelector(':scope > .particles')) return

    const particlesContainer = document.createElement('div')
    particlesContainer.className = 'particles'
    feature.appendChild(particlesContainer)
    createdParticles.push(particlesContainer)

    for (let i = 0; i < 15; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'

      const size = Math.random() * 10 + 5
      particle.style.width = \`\${size}px\`
      particle.style.height = \`\${size}px\`
      particle.style.left = \`\${Math.random() * 100}%\`
      particle.style.top = \`\${Math.random() * 100}%\`

      const hue = 240 + Math.random() * 60
      particle.style.background =
        \`hsla(\${hue}, 80%, 70%, \${0.2 + Math.random() * 0.3})\`

      particle.style.animationDelay =
        \`\${Math.random() * 5}s\`

      particle.style.animationDuration =
        \`\${10 + Math.random() * 20}s\`

      particlesContainer.appendChild(particle)
    }
  })
})

onBeforeUnmount(() => {
  if (intervalId) {
    window.clearInterval(intervalId)
    intervalId = null
  }

  if (timeoutId) {
    window.clearTimeout(timeoutId)
    timeoutId = null
  }

  for (const element of createdParticles) {
    element.remove()
  }

  createdParticles.length = 0
})
</script>`

  source =
    source.slice(0, start) +
    replacement +
    source.slice(end + '</script>'.length)

  if (hadCrLf) {
    source = source.replace(/\n/g, '\r\n')
  }

  fs.writeFileSync(indexFile, source, 'utf8')
}

function patchPackageJson() {
  backup(packageFile)

  const pkg = JSON.parse(
    fs.readFileSync(packageFile, 'utf8')
  )

  pkg.scripts ||= {}

  pkg.scripts['site:audit'] =
    'node tools/site-audit.mjs'

  pkg.scripts['migrate:legacy-images'] =
    'node tools/migrate-legacy-images.mjs'

  pkg.scripts['cleanup:legacy-deps'] =
    'node tools/cleanup-legacy-deps.mjs'

  fs.writeFileSync(
    packageFile,
    JSON.stringify(pkg, null, 2) + '\n',
    'utf8'
  )
}

patchConfig()
patchHomepageRuntime()
patchPackageJson()

console.log('V2.5.1 performance layer applied successfully.')
console.log('Windows CRLF/LF line endings are supported.')
console.log('Visual V2 files were not redesigned.')
console.log('Backups: .v25-backup/')

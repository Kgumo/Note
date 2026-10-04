import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

const root =
  process.cwd()

const configFile =
  path.join(
    root,
    '.vitepress',
    'config.mjs'
  )

const customCssFile =
  path.join(
    root,
    '.vitepress',
    'theme',
    'custom.css'
  )

const packageFile =
  path.join(
    root,
    'package.json'
  )

const gitignoreFile =
  path.join(
    root,
    '.gitignore'
  )

const backupDir =
  path.join(
    root,
    '.v29-backup'
  )

const cssStart =
  '/* MOBILE_COMPACT_V29_START */'
const cssEnd =
  '/* MOBILE_COMPACT_V29_END */'

function backup(file) {
  if (!fs.existsSync(file)) return

  fs.mkdirSync(
    backupDir,
    {
      recursive: true
    }
  )

  const target =
    path.join(
      backupDir,
      path
        .relative(root, file)
        .replaceAll('\\', '__')
        .replaceAll('/', '__')
    )

  if (!fs.existsSync(target)) {
    fs.copyFileSync(
      file,
      target
    )
  }
}

function normalize(value) {
  return value.replace(
    /\r\n/g,
    '\n'
  )
}

function restoreEol(
  value,
  hadCrLf
) {
  return hadCrLf
    ? value.replace(/\n/g, '\r\n')
    : value
}

function findMatchingBrace(
  source,
  openingBraceIndex
) {
  let depth = 0
  let mode = 'code'
  let quote = ''
  let escaped = false

  for (
    let i = openingBraceIndex;
    i < source.length;
    i += 1
  ) {
    const ch = source[i]
    const next = source[i + 1]

    if (mode === 'line-comment') {
      if (ch === '\n') {
        mode = 'code'
      }
      continue
    }

    if (mode === 'block-comment') {
      if (
        ch === '*' &&
        next === '/'
      ) {
        mode = 'code'
        i += 1
      }
      continue
    }

    if (mode === 'string') {
      if (escaped) {
        escaped = false
        continue
      }

      if (ch === '\\') {
        escaped = true
        continue
      }

      if (ch === quote) {
        mode = 'code'
        quote = ''
      }

      continue
    }

    if (mode === 'template') {
      if (escaped) {
        escaped = false
        continue
      }

      if (ch === '\\') {
        escaped = true
        continue
      }

      if (ch === '`') {
        mode = 'code'
      }

      continue
    }

    if (
      ch === '/' &&
      next === '/'
    ) {
      mode = 'line-comment'
      i += 1
      continue
    }

    if (
      ch === '/' &&
      next === '*'
    ) {
      mode = 'block-comment'
      i += 1
      continue
    }

    if (
      ch === "'" ||
      ch === '"'
    ) {
      mode = 'string'
      quote = ch
      continue
    }

    if (ch === '`') {
      mode = 'template'
      continue
    }

    if (ch === '{') {
      depth += 1
    } else if (ch === '}') {
      depth -= 1

      if (depth === 0) {
        return i
      }
    }
  }

  return -1
}

function addImport(
  source,
  statement,
  moduleNeedle
) {
  if (
    source.includes(
      moduleNeedle
    )
  ) {
    return source
  }

  const imports =
    [...source.matchAll(
      /^import\s+.*?(?:from\s+)?['"][^'"]+['"]\s*;?\s*$/gm
    )]

  if (!imports.length) {
    throw new Error(
      'No import block found in config.mjs'
    )
  }

  const last =
    imports[imports.length - 1]

  const at =
    last.index +
    last[0].length

  return (
    source.slice(0, at) +
    '\n' +
    statement +
    source.slice(at)
  )
}

function patchConfig() {
  backup(configFile)

  let source =
    fs.readFileSync(
      configFile,
      'utf8'
    )

  const hadCrLf =
    /\r\n/.test(source)

  source =
    normalize(source)

  source =
    addImport(
      source,
      "import { installObsidianCompat } from './utils/obsidian-compat.mjs'",
      "from './utils/obsidian-compat.mjs'"
    )

  if (
    !source.includes(
      'V29_OBSIDIAN_COMPAT'
    )
  ) {
    const pattern =
      /config\s*:\s*async\s*\(\s*md\s*\)\s*=>\s*\{/

    const match =
      pattern.exec(source)

    if (!match) {
      throw new Error(
        'Could not locate markdown.config callback.'
      )
    }

    const openingBrace =
      source.indexOf(
        '{',
        match.index
      )

    const closingBrace =
      findMatchingBrace(
        source,
        openingBrace
      )

    if (closingBrace < 0) {
      throw new Error(
        'Could not locate markdown.config closing brace.'
      )
    }

    const injection =
      "\n" +
      "      // V29_OBSIDIAN_COMPAT\n" +
      "      installObsidianCompat(md, {\n" +
      "        docsDir: path.resolve(__dirname, '../docs')\n" +
      "      })\n"

    // Install early in the callback. Inline rules are active later when
    // VitePress parses pages; existing image renderers still wrap the tokens.
    const insertAt =
      openingBrace + 1

    source =
      source.slice(
        0,
        insertAt
      ) +
      injection +
      source.slice(insertAt)
  }

  source =
    restoreEol(
      source,
      hadCrLf
    )

  fs.writeFileSync(
    configFile,
    source,
    'utf8'
  )

  const check =
    spawnSync(
      process.execPath,
      [
        '--check',
        configFile
      ],
      {
        cwd: root,
        encoding: 'utf8'
      }
    )

  if (check.status !== 0) {
    throw new Error(
      'config.mjs syntax validation failed:\n' +
      (
        check.stderr ||
        check.stdout ||
        ''
      )
    )
  }
}

function patchCss() {
  backup(customCssFile)

  let source =
    fs.readFileSync(
      customCssFile,
      'utf8'
    )

  const start =
    cssStart.replace(
      /[.*+?^${}()|[\]\\]/g,
      '\\$&'
    )

  const end =
    cssEnd.replace(
      /[.*+?^${}()|[\]\\]/g,
      '\\$&'
    )

  source =
    source
      .replace(
        new RegExp(
          `${start}[\\s\\S]*?${end}`,
          'g'
        ),
        ''
      )
      .trimEnd()

  const patch = `

${cssStart}

/* =========================================================
   V2.9 Mobile Compact Layout
   Desktop V2 remains unchanged.
   ========================================================= */

@media (max-width: 959px) {
  /* The old theme added 1.5rem around VPContent on top of VitePress'
     own component padding. On a 390px phone that made everything
     unnecessarily narrow. Let each VitePress component own its gutter. */
  body.sidebar-closed .VPContent,
  body:not(.sidebar-open) .VPContent {
    padding-left: 0 !important;
    padding-right: 0 !important;
  }

  body.sidebar-open .VPSidebar {
    width: min(88vw, 340px) !important;
  }
}

@media (max-width: 639px) {
  /* ---------- Hero ---------- */
  .VPHomeHero {
    padding:
      calc(
        var(--vp-nav-height) +
        var(--vp-layout-top-height, 0px) +
        30px
      )
      20px
      28px !important;
  }

  .VPHomeHero .container {
    max-width: 430px !important;
  }

  .VPHero .image {
    margin:
      -34px
      auto
      -8px !important;
  }

  .VPHero .image-container {
    width: 176px !important;
    height: 176px !important;
  }

  .VPHero .image-bg {
    width: 132px !important;
    height: 132px !important;
    filter: blur(30px) !important;
    opacity: 0.70;
  }

  .VPHero .image-src {
    max-width: 150px !important;
    max-height: 150px !important;
  }

  .VPHero.has-image .container {
    text-align: left !important;
  }

  .VPHero.has-image .name,
  .VPHero.has-image .text,
  .VPHero.has-image .tagline {
    margin-left: 0 !important;
    margin-right: 0 !important;
  }

  .VPHero .heading {
    gap: 0.22rem;
  }

  .VPHero .name {
    max-width: 100% !important;
    line-height: 1.08 !important;
    font-size: clamp(
      1.9rem,
      9vw,
      2.2rem
    ) !important;
    white-space: normal !important;
  }

  .VPHero .text {
    max-width: 100% !important;
    line-height: 1.10 !important;
    font-size: clamp(
      2rem,
      9.4vw,
      2.35rem
    ) !important;
    white-space: normal !important;
    word-break: keep-all;
    overflow-wrap: normal;
    text-wrap: balance;
  }

  .VPHero .text::after {
    bottom: -8px;
  }

  .VPHero .tagline {
    max-width: 100% !important;
    padding-top: 16px !important;
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
    text-wrap: pretty;
  }

  .VPHero .actions {
    display: grid !important;
    grid-template-columns: 1fr !important;
    gap: 10px !important;
    margin: 0 !important;
    padding-top: 20px !important;
  }

  .VPHero .action {
    width: 100%;
    padding: 0 !important;
  }

  .VPHero .VPButton {
    display: flex !important;
    width: 100% !important;
    min-height: 44px;
    align-items: center;
    justify-content: center;
    padding:
      0.64rem
      0.82rem !important;
    white-space: normal;
    text-align: center;
    line-height: 1.3;
    font-size: 0.82rem;
  }

  /* ---------- Three homepage entry cards ----------
     Show one full card + a hint of the next card instead of stacking all 3. */
  .VPFeatures {
    overflow: hidden;
    padding:
      0
      0
      0.2rem !important;
  }

  .VPFeatures .container {
    max-width: none !important;
  }

  .VPFeatures .items {
    flex-wrap: nowrap !important;
    gap: 10px;
    overflow-x: auto;
    margin: 0 !important;
    padding:
      0
      20px
      10px !important;
    scroll-padding-left: 20px;
    scroll-snap-type: x mandatory;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }

  .VPFeatures .items::-webkit-scrollbar {
    display: none;
  }

  .VPFeatures .item,
  .VPFeatures .item.grid-3 {
    flex:
      0
      0
      min(84vw, 330px) !important;
    width:
      min(84vw, 330px) !important;
    padding: 0 !important;
    scroll-snap-align: start;
  }

  .VPFeature {
    min-height: 0 !important;
    transform: none !important;
  }

  .VPFeature .box {
    min-height: 210px;
    padding: 20px !important;
  }

  .VPFeature .icon {
    margin-bottom: 16px !important;
    width: 44px !important;
    height: 44px !important;
  }

  .VPFeature .particle:nth-child(n + 7) {
    display: none;
  }

  /* ---------- System Map ----------
     Horizontal cards are much more natural than a 5-card vertical tower. */
  .integration-section {
    margin-top: 1.35rem !important;
    padding:
      0
      14px !important;
    contain-intrinsic-size: 860px;
  }

  .integration-shell {
    padding: 1rem !important;
    border-radius: 18px !important;
  }

  .integration-shell::before {
    width: 300px !important;
    height: 300px !important;
    right: -160px !important;
    top: -170px !important;
  }

  .section-heading {
    padding-bottom: 1.05rem !important;
  }

  .section-heading h2 {
    max-width: 100% !important;
    font-size: 1.65rem !important;
    line-height: 1.15 !important;
    text-wrap: balance;
  }

  .section-heading p {
    margin-top: 0.7rem !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
  }

  .stack-map {
    display: grid !important;
    grid-template-columns: none !important;
    grid-auto-flow: column;
    grid-auto-columns:
      minmax(
        238px,
        80vw
      );
    gap: 10px !important;
    overflow-x: auto;
    margin:
      0.9rem
      -0.25rem
      0 !important;
    padding:
      0.1rem
      0.25rem
      0.65rem;
    scroll-padding-left: 0.25rem;
    scroll-snap-type: x mandatory;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }

  .stack-map::-webkit-scrollbar {
    display: none;
  }

  .stack-card,
  .stack-output {
    grid-column: auto !important;
    min-height: 208px !important;
    scroll-snap-align: start;
    transform: none !important;
  }

  .stack-card {
    padding: 1rem !important;
  }

  .stack-card-body {
    margin-top: 1.35rem !important;
  }

  .stack-card h3 {
    font-size: 1.18rem !important;
  }

  .stack-output {
    padding: 1rem !important;
  }

  .output-mark {
    margin:
      0.9rem
      0
      0.45rem !important;
  }

  .lower-grid {
    gap: 10px !important;
    margin-top: 10px !important;
  }

  .pipeline-panel,
  .projects-panel {
    padding: 1rem !important;
  }

  .panel-heading {
    margin-bottom: 0.85rem !important;
  }

  .pipeline-list li {
    grid-template-columns:
      30px
      minmax(0, 1fr) !important;
    gap: 0.55rem !important;
    padding:
      0.72rem
      0 !important;
  }

  .pipeline-title-row {
    display: flex !important;
    align-items: flex-start !important;
    flex-direction: column;
    gap: 0.15rem !important;
  }

  .pipeline-title-row h4 {
    font-size: 0.96rem !important;
  }

  .pipeline-title-row span {
    display: block !important;
    margin-top: 0 !important;
    text-align: left !important;
    font-size: 0.64rem !important;
  }

  .pipeline-copy p {
    margin-top: 0.28rem !important;
    font-size: 0.83rem !important;
    line-height: 1.55 !important;
  }

  /* Project snapshots also become a swipe row. */
  .project-list {
    display: grid !important;
    grid-template-columns: none !important;
    grid-auto-flow: column;
    grid-auto-columns:
      minmax(
        240px,
        82vw
      );
    gap: 10px !important;
    overflow-x: auto;
    padding-bottom: 0.5rem;
    scroll-snap-type: x mandatory;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }

  .project-list::-webkit-scrollbar {
    display: none;
  }

  .project-card {
    min-height: 190px;
    padding: 0.9rem !important;
    scroll-snap-align: start;
    transform: none !important;
  }

  .project-topline {
    margin-bottom: 0.7rem !important;
  }

  .integration-footer {
    gap: 0.8rem !important;
    margin-top: 10px !important;
    padding: 0.9rem !important;
  }

  .integration-footer nav {
    display: flex !important;
    width: 100%;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    overflow-x: auto;
    padding-bottom: 0.25rem;
    scroll-snap-type: x proximity;
    scrollbar-width: none;
  }

  .integration-footer nav::-webkit-scrollbar {
    display: none;
  }

  .integration-footer a {
    flex: 0 0 auto;
    min-height: 40px;
    align-items: center;
    scroll-snap-align: start;
  }

  /* ---------- Recent updates ---------- */
  .recent-updates {
    margin:
      0.8rem
      auto
      3rem !important;
    padding:
      0
      18px !important;
  }

  .updates-heading {
    padding:
      1.45rem
      0
      0.95rem !important;
  }

  .updates-heading h2 {
    font-size: 1.65rem !important;
  }

  .updates-heading p {
    margin-top: 0.5rem !important;
    font-size: 0.84rem !important;
    line-height: 1.6 !important;
  }

  .update-row {
    grid-template-columns:
      28px
      minmax(0, 1fr)
      20px !important;
    gap: 0.5rem !important;
    min-height: 66px !important;
    padding:
      0
      0.55rem !important;
  }

  .update-row:hover {
    padding-left: 0.55rem !important;
  }

  .update-row h3 {
    display: -webkit-box;
    overflow: hidden;
    font-size: 0.91rem !important;
    line-height: 1.45 !important;
    white-space: normal !important;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
  }

  .row-number {
    font-size: 0.69rem !important;
  }

  .row-action {
    font-size: 0.96rem !important;
  }

  /* ---------- Normal documentation pages ---------- */
  .VPDoc {
    padding-left: 18px !important;
    padding-right: 18px !important;
  }

  .vp-doc {
    font-size: 15.5px;
  }

  .vp-doc h1 {
    font-size: 1.8rem;
    line-height: 1.2;
  }

  .vp-doc h2 {
    font-size: 1.42rem;
  }

  .vp-doc div[class*='language-'] {
    margin-left: -18px;
    margin-right: -18px;
    border-radius: 0;
  }

  .vp-doc table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
}

${cssEnd}
`

  fs.writeFileSync(
    customCssFile,
    source +
      patch +
      '\n',
    'utf8'
  )
}

function patchPackage() {
  backup(packageFile)

  const pkg =
    JSON.parse(
      fs.readFileSync(
        packageFile,
        'utf8'
      )
    )

  pkg.scripts ||= {}

  pkg.scripts['obsidian:audit'] =
    'node tools/obsidian-audit.mjs'

  fs.writeFileSync(
    packageFile,
    JSON.stringify(
      pkg,
      null,
      2
    ) +
      '\n',
    'utf8'
  )
}

function patchGitignore() {
  backup(gitignoreFile)

  let source =
    fs.existsSync(
      gitignoreFile
    )
      ? fs.readFileSync(
          gitignoreFile,
          'utf8'
        )
      : ''

  const additions = [
    '.v29-backup/'
  ]

  const existing =
    new Set(
      source.split(/\r?\n/)
    )

  for (
    const line
    of additions
  ) {
    if (existing.has(line)) {
      continue
    }

    if (
      source.length &&
      !source.endsWith('\n')
    ) {
      source += '\n'
    }

    source += `${line}\n`
  }

  fs.writeFileSync(
    gitignoreFile,
    source,
    'utf8'
  )
}

patchConfig()
patchCss()
patchPackage()
patchGitignore()

console.log('')
console.log(
  'V2.9 Obsidian compatibility + mobile layout applied.'
)
console.log(
  'Desktop V2 layout/components were not replaced.'
)
console.log(
  'Backups: .v29-backup/'
)

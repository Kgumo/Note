import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

const root = process.cwd()

const configFile =
  path.join(root, '.vitepress', 'config.mjs')

const packageFile =
  path.join(root, 'package.json')

const workflowFile =
  path.join(
    root,
    '.github',
    'workflows',
    'deploy.yml'
  )

const gitignoreFile =
  path.join(root, '.gitignore')

const backupRoot =
  path.join(root, '.v26-backup')

function backup(file) {
  if (!fs.existsSync(file)) return null

  fs.mkdirSync(
    backupRoot,
    { recursive: true }
  )

  const target =
    path.join(
      backupRoot,
      path
        .relative(root, file)
        .replaceAll('\\', '__')
        .replaceAll('/', '__')
    )

  if (!fs.existsSync(target)) {
    fs.copyFileSync(file, target)
  }

  return target
}

function normalize(source) {
  return source.replace(/\r\n/g, '\n')
}

function restoreLineEnding(
  source,
  hadCrLf
) {
  return hadCrLf
    ? source.replace(/\n/g, '\r\n')
    : source
}

/**
 * Finds the matching closing brace for a JavaScript block.
 * It skips quoted strings, template literals and comments, so braces inside
 * regex-like text, template strings and comments do not confuse the count.
 */
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
      if (ch === '*' && next === '/') {
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

      // For our purpose it is safe to ignore ${...} braces inside a template:
      // they are always paired before the template's closing backtick.
      if (ch === '`') {
        mode = 'code'
      }

      continue
    }

    if (ch === '/' && next === '/') {
      mode = 'line-comment'
      i += 1
      continue
    }

    if (ch === '/' && next === '*') {
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
      continue
    }

    if (ch === '}') {
      depth -= 1

      if (depth === 0) {
        return i
      }
    }
  }

  return -1
}

function locateMarkdownConfigBody(source) {
  const pattern =
    /config\s*:\s*async\s*\(\s*md\s*\)\s*=>\s*\{/

  const match =
    pattern.exec(source)

  if (!match) {
    throw new Error(
      'Could not locate markdown.config async callback in config.mjs'
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
      'Could not find the closing brace of markdown.config'
    )
  }

  return {
    openingBrace,
    closingBrace
  }
}

function addImport(
  source,
  statement,
  moduleNeedle
) {
  if (source.includes(moduleNeedle)) {
    return source
  }

  const imports =
    [...source.matchAll(
      /^import\s+.*?from\s+['"][^'"]+['"]\s*;?\s*$/gm
    )]

  if (!imports.length) {
    throw new Error(
      'No ES module imports found in config.mjs'
    )
  }

  const last =
    imports[imports.length - 1]

  const insertAt =
    last.index + last[0].length

  return (
    source.slice(0, insertAt) +
    '\n' +
    statement +
    source.slice(insertAt)
  )
}

function patchConfig() {
  const backupFile =
    backup(configFile)

  let source =
    fs.readFileSync(
      configFile,
      'utf8'
    )

  const hadCrLf =
    /\r\n/.test(source)

  source = normalize(source)

  source = addImport(
    source,
    "import { installMediaRenderer } from './utils/media-renderer.mjs'",
    "from './utils/media-renderer.mjs'"
  )

  if (!source.includes("qmake: 'makefile'")) {
    const markdownPattern =
      /markdown\s*:\s*\{\s*\n\s*lineNumbers\s*:\s*true\s*,?\s*\n/

    if (markdownPattern.test(source)) {
      source = source.replace(
        markdownPattern,
        (match) =>
          match +
          "    languageAlias: {\n" +
          "      qmake: 'makefile'\n" +
          "    },\n"
      )
    } else {
      console.warn(
        'qmake alias anchor not found; skipping that optional build-warning fix.'
      )
    }
  }

  if (!source.includes('V26_MEDIA_RENDERER')) {
    const {
      closingBrace
    } = locateMarkdownConfigBody(source)

    const injection =
      "\n\n" +
      "      // V26_MEDIA_RENDERER\n" +
      "      // Installed last so it wraps the existing V2/V2.5 image renderer\n" +
      "      // instead of depending on its exact source layout.\n" +
      "      installMediaRenderer(md, {\n" +
      "        rootDir: path.resolve(__dirname, '..'),\n" +
      "        docsDir: path.resolve(__dirname, '../docs'),\n" +
      "        manifestFile: path.resolve(__dirname, './cache/media-manifest.json'),\n" +
      "        siteBase,\n" +
      "        eagerImageMaxLine: 28\n" +
      "      })\n"

    source =
      source.slice(0, closingBrace) +
      injection +
      source.slice(closingBrace)
  }

  source =
    restoreLineEnding(
      source,
      hadCrLf
    )

  fs.writeFileSync(
    configFile,
    source,
    'utf8'
  )

  // Syntax-check the actual patched file. If it fails, restore the backup
  // immediately instead of leaving a broken config in the working tree.
  const check =
    spawnSync(
      process.execPath,
      ['--check', configFile],
      {
        cwd: root,
        encoding: 'utf8'
      }
    )

  if (check.status !== 0) {
    if (backupFile) {
      fs.copyFileSync(
        backupFile,
        configFile
      )
    }

    throw new Error(
      'Patched config.mjs failed syntax validation and was restored.\n' +
      (check.stderr || check.stdout || '')
    )
  }
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

  pkg.scripts['docs:dev'] =
    'npm run media:prepare && vitepress dev'

  pkg.scripts['docs:build'] =
    'npm run media:prepare && vitepress build'

  pkg.scripts['media:prepare'] =
    'node tools/media-pipeline.mjs prepare'

  pkg.scripts['media:audit'] =
    'node tools/media-pipeline.mjs audit'

  pkg.scripts['media:clean'] =
    'node tools/media-pipeline.mjs clean'

  pkg.scripts['media:mirror'] =
    'node tools/media-pipeline.mjs mirror'

  pkg.scripts['media:mirror:apply'] =
    'node tools/media-pipeline.mjs mirror --apply'

  pkg.scripts['site:audit'] =
    'npm run media:audit'

  pkg.devDependencies ||= {}
  pkg.devDependencies.sharp =
    '0.35.5'

  fs.writeFileSync(
    packageFile,
    JSON.stringify(
      pkg,
      null,
      2
    ) + '\n',
    'utf8'
  )
}

function patchWorkflow() {
  if (!fs.existsSync(workflowFile)) {
    console.warn(
      'GitHub Pages workflow not found; skipping media cache step.'
    )
    return
  }

  backup(workflowFile)

  let source =
    fs.readFileSync(
      workflowFile,
      'utf8'
    )

  const hadCrLf =
    /\r\n/.test(source)

  source = normalize(source)

  if (
    !source.includes(
      'Restore generated media cache'
    )
  ) {
    const setupPattern =
      /(\s+- name:\s*Setup Node\s*\n\s*uses:\s*actions\/setup-node@v4\s*\n\s*with:\s*\n(?:\s+.*\n)*?\s+cache:\s*npm\s*\n)/

    if (setupPattern.test(source)) {
      source = source.replace(
        setupPattern,
        (match) =>
          match +
          "\n" +
          "      - name: Restore generated media cache\n" +
          "        uses: actions/cache@v4\n" +
          "        with:\n" +
          "          path: |\n" +
          "            docs/public/_media\n" +
          "            .vitepress/cache/media-manifest.json\n" +
          "          key: media-${{ runner.os }}-${{ hashFiles('docs/**', 'media.config.mjs', 'tools/media-pipeline.mjs') }}\n" +
          "          restore-keys: |\n" +
          "            media-${{ runner.os }}-\n"
      )
    } else {
      console.warn(
        'Could not find setup-node block; CI media cache step skipped.'
      )
    }
  }

  source =
    restoreLineEnding(
      source,
      hadCrLf
    )

  fs.writeFileSync(
    workflowFile,
    source,
    'utf8'
  )
}

function patchGitignore() {
  backup(gitignoreFile)

  let source =
    fs.existsSync(gitignoreFile)
      ? fs.readFileSync(
          gitignoreFile,
          'utf8'
        )
      : ''

  const additions = [
    'docs/public/_media/',
    '.v26-backup/',
    '.v25-backup/'
  ]

  for (const line of additions) {
    const exists =
      source
        .split(/\r?\n/)
        .includes(line)

    if (!exists) {
      if (
        source.length &&
        !source.endsWith('\n')
      ) {
        source += '\n'
      }

      source += `${line}\n`
    }
  }

  fs.writeFileSync(
    gitignoreFile,
    source,
    'utf8'
  )
}

patchConfig()
patchPackage()
patchWorkflow()
patchGitignore()

console.log('')
console.log(
  'V2.6.1 media pipeline configuration applied successfully.'
)
console.log(
  'The renderer was injected structurally at the end of markdown.config.'
)
console.log(
  'Existing V2/V2.5 image behavior remains underneath it.'
)
console.log(
  'Backups: .v26-backup/'
)

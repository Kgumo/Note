import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

const root = process.cwd()

const configFile =
  path.join(root, '.vitepress', 'config.mjs')

const packageFile =
  path.join(root, 'package.json')

const indexFile =
  path.join(root, 'docs', 'index.md')

const buildIndexFile =
  path.join(root, 'docs', 'build', 'index.md')

const gitignoreFile =
  path.join(root, '.gitignore')

const backupDir =
  path.join(root, '.v28-backup')

function backup(file) {
  if (!fs.existsSync(file)) return

  fs.mkdirSync(
    backupDir,
    { recursive: true }
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
    fs.copyFileSync(file, target)
  }
}

function preserveEol(source, hadCrLf) {
  return hadCrLf
    ? source.replace(/\n/g, '\r\n')
    : source
}

function normalize(source) {
  return source.replace(/\r\n/g, '\n')
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

  source = normalize(source)

  if (!source.includes('const buildSidebar')) {
    const internshipPattern =
      /const\s+InternshipSidebar\s*=\s*set_sidebar\('Internship',\s*configPath\)\s*\n/

    if (!internshipPattern.test(source)) {
      throw new Error(
        'Could not locate InternshipSidebar declaration in config.mjs'
      )
    }

    source = source.replace(
      internshipPattern,
      (match) =>
        match +
        "\n" +
        "// 综合技术目录：顶层阶段默认展开，子目录仍可折叠。\n" +
        "const buildSidebar = set_sidebar('build', configPath).map((item) =>\n" +
        "  item.items\n" +
        "    ? { ...item, collapsed: false }\n" +
        "    : item\n" +
        ")\n"
    )
  }

  if (!source.includes("'/build/': buildSidebar")) {
    const aiMapping =
      /(\s*'\/AI\/'\s*:\s*aiSidebar\s*,?\s*\n)/

    if (!aiMapping.test(source)) {
      throw new Error(
        'Could not locate /AI/ sidebar mapping in config.mjs'
      )
    }

    source = source.replace(
      aiMapping,
      "$1      '/build/': buildSidebar,\n"
    )
  }

  source =
    preserveEol(
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
      ['--check', configFile],
      {
        cwd: root,
        encoding: 'utf8'
      }
    )

  if (check.status !== 0) {
    throw new Error(
      'config.mjs syntax validation failed:\n' +
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

  pkg.scripts['knowledge:prepare'] =
    'node tools/generate-build-graph.mjs'

  for (
    const scriptName
    of ['docs:dev', 'docs:build']
  ) {
    const existing =
      pkg.scripts[scriptName]

    if (!existing) continue

    if (
      !existing.includes(
        'npm run knowledge:prepare'
      )
    ) {
      pkg.scripts[scriptName] =
        `npm run knowledge:prepare && ${existing}`
    }
  }

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

function patchHomeCard() {
  backup(indexFile)

  let source =
    fs.readFileSync(
      indexFile,
      'utf8'
    )

  source = source.replace(
    /title:\s*["']知识图谱(?:\([^"\n]*\)|（[^"\n]*）)?["']/,
    'title: "知识图谱"'
  )

  source = source.replace(
    /details:\s*["']构建结构化知识体系网络["']/,
    'details: "连接 C++ / Qt / AI / ONNX 与综合技术实践"'
  )

  fs.writeFileSync(
    indexFile,
    source,
    'utf8'
  )
}

function patchBuildIndex() {
  if (!fs.existsSync(buildIndexFile)) return

  backup(buildIndexFile)

  let source =
    fs.readFileSync(
      buildIndexFile,
      'utf8'
    )

  // Current file has an empty frontmatter block.
  // Give VitePress a stable title without changing the article body.
  if (
    /^---\s*\r?\n\s*\r?\n---/.test(source)
  ) {
    source =
      source.replace(
        /^---\s*\r?\n\s*\r?\n---/,
        [
          '---',
          'title: 综合技术',
          'outline: [2, 3]',
          '---'
        ].join('\n')
      )
  }

  fs.writeFileSync(
    buildIndexFile,
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

  const line =
    '.vitepress/cache/build-graph.generated.mjs'

  const lines =
    source.split(/\r?\n/)

  if (!lines.includes(line)) {
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
patchPackage()
patchHomeCard()
patchBuildIndex()
patchGitignore()

console.log(
  'V2.8 build sidebar + knowledge graph configuration applied.'
)
console.log(
  'V2 homepage layout was not redesigned.'
)
console.log(
  'Backups: .v28-backup/'
)

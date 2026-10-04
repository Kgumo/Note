import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

const root = process.cwd()
const packageFile = path.join(root, 'package.json')

const obsolete = [
  'vitepress-plugin-mermaid',
  'vite-plugin-optimize-persist',
  'vite-plugin-package-config'
]

const pkg = JSON.parse(
  fs.readFileSync(packageFile, 'utf8')
)

const present = obsolete.filter(
  (name) =>
    pkg.dependencies?.[name] ||
    pkg.devDependencies?.[name]
)

if (!present.length) {
  console.log(
    'Known obsolete VitePress plugin dependencies are already gone.'
  )
  process.exit(0)
}

console.log(
  'Removing known obsolete packages:\n' +
  present.map((name) => `  - ${name}`).join('\n')
)

console.log(
  '\nThe `mermaid` package itself is kept ' +
  'because V2 lazy rendering still uses it.\n'
)

const npm = process.platform === 'win32'
  ? 'npm.cmd'
  : 'npm'

const result = spawnSync(
  npm,
  ['uninstall', '-D', ...present],
  {
    cwd: root,
    stdio: 'inherit'
  }
)

if (result.error) throw result.error

process.exit(result.status ?? 1)

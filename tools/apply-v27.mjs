import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()

const cssFile =
  path.join(root, '.vitepress', 'theme', 'custom.css')
const packageFile =
  path.join(root, 'package.json')
const workflowFile =
  path.join(
    root,
    '.github',
    'workflows',
    'deploy.yml'
  )

const backupDir =
  path.join(root, '.v27-backup')

const startMarker =
  '/* HOME_READABILITY_V27_START */'
const endMarker =
  '/* HOME_READABILITY_V27_END */'

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

function patchCss() {
  backup(cssFile)

  let source =
    fs.readFileSync(cssFile, 'utf8')

  // Idempotent: replace an older V2.7 block instead of stacking overrides.
  const escapedStart =
    startMarker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const escapedEnd =
    endMarker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  const oldBlock =
    new RegExp(
      `${escapedStart}[\\s\\S]*?${escapedEnd}`,
      'g'
    )

  source =
    source.replace(oldBlock, '').trimEnd()

  const patch = `

${startMarker}

/* =========================================================
   V2.7 Homepage readability
   Preserve the V2 layout/design; improve only type scale and contrast.
   ========================================================= */

/* Hero + built-in VitePress feature cards */
.VPHero .tagline {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 76%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 1.03rem;
  line-height: 1.72;
}

.VPFeature .title {
  color: var(--vp-c-text-1) !important;
  font-size: 1.08rem;
}

.VPFeature .details {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 74%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 0.96rem;
  line-height: 1.68;
}

.VPFeature .link-text {
  font-size: 0.90rem;
  font-weight: 650;
}

/* System-map section:
   V2 currently uses many 0.58–0.72rem labels; raise them without
   changing layout geometry. */
.integration-section {
  --v27-readable-muted:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 72%,
      var(--vp-c-text-2)
    );
  --v27-readable-meta:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 58%,
      var(--vp-c-text-2)
    );
}

.integration-section .section-kicker {
  font-size: 0.82rem !important;
}

.integration-section .heading-meta {
  color: var(--v27-readable-meta) !important;
  font-size: 0.73rem !important;
}

.integration-section .section-heading p {
  color: var(--v27-readable-muted) !important;
}

.integration-section .stack-index {
  font-size: 0.80rem !important;
}

.integration-section .stack-eyebrow {
  color: var(--v27-readable-meta) !important;
  font-size: 0.69rem !important;
  font-weight: 600;
}

.integration-section .stack-card p {
  color: var(--v27-readable-muted) !important;
  font-size: 0.94rem !important;
  line-height: 1.70 !important;
}

.integration-section .stack-tags span,
.integration-section .project-tags span,
.integration-section .output-chip {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 66%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 0.72rem !important;
}

.integration-section .output-status,
.integration-section .panel-kicker {
  font-size: 0.70rem !important;
}

.integration-section .stack-output p {
  color: var(--v27-readable-muted) !important;
  font-size: 0.91rem !important;
  line-height: 1.66 !important;
}

.integration-section .panel-count {
  color: var(--v27-readable-meta) !important;
  font-size: 0.68rem !important;
}

.integration-section .pipeline-index {
  font-size: 0.78rem !important;
}

.integration-section .pipeline-title-row h4 {
  font-size: 1.02rem !important;
}

.integration-section .pipeline-title-row span {
  color: var(--v27-readable-meta) !important;
  font-size: 0.69rem !important;
}

.integration-section .pipeline-copy p,
.integration-section .project-card p {
  color: var(--v27-readable-muted) !important;
  font-size: 0.90rem !important;
  line-height: 1.66 !important;
}

.integration-section .project-topline {
  color: var(--v27-readable-meta) !important;
  font-size: 0.67rem !important;
}

.integration-section .footer-label > span {
  font-size: 0.69rem !important;
}

.integration-section .footer-label small {
  color: var(--v27-readable-meta) !important;
  font-size: 0.83rem !important;
}

.integration-section .integration-footer a {
  color: var(--v27-readable-muted) !important;
  font-size: 0.85rem !important;
}

/* Slightly increase card/background separation in dark mode.
   This changes contrast, not layout or component design. */
.dark .integration-section .stack-card,
.dark .integration-section .pipeline-panel,
.dark .integration-section .projects-panel,
.dark .integration-section .project-card {
  border-color:
    color-mix(
      in srgb,
      var(--vp-c-brand-1, #5b6cff) 20%,
      var(--vp-c-divider)
    ) !important;
}

.dark .integration-section .stack-card,
.dark .integration-section .pipeline-panel,
.dark .integration-section .projects-panel {
  background:
    color-mix(
      in srgb,
      var(--vp-c-bg-soft) 92%,
      var(--vp-c-bg)
    ) !important;
}

/* Recent updates */
.recent-updates .updates-kicker {
  font-size: 0.75rem !important;
}

.recent-updates .updates-heading p {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 68%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 0.92rem !important;
  line-height: 1.70 !important;
}

.recent-updates .row-number {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 50%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 0.75rem !important;
}

.recent-updates .update-row time {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 56%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 0.79rem !important;
}

.recent-updates .update-row h3 {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 86%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 1.03rem !important;
  font-weight: 590 !important;
}

.recent-updates .row-action {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 52%,
      var(--vp-c-text-2)
    ) !important;
  font-size: 1.08rem !important;
}

@media (max-width: 720px) {
  .VPHero .tagline {
    font-size: 0.98rem;
  }

  .integration-section .stack-card p,
  .integration-section .stack-output p,
  .integration-section .pipeline-copy p,
  .integration-section .project-card p {
    font-size: 0.88rem !important;
  }

  .recent-updates .update-row h3 {
    font-size: 1rem !important;
  }
}

${endMarker}
`

  fs.writeFileSync(
    cssFile,
    source + patch + '\n',
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

  pkg.scripts['media:verify:dist'] =
    'node tools/verify-media-dist.mjs'

  // Keep V2.6's automatic media preparation, then verify that the generated
  // files survived the VitePress public->dist copy.
  pkg.scripts['docs:build'] =
    'npm run media:prepare && vitepress build && npm run media:verify:dist'

  fs.writeFileSync(
    packageFile,
    JSON.stringify(pkg, null, 2) + '\n',
    'utf8'
  )
}

function patchWorkflow() {
  if (!fs.existsSync(workflowFile)) return

  backup(workflowFile)

  let source =
    fs.readFileSync(
      workflowFile,
      'utf8'
    )

  if (
    !source.includes(
      'Media output is verified by docs:build'
    )
  ) {
    const needle =
      '      - name: Build with VitePress\n'

    if (source.includes(needle)) {
      source =
        source.replace(
          needle,
          '      # Media output is verified by docs:build before the Pages artifact is uploaded.\n' +
          needle
        )
    }
  }

  fs.writeFileSync(
    workflowFile,
    source,
    'utf8'
  )
}

patchCss()
patchPackage()
patchWorkflow()

console.log('V2.7 production/readability layer applied.')
console.log('V2 layout and component structure are unchanged.')
console.log('Backups: .v27-backup/')

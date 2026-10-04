export default {
  // Original files remain the source of truth for Obsidian.
  // Generated website derivatives go here and are gitignored.
  generatedPublicDir: 'docs/public/_media',
  manifestFile: '.vitepress/cache/media-manifest.json',

  // Only images actually referenced by Markdown are optimized.
  minBytes: 80 * 1024,
  responsiveWidths: [640, 960, 1280, 1600, 1920],
  maxWidth: 1920,

  // High enough for screenshots / diagrams while still cutting transfer size.
  webpQuality: 86,
  animatedWebpQuality: 80,
  animatedMaxWidth: 1280,

  // Do not use an optimized derivative unless it is meaningfully smaller.
  maxOutputRatio: 0.94,

  // GIF is inefficient enough that it is always considered for conversion.
  alwaysOptimizeGif: true,

  // External images are never mirrored automatically unless the domain is
  // explicitly allowlisted here. Your old Netlify domain is safe to migrate.
  mirrorDomains: [
    'obsidiannote.netlify.app'
  ],

  mirroredAssetDir: 'docs/assets/mirrored',

  // First article image gets high priority only when it appears near the top.
  eagerImageMaxLine: 28
}

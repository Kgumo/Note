import { defineConfig } from 'vitepress'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import fs from 'fs'
import { createRequire } from 'module'

import { enhanceMarkdownImages } from './utils/performance-images.mjs'
import { installMediaRenderer } from './utils/media-renderer.mjs'
import { installObsidianCompat } from './utils/obsidian-compat.mjs'
const require = createRequire(import.meta.url)
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_PATH = __dirname

function normalizeBase(value = '/') {
  let base = value.trim() || '/'
  if (!base.startsWith('/')) base = `/${base}`
  if (!base.endsWith('/')) base = `${base}/`
  return base.replace(/\/{2,}/g, '/')
}

// gukmo.cn uses a custom domain, so production defaults to '/'.
// Use VITEPRESS_BASE=/Note/ only when previewing under the GitHub project path.
const siteBase = normalizeBase(process.env.VITEPRESS_BASE || '/')

let set_sidebar
try {
  const utilsPath = path.resolve(ROOT_PATH, 'utils/auto_sidebar.cjs')

  if (!fs.existsSync(utilsPath)) {
    throw new Error(`文件不存在: ${utilsPath}`)
  }

  const sidebarModule = require(utilsPath)
  set_sidebar = sidebarModule.set_sidebar
} catch (error) {
  console.error('无法导入侧边栏模块:', error)
  set_sidebar = () => []
}

const configPath = path.resolve(__dirname, './utils/sidebar-config.json')
const cppSidebar = set_sidebar('C++', configPath)
const aiSidebar = set_sidebar('AI', configPath)
const PostgraduateSidebar = set_sidebar('Postgraduate', configPath)
const InternshipSidebar = set_sidebar('Internship', configPath)


// 综合技术目录：顶层阶段默认展开，子目录仍可折叠。
const buildSidebar = set_sidebar('build', configPath).map((item) =>
  item.items
    ? { ...item, collapsed: false }
    : item
)
export default defineConfig({
  title: '额滴笔记',
  description: '个人技术知识库 - C++ | Qt | AI',
  base: siteBase,

    head: [
      ['link', { rel: 'icon', href: `${siteBase}head.svg` }],
      ['link', {
        rel: 'preload',
        as: 'image',
        href: `${siteBase}logo.svg`,
        type: 'image/svg+xml'
      }]
    ],

  cleanUrls: true,
  lastUpdated: true,
  appearance: 'dark',

  themeConfig: {
    outlineTitle: '📚 本文目录',
    outline: [2, 6],
    smoothScroll: true,

    logo: '/whead.png',

    nav: [
      {
        text: '🏠 首页',
        link: '/',
        activeMatch: '^/$'
      },
      {
        text: '🌍 认知边界',
        link: '/我们只是通过无数的思维模型在给世界建模',
        activeMatch: '/我们只是通过无数的思维模型在给世界建模'
      },
      {
        text: '🧪 实验室',
        items: [
          { text: 'C++/Qt', link: '/C++/' },
          { text: 'AI研究', link: '/AI/' },
          { text: '综合技术', link: '/build/' }
        ]
      },
      {
        text: '🚤 跨越两岸',
        items: [
          { text: '考研', link: '/Postgraduate/' },
          { text: '实习', link: '/Internship/' }
        ],
        className: 'nav-right'
      },
      {
        text: '🔗 资源',
        link: '/resources',
        activeMatch: '/resources',
        className: 'nav-right'
      }
    ],

    sidebar: {
      '/C++/': cppSidebar,
      '/AI/': aiSidebar,
      '/build/': buildSidebar,
      '/Postgraduate/': PostgraduateSidebar,
      '/Internship/': InternshipSidebar
    },

    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/Kgumo'
      }
    ],

    footer: {
      message: '知识如风，常伴吾身',
      copyright: `Copyright © 2023-${new Date().getFullYear()} Kgumo`
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '🔍 搜索笔记...'
          }
        }
      }
    },

    editLink: {
      pattern: 'https://github.com/Kgumo/Note/edit/master/docs/:path',
      text: '✏️ 编辑此页'
    }
  },

  markdown: {
    lineNumbers: true,

    languageAlias: {
      qmake: 'makefile'
    },
    config: async (md) => {
      // V29_OBSIDIAN_COMPAT
      installObsidianCompat(md, {
        docsDir: path.resolve(__dirname, '../docs')
      })

      const { default: katex } = await import('markdown-it-katex')
      md.use(katex)

      md.core.ruler.push('clean-attributes', (state) => {
        state.tokens.forEach((token) => {
          if (token.attrs) {
            token.attrs = token.attrs.filter(([name]) => {
              return typeof name === 'string' && !/^\d+$/.test(name)
            })
          }
        })
      })

      // Mermaid is emitted as lightweight HTML during build.
      // The browser imports mermaid only when a diagram approaches the viewport.
      const defaultFence = md.renderer.rules.fence

      md.renderer.rules.fence = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const language = token.info.trim().split(/\s+/)[0]

        if (language === 'mermaid') {
          const source = md.utils.escapeHtml(token.content)

          return [
            '<div class="mermaid-diagram" data-mermaid-state="idle" role="img" aria-label="Mermaid diagram">',
            `<template class="mermaid-diagram__source">${source}</template>`,
            '<div class="mermaid-diagram__placeholder">Diagram</div>',
            '</div>'
          ].join('')
        }

        if (defaultFence) {
          return defaultFence(tokens, idx, options, env, self)
        }

        return self.renderToken(tokens, idx, options)
      }

      const defaultImageRule =
        md.renderer.rules.image ||
        ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options))

      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const srcIndex = token.attrIndex('src')

        if (srcIndex >= 0) {
          const src = token.attrs[srcIndex][1]

          if (src) {
            // Obsidian-friendly:
            // ![](images/xxx.png)
            // ![](./images/xxx.png)
            if (src.startsWith('images/') || src.startsWith('./images/')) {
              const relative = src
                .replace(/^\.\//, '')
                .replace(/^\/+/, '')

              token.attrs[srcIndex][1] = `${siteBase}${relative}`
            }
            // Legacy:
            // ![](public/images/xxx.png)
            else if (src.startsWith('public/') || src.startsWith('./public/')) {
              const relative = src
                .replace(/^\.\//, '')
                .replace(/^public\//, '')
                .replace(/^\/+/, '')

              token.attrs[srcIndex][1] = `${siteBase}${relative}`
            }
          }
        }

        token.attrSet('loading', 'lazy')
        token.attrSet('decoding', 'async')

        if (token.attrs) {
          token.attrs = token.attrs.filter((attr) =>
            Array.isArray(attr) &&
            attr.length === 2 &&
            typeof attr[0] === 'string'
          )
        }

        return defaultImageRule(tokens, idx, options, env, self)
      }

      // V25_IMAGE_PERFORMANCE
      enhanceMarkdownImages(md, {
        docsDir: path.resolve(__dirname, '../docs')
      })
    

      // V26_MEDIA_RENDERER
      // Installed last so it wraps the existing V2/V2.5 image renderer
      // instead of depending on its exact source layout.
      installMediaRenderer(md, {
        rootDir: path.resolve(__dirname, '..'),
        docsDir: path.resolve(__dirname, '../docs'),
        manifestFile: path.resolve(__dirname, './cache/media-manifest.json'),
        siteBase,
        eagerImageMaxLine: 28
      })
}
  },

  vite: {
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes('node_modules')) return

            const normalized = id.replace(/\\/g, '/')

            if (/\/node_modules\/(?:d3|d3-[^/]+)\//.test(normalized)) {
              return 'vendor-d3'
            }

            if (normalized.includes('/node_modules/element-plus/')) {
              return 'vendor-element-plus'
            }

            if (normalized.includes('/node_modules/katex/')) {
              return 'vendor-katex'
            }

            if (normalized.includes('/node_modules/langium/')) {
              return 'vendor-langium'
            }
          }
        }
      }
    },

    resolve: {
      alias: {
        'langium/lib/utils/cancellation': 'cancellation-shim',
        'langium-ast': 'langium/lib/ast',
        '@': path.resolve(__dirname, './'),
        '~': path.resolve(__dirname, '../../'),
        '@theme': path.resolve(__dirname, './theme')
      }
    },

    server: {
      fs: {
        allow: [
          path.resolve(__dirname, '../../'),
          __dirname
        ],
        deny: ['node_modules', '.git']
      }
    },

    // Do not eagerly prebundle heavy libraries just for opening the dev homepage.
    // D3 / Mermaid will be optimized only when their pages/features are actually used.
    optimizeDeps: {
      include: ['markdown-it'],
      exclude: ['vitepress'],
      esbuildOptions: {
        target: 'esnext'
      }
    }
  },

  tempDir: './.vitepress/.temp',
  srcDir: './docs',
  outDir: './dist'
})

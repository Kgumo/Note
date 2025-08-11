import { defineConfig } from 'vitepress';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'fs';
import { withMermaid } from 'vitepress-plugin-mermaid';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_PATH = __dirname;
const PROJECT_ROOT = path.resolve(__dirname, '../..');
// 动态导入侧边栏模块
let set_sidebar;
try {
  const utilsPath = path.resolve(ROOT_PATH, 'utils/auto_sidebar.cjs');
  
  if (!fs.existsSync(utilsPath)) {
    throw new Error(`文件不存在: ${utilsPath}`);
  }
  
  // 使用 require 替代 import
  const sidebarModule = require(utilsPath);
  set_sidebar = sidebarModule.set_sidebar;
} catch (error) {
  console.error('无法导入侧边栏模块:', error);
  set_sidebar = () => [];
}

// 动态生成侧边栏
const configPath = path.resolve(__dirname, './utils/sidebar-config.json');
const cppSidebar = set_sidebar("C++", configPath);
const aiSidebar = set_sidebar("AI", configPath);
const PostgraduateSidebar = set_sidebar("Postgraduate", configPath);
const InternshipSidebar = set_sidebar("Internship", configPath);

export default withMermaid(defineConfig({
  title: "额滴笔记",
  description: "个人技术知识库 - C++ | Qt | AI",
  base: "/Note/",
  assetsDir: 'assets',
  // 新增的head配置
  head: [
    ["link", { rel: "icon", href: "Note/head.svg" }],
    ["link", { 
      rel: "stylesheet", 
      href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap" 
    }],
    ["link", { 
    rel: "stylesheet", 
    href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap" 
  }],
  ],
  
  // 新增的cleanUrls配置
  cleanUrls: true,
  lastUpdated: true,
  appearance: 'dark',
  
  themeConfig: {
    // 保留所有自定义配置
    outlineTitle: "📚 本文目录",
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
          { text: "C++/Qt", link: '/C++/' },
          { text: "AI研究", link: '/AI/' }
        ]
      },
      {
        text: '🚤 跨越两岸',
        items: [
          { text: "考研", link: '/Postgraduate/' },
          { text: "实习", link: '/Internship/' }
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
      "/C++/": cppSidebar,
      "/AI/": aiSidebar,
      "/Postgraduate/": PostgraduateSidebar,
      "/Internship/": InternshipSidebar
    },
    
    // 新增的社交链接配置
    socialLinks: [
      { 
        icon: 'github',
        link: 'https://github.com/Kgumo' 
      },
    ],
    
    // 保留其他配置
    footer: {
      message: "知识如风，常伴吾身",
      copyright: `Copyright © 2023-${new Date().getFullYear()} Kgumo`
    },
    
    // 新增的搜索配置
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
    
    // 新增的编辑链接配置
    editLink: {
      pattern: 'https://github.com/Kgumo/Note/edit/main/docs/:path',
      text: '✏️ 编辑此页'
    }
  },
  
  markdown: {
    lineNumbers: true,
    config: async (md) => {
      const { default: katex } = await import('markdown-it-katex');
      md.use(katex);
    }
  },
  
  mermaid: {
    theme: 'dark',
    securityLevel: 'loose',
    fontFamily: "'Noto Serif SC', sans-serif",
    fontSize: 16,
    htmlLabels: true,
    flowchart: {
      nodeSpacing: 50,
      rankSpacing: 50
    }
  },
  
  vite: {
    build: {
      rollupOptions: {
        external: ['mermaid', 'vitepress-plugin-mermaid', /^mermaid/]
      }
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './'),
        '~': path.resolve(__dirname, '../../')
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
    optimizeDeps: {
      include: ['mermaid', 'vitepress-plugin-mermaid'],
      esbuildOptions: {
        target: 'esnext'
      }
    },
    ssr: {
      noExternal: ['mermaid']
    }
  },
  
  // 保留原始目录配置
  tempDir: './.vitepress/.temp',
  srcDir: "./docs",
  outDir: "./dist"
}));
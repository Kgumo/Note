import { defineConfig } from 'vitepress';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'fs';
import { withMermaid } from 'vitepress-plugin-mermaid'; // 导入官方插件
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

// 使用 withMermaid 包装配置
export default withMermaid(defineConfig({
  title: "额滴笔记",
  description: "个人技术知识库 - C++ | Qt | AI",
  base: process.env.NODE_ENV === 'production' ? '/Note/' : '/',
  assetsDir: 'assets',
  
  head: [
    // 使用 base 变量构建路径
    ["link", { 
      rel: "icon", 
      href: process.env.NODE_ENV === 'production' 
        ? '/Note/head.svg' 
        : '/head.svg' 
    }],
    
    // 添加 CSP 修复重定向问题
    ['meta', { 
      'http-equiv': 'Content-Security-Policy',
      content: 'upgrade-insecure-requests' 
    }]
  ],
  
  cleanUrls: true,
  lastUpdated: true,
  appearance: 'dark',
  
  // Mermaid 配置
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
  
  themeConfig: {
    outlineTitle: "📚 本文目录",
    outline: [2, 6],
    smoothScroll: true,
    
    logo: process.env.NODE_ENV === 'production' 
      ? '/Note/whead.png' 
      : '/whead.png',
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
    
    socialLinks: [
      { 
        icon: 'github',
        link: 'https://github.com/Kgumo' 
      },
    ],
    
    footer: {
      message: "知识如风，常伴吾身",
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
      pattern: 'https://github.com/Kgumo/Note/edit/main/docs/:path',
      text: '✏️ 编辑此页'
    }
  },
  
  markdown: {
    lineNumbers: true,
    config: async (md) => {
      const { default: katex } = await import('markdown-it-katex');
      md.use(katex);
      
      // 移除自定义的 Mermaid 渲染规则 - 插件会自动处理
    }
  },
  
  vite: {
    build: {
      rollupOptions: {}
    },
    resolve: {
      alias: {
        'langium/lib/utils/cancellation': 'cancellation-shim',
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
      },
      headers: {
        'Content-Type': 'application/javascript'
      }
    },
    optimizeDeps: {
      include: [
      // 确保 langium 相关依赖被正确优化
      'langium',
      'langium-ast'
    ],
      esbuildOptions: {
        target: 'esnext'
      }
    }
  },
  
  tempDir: './.vitepress/.temp',
  srcDir: "./docs",
  outDir: "./dist"
}));
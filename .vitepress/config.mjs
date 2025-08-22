import { defineConfig } from 'vitepress';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'fs';
import { withMermaid } from 'vitepress-plugin-mermaid'; // 导入官方插件
import { createRequire } from 'module';
import pkgConfig from 'vite-plugin-package-config';
import optimizePersist from 'vite-plugin-optimize-persist';
const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_PATH = __dirname;

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
  base: process.env.NODE_ENV === 'production' ? '/' : '/Note/',
  assetsDir: 'assets',
  
  head: [
    ["link", { rel: "icon", href: "/head.svg" }],
    ["link", { 
      rel: "stylesheet", 
      href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap" 
    }],
    // 移除手动添加的 CDN 脚本 - 插件会自动处理
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
      
      // 添加一个处理器来清理无效属性
      md.core.ruler.push('clean-attributes', (state) => {
        state.tokens.forEach((token) => {
          if (token.attrs) {
            // 过滤掉无效的属性名（如纯数字）
            token.attrs = token.attrs.filter(([name]) => {
              return !/^\d+$/.test(name) && typeof name === 'string';
            });
          }
        });
      });
      
      const defaultImageRule = md.renderer.rules.image;
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx];
        const srcIndex = token.attrIndex('src');
        
        if (srcIndex >= 0) {
          let src = token.attrs[srcIndex][1];
          
          // 处理相对路径的图片
          if (src && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('data:')) {
            // 移除可能的 public/ 前缀
            if (src.startsWith('public/')) {
              src = src.substring(7);
            }
            
            // 添加前导斜杠
            if (!src.startsWith('/')) {
              src = '/' + src;
            }
            
            // 确保只修改 src 属性，不修改其他属性
            token.attrs[srcIndex][1] = src;
          }
        }
        
        // 确保所有属性都是有效的键值对
        const validAttrs = token.attrs.filter(attr => 
          Array.isArray(attr) && attr.length === 2 && typeof attr[0] === 'string'
        );
        
        token.attrs = validAttrs;
        
        return defaultImageRule(tokens, idx, options, env, self);
      };
    }
  },
  
  vite: {
    plugins: [
      pkgConfig.default(),
      optimizePersist.default()
    ],
    build: {
      rollupOptions: {}
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
    optimizeDeps: {
      include: [
        // 确保 langium 相关依赖被正确优化
        'langium',
        
        // 修正依赖名称
        'markdown-it',
        'element-plus',
        '@vueuse/core',
        'd3',
        
        // 移除本地组件路径（Vite 无法优化本地 Vue 文件）
        // 改为优化其依赖的第三方库
      ],
      exclude: [
        'vitepress-plugin-mermaid',
        'vitepress' // 避免标记为 external
      ],
      esbuildOptions: {
        target: 'esnext',
        // 添加以下配置解决入口点问题
      }
    }
  },
  
  tempDir: './.vitepress/.temp',
  srcDir: "./docs",
  outDir: "./dist"
}));
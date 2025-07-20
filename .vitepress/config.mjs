import { defineConfig } from 'vitepress';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import markdownItKatex from 'markdown-it-katex';
import markdownItMermaid from '@liradb2000/markdown-it-mermaid';

// 获取当前文件路径
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 动态导入侧边栏模块
let set_sidebar;
try {
  const utilsPath = path.resolve(__dirname, './utils/auto_sidebar.mjs').replace(/\\/g, '/');
  console.log(`导入侧边栏模块: ${utilsPath}`);
  
  const utilsUrl = new URL(`file:///${utilsPath}`).href;
  const sidebarModule = await import(utilsUrl);
  set_sidebar = sidebarModule.set_sidebar;
} catch (error) {
  console.error('无法导入侧边栏模块:', error);
  set_sidebar = () => {
    console.warn('使用空侧边栏函数');
    return [];
  };
}

// 动态生成侧边栏
const cppSidebar = set_sidebar("C++");
const aiSidebar = set_sidebar("AI");

// 打印调试信息
console.log('C++ 侧边栏:', cppSidebar);
console.log('AI 侧边栏:', aiSidebar);

export default defineConfig({
  base: "/Note/",
  head: [
    ["link", { rel: "icon", href: "/Note/head.svg" }],
    ["link", { 
      rel: "stylesheet", 
      href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap" 
    }],
    ["link", {
      rel: "stylesheet",
      href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
      crossorigin: "anonymous"
    }],
    ["script", { 
      src: "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js" 
    }],
    ["script", {}, `
      document.addEventListener('DOMContentLoaded', function() {
        if (window.mermaid) {
          mermaid.initialize({
            startOnLoad: true,
            theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
            fontFamily: '"Noto Serif SC", serif',
            flowchart: { 
              useMaxWidth: true,
              htmlLabels: true,
              curve: 'basis'
            },
            securityLevel: 'loose'
          });
          mermaid.run();
        }
      });
    `]
  ],

  title: "额滴笔记",
  description: "个人技术知识库 - C++ | Qt | AI",
  
  appearance: 'dark',
  lastUpdated: true,
  
  themeConfig: {
    outlineTitle: "📚 本文目录",
    outline: [2, 6],
    
    logo: '/whead.png',
    nav: [
      { 
        text: '🏠 首页', 
        link: '/',
        activeMatch: '^/$'
      },
      { 
        text: '🧪 实验室', 
        items: [
          { text: "C++/Qt", link: '/C++/' },
          { text: "AI研究", link: '/AI/' }
        ]
      },
      { 
        text: '🔗 资源', 
        link: '/resources',
        activeMatch: '/resources'
      }
    ],

    sidebar: {
      "/C++/": cppSidebar,
      "/AI/": aiSidebar
    },

    footer: {
      message: "知识如风，常伴吾身",
      copyright: `Copyright © 2023-${new Date().getFullYear()} Kgumo`
    },

    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "🔍 搜索笔记...",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            displayDetails: true,
            resetButtonTitle: "清除搜索",
            backButtonTitle: "关闭搜索",
            noResultsText: "未找到相关结果",
            footer: {
              selectText: "选择",
              navigateText: "切换",
              closeText: "关闭"
            }
          }
        }
      }
    },

    editLink: {
      pattern: 'https://github.com/Kgumo/Note/edit/main/docs/:path',
      text: '✏️ 编辑此页'
    },

    socialLinks: [
      { 
        icon: {
          svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
        }, 
        link: 'https://github.com/Kgumo' 
      },
      {
        icon: 'discord',
        link: 'https://discord.gg/your-invite'
      }
    ],

    lastUpdated: {
      text: '📅 最后更新',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },
    
    darkModeSwitchLabel: '🌓 切换主题',
    returnToTopLabel: '👆 返回顶部',
    sidebarMenuLabel: '📖 目录',
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    }
  },
  
  markdown: {
  lineNumbers: true,
  config: (md) => {
    // 修复插件使用方式
    md.use(markdownItKatex.default || markdownItKatex);
    
    // 确保使用正确的导出方式
    const mermaidPlugin = markdownItMermaid.default || markdownItMermaid;
    md.use(mermaidPlugin, {
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }
    });
    
    // 自定义锚点渲染
    md.renderer.rules.heading_open = (tokens, idx, options, env, self) => {
      const token = tokens[idx];
      const level = token.tag.slice(1);
      const nextToken = tokens[idx + 1];
      const children = nextToken.content;
      const id = encodeURIComponent(children.toLowerCase().replace(/\s+/g, '-'));
      
      return `<${token.tag} id="${id}" class="heading-anchor">`;
    };
  }
},
  
  vite: {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './'),
      }
    },
    server: {
      fs: {
        allow: ['..']
      }
    }
  }
});
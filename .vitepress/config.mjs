import { defineConfig } from 'vitepress'
import { set_sidebar } from "./utils/auto_sidebar.mjs";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/Note/",
  head: [["link", { rel: "icon", href: "/Note/2.svg" }]],
  title: "额滴笔记",
  description: "A VitePress Site",
  themeConfig: {
    outlineTile: "目录",
    outline: [2,6],
    logo: '/1.png',
    nav: [
      { 
        text: 'Home', 
        items: [
          { text: "首页", link: '/' },
          { text: "笔记", link: '/markdown-examples' }
        ] 
      },
      { text: 'C++', link: '/C++/' },
      { text: '人工智能', link: '/AI/' }
    ],

    // 配置自动生成的侧边栏
    sidebar: {
      // 匹配所有以 /C++/ 开头的路径
      "/C++/": set_sidebar("/C++"),
      
      // 匹配所有以 /Artificial_lintelligence/ 开头的路径
      "/AI/": set_sidebar("/AI")
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Kgumo' }
    ],
    footer: {
      copyright: "Copyright © 2023-present My Awesome Project"
    },
       // 设置搜索框的样式
    search: {
      provider: "local",
      options: {
        translations: {
          button: {
            buttonText: "搜索文档",
            buttonAriaLabel: "搜索文档",
          },
          modal: {
            noResultsText: "无法找到相关结果",
            resetButtonTitle: "清除查询条件",
            footer: {
              selectText: "选择",
              navigateText: "切换",
            },
          },
        },
      },
    },
  }
})
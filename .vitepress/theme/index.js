import DefaultTheme from 'vitepress/theme'
import 'vitepress/dist/client/theme-default/styles/vars.css'
import 'vitepress/dist/client/theme-default/styles/base.css'
import { h } from 'vue'

export default {
  ...DefaultTheme,
  
  Layout: () => {
    // 简化布局处理
    return h(DefaultTheme.Layout)
  },
  
  enhanceApp({ app }) {
    // 可以注册全局组件
  }
}
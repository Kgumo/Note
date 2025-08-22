import DefaultTheme from 'vitepress/theme';
import RecentPosts from './components/RecentPosts.vue';
import CustomLayout from './Layout.vue';
import CustomNav from './components/CustomNav.vue';
import 'vite/modulepreload-polyfill';
import './custom.css';
import { h, defineAsyncComponent } from 'vue';

// 自定义组件
const HeroStats = {
  props: ['stats'],
  render() {
    return h('div', { class: 'hero-stats' }, 
      this.stats.map(stat => 
        h('div', { class: 'stat-item' }, [
          h('div', { class: 'stat-value' }, stat.value),
          h('div', { class: 'stat-label' }, stat.label)
        ])
      )
    );
  }
};

const FeatureProgress = {
  props: ['value', 'label'],
  render() {
    return h('div', { class: 'progress-container' }, [
      h('div', { 
        class: 'progress-bar',
        style: { width: `${this.value}%` }
      }),
      h('small', `${this.label}: ${this.value}%`)
    ]);
  }
};

const FeatureTags = {
  props: ['tags'],
  render() {
    return h('div', { class: 'feature-tags' }, 
      this.tags.map(tag => 
        h('span', { class: 'feature-tag' }, tag)
      )
    );
  }
};

// 全局路径修复函数
const fixBasePathGlobally = () => {
  if (import.meta.env.PROD && typeof window !== 'undefined') {
    // 修复所有资源路径
    document.querySelectorAll('link[href], script[src], img[src]').forEach(el => {
      const attr = el.href ? 'href' : 'src';
      const value = el[attr];
      
      if (value && value.includes('/Note/')) {
        el[attr] = value.replace('/Note/', '/');
      }
    });
    
    // 修复内联样式中的路径
    document.querySelectorAll('style').forEach(style => {
      style.textContent = style.textContent.replace(
        /url\(['"]?\/Note\//g, 
        'url(/'
      );
    });
  }
};

// 全局错误处理 - 处理无效属性错误
const setupGlobalErrorHandling = () => {
  if (typeof window === 'undefined') return;
  
  // 重写 setAttribute 方法以捕获错误
  const originalSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function(name, value) {
    try {
      return originalSetAttribute.call(this, name, value);
    } catch (e) {
      // 忽略无效属性名错误
      if (e.message && e.message.includes('is not a valid attribute name')) {
        console.warn('Ignored invalid attribute:', name);
        return;
      }
      throw e;
    }
  };
  
  // 添加全局错误事件监听器
  window.addEventListener('error', (e) => {
    // 忽略属性错误
    if (e.error && e.error.message && e.error.message.includes('is not a valid attribute name')) {
      e.preventDefault();
      console.warn('Ignored invalid attribute error:', e.error.message);
    }
  });
  
  // 添加未处理的 Promise 拒绝监听器
  window.addEventListener('unhandledrejection', (e) => {
    // 忽略属性相关的 Promise 拒绝
    if (e.reason && e.reason.message && e.reason.message.includes('is not a valid attribute name')) {
      e.preventDefault();
      console.warn('Ignored promise rejection due to attribute error:', e.reason.message);
    }
  });
};

// 初始化全局错误处理
if (typeof window !== 'undefined') {
  setupGlobalErrorHandling();
}

export default {
  extends: DefaultTheme,
  Layout: CustomLayout,
  enhanceApp({ app, router }) {
    // 生产环境路径修复
    if (import.meta.env.PROD) {
      // 路由变化后修复
      router.onAfterRouteChanged = () => {
        setTimeout(fixBasePathGlobally, 100);
      };
    }
    
    // 添加 Vue 错误处理
    app.config.errorHandler = (err, instance, info) => {
      // 忽略属性错误
      if (err.message && err.message.includes('is not a valid attribute name')) {
        console.warn('Ignored Vue attribute error:', err.message);
        return;
      }
      console.error('Vue error:', err);
    };
    
    // 注册组件
    app.component('KnowledgeGraph', defineAsyncComponent(() => 
      import('./components/KnowledgeGraph.vue')
    ));
    
    app.component('RecentPosts', RecentPosts);
    app.component('CustomNav', CustomNav);
    app.component('HeroStats', HeroStats);
    app.component('FeatureProgress', FeatureProgress);
    app.component('FeatureTags', FeatureTags);
  }
};
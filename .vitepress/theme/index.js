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



export default {
  extends: DefaultTheme,
  Layout: CustomLayout,
  enhanceApp({ app, router }) {

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
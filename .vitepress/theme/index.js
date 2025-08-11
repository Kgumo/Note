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

export default {
  extends: DefaultTheme,
  Layout: CustomLayout,
  enhanceApp({ app }) {
    // 修改为使用 defineAsyncComponent
    app.component('KnowledgeGraph', defineAsyncComponent(() => 
      import('./components/KnowledgeGraph.vue')
    ));
    
    // 注册其他组件 (保持不变)
    app.component('RecentPosts', RecentPosts);
    app.component('CustomNav', CustomNav);
    app.component('HeroStats', HeroStats);
    app.component('FeatureProgress', FeatureProgress);
    app.component('FeatureTags', FeatureTags);
  }
};
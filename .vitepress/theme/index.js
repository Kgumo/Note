import DefaultTheme from 'vitepress/theme';
import RecentPosts from './components/RecentPosts.vue';
import CustomLayout from './Layout.vue';
import CustomNav from './components/CustomNav.vue';
import { defineClientComponent } from 'vitepress';
import './custom.css';
import { h } from 'vue';

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
    // 使用 defineClientComponent 注册知识图谱组件
    app.component('KnowledgeGraph', defineClientComponent(() => 
      import('./components/KnowledgeGraph.vue')
    ));
    
    // 注册其他组件
    app.component('RecentPosts', RecentPosts);
    app.component('CustomNav', CustomNav);
    
    // 注册自定义渲染组件
    app.component('HeroStats', HeroStats);
    app.component('FeatureProgress', FeatureProgress);
    app.component('FeatureTags', FeatureTags);
  }
};
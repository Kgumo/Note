import DefaultTheme from 'vitepress/theme';
import CustomLayout from './Layout.vue';
import './custom.css';
import { h, defineAsyncComponent } from 'vue';

const KnowledgeGraph = defineAsyncComponent(() => import('./components/KnowledgeGraph.vue'));
const RecentPosts = defineAsyncComponent(() => import('./components/RecentPosts.vue'));
const CustomNav = defineAsyncComponent(() => import('./components/CustomNav.vue'));
const IntegrationSection = defineAsyncComponent(() => import('./components/IntegrationSection.vue'));

const HeroStats = {
  props: ['stats'],
  render() {
    return h('div', { class: 'hero-stats' },
      this.stats.map((stat) =>
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
      this.tags.map((tag) => h('span', { class: 'feature-tag' }, tag))
    );
  }
};

export default {
  extends: DefaultTheme,
  Layout: CustomLayout,
  enhanceApp({ app }) {
    app.component('KnowledgeGraph', KnowledgeGraph);
    app.component('RecentPosts', RecentPosts);
    app.component('CustomNav', CustomNav);
    app.component('HeroStats', HeroStats);
    app.component('FeatureProgress', FeatureProgress);
    app.component('FeatureTags', FeatureTags);
    app.component('IntegrationSection', IntegrationSection);
  }
};

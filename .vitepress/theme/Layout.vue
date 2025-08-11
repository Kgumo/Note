<!-- .vitepress/theme/Layout.vue -->
<template>
  <VPContent>
    <template #layout-top>
      <button class="sidebar-toggle" @click="toggleSidebar" :aria-label="isSidebarOpen ? '隐藏侧边栏' : '显示侧边栏'">
        <svg v-if="isSidebarOpen" class="toggle-icon" viewBox="0 0 24 24">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
        <svg v-else class="toggle-icon" viewBox="0 0 24 24">
          <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
        </svg>
      </button>
    </template>
    <template #layout-bottom>
      <div class="vp-bg-elements">
        <div class="vp-floating-element vp-element-1"></div>
        <div class="vp-floating-element vp-element-2"></div>
        <div class="vp-floating-element vp-element-3"></div>
      </div>
      <div class="vp-grid-overlay"></div>
      <div v-if="isSidebarOpen" class="vp-sidebar-mask" @click="toggleSidebar"></div>
    </template>
  </VPContent>
</template>

<script>
import { ref, onMounted } from 'vue';
import DefaultTheme from 'vitepress/theme';
import Layout from 'vitepress/dist/client/theme-default/Layout.vue'
import Mermaid from './mermaid.vue'

export default {
  name: 'CustomLayout',
  
  setup() {
    const isSidebarOpen = ref(false);
    
    const toggleSidebar = () => {
      isSidebarOpen.value = !isSidebarOpen.value;
      updateBodyClass();
    };
    
    const updateBodyClass = () => {
      if (typeof document === 'undefined') return;
      document.body.classList.toggle('sidebar-open', isSidebarOpen.value);
      document.body.classList.toggle('sidebar-closed', !isSidebarOpen.value);
    };
    
    onMounted(() => {
      isSidebarOpen.value = false;
      updateBodyClass();
    });
    
    return {
      isSidebarOpen,
      toggleSidebar
    };
  },
  
  components: {
    VPContent: DefaultTheme.Layout,
    Layout,
    Mermaid
  }
};
</script>

<style scoped>
.sidebar-toggle {
  position: fixed;
  top: 20px;
  left: 24px;
  z-index: 1001;
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--vp-c-bg);
  border-radius: 50%;
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  cursor: pointer;
  transition: all 0.3s ease;
}

.sidebar-toggle:hover {
  background: var(--vp-c-bg-soft);
  transform: scale(1.08);
}

.toggle-icon {
  width: 24px;
  height: 24px;
  fill: var(--vp-c-text-1);
}

.vp-sidebar-mask {
  display: none;
}

@media (max-width: 959px) {
  .sidebar-toggle {
    top: 16px;
    left: 16px;
    width: 40px;
    height: 40px;
  }
  
  .toggle-icon {
    width: 22px;
    height: 22px;
  }
  
  .vp-sidebar-mask {
    display: block;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
    backdrop-filter: blur(4px);
  }
}
</style>
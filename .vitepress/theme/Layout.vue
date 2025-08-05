<script setup>
import { onMounted, ref } from 'vue'
import DefaultTheme from 'vitepress/theme'

const { Layout: DefaultLayout } = DefaultTheme

// 侧边栏状态
const isSidebarOpen = ref(true)

// 切换侧边栏状态
const toggleSidebar = () => {
  isSidebarOpen.value = !isSidebarOpen.value
  localStorage.setItem('sidebar-state', isSidebarOpen.value ? 'open' : 'closed')
  updateBodyClass()
}

// 更新 body 类
const updateBodyClass = () => {
  if (isSidebarOpen.value) {
    document.body.classList.remove('sidebar-closed')
    document.body.classList.add('sidebar-open')
  } else {
    document.body.classList.remove('sidebar-open')
    document.body.classList.add('sidebar-closed')
  }
}

// 检查是否是移动端
const isMobile = () => {
  return window.innerWidth <= 959
}

onMounted(() => {
  // 检查是否是移动端
  const mobile = isMobile()
  
  // 移动端默认关闭侧边栏
  if (mobile) {
    isSidebarOpen.value = false
    localStorage.setItem('sidebar-state', 'closed')
  } else {
    // 从 localStorage 读取状态
    const savedState = localStorage.getItem('sidebar-state')
    if (savedState === 'closed') {
      isSidebarOpen.value = false
    }
  }
  
  // 确保body元素存在
  if (document.body) {
    updateBodyClass()
  } else {
    // 如果body未加载完成，延迟执行
    setTimeout(updateBodyClass, 100)
  }
})
</script>

<template>
  <DefaultLayout>
    <template #layout-top>
      <!-- 侧边栏切换按钮 -->
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
      <!-- 背景装饰元素 -->
      <div class="vp-bg-elements">
        <div class="vp-floating-element vp-element-1"></div>
        <div class="vp-floating-element vp-element-2"></div>
        <div class="vp-floating-element vp-element-3"></div>
      </div>
      
      <!-- 网格覆盖层 -->
      <div class="vp-grid-overlay"></div>
      
      <!-- 添加遮罩层 -->
      <div v-if="isSidebarOpen" class="vp-sidebar-mask" @click="toggleSidebar"></div>
    </template>
  </DefaultLayout>
</template>

<style scoped>
.sidebar-toggle {
  position: fixed;
  top: 24px;
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

/* 遮罩层样式 */
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
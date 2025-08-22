<script setup>
import { useData, useRoute } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { ref, onMounted, watch } from 'vue'

const { Layout } = DefaultTheme
const { theme } = useData()
const route = useRoute()
const instance = getCurrentInstance()
// 修复属性处理 - 更严格的过滤
// 修复属性处理
const cleanProps = (props) => {
  if (instance && instance.appContext.config.globalProperties.$validateAttributes) {
    return instance.appContext.config.globalProperties.$validateAttributes(props, 'CustomNav');
  }
  
  // 备用方案
  const clean = {}
  for (const key in props) {
    if (Object.prototype.hasOwnProperty.call(props, key)) {
      // 确保属性名是有效的，且不是数字
      if (typeof key === 'string' && key.match(/^[a-zA-Z_$][a-zA-Z0-9_$]*$/) && !/^\d+$/.test(key)) {
        clean[key] = props[key]
      } else {
        console.warn(`Invalid attribute name: ${key}`)
      }
    }
  }
  return clean
}


// 响应式导航状态
const isMobileMenuOpen = ref(false)
const currentPath = ref('/')

// 更新当前路径
// 添加路由监听器，在路由变化时重置状态
watch(() => route.path, (newPath) => {
  currentPath.value = newPath
  isMobileMenuOpen.value = false
}, { immediate: true })

// 处理移动端菜单切换
const toggleMobileMenu = () => {
  isMobileMenuOpen.value = !isMobileMenuOpen.value
}

// 检查是否为活动链接
const isActiveLink = (link) => {
  return currentPath.value === link || currentPath.value.startsWith(link + '/')
}
</script>

<template>
  <Layout>
    <template #nav-bar-title-before>
      <!-- 左侧导航项 -->
      <div class="nav-left">
        <template v-for="item in theme.nav" :key="item.text">
          <div v-if="!item.className || item.className !== 'nav-right'" class="nav-item">
            <a
              v-if="!item.items"
              class="nav-link"
              :href="item.link"
              :class="{ active: isActiveLink(item.link) }"
              v-bind="cleanProps({})"
            >
              {{ item.text }}
            </a>
            
            <div v-else class="nav-dropdown">
              <button class="nav-dropdown-toggle" v-bind="cleanProps({})">
                {{ item.text }}
                <span class="dropdown-arrow">▼</span>
              </button>
              <div class="nav-dropdown-content">
                <a
                  v-for="subItem in item.items"
                  :key="subItem.text"
                  :href="subItem.link"
                  class="nav-dropdown-item"
                  :class="{ active: isActiveLink(subItem.link) }"
                  v-bind="cleanProps({})"
                >
                  {{ subItem.text }}
                </a>
              </div>
            </div>
          </div>
        </template>
      </div>
    </template>
    
    <template #nav-bar-title-after>
      <!-- 右侧导航项 - 使用flex容器包裹 -->
      <div class="nav-right-container">
        <div class="nav-right">
          <template v-for="item in theme.nav" :key="item.text">
            <div v-if="item.className === 'nav-right'" class="nav-item">
              <a
                v-if="!item.items"
                class="nav-link"
                :href="item.link"
                :class="{ active: isActiveLink(item.link) }"
                v-bind="cleanProps({})"
              >
                {{ item.text }}
              </a>
              
              <div v-else class="nav-dropdown">
                <button class="nav-dropdown-toggle" v-bind="cleanProps({})">
                  {{ item.text }}
                  <span class="dropdown-arrow">▼</span>
                </button>
                <div class="nav-dropdown-content">
                  <a
                    v-for="subItem in item.items"
                    :key="subItem.text"
                    :href="subItem.link"
                    class="nav-dropdown-item"
                    :class="{ active: isActiveLink(subItem.link) }"
                    v-bind="cleanProps({})"
                  >
                    {{ subItem.text }}
                  </a>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>
    </template>
    
    <template #nav-bar-content-after>
      <!-- 移动端菜单按钮 -->
      <div class="menu-container">
        <button class="menu-toggle" @click="toggleMobileMenu" v-bind="cleanProps({})">☰</button>
        
        <!-- 移动端菜单 -->
        <div v-if="isMobileMenuOpen" class="mobile-menu">
          <div class="mobile-menu-content">
            <template v-for="item in theme.nav" :key="item.text">
              <div class="mobile-nav-item">
                <a
                  v-if="!item.items"
                  class="mobile-nav-link"
                  :href="item.link"
                  :class="{ active: isActiveLink(item.link) }"
                  @click="toggleMobileMenu"
                  v-bind="cleanProps({})"
                >
                  {{ item.text }}
                </a>
                
                <div v-else class="mobile-nav-dropdown">
                  <div class="mobile-nav-dropdown-header">
                    {{ item.text }}
                  </div>
                  <div class="mobile-nav-dropdown-content">
                    <a
                      v-for="subItem in item.items"
                      :key="subItem.text"
                      :href="subItem.link"
                      class="mobile-nav-dropdown-item"
                      :class="{ active: isActiveLink(subItem.link) }"
                      @click="toggleMobileMenu"
                      v-bind="cleanProps({})"
                    >
                      {{ subItem.text }}
                    </a>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </template>
  </Layout>
</template>

<style scoped>
/* 导航项基本样式 */
.nav-item {
  position: relative;
  height: 100%;
  display: flex;
  align-items: center;
}

.nav-link {
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 1rem;
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: color 0.2s ease;
  font-weight: 500;
}

.nav-link:hover {
  color: var(--vp-c-brand);
}

.nav-link.active {
  color: var(--vp-c-brand);
  border-bottom: 2px solid var(--vp-c-brand);
}

/* 下拉菜单样式 */
.nav-dropdown {
  position: relative;
  height: 100%;
  display: flex;
  align-items: center;
}

.nav-dropdown-toggle {
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 1rem;
  background: none;
  border: none;
  color: var(--vp-c-text-1);
  cursor: pointer;
  font-weight: 500;
  font-size: inherit;
  transition: color 0.2s ease;
}

.nav-dropdown-toggle:hover {
  color: var(--vp-c-brand);
}

.dropdown-arrow {
  margin-left: 0.5rem;
  font-size: 0.7em;
}

.nav-dropdown-content {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 180px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: all 0.3s ease;
}

.nav-dropdown:hover .nav-dropdown-content {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.nav-dropdown-item {
  display: block;
  padding: 0.75rem 1.5rem;
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: background 0.2s ease;
}

.nav-dropdown-item:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand);
}

/* 右侧导航容器 */
.nav-right-container {
  display: flex;
  justify-content: flex-end;
  flex-grow: 1;
}

.nav-right {
  display: flex;
  gap: 1rem;
  height: 100%;
}

/* 移动端菜单 */
.menu-container {
  display: none;
  position: relative;
}

.menu-toggle {
  background: none;
  border: none;
  color: var(--vp-c-text-1);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0.5rem;
}

.mobile-menu {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  z-index: 1000;
  display: flex;
  justify-content: flex-end;
}

.mobile-menu-content {
  width: 70%;
  height: 100%;
  background: var(--vp-c-bg);
  padding: 2rem 1rem;
  overflow-y: auto;
}

.mobile-nav-item {
  margin-bottom: 1rem;
}

.mobile-nav-link {
  display: block;
  padding: 0.75rem 1rem;
  color: var(--vp-c-text-1);
  text-decoration: none;
  border-radius: 4px;
  transition: background 0.2s ease;
}

.mobile-nav-link:hover,
.mobile-nav-link.active {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand);
}

.mobile-nav-dropdown-header {
  padding: 0.75rem 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.mobile-nav-dropdown-content {
  padding-left: 1rem;
}

.mobile-nav-dropdown-item {
  display: block;
  padding: 0.5rem 1rem;
  color: var(--vp-c-text-1);
  text-decoration: none;
  border-radius: 4px;
  transition: background 0.2s ease;
}

.mobile-nav-dropdown-item:hover,
.mobile-nav-dropdown-item.active {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-brand);
}

@media (max-width: 768px) {
  .nav-left,
  .nav-right-container {
    display: none;
  }
  
  .menu-container {
    display: block;
  }
}
</style>
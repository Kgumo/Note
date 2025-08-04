<script setup>
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

const { Layout } = DefaultTheme
const { theme } = useData()
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
              :class="{ active: $route.path === item.link }"
            >
              {{ item.text }}
            </a>
            
            <div v-else class="nav-dropdown">
              <button class="nav-dropdown-toggle">
                {{ item.text }}
                <span class="dropdown-arrow">▼</span>
              </button>
              <div class="nav-dropdown-content">
                <a
                  v-for="subItem in item.items"
                  :key="subItem.text"
                  :href="subItem.link"
                  class="nav-dropdown-item"
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
                :class="{ active: $route.path === item.link }"
              >
                {{ item.text }}
              </a>
              
              <div v-else class="nav-dropdown">
                <button class="nav-dropdown-toggle">
                  {{ item.text }}
                  <span class="dropdown-arrow">▼</span>
                </button>
                <div class="nav-dropdown-content">
                  <a
                    v-for="subItem in item.items"
                    :key="subItem.text"
                    :href="subItem.link"
                    class="nav-dropdown-item"
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
        <button class="menu-toggle">☰</button>
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

/* 移动端菜单按钮 */
.menu-container {
  display: none;
}

.menu-toggle {
  background: none;
  border: none;
  color: var(--vp-c-text-1);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0.5rem;
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
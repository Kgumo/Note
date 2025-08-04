<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  posts: {
    type: Array,
    required: true,
    default: () => []
  },
  loadingDelay: {
    type: Number,
    default: 800
  }
})

const isLoading = ref(true)
const randomAngles = props.posts.map(() => 
  Math.random() * 10 - 5 // -5° 到 +5° 的随机倾斜
)

onMounted(() => {
  setTimeout(() => {
    isLoading.value = false
  }, props.loadingDelay)
})

// 修改后的链接处理函数
function handleLinkClick(event, path) {
  event.preventDefault();
  
  // 移除本地路径前缀
  const cleanPath = path.replace('D:/0.Project/Note', '');
  
  // 确保路径格式正确
  let targetPath = cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
  targetPath = targetPath.replace(/\.md$/, '');
  
  // 添加 base 路径
  const base = '/Note/'
  if (!targetPath.startsWith(base)) {
    targetPath = `${base}${targetPath.replace(/^\//, '')}`
  }
  
  // 使用原生导航
  window.location.href = targetPath;
}
</script>

<template>
  <section class="recent-updates-3d">
    <h2 class="section-title">
      <span class="icon">🌀</span>
      <span class="text">动态更新</span>
      <span class="divider"></span>
    </h2>

    <!-- 骨架屏 -->
    <div v-if="isLoading" class="skeleton-grid">
      <div v-for="i in 6" :key="i" class="skeleton-card">
        <div class="skeleton-line lg"></div>
        <div class="skeleton-line md"></div>
        <div class="skeleton-line sm"></div>
      </div>
    </div>

    <!-- 3D卡片内容 -->
    <div v-else class="posts-3d-grid">
      <article 
        v-for="(post, index) in posts" 
        :key="post.link"
        :style="{
          '--rotate-angle': `${randomAngles[index]}deg`,
          '--hue-rotate': `${index * 12}deg`
        }"
        class="post-3d-card"
      >
        <div class="card-inner">
          <div class="card-front">
            <time class="post-date">{{ post.date }}</time>
            <h3 class="post-title">{{ post.title }}</h3>
            <div class="post-badge">New</div>
          </div>
          <div class="card-back">
            <!-- 修改为绝对定位的链接 -->
            <a 
              :href="post.link" 
              class="card-link"
              @click="handleLinkClick($event, post.link)"
            >
              <div class="link-content">
                <svg class="link-icon" viewBox="0 0 24 24">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
                </svg>
                <span>立即阅读</span>
              </div>
            </a>
          </div>
        </div>
        <div class="card-glare"></div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.recent-updates-3d {
  max-width: 1300px;
  margin: 6rem auto;
  padding: 0 2rem;
  perspective: 2000px;
}

.section-title {
  display: flex;
  align-items: center;
  font-size: 2rem;
  margin-bottom: 4rem;
  color: var(--vp-c-text-1);
  position: relative;
}

.section-title .icon {
  margin-right: 15px;
  font-size: 1.8em;
  animation: spin 8s linear infinite;
}

.section-title .divider {
  flex-grow: 1;
  height: 2px;
  margin-left: 25px;
  background: linear-gradient(
    90deg,
    var(--vp-c-brand),
    transparent 80%
  );
}

/* 骨架屏样式 */
.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 30px;
}

.skeleton-card {
  height: 220px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 1.5rem;
  overflow: hidden;
  position: relative;
}

.skeleton-line {
  height: 16px;
  background: var(--vp-c-bg-soft-down);
  border-radius: 4px;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}

.skeleton-line::after {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(var(--vp-c-brand-rgb), 0.1),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

.skeleton-line.lg { width: 80%; }
.skeleton-line.md { width: 60%; }
.skeleton-line.sm { width: 40%; }

/* 3D卡片容器 */
.posts-3d-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 30px;
}

.post-3d-card {
  height: 240px;
  transform-style: preserve-3d;
  transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
  position: relative;
  transform: rotateY(var(--rotate-angle));
}

.post-3d-card:hover {
  transform: rotateY(0) translateY(-10px) scale(1.05);
  filter: hue-rotate(var(--hue-rotate)) brightness(1.1);
  z-index: 10;
}

.card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transform-style: preserve-3d;
  transition: transform 0.8s;
  border-radius: 16px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
}

.post-3d-card:hover .card-inner {
  transform: rotateY(180deg);
}

.card-front,
.card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
  border-radius: inherit;
  display: flex;
  flex-direction: column;
}

.card-front {
  background: linear-gradient(
    135deg,
    var(--vp-c-bg-soft-up),
    var(--vp-c-bg-soft)
  );
  justify-content: space-between;
  position: relative;
  padding: 2rem; /* 保持内边距 */
}

/* 添加卡片内部渐变效果 */
.card-front::after {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    135deg,
    rgba(255, 255, 255, 0.1),
    rgba(0, 0, 0, 0.1)
  );
  border-radius: inherit;
  z-index: 1;
}

/* 修复：卡片背面改为绝对定位 */
.card-back {
   background: linear-gradient(
    135deg,
    var(--vp-c-brand),
    var(--vp-c-brand-dark)
  );
  transform: rotateY(180deg);
  color: white;
  
  /* 修复定位 */
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex; /* 添加flex布局 */
  align-items: center; /* 垂直居中 */
  justify-content: center; /* 水平居中 */
  z-index: 2; /* 确保在炫光层之上 */
}

/* 确保链接覆盖整个卡片背面 */
.card-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: inherit;
  text-decoration: none;
  z-index: 3; /* 高于炫光层 */
  position: relative; /* 添加相对定位 */
}

/* 添加链接内容容器 */
.link-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  font-size: 1.2rem;
  font-weight: 500;
}

.post-date {
  font-size: 0.95rem;
  color: var(--vp-c-brand);
  font-weight: 600;
  letter-spacing: 0.5px;
  position: relative;
  z-index: 2;
}

.post-title {
  font-size: 1.4rem;
  line-height: 1.4;
  margin: 1rem 0;
  color: var(--vp-c-text-1);
  position: relative;
  z-index: 2;
}

.post-badge {
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  background: var(--vp-c-brand);
  color: white;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: bold;
  animation: pulse 2s infinite;
  z-index: 2;
}

.link-icon {
  width: 48px;
  height: 48px;
  stroke: white;
  stroke-width: 1.5;
  stroke-linecap: round;
  stroke-linejoin: round;
  fill: none;
  margin-bottom: 1rem;
}

.card-glare {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: inherit;
  background: radial-gradient(
    circle at 70% 30%,
    rgba(255, 255, 255, 0.2),
    transparent 50%
  );
  opacity: 0;
  transition: opacity 0.3s;
  z-index: 1; /* 低于卡片内容 */
  pointer-events: none; /* 关键：禁止炫光层接收鼠标事件 */
}

.post-3d-card:hover .card-glare {
  opacity: 1;
}

/* 动画定义 */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

@keyframes shimmer {
  100% {
    left: 100%;
  }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .posts-3d-grid,
  .skeleton-grid {
    grid-template-columns: 1fr;
  }
  
  .post-3d-card {
    height: 200px;
  }
  
  .section-title {
    font-size: 1.6rem;
  }
  
  /* 移动端链接文字大小调整 */
  .link-content {
    font-size: 1rem;
    padding: 1rem;
  }
  
  .link-icon {
    width: 36px;
    height: 36px;
  }
  
  /* 修复移动端卡片背面显示 */
  .card-back {
    padding: 1rem;
  }
}
</style>
<script setup>
import { withBase } from 'vitepress'

const props = defineProps({
  posts: {
    type: Array,
    required: true,
    default: () => []
  }
})

function normalizePath(filePath) {
  if (!filePath) return withBase('/')

  let cleanPath = filePath
    .replace(/^.*[\\/]Note[\\/]/, '')
    .replace(/\\/g, '/')
    .replace(/\.md$/i, '')

  if (!cleanPath.startsWith('/')) cleanPath = '/' + cleanPath
  if (cleanPath.endsWith('/index')) cleanPath = cleanPath.slice(0, -6)
  if (!cleanPath) cleanPath = '/'

  return withBase(cleanPath)
}

function cardAngle(index) {
  return ((index * 7) % 11) - 5
}
</script>

<template>
  <section class="recent-updates-3d">
    <h2 class="section-title">
      <span class="icon">🌀</span>
      <span class="text">动态更新</span>
      <span class="divider"></span>
    </h2>

    <div class="posts-3d-grid">
      <article
        v-for="(post, index) in props.posts"
        :key="post.link"
        :style="{
          '--rotate-angle': `${cardAngle(index)}deg`,
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
            <a :href="normalizePath(post.link)" class="card-link">
              <div class="link-content">
                <svg class="link-icon" viewBox="0 0 24 24" aria-hidden="true">
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
  background: linear-gradient(90deg, var(--vp-c-brand), transparent 80%);
}

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
  background: linear-gradient(135deg, var(--vp-c-bg-soft-up), var(--vp-c-bg-soft));
  justify-content: space-between;
  position: relative;
  padding: 2rem;
}

.card-front::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(0, 0, 0, 0.1));
  border-radius: inherit;
  z-index: 1;
}

.card-back {
  background: linear-gradient(135deg, var(--vp-c-brand), var(--vp-c-brand-dark));
  transform: rotateY(180deg);
  color: white;
  position: absolute;
  inset: 0;
  align-items: center;
  justify-content: center;
  z-index: 2;
}

.card-link {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: inherit;
  text-decoration: none;
  z-index: 3;
  position: relative;
}

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
  inset: 0;
  border-radius: inherit;
  background: radial-gradient(circle at 70% 30%, rgba(255, 255, 255, 0.2), transparent 50%);
  opacity: 0;
  transition: opacity 0.3s;
  z-index: 1;
  pointer-events: none;
}

.post-3d-card:hover .card-glare {
  opacity: 1;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

@media (max-width: 768px) {
  .posts-3d-grid { grid-template-columns: 1fr; }
  .post-3d-card { height: 200px; }
  .section-title { font-size: 1.6rem; }
  .link-content { font-size: 1rem; padding: 1rem; }
  .link-icon { width: 36px; height: 36px; }
  .card-back { padding: 1rem; }
}
</style>

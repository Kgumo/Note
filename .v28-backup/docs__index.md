---
layout: home

hero:
  name: "额滴笔记"
  text: "从“Hello World”到此刻与未来"
  tagline: "吾魂兮无求乎永生 竭尽兮人事之所能"
  image:
    src: /logo.svg
    alt: 知识图谱
  actions:
    - theme: brand
      text: 📝 C++与Qt开发体系概览
      link: /C++/
    - theme: alt
      text: ⚙️ 人工智能（AI）及其核心技术体系
      link: /AI/

features:
  - icon: 🧠
    title: "咕泡教育"
    details: "系统学习C++及Qt开发技术"
    link: /C++/
    linkText: "开始学习 →"
  - icon: 🚀
    title: "驱风计划"
    details: "深入探索人工智能核心领域"
    link: /AI/
    linkText: "探索技术 →"
  - icon: 🧩
    title: "知识图谱(已更新机器学习)"
    details: "构建结构化知识体系网络"
    link: /knowledge-graph/
    linkText: "查看图谱 →"
---

<IntegrationSection />

<style>
.custom-hero-title {
  background: linear-gradient(120deg, 
    var(--vp-c-brand) 0%, 
    var(--vp-c-brand-light) 30%,
    var(--vp-c-brand) 70%,
    var(--vp-c-brand-darker) 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 3.5rem !important;
  font-weight: 900;
  letter-spacing: -0.5px;
  text-shadow: 0 2px 10px rgba(100, 108, 255, 0.2);
  animation: name-gradient 8s ease infinite;
  background-size: 300% 300%;
  display: block !important;
  margin-bottom: 1rem;
}

@keyframes name-gradient {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* ========== 优化后的光晕效果 ========== */
.VPHero .image-container {
  position: relative;
  display: inline-block;
}

.VPHero .image {
  position: relative;
  z-index: 2;
  transition: transform 0.5s ease;
}

.VPHero .image:hover {
  transform: scale(1.03);
}

/* SVG 线条颜色修改 */
.VPHero .image img {
  /* 将黑色线条改为品牌色 */
  filter: 
    brightness(0) 
    invert(0.15) 
    sepia(1) 
    saturate(3000%) 
    hue-rotate(240deg) 
    brightness(0.9) 
    contrast(1.1);
  transition: filter 0.5s ease;
}

.VPHero .image:hover img {
  /* 悬停时增加饱和度 */
  filter: 
    brightness(0) 
    invert(0.1) 
    sepia(1) 
    saturate(4000%) 
    hue-rotate(240deg) 
    brightness(1) 
    contrast(1.2);
}

/* 更小更精致的光晕效果 */
.VPHero .image-container::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 110%; /* 进一步缩小光晕尺寸 */
  height: 110%; /* 进一步缩小光晕尺寸 */
  background: radial-gradient(
    circle at center,
    rgba(255, 255, 255, 0.5) 0%, /* 降低不透明度 */
    rgba(255, 255, 255, 0.3) 30%, /* 更快过渡到透明 */
    rgba(255, 255, 255, 0.1) 50%,
    transparent 70% /* 更早过渡到完全透明 */
  );
  border-radius: 50%;
  z-index: 1;
  animation: subtle-glow 4s ease-in-out infinite;
  opacity: 0.5; /* 降低基础不透明度 */
}

/* 更微妙的动画 */
@keyframes subtle-glow {
  0% {
    opacity: 0.4;
    transform: translate(-50%, -50%) scale(1);
  }
  50% {
    opacity: 0.6; /* 更小的不透明度变化 */
    transform: translate(-50%, -50%) scale(1.02); /* 更小的缩放变化 */
  }
  100% {
    opacity: 0.4;
    transform: translate(-50%, -50%) scale(1);
  }
}

/* 添加微弱的品牌色光晕 */
.VPHero .image-container::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 105%;
  height: 105%;
  background: radial-gradient(
    circle at center,
    rgba(79, 70, 229, 0.15) 0%,
    rgba(79, 70, 229, 0.05) 40%,
    transparent 70%
  );
  border-radius: 50%;
  z-index: 1;
  opacity: 0.3;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .VPHero .image-container::before {
    width: 105%;
    height: 105%;
  }
  
  .VPHero .image-container::after {
    width: 100%;
    height: 100%;
  }
  
  .VPHero .image img {
    filter: 
      brightness(0) 
      invert(0.15) 
      sepia(1) 
      saturate(2500%) 
      hue-rotate(240deg) 
      brightness(0.95) 
      contrast(1.05);
  }
}
</style>

<!-- 自定义标题 -->
<div class="custom-hero-title"></div>

<script setup>
// V25_HOME_RUNTIME_CLEANUP
import { onBeforeUnmount, onMounted } from 'vue'

let intervalId = null
let timeoutId = null
const createdParticles = []

onMounted(() => {
  const taglines = [
    "代码是写给人看的，只是顺便让机器能运行",
    "Stay hungry, stay foolish",
    "求知若饥，虚心若愚",
    "技术是解决问题的艺术",
    "吾魂兮无求乎永生 竭尽兮人事之所能"
  ]

  let current = 0
  const el = document.querySelector('.VPHero .tagline')

  const changeTagline = () => {
    if (!el) return

    current = (current + 1) % taglines.length
    el.style.opacity = 0

    if (timeoutId) {
      window.clearTimeout(timeoutId)
    }

    timeoutId = window.setTimeout(() => {
      if (!el.isConnected) return
      el.textContent = taglines[current]
      el.style.opacity = 1
    }, 500)
  }

  changeTagline()
  intervalId = window.setInterval(changeTagline, 5000)

  const features = document.querySelectorAll('.VPFeature')

  features.forEach((feature) => {
    if (feature.querySelector(':scope > .particles')) return

    const particlesContainer = document.createElement('div')
    particlesContainer.className = 'particles'
    feature.appendChild(particlesContainer)
    createdParticles.push(particlesContainer)

    for (let i = 0; i < 15; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'

      const size = Math.random() * 10 + 5
      particle.style.width = `${size}px`
      particle.style.height = `${size}px`
      particle.style.left = `${Math.random() * 100}%`
      particle.style.top = `${Math.random() * 100}%`

      const hue = 240 + Math.random() * 60
      particle.style.background =
        `hsla(${hue}, 80%, 70%, ${0.2 + Math.random() * 0.3})`

      particle.style.animationDelay =
        `${Math.random() * 5}s`

      particle.style.animationDuration =
        `${10 + Math.random() * 20}s`

      particlesContainer.appendChild(particle)
    }
  })
})

onBeforeUnmount(() => {
  if (intervalId) {
    window.clearInterval(intervalId)
    intervalId = null
  }

  if (timeoutId) {
    window.clearTimeout(timeoutId)
    timeoutId = null
  }

  for (const element of createdParticles) {
    element.remove()
  }

  createdParticles.length = 0
})
</script>

<RecentPosts :posts="[
  { title: '持续更新AI（驱风计划）', date: '2025-2026', link: '/AI' },
  { title: '更新4.C++研修阶段中C++/python数据库的连接', date: '2025-12-11', link: '/C++/4.C++研修阶段/第1节数据库' },
  { title: '更新了build,开始C++ ONNX推理的学习', date: '2025-10-4', link: '/build' },
  { title: 'resources', date: '2025-10-2', link: '/resources' },
  { title: '填充知识图谱', date: '2025-8-23', link: '/knowledge-graph' },
  { title: '更新C++第三阶段实战', date: '2025-8-6', link: '/C++/3.C++进阶课程/第6节实战' }
]"/>
---
layout: home

hero:
  name: "额滴笔记"
  text: "从“Hello World”到此刻与未来"
  tagline: "吾魂兮无求乎永生 竭尽兮人事之所能"
  image:
    src: /10.png
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
    title: "知识图谱"
    details: "构建结构化知识体系网络"
    link: /knowledge-graph/
    linkText: "查看图谱 →"
---

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
</style>

<!-- 自定义标题 -->
<div class="custom-hero-title"></div>

<script setup>
import { onMounted } from 'vue'

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
    current = (current + 1) % taglines.length
    el.style.opacity = 0
    setTimeout(() => {
      el.textContent = taglines[current]
      el.style.opacity = 1
    }, 500)
  }
  
  changeTagline()
  setInterval(changeTagline, 5000)
  
  // 添加功能卡片的粒子效果
  const features = document.querySelectorAll('.VPFeature')
  
  features.forEach(feature => {
    const particlesContainer = document.createElement('div')
    particlesContainer.className = 'particles'
    feature.appendChild(particlesContainer)
    
    // 创建粒子
    for (let i = 0; i < 15; i++) {
      const particle = document.createElement('div')
      particle.className = 'particle'
      
      // 随机位置和大小
      const size = Math.random() * 10 + 5
      particle.style.width = `${size}px`
      particle.style.height = `${size}px`
      particle.style.left = `${Math.random() * 100}%`
      particle.style.top = `${Math.random() * 100}%`
      
      // 随机颜色
      const hue = 240 + Math.random() * 60
      particle.style.background = `hsla(${hue}, 80%, 70%, ${0.2 + Math.random() * 0.3})`
      
      // 随机动画延迟
      particle.style.animationDelay = `${Math.random() * 5}s`
      particle.style.animationDuration = `${10 + Math.random() * 20}s`
      
      particlesContainer.appendChild(particle)
    }
  })
})
</script>

<RecentPosts :posts="[
  { title: '集成学习：从Bagging到Boosting', date: '2025-7-31', link: 'D:/0.Project/Note/AI/机器学习/09.集成学习/index' },
  { title: '更新了🔗 资源', date: '2025-8-2', link: 'D:/0.Project/Note/resources.md' }
]"/>
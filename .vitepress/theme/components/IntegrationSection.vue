<template>
    <div class="integration-section">
        <div class="container">
            <h2 class="section-title">C++ / Qt / AI 综合技术体系</h2>
            <p class="section-description">
                探索现代C++、Qt框架与人工智能技术的深度融合，构建下一代智能跨平台应用
            </p>

            <div class="integration-grid">
                <!-- 核心融合展示 -->
                <div class="core-integration">
                    <div class="tech-circle cpp">
                        <div class="tech-icon">C++</div>
                    </div>
                    <div class="connector"></div>
                    <div class="tech-circle qt">
                        <div class="tech-icon">Qt</div>
                    </div>
                    <div class="connector"></div>
                    <div class="tech-circle ai">
                        <div class="tech-icon">AI</div>
                    </div>
                    <div class="fusion-center">
                        <div class="fusion-icon">⚡</div>
                        <div class="fusion-text">智能应用</div>
                    </div>
                </div>
            </div>

            <div class="applications-grid">
                <div class="application-card">
                    <h3>计算机视觉</h3>
                    <div class="tech-tags">
                        <span class="tag cpp-tag">OpenCV</span>
                        <span class="tag qt-tag">Qt Widgets</span>
                        <span class="tag ai-tag">CNN</span>
                    </div>
                    <p>使用Qt构建直观的视觉分析界面，C++优化图像处理性能，AI算法实现智能识别</p>
                </div>

                <div class="application-card">
                    <h3>智能工业控制</h3>
                    <div class="tech-tags">
                        <span class="tag cpp-tag">STL</span>
                        <span class="tag qt-tag">QML</span>
                        <span class="tag ai-tag">强化学习</span>
                    </div>
                    <p>Qt创建响应式控制面板，C++处理实时数据，AI优化控制策略与预测性维护</p>
                </div>

                <div class="application-card">
                    <h3>数据分析平台</h3>
                    <div class="tech-tags">
                        <span class="tag cpp-tag">高性能计算</span>
                        <span class="tag qt-tag">图表可视化</span>
                        <span class="tag ai-tag">机器学习</span>
                    </div>
                    <p>利用C++处理大规模数据，Qt实现交互式可视化，AI提供预测分析与模式识别</p>
                </div>
            </div>

            <div class="action-buttons">
                <a class="action-link" href="/Note/C++/">
                    <span class="action-icon">📚</span>
                    <span class="action-text">开始学习C++/Qt</span>
                </a>
                <a class="action-link" href="/Note/AI/">
                    <span class="action-icon">🧠</span>
                    <span class="action-text">探索AI技术</span>
                </a>
                <a class="action-link" href="/Note/build/">
                    <span class="action-icon">📚</span>
                    <span class="action-text">学习综合开发指南</span>
                </a>
                <a class="action-link" href="/Note/build/">
                    <span class="action-icon">🚀</span>
                    <span class="action-text">查看实战项目</span>
                </a>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'IntegrationSection'
}
</script>

<style scoped>
.integration-section {
    /* 去掉上方空白，减少 padding-top */
    padding: 2rem 2rem 3rem;
    /* 与其他组件保持距离 */
    margin: 1.5rem 0;
    /* 使用你现有的背景渐变变量 */
    background: linear-gradient(135deg, var(--vp-bg-gradient-start, #f9f9ff) 0%, var(--vp-bg-gradient-end, #f0f4ff) 100%);
    border-top: 1px solid var(--vp-line-color, #eaeefb);
    border-bottom: 2px solid var(--vp-line-color, #eaeefb);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
}

/* 添加微妙的背景纹理 */
.integration-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(66, 184, 131, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(79, 70, 229, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.container {
    max-width: 960px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

.section-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #42b883 0%, #347474 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    /* 减少标题上方的空间 */
    margin-top: 0;
}

.section-description {
    text-align: center;
    font-size: 1.1rem;
    color: var(--vp-c-text-2, #5d6b83);
    max-width: 700px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

.integration-grid {
    display: flex;
    justify-content: center;
    margin-bottom: 3rem;
}

.core-integration {
    display: flex;
    align-items: center;
    position: relative;
}

.tech-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    z-index: 2;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.tech-circle:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.tech-icon {
    font-weight: 700;
    font-size: 1.2rem;
    color: white;
}

.cpp {
    background: linear-gradient(135deg, #6599db 0%, #3b6cb0 100%);
}

.qt {
    background: linear-gradient(135deg, #41d1ff 0%, #2a96c7 100%);
}

.ai {
    background: linear-gradient(135deg, #ff6b6b 0%, #c74141 100%);
}

.connector {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #3b6cb0, #2a96c7, #c74141);
    position: relative;
}

.connector::after {
    content: "→";
    position: absolute;
    top: -10px;
    right: -5px;
    color: var(--vp-c-text-2, #5d6b83);
    font-size: 1.2rem;
}

.fusion-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--vp-c-bg-soft, white);
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(66, 184, 131, 0.4);
    }

    70% {
        box-shadow: 0 0 0 15px rgba(66, 184, 131, 0);
    }

    100% {
        box-shadow: 0 0 0 0 rgba(66, 184, 131, 0);
    }
}

.fusion-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.fusion-text {
    font-weight: 600;
    color: #42b883;
    font-size: 0.9rem;
}

.applications-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin-bottom: 3rem;
}

.application-card {
    background: var(--vp-c-bg-soft, white);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
    border: 1px solid var(--vp-line-color, transparent);
}

.application-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.application-card h3 {
    margin-top: 0;
    color: var(--vp-c-text-1, #2c3e50);
    font-size: 1.3rem;
}

.tech-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
}

.tag {
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.cpp-tag {
    background-color: var(--cpp-tag-bg, #ebf3ff);
    color: var(--cpp-tag-color, #3b6cb0);
}

.qt-tag {
    background-color: var(--qt-tag-bg, #e6f7ff);
    color: var(--qt-tag-color, #2a96c7);
}

.ai-tag {
    background-color: var(--ai-tag-bg, #ffebeb);
    color: var(--ai-tag-color, #c74141);
}

.application-card p {
    color: var(--vp-c-text-2, #5d6b83);
    line-height: 1.6;
    margin-bottom: 0;
}

.action-buttons {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}

.action-link {
    display: inline-flex;
    align-items: center;
    padding: 0.8rem 1.5rem;
    background: var(--vp-c-bg-soft, white);
    border-radius: 8px;
    color: #42b883;
    text-decoration: none;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(66, 184, 131, 0.15);
    transition: all 0.3s ease;
    border: 1px solid var(--vp-line-color, transparent);
}

.action-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(66, 184, 131, 0.25);
    color: #359f74;
}

.action-icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
}

/* ================== 暗色主题完全重写 ================== */
.dark .integration-section {
    /* 使用你的暗色背景渐变变量 */
    background: linear-gradient(135deg, var(--vp-bg-gradient-start) 0%, var(--vp-bg-gradient-end) 100%);
    border-color: var(--vp-line-color);
}

/* 暗色主题背景纹理 */
.dark .integration-section::before {
    background:
        radial-gradient(circle at 20% 30%, rgba(79, 70, 229, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(74, 222, 128, 0.08) 0%, transparent 50%);
}

.dark .section-title {
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.dark .section-description {
    color: #d1d5db;
}

.dark .tech-circle {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.dark .tech-circle:hover {
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
}

.dark .connector::after {
    color: #9ca3af;
}

.dark .fusion-center {
    background: #1f2937;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    border: 1px solid var(--vp-line-color);
}

.dark .fusion-text {
    color: #4ade80;
}

/* 暗色主题脉冲动画 */
.dark .fusion-center {
    animation: dark-pulse 3s infinite;
}

@keyframes dark-pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4);
    }

    70% {
        box-shadow: 0 0 0 15px rgba(79, 70, 229, 0);
    }

    100% {
        box-shadow: 0 0 0 0 rgba(79, 70, 229, 0);
    }
}

.dark .application-card {
    background: #1f2937;
    border-color: var(--vp-line-color);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
}

.dark .application-card:hover {
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
    border-color: rgba(79, 70, 229, 0.25);
}

.dark .application-card h3 {
    color: #f9fafb;
}

.dark .application-card p {
    color: #d1d5db;
}

/* 暗色主题标签 */
.dark .cpp-tag {
    background-color: rgba(59, 108, 176, 0.2);
    color: #60a5fa;
}

.dark .qt-tag {
    background-color: rgba(42, 150, 199, 0.2);
    color: #38bdf8;
}

.dark .ai-tag {
    background-color: rgba(239, 68, 68, 0.2);
    color: #f87171;
}

.dark .action-link {
    background: #1f2937;
    border-color: var(--vp-line-color);
    color: #4ade80;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

.dark .action-link:hover {
    border-color: rgba(79, 70, 229, 0.4);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3);
    color: #22c55e;
    background: #374151;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .integration-section {
        padding: 1.5rem 1rem 2rem;
        margin: 1rem 0;
    }

    .section-title {
        font-size: 1.8rem;
    }

    .core-integration {
        flex-direction: column;
    }

    .connector {
        width: 3px;
        height: 40px;
    }

    .connector::after {
        content: "↓";
        top: auto;
        bottom: -15px;
        right: 50%;
        transform: translateX(50%);
    }

    .fusion-center {
        position: relative;
        top: auto;
        left: auto;
        transform: none;
        margin: 1.5rem auto;
    }

    .action-buttons {
        flex-direction: column;
        align-items: center;
    }

    .action-link {
        width: 100%;
        justify-content: center;
        max-width: 280px;
    }
}

.section-title {
  text-align: center;
  font-size: 2.2rem;
  font-weight: 700;
  /* 增加下方距离 */
  margin-bottom: 2rem;
  background: linear-gradient(135deg, #42b883 0%, #347474 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-top: 0;
  /* 改善字体渲染 */
  letter-spacing: 0.02em;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
  /* 确保渐变文字在所有浏览器中显示正确 */
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.section-description {
  text-align: center;
  font-size: 1.1rem;
  color: var(--vp-c-text-2, #5d6b83);
  max-width: 700px;
  /* 调整上方距离，现在标题和描述之间有更大间距 */
  margin: 0 auto 2.5rem;
  line-height: 1.6;
  /* 改善字体渲染 */
  letter-spacing: 0.01em;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
}

/* 暗色主题下的标题 */
.dark .section-title {
  background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  /* 确保暗色主题下也有良好的字体渲染 */
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.dark .section-description {
  color: #d1d5db;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .section-title {
    font-size: 1.8rem;
    /* 移动端稍微减少间距 */
    margin-bottom: 1.5rem;
    /* 移动端改善可读性 */
    letter-spacing: 0.01em;
  }
  
  .section-description {
    font-size: 1rem;
    /* 移动端调整行高 */
    line-height: 1.7;
  }
}

/* 如果浏览器不支持背景剪切，提供后备方案 */
@supports not (-webkit-background-clip: text) {
  .section-title {
    background: none;
    color: #42b883;
  }
  
  .dark .section-title {
    color: #4ade80;
  }
}
</style>
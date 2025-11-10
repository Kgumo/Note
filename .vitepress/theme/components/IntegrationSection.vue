<template>
    <div class="integration-section">
        <div class="container">
            <h2 class="section-title">C++ / Qt / AI 综合技术体系</h2>
            <p class="section-description">
                从模型训练到应用部署的完整解决方案，让AI能力无缝集成到桌面应用中
            </p>

            <div class="integration-grid">
                <!-- 核心融合展示 -->
                <div class="core-integration">
                    <div class="tech-circle cpp">
                        <div class="tech-icon">C++</div>
                        <div class="particles">
                            <div class="particle" v-for="n in 8" :key="n" :style="getCircleParticleStyle(n)"></div>
                        </div>
                    </div>
                    <div class="connector"></div>
                    <div class="tech-circle qt">
                        <div class="tech-icon">Qt</div>
                        <div class="particles">
                            <div class="particle" v-for="n in 8" :key="n" :style="getCircleParticleStyle(n)"></div>
                        </div>
                    </div>
                    <div class="connector"></div>
                    <div class="tech-circle ai">
                        <div class="tech-icon">AI</div>
                        <div class="particles">
                            <div class="particle" v-for="n in 8" :key="n" :style="getCircleParticleStyle(n)"></div>
                        </div>
                    </div>
                    <div class="fusion-center">
                        <div class="fusion-icon">🚀</div>
                        <div class="fusion-text">智能应用</div>
                        <div class="particles">
                            <div class="particle" v-for="n in 12" :key="n" :style="getFusionParticleStyle(n)"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 开发流程移到卡片前面 -->
            <div class="workflow-section">
                <h3 class="workflow-title">开发流程 - 从想法到产品</h3>
                <div class="workflow-steps">
                    <div class="workflow-step">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <h4>Python原型开发</h4>
                            <p>快速验证AI模型效果</p>
                            <div class="step-tools">Jupyter · PyTorch · Scikit-learn</div>
                        </div>
                        <div class="step-particles"></div>
                    </div>
                    <div class="workflow-step">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <h4>模型转换优化</h4>
                            <p>转换为ONNX格式</p>
                            <div class="step-tools">ONNX · Model Optimizer</div>
                        </div>
                        <div class="step-particles"></div>
                    </div>
                    <div class="workflow-step">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <h4>C++集成部署</h4>
                            <p>高性能推理引擎</p>
                            <div class="step-tools">ONNX Runtime · C++17</div>
                        </div>
                        <div class="step-particles"></div>
                    </div>
                    <div class="workflow-step">
                        <div class="step-number">4</div>
                        <div class="step-content">
                            <h4>Qt界面开发</h4>
                            <p>美观的桌面应用</p>
                            <div class="step-tools">Qt Widgets · QML · Qt Charts</div>
                        </div>
                        <div class="step-particles"></div>
                    </div>
                </div>
            </div>

            <!-- 应用卡片 -->
            <div class="applications-grid">
                <!-- 图像识别桌面应用卡片 -->
                <div class="application-card">
                    <div class="card-header">
                        <div class="card-title-section">
                            <h3>图像识别桌面应用</h3>
                            <div class="card-badge">实战项目</div>
                        </div>
                    </div>
                    
                    <div class="card-content">
                        <div class="tech-section">
                            <h4 class="section-subtitle">技术架构</h4>
                            <div class="tech-stack">
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">训练框架</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">PyTorch</span>
                                            <span class="stack-tag">TensorFlow</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">使用Python进行模型训练和验证</p>
                                </div>
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">推理引擎</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">ONNX Runtime</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">C++高性能推理，支持CPU/GPU加速</p>
                                </div>
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">界面开发</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">Qt Widgets</span>
                                            <span class="stack-tag">OpenCV</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">跨平台GUI应用，支持图像实时处理</p>
                                </div>
                            </div>
                        </div>

                        <div class="features-section">
                            <h4 class="section-subtitle">核心功能</h4>
                            <div class="features-grid">
                                <div class="feature-item">
                                    <div class="feature-icon">🖼️</div>
                                    <div class="feature-content">
                                        <div class="feature-title">拖拽图片识别</div>
                                        <div class="feature-desc">支持常见图片格式，即时显示识别结果</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">📷</div>
                                    <div class="feature-content">
                                        <div class="feature-title">实时摄像头检测</div>
                                        <div class="feature-desc">USB摄像头实时画面分析与识别</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">🔒</div>
                                    <div class="feature-content">
                                        <div class="feature-title">本地化部署</div>
                                        <div class="feature-desc">数据完全本地处理，保护隐私安全</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">💻</div>
                                    <div class="feature-content">
                                        <div class="feature-title">跨平台运行</div>
                                        <div class="feature-desc">Windows、Linux、macOS全平台支持</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="use-cases-section">
                            <h4 class="section-subtitle">应用场景</h4>
                            <div class="use-cases">
                                <div class="use-case-item">
                                    <div class="use-case-icon">🏭</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">工业质检</div>
                                        <div class="use-case-desc">产品缺陷检测，质量自动分级</div>
                                    </div>
                                </div>
                                <div class="use-case-item">
                                    <div class="use-case-icon">🏥</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">医疗影像</div>
                                        <div class="use-case-desc">X光片分析，病灶区域识别</div>
                                    </div>
                                </div>
                                <div class="use-case-item">
                                    <div class="use-case-icon">🔒</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">安防监控</div>
                                        <div class="use-case-desc">人脸识别，异常行为检测</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="particles">
                        <div class="particle" v-for="n in 15" :key="n" :style="getCardParticleStyle(n, 0)"></div>
                    </div>
                </div>

                <!-- 智能工业检测系统卡片 -->
                <div class="application-card">
                    <div class="card-header">
                        <div class="card-title-section">
                            <h3>智能工业检测系统</h3>
                            <div class="card-badge">企业级</div>
                        </div>
                    </div>
                    
                    <div class="card-content">
                        <div class="tech-section">
                            <h4 class="section-subtitle">技术架构</h4>
                            <div class="tech-stack">
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">视觉算法</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">OpenCV</span>
                                            <span class="stack-tag">Halcon</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">工业级图像处理与特征提取</p>
                                </div>
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">推理引擎</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">ONNX Runtime</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">深度学习模型部署与优化</p>
                                </div>
                                <div class="stack-item">
                                    <div class="stack-header">
                                        <span class="stack-label">界面开发</span>
                                        <div class="stack-tags">
                                            <span class="stack-tag">Qt Widgets</span>
                                            <span class="stack-tag">QCustomPlot</span>
                                        </div>
                                    </div>
                                    <p class="stack-description">专业级工业控制界面</p>
                                </div>
                            </div>
                        </div>

                        <div class="features-section">
                            <h4 class="section-subtitle">核心功能</h4>
                            <div class="features-grid">
                                <div class="feature-item">
                                    <div class="feature-icon">🔍</div>
                                    <div class="feature-content">
                                        <div class="feature-title">高精度检测</div>
                                        <div class="feature-desc">亚像素级精度，微米级缺陷识别</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">⚡</div>
                                    <div class="feature-content">
                                        <div class="feature-title">高速处理</div>
                                        <div class="feature-desc">毫秒级响应，满足产线实时需求</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">📊</div>
                                    <div class="feature-content">
                                        <div class="feature-title">数据分析</div>
                                        <div class="feature-desc">生产数据统计与质量分析</div>
                                    </div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-icon">🔧</div>
                                    <div class="feature-content">
                                        <div class="feature-title">参数配置</div>
                                        <div class="feature-desc">灵活的参数调整与算法配置</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="use-cases-section">
                            <h4 class="section-subtitle">应用场景</h4>
                            <div class="use-cases">
                                <div class="use-case-item">
                                    <div class="use-case-icon">🔩</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">零件检测</div>
                                        <div class="use-case-desc">尺寸测量，外观缺陷检测</div>
                                    </div>
                                </div>
                                <div class="use-case-item">
                                    <div class="use-case-icon">📦</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">包装检测</div>
                                        <div class="use-case-desc">标签识别，包装完整性检查</div>
                                    </div>
                                </div>
                                <div class="use-case-item">
                                    <div class="use-case-icon">🔌</div>
                                    <div class="use-case-content">
                                        <div class="use-case-title">电子元件</div>
                                        <div class="use-case-desc">PCB板检测，元器件定位</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="particles">
                        <div class="particle" v-for="n in 15" :key="n" :style="getCardParticleStyle(n, 1)"></div>
                    </div>
                </div>
            </div>

            <!-- 背景浮动粒子 -->
            <div class="background-particles">
                <div class="particle" v-for="n in 25" :key="n" :style="getBackgroundParticleStyle(n)"></div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'IntegrationSection',
    methods: {
        getBackgroundParticleStyle(index) {
            const sizes = [2, 3, 4];
            const colors = ['#3b82f6', '#6366f1', '#8b5cf6'];
            
            return {
                '--tx': `${(Math.random() - 0.5) * 200}px`,
                '--ty': `${(Math.random() - 0.5) * 200}px`,
                '--duration': `${15 + Math.random() * 15}s`,
                '--delay': `${Math.random() * 5}s`,
                width: `${sizes[index % sizes.length]}px`,
                height: `${sizes[index % sizes.length]}px`,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                background: colors[index % colors.length],
                opacity: 0.6 + Math.random() * 0.4
            };
        },
        getCardParticleStyle(index, cardIndex) {
            const sizes = [2, 3];
            const colors = [
                '#3b82f6',
                '#6366f1', 
                '#8b5cf6',
                '#10b981',
                '#f59e0b'
            ];
            
            return {
                '--tx': `${(Math.random() - 0.5) * 100}px`,
                '--ty': `${(Math.random() - 0.5) * 100}px`,
                '--duration': `${8 + Math.random() * 8}s`,
                '--delay': `${Math.random() * 3}s`,
                width: `${sizes[index % sizes.length]}px`,
                height: `${sizes[index % sizes.length]}px`,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                background: colors[(index + cardIndex) % colors.length],
                opacity: 0
            };
        },
        getCircleParticleStyle(index) {
            const angles = [0, 45, 90, 135, 180, 225, 270, 315];
            return {
                '--tx': `${Math.cos(angles[index] * Math.PI / 180) * 60}px`,
                '--ty': `${Math.sin(angles[index] * Math.PI / 180) * 60}px`,
                '--duration': `${3 + index}s`,
                '--delay': `${index * 0.5}s`,
                width: '3px',
                height: '3px',
                left: '50%',
                top: '50%',
                background: 'rgba(255, 255, 255, 0.8)',
                opacity: 0
            };
        },
        getFusionParticleStyle(index) {
            const radius = 40;
            const angle = (index / 12) * 360;
            return {
                '--tx': `${Math.cos(angle * Math.PI / 180) * radius}px`,
                '--ty': `${Math.sin(angle * Math.PI / 180) * radius}px`, 
                '--duration': `${4 + (index % 3)}s`,
                '--delay': `${(index * 0.3) % 2}s`,
                width: '4px',
                height: '4px',
                left: '50%',
                top: '50%',
                background: '#3b82f6',
                opacity: 0
            };
        }
    },
    mounted() {
        // 卡片悬停时显示粒子
        const cards = document.querySelectorAll('.application-card');
        cards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                const particles = card.querySelectorAll('.particle');
                particles.forEach(particle => {
                    particle.style.opacity = '0.7';
                });
            });
            card.addEventListener('mouseleave', () => {
                const particles = card.querySelectorAll('.particle');
                particles.forEach(particle => {
                    particle.style.opacity = '0';
                });
            });
        });

        // 技术圆圈始终显示粒子
        const techCircles = document.querySelectorAll('.tech-circle, .fusion-center');
        techCircles.forEach(circle => {
            const particles = circle.querySelectorAll('.particle');
            particles.forEach(particle => {
                particle.style.opacity = '0.7';
            });
        });
    }
}
</script>

<style scoped>
/* 基础样式 */
.integration-section {
    padding: 2rem 2rem 3rem;
    margin: 1.5rem 0;
    background: #202127;
    border-top: 1px solid rgba(59, 130, 246, 0.2);
    border-bottom: 2px solid rgba(59, 130, 246, 0.2);
    border-radius: 14px;
    position: relative;
    overflow: hidden;
}

.integration-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.08) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
}

.section-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 0;
    letter-spacing: 0.02em;
    font-family: var(--vp-font-family-base);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.section-description {
    text-align: center;
    font-size: 1.1rem;
    color: #d1d5db;
    max-width: 800px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
    letter-spacing: 0.01em;
    font-family: var(--vp-font-family-base);
}

/* 技术圆圈样式 */
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
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    overflow: hidden;
}

.tech-circle:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
}

.tech-icon {
    font-weight: 700;
    font-size: 1.2rem;
    color: white;
    position: relative;
    z-index: 2;
}

.cpp {
    background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
}

.qt {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
}

.ai {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
}

.connector {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #6366f1);
    position: relative;
}

.connector::after {
    content: "→";
    position: absolute;
    top: -10px;
    right: -5px;
    color: #9ca3af;
    font-size: 1.2rem;
}

.fusion-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #2d2d35;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    animation: pulse 3s infinite;
    overflow: hidden;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4);
    }
    70% {
        box-shadow: 0 0 0 15px rgba(59, 130, 246, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
    }
}

.fusion-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 2;
}

.fusion-text {
    font-weight: 600;
    color: #3b82f6;
    font-size: 0.9rem;
    position: relative;
    z-index: 2;
}

/* 粒子系统 */
@keyframes particle-float {
  0% {
    transform: translate(0, 0) rotate(0deg);
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
  100% {
    transform: translate(var(--tx, 100px), var(--ty, -100px)) rotate(360deg);
    opacity: 0;
  }
}

.particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  pointer-events: none;
  z-index: 1;
  border-radius: inherit;
}

.particle {
  position: absolute;
  border-radius: 50%;
  animation: particle-float var(--duration, 15s) linear infinite;
  animation-delay: var(--delay, 0s);
  transition: opacity 0.3s ease;
}

.background-particles {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
}

.background-particles .particle {
  opacity: 0.3;
}

.application-card .particles .particle {
  opacity: 0;
  transition: opacity 0.5s ease;
}

.application-card:hover .particles .particle {
  opacity: 0.7;
}

.tech-circle .particles .particle,
.fusion-center .particles .particle {
  opacity: 0.7;
}

/* 工作流程部分 */
.workflow-section {
    margin-bottom: 3rem;
    padding: 2.5rem;
    background: #2d2d35;
    border-radius: 12px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(59, 130, 246, 0.2);
    position: relative;
    overflow: hidden;
}

.workflow-section::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
}

.workflow-title {
    text-align: center;
    font-size: 1.6rem;
    margin-bottom: 2.5rem;
    color: #f9fafb;
    font-weight: 700;
    position: relative;
    z-index: 2;
}

.workflow-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 2rem;
}

.workflow-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 1.5rem;
    border-radius: 10px;
    background: rgba(59, 130, 246, 0.08);
    transition: all 0.3s ease;
    border: 1px solid rgba(59, 130, 246, 0.15);
    position: relative;
    overflow: hidden;
}

.workflow-step:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
    background: rgba(59, 130, 246, 0.12);
}

.step-number {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    position: relative;
    z-index: 2;
}

.step-content h4 {
    margin: 0 0 0.8rem 0;
    color: #f9fafb;
    font-size: 1.2rem;
    font-weight: 600;
    position: relative;
    z-index: 2;
}

.step-content p {
    color: #d1d5db;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.5;
    position: relative;
    z-index: 2;
}

.step-tools {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #d1d5db;
    background: rgba(59, 130, 246, 0.12);
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    display: inline-block;
    position: relative;
    z-index: 2;
}

.step-particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
    border-radius: 10px;
}

.step-particles::before,
.step-particles::after {
    content: '';
    position: absolute;
    width: 4px;
    height: 4px;
    background: #3b82f6;
    border-radius: 50%;
    opacity: 0;
    transition: all 0.5s ease;
}

.workflow-step:hover .step-particles::before,
.workflow-step:hover .step-particles::after {
    opacity: 0.4;
    animation: step-particle-rotate 4s linear infinite;
}

.step-particles::before {
    top: 15%;
    left: 15%;
    animation-delay: 0s;
}

.step-particles::after {
    bottom: 15%;
    right: 15%;
    animation-delay: 2s;
}

@keyframes step-particle-rotate {
    0% {
        transform: rotate(0deg) translateX(15px) rotate(0deg);
    }
    100% {
        transform: rotate(360deg) translateX(15px) rotate(-360deg);
    }
}

/* 应用卡片样式 */
.applications-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
}

.application-card {
    background: #2d2d35;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    border: 1px solid rgba(59, 130, 246, 0.2);
    position: relative;
    overflow: hidden;
}

.application-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 35px rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.4);
}

.application-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.5s ease;
}

.application-card:hover::before {
    transform: scaleX(1);
}

/* 卡片头部 */
.card-header {
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 2;
}

.card-title-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.card-title-section h3 {
    margin: 0;
    color: #f9fafb;
    font-size: 1.6rem;
    font-weight: 700;
}

.card-badge {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* 卡片内容区域 */
.card-content {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

.section-subtitle {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 1rem;
    border-bottom: 2px solid rgba(59, 130, 246, 0.2);
    padding-bottom: 0.5rem;
}

/* 技术栈部分 */
.tech-stack {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.stack-item {
    background: rgba(59, 130, 246, 0.08);
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid rgba(59, 130, 246, 0.15);
}

.stack-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.stack-label {
    font-size: 0.9rem;
    color: #d1d5db;
    font-weight: 500;
}

.stack-tags {
    display: flex;
    gap: 0.5rem;
}

.stack-tag {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.stack-description {
    font-size: 0.85rem;
    color: #9ca3af;
    margin: 0;
}

/* 功能部分 */
.features-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

.feature-item {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.8rem;
    background: rgba(59, 130, 246, 0.05);
    border-radius: 8px;
    transition: all 0.3s ease;
}

.feature-item:hover {
    background: rgba(59, 130, 246, 0.1);
    transform: translateY(-2px);
}

.feature-icon {
    font-size: 1.5rem;
    margin-top: 0.2rem;
}

.feature-content {
    flex: 1;
}

.feature-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 0.2rem;
}

.feature-desc {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.4;
}

/* 应用场景部分 */
.use-cases {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

.use-case-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 1rem;
    background: rgba(59, 130, 246, 0.05);
    border-radius: 8px;
    transition: all 0.3s ease;
}

.use-case-item:hover {
    background: rgba(59, 130, 246, 0.1);
    transform: translateY(-2px);
}

.use-case-icon {
    font-size: 1.5rem;
}

.use-case-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 0.2rem;
}

.use-case-desc {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.4;
}

/* ================== 白色主题支持 ================== */
:root {
  --vp-c-bg: #ffffff;
  --vp-c-bg-soft: #f9fafb;
  --vp-c-bg-alt: #f1f5f9;
  --vp-c-text: #1f2937;
  --vp-c-text-soft: #6b7280;
  --vp-c-text-mute: #9ca3af;
  --vp-c-divider: #e5e7eb;
  --vp-c-brand: #3b82f6;
  --vp-c-brand-light: #60a5fa;
  --vp-c-brand-dark: #1d4ed8;
}

.dark {
  --vp-c-bg: #202127;
  --vp-c-bg-soft: #2d2d35;
  --vp-c-bg-alt: #374151;
  --vp-c-text: #f9fafb;
  --vp-c-text-soft: #d1d5db;
  --vp-c-text-mute: #9ca3af;
  --vp-c-divider: #4b5563;
  --vp-c-brand: #3b82f6;
  --vp-c-brand-light: #60a5fa;
  --vp-c-brand-dark: #1d4ed8;
}

/* 白色主题样式 */
:not(.dark) .integration-section {
    background: var(--vp-c-bg);
    border-color: var(--vp-c-divider);
}

:not(.dark) .integration-section::before {
    background:
        radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.05) 0%, transparent 50%);
}

:not(.dark) .section-description {
    color: var(--vp-c-text-soft);
}

:not(.dark) .application-card {
    background: var(--vp-c-bg-soft);
    border-color: var(--vp-c-divider);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
}

:not(.dark) .application-card:hover {
    box-shadow: 0 15px 35px rgba(59, 130, 246, 0.1);
    border-color: rgba(59, 130, 246, 0.3);
}

:not(.dark) .card-title-section h3,
:not(.dark) .section-subtitle,
:not(.dark) .feature-title,
:not(.dark) .use-case-title,
:not(.dark) .stack-label {
    color: var(--vp-c-text);
}

:not(.dark) .stack-description,
:not(.dark) .feature-desc,
:not(.dark) .use-case-desc,
:not(.dark) .stat-label {
    color: var(--vp-c-text-soft);
}

:not(.dark) .workflow-section {
    background: var(--vp-c-bg-soft);
    border-color: var(--vp-c-divider);
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
}

:not(.dark) .workflow-step {
    background: rgba(59, 130, 246, 0.05);
    border-color: rgba(59, 130, 246, 0.1);
}

:not(.dark) .workflow-step:hover {
    background: rgba(59, 130, 246, 0.08);
}

:not(.dark) .step-content h4 {
    color: var(--vp-c-text);
}

:not(.dark) .step-content p {
    color: var(--vp-c-text-soft);
}

/* 响应式设计 */
@media (max-width: 768px) {
    .integration-section {
        padding: 1.5rem 1rem 2rem;
        margin: 1rem 0;
    }

    .section-title {
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }

    .section-description {
        font-size: 1rem;
        line-height: 1.7;
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

    .applications-grid {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }

    .application-card {
        padding: 1.5rem;
    }

    .card-title-section {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }

    .features-grid {
        grid-template-columns: 1fr;
    }

    .use-cases {
        grid-template-columns: 1fr;
    }

    .workflow-steps {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }

    .workflow-step {
        flex-direction: row;
        text-align: left;
        padding: 1.2rem;
    }

    .step-number {
        margin-bottom: 0;
        margin-right: 1rem;
        font-size: 2.5rem;
    }

    .workflow-section {
        padding: 1.5rem;
    }
    
    /* 移动端减少粒子数量 */
    .background-particles .particle:nth-child(n+15) {
        display: none;
    }
    
    .application-card .particles .particle:nth-child(n+8) {
        display: none;
    }
}

@supports not (-webkit-background-clip: text) {
    .section-title {
        background: none;
        color: #3b82f6;
    }
}
</style>
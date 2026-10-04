<script setup>
import { withBase } from 'vitepress'

const stack = [
  {
    index: '01',
    eyebrow: 'MODEL LAB',
    title: 'Python / AI',
    description: '完成数据处理、模型训练与效果验证。',
    tags: ['PyTorch', 'Jupyter', 'OpenCV']
  },
  {
    index: '02',
    eyebrow: 'EXCHANGE',
    title: 'ONNX',
    description: '把实验模型转换为稳定、可迁移的中间格式。',
    tags: ['ONNX', 'Graph', 'Optimize']
  },
  {
    index: '03',
    eyebrow: 'RUNTIME',
    title: 'C++',
    description: '高性能推理、资源管理与工程化运行时。',
    tags: ['C++17', 'ORT', 'CUDA']
  },
  {
    index: '04',
    eyebrow: 'INTERFACE',
    title: 'Qt',
    description: '把算法能力包装成真正可使用的桌面产品。',
    tags: ['Widgets', 'QML', 'Charts']
  }
]

const pipeline = [
  {
    index: '01',
    title: 'Prototype',
    description: '先验证问题是否值得解决，而不是先写完整工程。',
    tools: 'Python · PyTorch · Scikit-learn'
  },
  {
    index: '02',
    title: 'Export',
    description: '固化输入输出、转换模型并完成精度对齐。',
    tools: 'ONNX · Model Optimizer'
  },
  {
    index: '03',
    title: 'Runtime',
    description: '把模型放进 C++ 运行时，处理性能、线程与资源生命周期。',
    tools: 'ONNX Runtime · C++17'
  },
  {
    index: '04',
    title: 'Product',
    description: '加入交互、可视化与业务流程，形成可交付应用。',
    tools: 'Qt Widgets · QML'
  }
]

const projects = [
  {
    code: 'CV / DESKTOP',
    title: '图像识别桌面应用',
    description: '从模型训练到本地推理，把视觉模型真正装进桌面软件。',
    tags: ['PyTorch', 'ONNX Runtime', 'OpenCV', 'Qt'],
    metric: 'LOCAL FIRST'
  },
  {
    code: 'VISION / INDUSTRY',
    title: '智能工业检测系统',
    description: '围绕实时视觉、缺陷检测和生产数据构建工程化链路。',
    tags: ['OpenCV', 'ONNX', 'C++', 'QCustomPlot'],
    metric: 'REAL-TIME'
  }
]

const links = [
  { label: 'C++ / Qt 学习路径', href: '/C++/' },
  { label: 'AI 研究笔记', href: '/AI/' },
  { label: '综合技术实践', href: '/build/' }
]
</script>

<template>
  <section class="integration-section" aria-labelledby="integration-title">
    <div class="integration-shell">
      <header class="section-heading">
        <div>
          <div class="section-kicker">
            <span class="signal-dot"></span>
            SYSTEM MAP / END-TO-END
          </div>
          <h2 id="integration-title">C++ / Qt / AI 综合技术体系</h2>
          <p>
            从模型训练、格式转换、高性能推理到桌面交付。
            这里不是技术名词的堆叠，而是一条可以真正落地的工程路径。
          </p>
        </div>

        <div class="heading-meta" aria-hidden="true">
          <span>MODEL</span>
          <i></i>
          <span>RUNTIME</span>
          <i></i>
          <span>PRODUCT</span>
        </div>
      </header>

      <div class="stack-map">
        <article
          v-for="item in stack"
          :key="item.index"
          class="stack-card"
        >
          <div class="stack-card-top">
            <span class="stack-index">{{ item.index }}</span>
            <span class="stack-eyebrow">{{ item.eyebrow }}</span>
          </div>

          <div class="stack-card-body">
            <h3>{{ item.title }}</h3>
            <p>{{ item.description }}</p>
          </div>

          <div class="stack-tags">
            <span v-for="tag in item.tags" :key="tag">{{ tag }}</span>
          </div>
        </article>

        <article class="stack-output">
          <div class="output-status">
            <span class="signal-dot"></span>
            OUTPUT
          </div>
          <div class="output-mark">→</div>
          <h3>Intelligent<br />Desktop App</h3>
          <p>把算法能力变成可以运行、交互和交付的产品。</p>
          <span class="output-chip">AI × C++ × Qt</span>
        </article>
      </div>

      <div class="lower-grid">
        <section class="pipeline-panel">
          <div class="panel-heading">
            <div>
              <span class="panel-kicker">DELIVERY PIPELINE</span>
              <h3>从想法到产品</h3>
            </div>
            <span class="panel-count">04 STAGES</span>
          </div>

          <ol class="pipeline-list">
            <li v-for="step in pipeline" :key="step.index">
              <span class="pipeline-index">{{ step.index }}</span>
              <div class="pipeline-copy">
                <div class="pipeline-title-row">
                  <h4>{{ step.title }}</h4>
                  <span>{{ step.tools }}</span>
                </div>
                <p>{{ step.description }}</p>
              </div>
            </li>
          </ol>
        </section>

        <section class="projects-panel">
          <div class="panel-heading">
            <div>
              <span class="panel-kicker">PROJECT SNAPSHOTS</span>
              <h3>工程落点</h3>
            </div>
            <span class="panel-count">02 TRACKS</span>
          </div>

          <div class="project-list">
            <article
              v-for="project in projects"
              :key="project.code"
              class="project-card"
            >
              <div class="project-topline">
                <span>{{ project.code }}</span>
                <strong>{{ project.metric }}</strong>
              </div>
              <h4>{{ project.title }}</h4>
              <p>{{ project.description }}</p>
              <div class="project-tags">
                <span v-for="tag in project.tags" :key="tag">{{ tag }}</span>
              </div>
            </article>
          </div>
        </section>
      </div>

      <footer class="integration-footer">
        <div class="footer-label">
          <span>EXPLORE</span>
          <small>沿着路径继续深入</small>
        </div>
        <nav aria-label="技术体系入口">
          <a
            v-for="item in links"
            :key="item.href"
            :href="withBase(item.href)"
          >
            {{ item.label }}
            <span aria-hidden="true">↗</span>
          </a>
        </nav>
      </footer>
    </div>
  </section>
</template>

<style scoped>
.integration-section {
  --panel: color-mix(in srgb, var(--vp-c-bg-soft) 78%, transparent);
  --panel-strong: color-mix(in srgb, var(--vp-c-bg-soft) 92%, var(--vp-c-bg));
  --line: color-mix(in srgb, var(--vp-c-divider) 72%, transparent);
  --line-brand: color-mix(in srgb, var(--vp-c-brand-1, #5b6cff) 34%, transparent);
  --muted: var(--vp-c-text-2);
  --brand: var(--vp-c-brand-1, #5b6cff);
  --brand-soft: color-mix(in srgb, var(--brand) 14%, transparent);

  max-width: 1360px;
  margin: 2.6rem auto 0;
  padding: 0 2rem;
  content-visibility: auto;
  contain-intrinsic-size: 1050px;
}

.integration-shell {
  position: relative;
  overflow: hidden;
  padding: clamp(1.4rem, 3vw, 2.6rem);
  border: 1px solid var(--line);
  border-radius: 26px;
  background:
    linear-gradient(180deg,
      color-mix(in srgb, var(--vp-c-bg-soft) 82%, transparent),
      color-mix(in srgb, var(--vp-c-bg) 92%, transparent)
    );
  box-shadow:
    0 22px 70px rgba(0, 0, 0, 0.16),
    inset 0 1px 0 rgba(255, 255, 255, 0.035);
}

.integration-shell::before {
  content: "";
  position: absolute;
  width: 520px;
  height: 520px;
  right: -220px;
  top: -260px;
  border-radius: 50%;
  background: radial-gradient(
    circle,
    color-mix(in srgb, var(--brand) 18%, transparent),
    transparent 67%
  );
  pointer-events: none;
}

.integration-shell::after {
  content: "";
  position: absolute;
  inset: 0;
  opacity: 0.25;
  background-image:
    linear-gradient(var(--line) 1px, transparent 1px),
    linear-gradient(90deg, var(--line) 1px, transparent 1px);
  background-size: 36px 36px;
  mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.55), transparent 52%);
  pointer-events: none;
}

.section-heading,
.stack-map,
.lower-grid,
.integration-footer {
  position: relative;
  z-index: 1;
}

.section-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--line);
}

.section-kicker,
.panel-kicker,
.output-status,
.stack-eyebrow,
.panel-count,
.project-topline,
.footer-label > span {
  font-family: var(--vp-font-family-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.section-kicker {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.85rem;
  color: var(--brand);
  font-size: 0.72rem;
  font-weight: 700;
}

.signal-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 0 5px var(--brand-soft);
}

.section-heading h2 {
  margin: 0;
  max-width: 760px;
  color: var(--vp-c-text-1);
  font-size: clamp(1.85rem, 4vw, 3.15rem);
  line-height: 1.12;
  letter-spacing: -0.035em;
  font-weight: 760;
}

.section-heading p {
  max-width: 720px;
  margin: 1rem 0 0;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.8;
}

.heading-meta {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding-bottom: 0.2rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.64rem;
  letter-spacing: 0.08em;
  white-space: nowrap;
}

.heading-meta i {
  display: block;
  width: 30px;
  height: 1px;
  background: var(--line-brand);
}

.stack-map {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)) minmax(190px, 1.05fr);
  gap: 0.8rem;
  margin-top: 1.25rem;
}

.stack-card,
.stack-output {
  min-height: 230px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--panel);
}

.stack-card {
  display: flex;
  flex-direction: column;
  padding: 1.15rem;
  transition:
    transform 180ms ease,
    border-color 180ms ease,
    background 180ms ease;
}

.stack-card:hover {
  transform: translateY(-3px);
  border-color: var(--line-brand);
  background: var(--panel-strong);
}

.stack-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}

.stack-index {
  color: var(--brand);
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  font-weight: 700;
}

.stack-eyebrow {
  color: var(--vp-c-text-3);
  font-size: 0.58rem;
}

.stack-card-body {
  margin-top: 2.2rem;
}

.stack-card h3,
.stack-output h3,
.panel-heading h3,
.project-card h4,
.pipeline-title-row h4 {
  margin: 0;
}

.stack-card h3 {
  color: var(--vp-c-text-1);
  font-size: 1.35rem;
  letter-spacing: -0.02em;
}

.stack-card p {
  margin: 0.65rem 0 0;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.65;
}

.stack-tags,
.project-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.stack-tags {
  margin-top: auto;
  padding-top: 1.2rem;
}

.stack-tags span,
.project-tags span,
.output-chip {
  border: 1px solid var(--line);
  border-radius: 999px;
  background: color-mix(in srgb, var(--vp-c-bg) 52%, transparent);
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.64rem;
  line-height: 1;
}

.stack-tags span,
.project-tags span {
  padding: 0.38rem 0.55rem;
}

.stack-output {
  position: relative;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 1.15rem;
  border-color: var(--line-brand);
  background:
    linear-gradient(145deg,
      color-mix(in srgb, var(--brand) 16%, var(--panel)),
      var(--panel)
    );
}

.stack-output::after {
  content: "";
  position: absolute;
  width: 150px;
  height: 150px;
  right: -60px;
  bottom: -70px;
  border: 1px solid var(--line-brand);
  border-radius: 50%;
  box-shadow:
    0 0 0 22px color-mix(in srgb, var(--brand) 4%, transparent),
    0 0 0 44px color-mix(in srgb, var(--brand) 3%, transparent);
}

.output-status {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--brand);
  font-size: 0.6rem;
  font-weight: 700;
}

.output-mark {
  margin: 1.35rem 0 0.6rem;
  color: var(--brand);
  font-family: var(--vp-font-family-mono);
  font-size: 1.4rem;
}

.stack-output h3 {
  color: var(--vp-c-text-1);
  font-size: 1.25rem;
  line-height: 1.25;
}

.stack-output p {
  max-width: 230px;
  margin: 0.65rem 0 1rem;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.6;
}

.output-chip {
  align-self: flex-start;
  margin-top: auto;
  padding: 0.45rem 0.65rem;
  border-color: var(--line-brand);
  color: var(--brand);
}

.lower-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
  gap: 0.8rem;
  margin-top: 0.8rem;
}

.pipeline-panel,
.projects-panel {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--panel);
  padding: 1.25rem;
}

.panel-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.15rem;
}

.panel-kicker {
  display: block;
  margin-bottom: 0.4rem;
  color: var(--brand);
  font-size: 0.6rem;
  font-weight: 700;
}

.panel-heading h3 {
  color: var(--vp-c-text-1);
  font-size: 1.15rem;
  letter-spacing: -0.015em;
}

.panel-count {
  color: var(--vp-c-text-3);
  font-size: 0.58rem;
}

.pipeline-list {
  list-style: none;
  margin: 0;
  padding: 0;
  border-top: 1px solid var(--line);
}

.pipeline-list li {
  display: grid;
  grid-template-columns: 46px 1fr;
  gap: 0.85rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--line);
}

.pipeline-list li:last-child {
  border-bottom: 0;
  padding-bottom: 0.15rem;
}

.pipeline-index {
  color: var(--brand);
  font-family: var(--vp-font-family-mono);
  font-size: 0.68rem;
  padding-top: 0.14rem;
}

.pipeline-title-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
}

.pipeline-title-row h4 {
  color: var(--vp-c-text-1);
  font-size: 0.95rem;
}

.pipeline-title-row span {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.6rem;
  text-align: right;
}

.pipeline-copy p {
  margin: 0.35rem 0 0;
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.58;
}

.project-list {
  display: grid;
  gap: 0.7rem;
}

.project-card {
  position: relative;
  overflow: hidden;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 13px;
  background: color-mix(in srgb, var(--vp-c-bg) 48%, transparent);
  transition: border-color 180ms ease, transform 180ms ease;
}

.project-card:hover {
  transform: translateY(-2px);
  border-color: var(--line-brand);
}

.project-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
  color: var(--vp-c-text-3);
  font-size: 0.56rem;
}

.project-topline strong {
  color: var(--brand);
  font-weight: 700;
}

.project-card h4 {
  color: var(--vp-c-text-1);
  font-size: 1rem;
}

.project-card p {
  margin: 0.48rem 0 0.9rem;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.6;
}

.integration-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  margin-top: 0.8rem;
  padding: 1.1rem 1.25rem;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: color-mix(in srgb, var(--vp-c-bg) 46%, transparent);
}

.footer-label {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.footer-label > span {
  color: var(--brand);
  font-size: 0.58rem;
  font-weight: 700;
}

.footer-label small {
  color: var(--vp-c-text-3);
  font-size: 0.74rem;
}

.integration-footer nav {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.5rem;
}

.integration-footer a {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.58rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  color: var(--vp-c-text-2);
  background: var(--panel);
  text-decoration: none;
  font-size: 0.76rem;
  transition:
    color 160ms ease,
    border-color 160ms ease,
    background 160ms ease;
}

.integration-footer a:hover {
  color: var(--vp-c-text-1);
  border-color: var(--line-brand);
  background: var(--brand-soft);
}

.integration-footer a span {
  color: var(--brand);
}

@media (max-width: 1120px) {
  .stack-map {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stack-output {
    grid-column: 1 / -1;
    min-height: 180px;
  }

  .lower-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .integration-section {
    margin-top: 1.8rem;
    padding: 0 1rem;
  }

  .integration-shell {
    padding: 1.1rem;
    border-radius: 20px;
  }

  .section-heading {
    display: block;
    padding-bottom: 1.4rem;
  }

  .heading-meta {
    display: none;
  }

  .section-heading h2 {
    font-size: 1.8rem;
  }

  .section-heading p {
    font-size: 0.9rem;
  }

  .stack-map {
    grid-template-columns: 1fr;
  }

  .stack-output {
    grid-column: auto;
  }

  .stack-card,
  .stack-output {
    min-height: 190px;
  }

  .pipeline-title-row {
    display: block;
  }

  .pipeline-title-row span {
    display: block;
    margin-top: 0.3rem;
    text-align: left;
  }

  .integration-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .integration-footer nav {
    justify-content: flex-start;
  }
}

@media (prefers-reduced-motion: reduce) {
  .stack-card,
  .project-card,
  .integration-footer a {
    transition: none;
  }
}
</style>

<!-- MOBILE_INTEGRATION_V291_START -->
<style scoped>
@media (max-width: 720px) {
  /*
   * Mobile hierarchy:
   * 1) stack-map = horizontal swipe row
   * 2) lower-grid = horizontal swipe row containing Delivery + Projects
   * 3) content inside each panel remains vertical and wraps normally
   */

  .integration-section {
    overflow: hidden;
  }

  .integration-shell {
    min-width: 0;
  }

  .stack-map {
    display: grid !important;
    grid-template-columns: none !important;
    grid-auto-flow: column !important;
    grid-auto-columns: calc(100% - 18px) !important;
    gap: 0.65rem !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding-bottom: 0.65rem !important;
    scroll-snap-type: x mandatory;
    overscroll-behavior-inline: contain;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }

  .stack-map::-webkit-scrollbar,
  .lower-grid::-webkit-scrollbar {
    display: none;
  }

  .stack-card,
  .stack-output {
    grid-column: auto !important;
    min-width: 0 !important;
    scroll-snap-align: start;
    scroll-snap-stop: always;
  }

  /*
   * Delivery Pipeline and Project Snapshots are peers.
   * Swipe between the two panels instead of allowing the page itself
   * to overflow horizontally.
   */
  .lower-grid {
    display: grid !important;
    grid-template-columns: none !important;
    grid-auto-flow: column !important;
    grid-auto-columns: calc(100% - 18px) !important;
    align-items: start !important;
    gap: 0.65rem !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding-bottom: 0.65rem !important;
    scroll-snap-type: x mandatory;
    overscroll-behavior-inline: contain;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }

  .pipeline-panel,
  .projects-panel {
    box-sizing: border-box !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    padding: 1rem !important;
    overflow: hidden !important;
    scroll-snap-align: start;
    scroll-snap-stop: always;
  }

  /*
   * The desktop Pipeline row uses multiple columns.
   * On a narrow phone the right-hand technology string was forcing
   * the row wider than the panel. Collapse it to a compact vertical block.
   */
  .pipeline-list {
    min-width: 0 !important;
  }

  .pipeline-list li {
    grid-template-columns: 30px minmax(0, 1fr) !important;
    gap: 0.6rem !important;
    min-width: 0 !important;
    padding: 0.85rem 0 !important;
  }

  .pipeline-index,
  .pipeline-copy {
    min-width: 0 !important;
  }

  .pipeline-title-row {
    display: flex !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    flex-direction: column !important;
    gap: 0.2rem !important;
    min-width: 0 !important;
    width: 100% !important;
  }

  .pipeline-title-row h4,
  .pipeline-title-row span,
  .pipeline-copy p {
    min-width: 0 !important;
    max-width: 100% !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: normal !important;
    text-align: left !important;
  }

  .pipeline-title-row span {
    display: block !important;
    margin-top: 0 !important;
    line-height: 1.5 !important;
  }

  .pipeline-copy p {
    line-height: 1.58 !important;
  }

  /*
   * Do not create a nested horizontal swipe inside Project Snapshots.
   * The parent lower-grid already owns the horizontal gesture.
   */
  .project-list {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) !important;
    grid-auto-flow: row !important;
    gap: 0.65rem !important;
    width: 100% !important;
    min-width: 0 !important;
    overflow: visible !important;
  }

  .project-card {
    box-sizing: border-box !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
  }

  .project-card p,
  .project-topline,
  .project-tags {
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
  }
}
</style>
<!-- MOBILE_INTEGRATION_V291_END -->


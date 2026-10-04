<template>
  <section class="knowledge-graph" aria-label="知识图谱">
    <header class="graph-toolbar">
      <div class="graph-intro">
        <span class="graph-kicker">KNOWLEDGE SYSTEM</span>
        <strong>知识之间不是目录关系，而是连接关系</strong>
        <small>
          综合技术分支会根据 docs/build 自动更新；带链接的节点可直接进入笔记。
        </small>
      </div>

      <div class="graph-controls">
        <label class="search-box">
          <span class="sr-only">搜索知识节点</span>
          <input
            v-model.trim="searchTerm"
            type="search"
            placeholder="搜索节点..."
            autocomplete="off"
          />
        </label>

        <button type="button" @click="resetView">
          重置视图
        </button>

        <button type="button" @click="togglePhysics">
          {{ physicsEnabled ? '暂停模拟' : '启动模拟' }}
        </button>
      </div>
    </header>

    <div class="graph-meta" aria-hidden="true">
      <span>{{ nodeCount }} NODES</span>
      <i></i>
      <span>{{ linkCount }} LINKS</span>
      <i></i>
      <span>BUILD AUTO-SYNC</span>
    </div>

    <div
      ref="graphContainer"
      class="graph-container"
    ></div>

    <div
      v-if="loading"
      class="graph-status"
      role="status"
    >
      加载知识图谱中...
    </div>

    <div
      v-else-if="error"
      class="graph-status graph-error"
      role="alert"
    >
      图谱加载失败：{{ error }}
    </div>
  </section>
</template>

<script setup>
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
  watch
} from 'vue'

import { withBase } from 'vitepress'

import {
  drag,
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  scaleOrdinal,
  select,
  zoom,
  zoomIdentity
} from 'd3'

import {
  baseLinks,
  baseNodes
} from '../data/knowledgeGraphBase.js'

import {
  buildLinks,
  buildNodes
} from '../../cache/build-graph.generated.mjs'

const graphContainer = ref(null)
const physicsEnabled = ref(true)
const searchTerm = ref('')
const loading = ref(false)
const error = ref(null)

let simulation = null
let svgSelection = null
let rootGroup = null
let linkSelection = null
let nodeSelection = null
let labelSelection = null
let zoomBehavior = null
let resizeObserver = null
let resizeFrame = 0

function uniqueNodes(nodes) {
  const map = new Map()

  for (const node of nodes) {
    if (!map.has(node.id)) {
      map.set(node.id, node)
    }
  }

  return [...map.values()]
}

const allNodes =
  uniqueNodes([
    ...baseNodes,
    ...buildNodes
  ])

const nodeIds =
  new Set(
    allNodes.map((node) => node.id)
  )

const allLinks =
  [...baseLinks, ...buildLinks]
    .filter(
      (link) =>
        nodeIds.has(link.source) &&
        nodeIds.has(link.target)
    )

const nodeCount = computed(
  () => allNodes.length
)

const linkCount = computed(
  () => allLinks.length
)

const groupColors = new Map([
  ['language', '#4f7cff'],
  ['framework', '#7a68ff'],
  ['library', '#5f82b8'],
  ['domain', '#5d8cff'],
  ['integration', '#8275ff'],
  ['exchange', '#6f7fff'],
  ['runtime', '#4f94cf'],
  ['tooling', '#718096'],
  ['acceleration', '#9a68c7'],
  ['experience', '#6d78d8'],
  ['stage', '#745fd1'],
  ['folder', '#65758f'],
  ['note', '#566780'],
  ['method', '#548d8d'],
  ['algorithm', '#668f63'],
  ['model', '#6679a8'],
  ['technique', '#806aa3'],
  ['problem', '#a26969'],
  ['theory', '#68816d'],
  ['concept', '#557b9d'],
  ['principle', '#8f7655'],
  ['activity', '#68815c'],
  ['subject', '#69768f'],
  ['resource', '#8a7c55']
])

const colorScale =
  scaleOrdinal(
    [...groupColors.keys()],
    [...groupColors.values()]
  )

function nodeRadius(node) {
  if (node.id === 'build') return 23
  if (node.level === 1) return 20
  if (node.level === 2) return 15
  if (node.level === 3) return 12
  return 9
}

function nodeFontSize(node) {
  if (node.id === 'build') return 15
  if (node.level === 1) return 14
  if (node.level === 2) return 12
  return 10
}

function graphDimensions() {
  const width =
    Math.max(
      graphContainer.value?.clientWidth || 0,
      360
    )

  const height =
    Math.max(
      Math.min(
        window.innerHeight * 0.74,
        780
      ),
      540
    )

  return {
    width,
    height
  }
}

function cloneGraphData() {
  return {
    nodes:
      allNodes.map((node) => ({
        ...node
      })),
    links:
      allLinks.map((link) => ({
        ...link
      }))
  }
}

function linkId(value) {
  return typeof value === 'object'
    ? value?.id
    : value
}

function destroyGraph() {
  simulation?.stop()
  simulation = null

  svgSelection?.on('.zoom', null)
  svgSelection?.remove()

  svgSelection = null
  rootGroup = null
  linkSelection = null
  nodeSelection = null
  labelSelection = null
  zoomBehavior = null
}

function buildGraph() {
  if (!graphContainer.value) return

  destroyGraph()
  loading.value = true
  error.value = null

  try {
    const {
      width,
      height
    } = graphDimensions()

    const {
      nodes,
      links
    } = cloneGraphData()

    svgSelection =
      select(graphContainer.value)
        .append('svg')
        .attr('class', 'graph-svg')
        .attr('role', 'img')
        .attr(
          'aria-label',
          'C++、Qt、AI、ONNX 与综合技术知识图谱'
        )
        .attr('width', '100%')
        .attr('height', '100%')
        .attr(
          'viewBox',
          `0 0 ${width} ${height}`
        )
        .attr(
          'preserveAspectRatio',
          'xMidYMid meet'
        )

    rootGroup =
      svgSelection
        .append('g')
        .attr('class', 'graph-root')

    zoomBehavior =
      zoom()
        .scaleExtent([0.3, 4.5])
        .on('zoom', (event) => {
          rootGroup?.attr(
            'transform',
            event.transform
          )
        })

    svgSelection.call(zoomBehavior)

    linkSelection =
      rootGroup
        .append('g')
        .attr('class', 'graph-links')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr(
          'class',
          (link) =>
            `graph-link graph-link--${link.kind || 'concept'}`
        )
        .attr(
          'stroke-width',
          (link) =>
            Math.max(
              1,
              Math.sqrt(link.value || 6) * 0.9
            )
        )

    nodeSelection =
      rootGroup
        .append('g')
        .attr('class', 'graph-nodes')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr(
          'class',
          (node) =>
            [
              'graph-node',
              node.link
                ? 'graph-node--linked'
                : ''
            ].filter(Boolean).join(' ')
        )
        .attr(
          'r',
          (node) => nodeRadius(node)
        )
        .attr(
          'fill',
          (node) =>
            colorScale(node.group)
        )
        .attr('tabindex', 0)
        .attr('role', 'button')
        .attr(
          'aria-label',
          (node) =>
            node.link
              ? `${node.name}，打开笔记`
              : node.name
        )

    nodeSelection
      .append('title')
      .text(
        (node) =>
          node.link
            ? `${node.name} · 点击打开`
            : node.name
      )

    labelSelection =
      rootGroup
        .append('g')
        .attr('class', 'graph-labels')
        .selectAll('text')
        .data(nodes)
        .join('text')
        .attr('class', 'graph-label')
        .text((node) => node.name)
        .attr(
          'font-size',
          (node) => nodeFontSize(node)
        )
        .attr(
          'dx',
          (node) => nodeRadius(node) + 6
        )
        .attr('dy', '0.35em')

    nodeSelection
      .call(
        drag()
          .on('start', dragStarted)
          .on('drag', dragged)
          .on('end', dragEnded)
      )
      .on(
        'mouseenter',
        (_, node) =>
          highlightConnected(node)
      )
      .on(
        'mouseleave',
        () =>
          applySearch(searchTerm.value)
      )
      .on(
        'focus',
        (_, node) =>
          highlightConnected(node)
      )
      .on(
        'blur',
        () =>
          applySearch(searchTerm.value)
      )
      .on(
        'click',
        (event, node) => {
          if (
            event.defaultPrevented ||
            !node.link
          ) {
            return
          }

          window.location.href =
            withBase(node.link)
        }
      )
      .on(
        'keydown',
        (event, node) => {
          if (
            (event.key === 'Enter' ||
             event.key === ' ') &&
            node.link
          ) {
            event.preventDefault()
            window.location.href =
              withBase(node.link)
          }
        }
      )

    simulation =
      forceSimulation(nodes)
        .force(
          'link',
          forceLink(links)
            .id((node) => node.id)
            .distance(
              (link) => {
                if (link.kind === 'bridge') {
                  return 128
                }

                if (link.kind === 'semantic') {
                  return 105
                }

                return Math.max(
                  68,
                  118 - (link.value || 6) * 4
                )
              }
            )
        )
        .force(
          'charge',
          forceManyBody()
            .strength((node) =>
              node.id === 'build'
                ? -520
                : -285
            )
        )
        .force(
          'center',
          forceCenter(
            width / 2,
            height / 2
          )
        )
        .force(
          'collision',
          forceCollide()
            .radius(
              (node) =>
                nodeRadius(node) + 9
            )
        )
        .on('tick', ticked)

    applySearch(searchTerm.value)
  } catch (cause) {
    console.error(
      '[KnowledgeGraph] init failed:',
      cause
    )

    error.value =
      cause instanceof Error
        ? cause.message
        : '图谱初始化失败'
  } finally {
    loading.value = false
  }
}

function ticked() {
  linkSelection
    ?.attr(
      'x1',
      (link) => link.source.x
    )
    .attr(
      'y1',
      (link) => link.source.y
    )
    .attr(
      'x2',
      (link) => link.target.x
    )
    .attr(
      'y2',
      (link) => link.target.y
    )

  nodeSelection
    ?.attr(
      'cx',
      (node) => node.x
    )
    .attr(
      'cy',
      (node) => node.y
    )

  labelSelection
    ?.attr(
      'x',
      (node) => node.x
    )
    .attr(
      'y',
      (node) => node.y
    )
}

function dragStarted(event, node) {
  if (
    !event.active &&
    simulation &&
    physicsEnabled.value
  ) {
    simulation
      .alphaTarget(0.22)
      .restart()
  }

  node.fx = node.x
  node.fy = node.y
}

function dragged(event, node) {
  node.fx = event.x
  node.fy = event.y
}

function dragEnded(event, node) {
  if (
    !event.active &&
    simulation
  ) {
    simulation.alphaTarget(0)
  }

  node.fx = null
  node.fy = null
}

function connectedIdsFor(node) {
  const ids =
    new Set([node.id])

  for (const link of allLinks) {
    if (link.source === node.id) {
      ids.add(link.target)
    }

    if (link.target === node.id) {
      ids.add(link.source)
    }
  }

  return ids
}

function highlightConnected(node) {
  if (!nodeSelection) return

  const connected =
    connectedIdsFor(node)

  nodeSelection.attr(
    'opacity',
    (candidate) =>
      connected.has(candidate.id)
        ? 1
        : 0.12
  )

  labelSelection?.attr(
    'opacity',
    (candidate) =>
      connected.has(candidate.id)
        ? 1
        : 0.12
  )

  linkSelection?.attr(
    'opacity',
    (link) => {
      const source = linkId(link.source)
      const target = linkId(link.target)

      return (
        source === node.id ||
        target === node.id
      )
        ? 0.95
        : 0.05
    }
  )
}

function applySearch(value) {
  if (!nodeSelection) return

  const term =
    value
      .trim()
      .toLocaleLowerCase()

  if (!term) {
    nodeSelection.attr('opacity', 1)
    labelSelection?.attr('opacity', 1)
    linkSelection?.attr(
      'opacity',
      (link) =>
        link.kind === 'semantic'
          ? 0.22
          : 0.44
    )
    return
  }

  const matches =
    new Set(
      allNodes
        .filter((node) =>
          `${node.name} ${node.source || ''}`
            .toLocaleLowerCase()
            .includes(term)
        )
        .map((node) => node.id)
    )

  nodeSelection.attr(
    'opacity',
    (node) =>
      matches.has(node.id)
        ? 1
        : 0.10
  )

  labelSelection?.attr(
    'opacity',
    (node) =>
      matches.has(node.id)
        ? 1
        : 0.10
  )

  linkSelection?.attr(
    'opacity',
    0.04
  )
}

function resetView() {
  if (
    !svgSelection ||
    !zoomBehavior
  ) {
    return
  }

  svgSelection.call(
    zoomBehavior.transform,
    zoomIdentity
  )
}

function togglePhysics() {
  if (!simulation) return

  physicsEnabled.value =
    !physicsEnabled.value

  if (physicsEnabled.value) {
    simulation
      .alpha(0.28)
      .restart()
  } else {
    simulation.stop()
  }
}

function resizeGraph() {
  if (
    !svgSelection ||
    !simulation
  ) {
    return
  }

  const {
    width,
    height
  } = graphDimensions()

  svgSelection.attr(
    'viewBox',
    `0 0 ${width} ${height}`
  )

  simulation.force(
    'center',
    forceCenter(
      width / 2,
      height / 2
    )
  )

  if (physicsEnabled.value) {
    simulation
      .alpha(0.10)
      .restart()
  }
}

function scheduleResize() {
  cancelAnimationFrame(resizeFrame)
  resizeFrame =
    requestAnimationFrame(resizeGraph)
}

watch(
  searchTerm,
  applySearch
)

onMounted(() => {
  buildGraph()

  if (
    typeof ResizeObserver !== 'undefined' &&
    graphContainer.value
  ) {
    resizeObserver =
      new ResizeObserver(
        scheduleResize
      )

    resizeObserver.observe(
      graphContainer.value
    )
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(resizeFrame)
  resizeObserver?.disconnect()
  resizeObserver = null
  destroyGraph()
})
</script>

<style scoped>
.knowledge-graph {
  position: relative;
  width: 100%;
  overflow: hidden;
  margin: 1.5rem 0 2.5rem;
  border: 1px solid
    color-mix(
      in srgb,
      var(--vp-c-divider) 86%,
      transparent
    );
  border-radius: 18px;
  background:
    radial-gradient(
      circle at 72% 0%,
      color-mix(
        in srgb,
        var(--vp-c-brand-1) 9%,
        transparent
      ),
      transparent 36%
    ),
    color-mix(
      in srgb,
      var(--vp-c-bg-soft) 86%,
      transparent
    );
}

.graph-toolbar {
  position: relative;
  z-index: 5;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1.25rem;
  padding: 1.15rem 1.2rem 0.95rem;
  border-bottom: 1px solid
    color-mix(
      in srgb,
      var(--vp-c-divider) 72%,
      transparent
    );
}

.graph-intro {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 0.22rem;
}

.graph-kicker {
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.70rem;
  font-weight: 700;
  letter-spacing: 0.10em;
}

.graph-intro strong {
  color: var(--vp-c-text-1);
  font-size: 1rem;
}

.graph-intro small {
  color:
    color-mix(
      in srgb,
      var(--vp-c-text-1) 62%,
      var(--vp-c-text-2)
    );
  font-size: 0.80rem;
}

.graph-controls {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  gap: 0.45rem;
}

.graph-controls button,
.search-box input {
  height: 36px;
  border: 1px solid
    color-mix(
      in srgb,
      var(--vp-c-divider) 86%,
      transparent
    );
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background:
    color-mix(
      in srgb,
      var(--vp-c-bg) 92%,
      transparent
    );
  font: inherit;
  font-size: 0.78rem;
}

.graph-controls button {
  padding: 0 0.75rem;
  cursor: pointer;
}

.search-box input {
  width: min(220px, 26vw);
  padding: 0 0.75rem;
  outline: none;
}

.search-box input:focus {
  border-color: var(--vp-c-brand-1);
}

.graph-meta {
  position: absolute;
  top: 78px;
  right: 20px;
  z-index: 4;
  display: flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.61rem;
  letter-spacing: 0.08em;
  pointer-events: none;
}

.graph-meta i {
  width: 24px;
  height: 1px;
  background:
    color-mix(
      in srgb,
      var(--vp-c-brand-1) 34%,
      transparent
    );
}

.graph-container {
  width: 100%;
  height: clamp(540px, 72vh, 780px);
}

.graph-status {
  position: absolute;
  inset: 55% auto auto 50%;
  z-index: 8;
  transform: translate(-50%, -50%);
  padding: 0.65rem 0.85rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 9px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font-size: 0.84rem;
}

.graph-error {
  border-color:
    color-mix(
      in srgb,
      #dc3545 55%,
      var(--vp-c-divider)
    );
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

.graph-container :deep(.graph-link) {
  stroke: var(--vp-c-text-3);
}

.graph-container :deep(.graph-link--semantic) {
  stroke-dasharray: 4 5;
}

.graph-container :deep(.graph-link--bridge) {
  stroke:
    color-mix(
      in srgb,
      var(--vp-c-brand-1) 60%,
      var(--vp-c-text-3)
    );
}

.graph-container :deep(.graph-node) {
  stroke: var(--vp-c-bg);
  stroke-width: 2px;
  cursor: grab;
  transition:
    opacity 120ms ease,
    stroke-width 120ms ease;
}

.graph-container :deep(.graph-node--linked) {
  cursor: pointer;
}

.graph-container :deep(.graph-node:hover),
.graph-container :deep(.graph-node:focus) {
  stroke: var(--vp-c-text-1);
  stroke-width: 3px;
  outline: none;
}

.graph-container :deep(.graph-label) {
  fill: var(--vp-c-text-1);
  stroke: var(--vp-c-bg);
  stroke-width: 3px;
  paint-order: stroke;
  stroke-linecap: round;
  stroke-linejoin: round;
  pointer-events: none;
  font-weight: 650;
  transition: opacity 120ms ease;
}

@media (max-width: 760px) {
  .graph-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .graph-controls {
    width: 100%;
    flex-wrap: wrap;
  }

  .search-box {
    flex: 1 1 180px;
  }

  .search-box input {
    width: 100%;
  }

  .graph-meta {
    display: none;
  }

  .graph-container {
    height: 66vh;
    min-height: 520px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .graph-container :deep(.graph-node),
  .graph-container :deep(.graph-label) {
    transition: none;
  }
}
</style>

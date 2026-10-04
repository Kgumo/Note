<template>
  <span class="mermaid-runtime" aria-hidden="true"></span>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, watch } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

let observer = null
let themeObserver = null
let mermaidPromise = null
let sequence = 0
let destroyed = false

function currentTheme() {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'default'
}

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then((module) => module.default ?? module)
  }
  return mermaidPromise
}

function getSource(node) {
  if (node.__mermaidSource) return node.__mermaidSource

  const sourceNode = node.querySelector('.mermaid-diagram__source')
  const source = sourceNode?.content?.textContent ?? sourceNode?.textContent ?? ''
  node.__mermaidSource = source
  return source
}

async function renderDiagram(node) {
  if (
    destroyed ||
    !node ||
    node.dataset.mermaidState === 'rendering' ||
    node.dataset.mermaidState === 'rendered'
  ) {
    return
  }

  const source = getSource(node)
  if (!source.trim()) return

  node.dataset.mermaidState = 'rendering'

  try {
    const mermaid = await loadMermaid()

    mermaid.initialize({
      startOnLoad: false,
      theme: currentTheme(),
      securityLevel: 'loose',
      fontFamily: "'Noto Serif SC', sans-serif",
      flowchart: {
        nodeSpacing: 50,
        rankSpacing: 50,
        htmlLabels: true
      }
    })

    const id = `mermaid-lazy-${Date.now()}-${sequence++}`
    const { svg, bindFunctions } = await mermaid.render(id, source)

    if (destroyed || !node.isConnected) return

    node.innerHTML = svg
    node.dataset.mermaidState = 'rendered'
    bindFunctions?.(node)
  } catch (error) {
    console.error('[Mermaid] render failed:', error)
    node.textContent = 'Mermaid 图表渲染失败'
    node.dataset.mermaidState = 'error'
  }
}

function ensureIntersectionObserver() {
  if (observer || typeof IntersectionObserver === 'undefined') return

  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        observer.unobserve(entry.target)
        renderDiagram(entry.target)
      }
    },
    {
      rootMargin: '320px 0px'
    }
  )
}

function registerDiagrams() {
  if (typeof document === 'undefined') return

  const nodes = document.querySelectorAll(
    '.mermaid-diagram[data-mermaid-state="idle"], .mermaid-diagram:not([data-mermaid-state])'
  )

  if (!nodes.length) return

  ensureIntersectionObserver()

  for (const node of nodes) {
    node.dataset.mermaidState = 'idle'

    if (observer) {
      observer.observe(node)
    } else {
      // Old browsers: still lazy-load by route, without IntersectionObserver.
      renderDiagram(node)
    }
  }
}

async function refreshAfterNavigation() {
  await nextTick()
  registerDiagrams()
}

async function rerenderForTheme() {
  await nextTick()

  const nodes = document.querySelectorAll('.mermaid-diagram')

  for (const node of nodes) {
    // Source is cached on the wrapper before the generated SVG replaces the template.
    getSource(node)
    node.dataset.mermaidState = 'idle'
    await renderDiagram(node)
  }
}

onMounted(async () => {
  await refreshAfterNavigation()

  themeObserver = new MutationObserver((mutations) => {
    if (mutations.some((item) => item.attributeName === 'class')) {
      rerenderForTheme()
    }
  })

  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class']
  })
})

watch(
  () => route.path,
  () => refreshAfterNavigation()
)

onBeforeUnmount(() => {
  destroyed = true
  observer?.disconnect()
  themeObserver?.disconnect()
})
</script>

<style>
.mermaid-runtime {
  display: none;
}

.mermaid-diagram {
  position: relative;
  overflow-x: auto;
  min-height: 160px;
  margin: 1.5rem 0;
  padding: 1.1rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, transparent);
}

.mermaid-diagram svg {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}

.mermaid-diagram__placeholder {
  display: flex;
  min-height: 120px;
  align-items: center;
  justify-content: center;
  gap: 0.65rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
}

.mermaid-diagram__placeholder::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--vp-c-brand-1, #5b6cff);
  box-shadow: 0 0 0 6px color-mix(in srgb, var(--vp-c-brand-1, #5b6cff) 12%, transparent);
}
</style>

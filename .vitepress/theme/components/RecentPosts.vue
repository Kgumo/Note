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

function rowNumber(index) {
  return String(index + 1).padStart(2, '0')
}
</script>

<template>
  <section class="recent-updates" aria-labelledby="recent-updates-title">
    <header class="updates-heading">
      <div>
        <span class="updates-kicker">CHANGELOG / NOTES</span>
        <h2 id="recent-updates-title">最近更新</h2>
      </div>
      <p>持续记录正在学习、正在构建，以及刚刚补完的内容。</p>
    </header>

    <div class="updates-list">
      <a
        v-for="(post, index) in props.posts"
        :key="post.link"
        :href="normalizePath(post.link)"
        class="update-row"
        :class="{ featured: index === 0 }"
      >
        <span class="row-number">{{ rowNumber(index) }}</span>
        <time>{{ post.date }}</time>
        <h3>{{ post.title }}</h3>
        <span class="row-action" aria-hidden="true">↗</span>
      </a>
    </div>
  </section>
</template>

<style scoped>
.recent-updates {
  max-width: 1360px;
  margin: 1rem auto 6rem;
  padding: 0 2rem;
  content-visibility: auto;
  contain-intrinsic-size: 620px;
}

.updates-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 2rem;
  padding: 2.4rem 0 1.35rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.updates-kicker {
  display: block;
  margin-bottom: 0.45rem;
  color: var(--vp-c-brand-1, #5b6cff);
  font-family: var(--vp-font-family-mono);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.11em;
}

.updates-heading h2 {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: clamp(1.65rem, 3vw, 2.35rem);
  line-height: 1.1;
  letter-spacing: -0.035em;
}

.updates-heading p {
  max-width: 420px;
  margin: 0;
  color: var(--vp-c-text-3);
  font-size: 0.82rem;
  line-height: 1.65;
  text-align: right;
}

.updates-list {
  border-bottom: 1px solid var(--vp-c-divider);
}

.update-row {
  position: relative;
  display: grid;
  grid-template-columns: 60px 150px minmax(0, 1fr) 30px;
  align-items: center;
  gap: 1rem;
  min-height: 84px;
  padding: 0 0.9rem;
  border-top: 1px solid color-mix(in srgb, var(--vp-c-divider) 78%, transparent);
  color: inherit;
  text-decoration: none;
  transition:
    background 160ms ease,
    padding 160ms ease;
}

.update-row::before {
  content: "";
  position: absolute;
  left: 0;
  top: 22%;
  bottom: 22%;
  width: 2px;
  border-radius: 999px;
  background: var(--vp-c-brand-1, #5b6cff);
  transform: scaleY(0);
  transition: transform 160ms ease;
}

.update-row:hover,
.update-row.featured {
  background:
    linear-gradient(
      90deg,
      color-mix(in srgb, var(--vp-c-brand-1, #5b6cff) 8%, transparent),
      transparent 72%
    );
}

.update-row:hover {
  padding-left: 1.15rem;
}

.update-row:hover::before,
.update-row.featured::before {
  transform: scaleY(1);
}

.row-number {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.66rem;
}

.update-row time {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.7rem;
}

.update-row h3 {
  overflow: hidden;
  margin: 0;
  color: var(--vp-c-text-2);
  font-size: 0.95rem;
  font-weight: 560;
  line-height: 1.5;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: color 160ms ease;
}

.update-row:hover h3,
.update-row.featured h3 {
  color: var(--vp-c-text-1);
}

.row-action {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 1rem;
  transition:
    color 160ms ease,
    transform 160ms ease;
}

.update-row:hover .row-action {
  color: var(--vp-c-brand-1, #5b6cff);
  transform: translate(2px, -2px);
}

@media (max-width: 720px) {
  .recent-updates {
    margin-bottom: 4rem;
    padding: 0 1rem;
  }

  .updates-heading {
    display: block;
    padding-top: 2rem;
  }

  .updates-heading p {
    margin-top: 0.7rem;
    text-align: left;
  }

  .update-row {
    grid-template-columns: 38px minmax(0, 1fr) 24px;
    gap: 0.65rem;
    min-height: 88px;
  }

  .update-row time {
    display: none;
  }

  .update-row h3 {
    white-space: normal;
  }
}

@media (prefers-reduced-motion: reduce) {
  .update-row,
  .update-row::before,
  .update-row h3,
  .row-action {
    transition: none;
  }
}
</style>

# Note 网站优化包

这是一个 **overlay（覆盖包）**，用于覆盖到你现有的 `Kgumo/Note` 仓库根目录。

之所以不是完全独立仓库，是因为你上传的压缩包没有包含根目录 `.vitepress` 里未修改的文件（例如 `Layout.vue`、`custom.css`、`IntegrationSection.vue`、sidebar utils 等）。本包包含你上传的本地 Markdown/资源，并加入本轮修改过的文件；覆盖到现有仓库或 fresh clone 后即可使用。

## 已做的修改

1. Markdown 图片默认增加 `loading="lazy"` 和 `decoding="async"`。
2. 不再把所有相对图片路径强制改成根路径；仅兼容你现有的 `public/...` 写法。
3. 保留自定义域名 `gukmo.cn` 所需的 `base: '/'`，并支持 `VITEPRESS_BASE` 环境变量。
4. 删除运行时 DOM 路径扫描和 `Element.prototype.setAttribute` monkey patch。
5. 首页 RecentPosts：
   - 删除人为 800ms 骨架屏等待；
   - 删除 `Math.random()` SSR/hydration 不确定性；
   - 删除 `window.location.href` 整页刷新，让 VitePress 接管站内导航。
6. 首页/知识图谱等全局组件改成异步组件，减少普通文章页首屏 JS。
7. CI 改为 `npm ci --legacy-peer-deps`，不再删除 lockfile、不再安装 `@latest`、不再强行把 Mermaid 降到 10.9.0。
8. 编辑链接从 `main` 修正为 `master`。
9. 新增根目录 `.gitignore`。
10. 17 张大 PNG 生成 WebP，并把相应 Markdown 引用切换到 WebP；原 PNG 保留。
11. 新增 `npm run fix:knowledge-graph`，修复重复 K-Means 节点和 resize 后 SVG/D3 资源未完整清理的问题。

## 图片优化结果

- 参与转换：17 张
- 原 PNG 合计：12.26 MB
- WebP 合计：2.62 MB
- 减少：9.64 MB（78.6%）
- 修改 Markdown：7 个文件

`640.gif`（约 1.4 MB）仍然保留为 GIF；它有 108 帧，本轮先通过 lazy loading 延迟加载，避免为了压缩动画引入兼容性风险。

## 使用方式

### 1. 覆盖到现有仓库

把本 zip 的内容直接解压到 `Note` 仓库根目录，允许覆盖同名文件。

### 2. 修复 KnowledgeGraph

```bash
npm run fix:knowledge-graph
```

### 3. 本地验证

```bash
npm ci --legacy-peer-deps
npm run docs:build
npm run docs:preview
```

### 4. 清理已被 Git 跟踪的构建产物（建议只做一次）

```bash
git rm -r --cached node_modules dist
git add .gitignore
```

如果某个目录本来就没有被 Git 跟踪，对应命令报 pathspec 不存在可以忽略。

### 5. 提交

```bash
git add .
git commit -m "perf: optimize image loading and site runtime"
git push origin master
```

## 说明

`docs/public/CNAME` 是 `gukmo.cn`，所以生产部署默认根路径 `/` 是正确的。只有你明确要从 `https://kgumo.github.io/Note/` 这类项目子路径预览时，才需要：

```bash
VITEPRESS_BASE=/Note/ npm run docs:build
```

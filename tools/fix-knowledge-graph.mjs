import fs from 'node:fs';
import path from 'node:path';

const file = path.resolve('.vitepress/theme/components/KnowledgeGraph.vue');

if (!fs.existsSync(file)) {
  console.error(`未找到 ${file}`);
  console.error('请在 Note 仓库根目录执行：npm run fix:knowledge-graph');
  process.exit(1);
}

let source = fs.readFileSync(file, 'utf8');
const original = source;

const duplicateKMeans = `        { id: "kmeans", name: "K-Means聚类", group: "algorithm", level: 4 },
        { id: "kmeans", name: "K-Means聚类", group: "algorithm", level: 4 },`;

source = source.replace(
  duplicateKMeans,
  `        { id: "kmeans", name: "K-Means聚类", group: "algorithm", level: 4 },`
);

// The second cleanup() is the effective declaration in the current source.
// Make it fully release the D3 simulation, SVG and references so resize does not accumulate SVGs.
const weakCleanup = `    function cleanup() {
      if (simulation.value) {
        simulation.value.stop();
      }
      window.removeEventListener('resize', handleResize);
    }`;

const robustCleanup = `    function cleanup() {
      if (simulation.value) {
        simulation.value.stop();
        simulation.value = null;
      }

      window.removeEventListener('resize', handleResize);

      if (graphContainer.value) {
        d3.select(graphContainer.value).selectAll('svg').remove();
      }

      svg.value = null;
      link.value = null;
      node.value = null;
      zoom.value = null;
    }`;

source = source.replace(weakCleanup, robustCleanup);

if (source === original) {
  console.log('KnowledgeGraph.vue 没有需要应用的已知修复，可能已经修过。');
  process.exit(0);
}

const backupDir = path.resolve('.optimization-backup');
fs.mkdirSync(backupDir, { recursive: true });
fs.copyFileSync(file, path.join(backupDir, 'KnowledgeGraph.vue'));
fs.writeFileSync(file, source, 'utf8');

console.log('KnowledgeGraph 修复完成：');
console.log('- 删除重复的 kmeans 节点');
console.log('- resize/重建时完整清理 D3 simulation 和 SVG');
console.log('- 原文件备份到 .optimization-backup/KnowledgeGraph.vue');

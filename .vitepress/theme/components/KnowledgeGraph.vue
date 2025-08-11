<template>
  <div class="knowledge-graph">
    <div class="graph-controls">
      <button @click="resetView">重置视图</button>
      <button @click="togglePhysics">{{ physicsEnabled ? '暂停' : '启动' }}物理模拟</button>
      <div class="search-box">
        <input v-model="searchTerm" placeholder="搜索节点..." />
      </div>
    </div>
    <div ref="graphContainer" class="graph-container"></div>
    <div v-if="loading" class="graph-loading">加载知识图谱中...</div>
    <div v-else-if="error" class="graph-error">图表加载失败: {{ error }}</div>
  </div>
</template>

<script>
import { onMounted, ref, watch, onBeforeUnmount } from 'vue';
import * as d3 from 'd3';

export default {
  setup() {
    const graphContainer = ref(null);
    const physicsEnabled = ref(true);
    const searchTerm = ref('');
    const loading = ref(false);
    const error = ref(null);
    
    // 示例知识图谱数据
    const graphData = {
      nodes: [
        { id: "c++", name: "C++", group: "language", level: 1 },
        { id: "qt", name: "Qt框架", group: "framework", level: 2 },
        { id: "stl", name: "STL", group: "library", level: 2 },
        { id: "ai", name: "人工智能", group: "domain", level: 1 },
        { id: "ml", name: "机器学习", group: "domain", level: 2 },
        { id: "dl", name: "深度学习", group: "domain", level: 2 },
        { id: "cv", name: "计算机视觉", group: "domain", level: 2 },
        { id: "python", name: "Python", group: "language", level: 2 },
        { id: "pytorch", name: "PyTorch", group: "framework", level: 3 },
        { id: "tensorflow", name: "TensorFlow", group: "framework", level: 3 },
        { id: "interview", name: "面试准备", group: "activity", level: 1 },
        { id: "leetcode", name: "LeetCode", group: "resource", level: 3 },
        { id: "system-design", name: "系统设计", group: "topic", level: 2 },
        { id: "postgraduate", name: "计算机考研", group: "activity", level: 1 },
        { id: "ds", name: "数据结构", group: "subject", level: 2 },
        { id: "os", name: "操作系统", group: "subject", level: 2 },
        { id: "network", name: "计算机网络", group: "subject", level: 2 },
        { id: "resources", name: "学习资源", group: "resource", level: 1 },
        { id: "books", name: "推荐书籍", group: "resource", level: 2 },
        { id: "courses", name: "在线课程", group: "resource", level: 2 },
        { id: "projects", name: "项目经验", group: "experience", level: 1 }
      ],
      links: [
        { source: "c++", target: "qt", value: 8 },
        { source: "c++", target: "stl", value: 9 },
        { source: "ai", target: "ml", value: 10 },
        { source: "ai", target: "dl", value: 10 },
        { source: "ai", target: "cv", value: 8 },
        { source: "ml", target: "python", value: 8 },
        { source: "dl", target: "pytorch", value: 9 },
        { source: "dl", target: "tensorflow", value: 9 },
        { source: "interview", target: "leetcode", value: 10 },
        { source: "interview", target: "system-design", value: 9 },
        { source: "interview", target: "c++", value: 8 },
        { source: "postgraduate", target: "ds", value: 10 },
        { source: "postgraduate", target: "os", value: 10 },
        { source: "postgraduate", target: "network", value: 9 },
        { source: "resources", target: "books", value: 10 },
        { source: "resources", target: "courses", value: 10 },
        { source: "resources", target: "leetcode", value: 8 },
        { source: "projects", target: "c++", value: 9 },
        { source: "projects", target: "qt", value: 8 },
        { source: "projects", target: "ai", value: 7 },
        { source: "ml", target: "projects", value: 7 },
        { source: "dl", target: "projects", value: 7 },
        { source: "postgraduate", target: "interview", value: 8 }
      ]
    };

    // 所有 D3 相关变量
    const simulation = ref(null);
    const svg = ref(null);
    const link = ref(null);
    const node = ref(null);
    const zoom = ref(null);
    
    onMounted(() => {
      initGraph();
    });
    
    onBeforeUnmount(() => {
      cleanup();
    });
    
    function initGraph() {
      if (!graphContainer.value) return;
      
      loading.value = true;
      error.value = null;
      
      try {
        const container = graphContainer.value;
        const width = container.clientWidth;
        const height = Math.max(600, window.innerHeight * 0.8);
        
        // 清除现有SVG内容
        d3.select(container).selectAll("svg").remove();
        
        // 颜色映射
        const colorScale = d3.scaleOrdinal()
          .domain(["language", "framework", "library", "domain", "activity", "resource", "topic", "subject", "experience"])
          .range(["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f"]);
        
        // 创建SVG容器
        const svgEl = d3.select(container)
          .append("svg")
          .attr("xmlns", "http://www.w3.org/2000/svg")
          .attr("width", width)
          .attr("height", height)
          .call(zoom.value = d3.zoom().on("zoom", zoomed))
          .append("g");
        
        svg.value = svgEl;
        
        // 创建力导向图模拟
        simulation.value = d3.forceSimulation(graphData.nodes)
          .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(d => 100 - d.value * 5))
          .force("charge", d3.forceManyBody().strength(-300))
          .force("center", d3.forceCenter(width / 2, height / 2))
          .force("collision", d3.forceCollide().radius(d => getNodeRadius(d.level) + 5));
        
        // 创建连线
        link.value = svgEl.append("g")
          .attr("class", "links")
          .selectAll("line")
          .data(graphData.links)
          .enter().append("line")
          .attr("stroke", "#999")
          .attr("stroke-opacity", 0.6)
          .attr("stroke-width", d => Math.sqrt(d.value))
          .attr("stroke-dasharray", d => d.value > 8 ? "0" : "5,5");
        
        // 创建节点
        node.value = svgEl.append("g")
          .attr("class", "nodes")
          .selectAll("circle")
          .data(graphData.nodes)
          .enter().append("circle")
          .attr("r", d => getNodeRadius(d.level))
          .attr("fill", d => colorScale(d.group))
          .attr("stroke", "#fff")
          .attr("stroke-width", 1.5)
          .attr("data-id", d => d.id)
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended))
          .on("click", nodeClicked);
        
        // 添加节点标签
        const labels = svgEl.append("g")
          .attr("class", "labels")
          .selectAll("text")
          .data(graphData.nodes)
          .enter().append("text")
          .text(d => d.name)
          .attr("font-size", d => getFontSize(d.level))
          .attr("dx", d => getNodeRadius(d.level) + 5)
          .attr("dy", "0.35em")
          .attr("fill", "#333")
          .attr("pointer-events", "none");
        
        // 添加悬停效果
        node.value.on("mouseover", function(event, d) {
          d3.select(this).attr("stroke", "#000").attr("stroke-width", 2);
          highlightConnected(d);
        }).on("mouseout", function(event, d) {
          d3.select(this).attr("stroke", "#fff").attr("stroke-width", 1.5);
          resetHighlight();
        });
        
        // 更新模拟
        simulation.value.on("tick", () => {
          link.value
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
          
          node.value
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);
          
          labels
            .attr("x", d => d.x)
            .attr("y", d => d.y);
        });
        
        // 添加窗口大小变化监听
        window.addEventListener('resize', handleResize);
      } catch (e) {
        console.error('知识图谱初始化失败:', e);
        error.value = '图表初始化失败';
      } finally {
        loading.value = false;
      }
    }
    
    function cleanup() {
      if (simulation.value) {
        simulation.value.stop();
      }
      window.removeEventListener('resize', handleResize);
    }
    
    function handleResize() {
      if (graphContainer.value) {
        initGraph();
      }
    }
    
    function getNodeRadius(level) {
      return level === 1 ? 20 : level === 2 ? 15 : 10;
    }
    
    function getFontSize(level) {
      return level === 1 ? "14px" : level === 2 ? "12px" : "10px";
    }
    
    function highlightConnected(d) {
      if (!node.value) return;
      
      // 高亮当前节点
      node.value.attr("opacity", 0.2);
      d3.select(`[data-id="${d.id}"]`).attr("opacity", 1);
      
      // 高亮相连节点
      const connectedIds = new Set();
      connectedIds.add(d.id);
      
      graphData.links.forEach(link => {
        if (link.source.id === d.id) {
          connectedIds.add(link.target.id);
        }
        if (link.target.id === d.id) {
          connectedIds.add(link.source.id);
        }
      });
      
      connectedIds.forEach(id => {
        d3.select(`[data-id="${id}"]`).attr("opacity", 1);
      });
      
      // 高亮相关连线
      if (link.value) {
        link.value.attr("opacity", l => {
          return (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1;
        });
      }
    }
    
    function resetHighlight() {
      if (node.value) node.value.attr("opacity", 1);
      if (link.value) link.value.attr("opacity", 0.6);
    }
    
    function nodeClicked(event, d) {
      console.log(`点击了节点: ${d.name}`);
      // 实际使用中可添加导航逻辑
      // this.$router.push(`/path/to/${d.id}`);
    }
    
    function dragstarted(event, d) {
      if (!event.active && simulation.value) simulation.value.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active && simulation.value) simulation.value.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    
    function zoomed(event) {
      if (svg.value) svg.value.attr("transform", event.transform);
    }
    
    function resetView() {
      if (svg.value) {
        svg.value.transition()
          .duration(750)
          .attr("transform", "translate(0,0) scale(1)");
      }
    }
    
    function togglePhysics() {
      if (!simulation.value) return;
      
      physicsEnabled.value = !physicsEnabled.value;
      if (physicsEnabled.value) {
        simulation.value.alpha(0.3).restart();
      } else {
        simulation.value.stop();
      }
    }
    
    // 搜索功能
    watch(searchTerm, (newTerm) => {
      if (!node.value) return;
      
      if (!newTerm) {
        resetHighlight();
        return;
      }
      
      const term = newTerm.toLowerCase();
      node.value.attr("opacity", d => 
        d.name.toLowerCase().includes(term) ? 1 : 0.2
      );
    });
    
    return {
      graphContainer,
      physicsEnabled,
      searchTerm,
      loading,
      error,
      resetView,
      togglePhysics
    };
  }
};
</script>

<style scoped>
.knowledge-graph {
  width: 100%;
  height: 80vh;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  position: relative;
  overflow: hidden;
  margin: 2rem 0;
}

.graph-controls {
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 10;
  display: flex;
  gap: 10px;
  background: rgba(var(--vp-c-bg-rgb), 0.8);
  padding: 10px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.graph-controls button {
  padding: 6px 12px;
  background: var(--vp-c-brand);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background 0.3s;
}

.graph-controls button:hover {
  background: var(--vp-c-brand-light);
}

.search-box {
  display: flex;
  align-items: center;
}

.search-box input {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
}

.graph-container {
  width: 100%;
  height: 100%;
}

.graph-loading,
.graph-error {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 12px 24px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  border-radius: 8px;
  font-size: 1.2rem;
  z-index: 10;
  text-align: center;
}

.graph-error {
  background: rgba(220, 53, 69, 0.8);
  max-width: 80%;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .knowledge-graph {
    height: 60vh;
  }
  
  .graph-controls {
    flex-direction: column;
    align-items: flex-end;
  }
  
  .graph-error {
    font-size: 1rem;
    padding: 8px 16px;
  }
}
</style>
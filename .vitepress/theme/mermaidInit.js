// docs/.vitepress/theme/mermaidInit.js
export default {
  mounted() {
    // 确保 Mermaid 可用
    if (typeof window.mermaid !== 'undefined') {
      window.mermaid.initialize({ 
        startOnLoad: true,
        theme: 'dark',
        fontFamily: "'Noto Serif SC', sans-serif",
        securityLevel: 'loose'
      });
      
      // 重绘所有图表
      setTimeout(() => {
        const mermaids = document.querySelectorAll('.mermaid');
        if (mermaids.length > 0) {
          window.mermaid.init(undefined, mermaids);
        }
      }, 500);
    } else {
      console.warn('Mermaid not loaded!');
    }
  },
  
  updated() {
    this.mounted();
  }
}
// .vitepress/theme/mermaid.js
export default {
  mounted() {
    this.initializeMermaid();
  },
  
  updated() {
    this.initializeMermaid();
  },
  
  methods: {
    initializeMermaid() {
      if (typeof window.mermaid !== 'undefined') {
        try {
          window.mermaid.initialize({
            theme: 'dark',
            securityLevel: 'loose',
            fontFamily: "'Noto Serif SC', sans-serif",
            fontSize: 16,
            htmlLabels: true,
            flowchart: {
              nodeSpacing: 50,
              rankSpacing: 50
            },
            startOnLoad: true
          });
          
          // 手动初始化所有图表
          setTimeout(() => {
            const mermaids = document.querySelectorAll('.mermaid');
            if (mermaids.length > 0) {
              window.mermaid.init(undefined, mermaids);
            }
          }, 500);
        } catch (e) {
          console.error('Mermaid initialization error:', e);
        }
      } else {
        console.warn('Mermaid not loaded!');
      }
    }
  }
}
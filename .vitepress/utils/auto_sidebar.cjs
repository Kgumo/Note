const path = require("path");
const fs = require("fs");

const ROOT_PATH = path.resolve(__dirname, '../..');
const DOCS_PATH = path.resolve(ROOT_PATH, 'docs');

// 确保 docs 目录存在
if (!fs.existsSync(DOCS_PATH)) {
  fs.mkdirSync(DOCS_PATH, { recursive: true });
}

// 默认配置
const DEFAULT_CONFIG = {
  noOverviewDirs: [],
  sortRules: new Map([
    ['index.md', 0],
    ['getting-started.md', 1],
    ['installation.md', 2],
    ['configuration.md', 3],
    ['advanced.md', 4]
  ]),
  ignorePatterns: [
    '.vitepress', 'node_modules', '.idea', '.git',
    'assets', 'public', 'dist', '*.tmp', '*.log',
    '*.html', '*.css', '*.json', '*.js', '*.mjs'
  ]
};

// 缓存机制
const MAX_CACHE_SIZE = 20;
const sidebarCache = new Map();

// 安全路径解析
const resolveSafePath = (segment = "") => {
  const cleanSegment = segment
    .replace(/^\/+/, '')
    .replace(/\\/g, '/');
  
  const fullPath = path.join(DOCS_PATH, cleanSegment);
  
  if (!fullPath.startsWith(DOCS_PATH)) {
    console.warn(`路径越界: ${fullPath}`);
    return DOCS_PATH;
  }
  
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
  
  return fullPath;
};

// 判断是否是文件夹
const isDirectory = (path) => {
  try {
    return fs.statSync(path).isDirectory();
  } catch (error) {
    return false;
  }
};

// 判断是否应该忽略
const shouldIgnore = (name, ignorePatterns) => {
  return ignorePatterns.some(pattern => {
    if (pattern.startsWith('*.')) {
      const ext = pattern.slice(1);
      return name.endsWith(ext);
    }
    return name === pattern;
  });
};

// 获取排序权重
const getSortWeight = (name, sortRules) => {
  const lowerName = name.toLowerCase();
  const numMatch = lowerName.match(/^(\d+)\./);
  if (numMatch) {
    return parseInt(numMatch[1], 10);
  }
  return sortRules.get(lowerName) || 100;
};

// 递归生成侧边栏
function generateSidebarItems(basePath, relativePath, config) {
  try {
    if (!fs.existsSync(basePath)) {
      return [];
    }
    
    let files;
    try {
      files = fs.readdirSync(basePath);
    } catch (error) {
      return [];
    }
    
    const validFiles = files
      .filter(file => {
        if (file.startsWith('.') || file.startsWith('_')) return false;
        return !shouldIgnore(file, config.ignorePatterns);
      })
      .sort((a, b) => {
        const aWeight = getSortWeight(a, config.sortRules);
        const bWeight = getSortWeight(b, config.sortRules);
        if (aWeight !== bWeight) return aWeight - bWeight;
        return a.localeCompare(b, undefined, { numeric: true });
      });
    
    const sidebarItems = [];
    const skipOverview = config.noOverviewDirs.some(dir => 
      relativePath.endsWith(dir)
    );
    const hasIndex = validFiles.includes("index.md") || validFiles.includes("README.md");
    
    if (hasIndex && !skipOverview) {
      sidebarItems.push({
        text: "概述",
        link: `${relativePath}/`
      });
    }
    
    for (const file of validFiles) {
      const fullPath = path.join(basePath, file);
      const fileExt = path.extname(file).toLowerCase();
      
      if (isDirectory(fullPath)) {
        const childItems = generateSidebarItems(
          fullPath, 
          `${relativePath}/${file}`,
          config
        );
        
        if (childItems.length > 0) {
          sidebarItems.push({
            text: formatTitle(file),
            collapsible: true,
            collapsed: config.collapsed !== false,
            items: childItems,
          });
        }
      } 
      else if (fileExt === '.md') {
        if ((file === 'index.md' || file === 'README.md') && !skipOverview) continue;
        
        const name = file.replace(/\.md$/i, "");
        sidebarItems.push({
          text: formatTitle(name),
          link: `${relativePath}/${name}`,
        });
      }
    }
    
    return sidebarItems;
  } catch (error) {
    return [];
  }
}

// 格式化标题
function formatTitle(name) {
  return name
    .replace(/\.md$/i, "")
    .replace(/^\d+\./, "")
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// 主导出函数
module.exports.set_sidebar = (pathname = "", configPath) => {
  const cleanPathname = pathname.replace(/\\/g, '/').replace(/^\/+/, '');
  const cacheKey = `sidebar-${cleanPathname}`;
  
  if (sidebarCache.has(cacheKey)) {
    return sidebarCache.get(cacheKey);
  }
  
  const config = {...DEFAULT_CONFIG};
  
  if (configPath && fs.existsSync(configPath)) {
    try {
      const customConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      
      if (Array.isArray(customConfig.ignorePatterns)) {
        config.ignorePatterns = [
          ...new Set([...config.ignorePatterns, ...customConfig.ignorePatterns])
        ];
      }
      
      if (Array.isArray(customConfig.noOverviewDirs)) {
        config.noOverviewDirs = [
          ...new Set([...config.noOverviewDirs, ...customConfig.noOverviewDirs])
        ];
      }
      
      if (customConfig.sortRules && typeof customConfig.sortRules === 'object') {
        const customSortMap = new Map(Object.entries(customConfig.sortRules));
        config.sortRules = new Map([...config.sortRules, ...customSortMap]);
      }
    } catch (error) {
      console.error('加载配置文件错误:', error);
    }
  }
  
  let safePath;
  try {
    safePath = resolveSafePath(cleanPathname);
  } catch (error) {
    return [];
  }
  
  if (!fs.existsSync(safePath)) {
    return [];
  }
  
  const sidebar = generateSidebarItems(
    safePath, 
    `/${cleanPathname.replace(/\\/g, '/')}`, 
    config
  );
  
  sidebarCache.set(cacheKey, sidebar);
  
  if (sidebarCache.size > MAX_CACHE_SIZE) {
    const oldestKey = [...sidebarCache.keys()][0];
    sidebarCache.delete(oldestKey);
  }
  
  return sidebar;
};
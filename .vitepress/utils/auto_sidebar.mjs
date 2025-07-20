import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

// 获取当前模块的绝对路径
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 项目根目录
const ROOT_PATH = path.join(__dirname, "../..");
// 修复点：文档目录直接使用项目根目录
const DOCS_PATH = ROOT_PATH; 

// 白名单和忽略规则
const IGNORE_LIST = [
  ".vitepress",
  "node_modules",
  ".idea",
  ".git",
  "assets",
  "public",
  "dist",
  "*.tmp",
  "*.log"
];

// 排序规则
const SORT_ORDER = {
  "index.md": 0,
  "README.md": 0,
  "getting-started.md": 1,
  "installation.md": 2,
  "configuration.md": 3,
  "advanced.md": 4
};

// 调试日志：显示文档路径
console.log(`文档根目录: ${DOCS_PATH}`);

// 判断是否是文件夹
const isDirectory = (path) => {
  try {
    return fs.lstatSync(path).isDirectory();
  } catch {
    return false;
  }
};

// 判断是否应该忽略
const shouldIgnore = (name) => {
  return IGNORE_LIST.some(pattern => {
    if (pattern.startsWith('*')) {
      return name.endsWith(pattern.slice(1));
    }
    return name === pattern;
  });
};

// 获取排序权重
const getSortWeight = (name) => {
  const lowerName = name.toLowerCase();
  return SORT_ORDER[lowerName] ?? 100;
};

// 递归生成侧边栏
function generateSidebarItems(basePath, relativePath) {
  const sidebarItems = [];
  
  // 读取目录内容
  let files;
  try {
    files = fs.readdirSync(basePath);
  } catch (error) {
    console.error(`读取目录失败: ${basePath}`, error);
    return [];
  }
  
  // 过滤和排序
  const validFiles = files
    .filter(file => !file.startsWith('.') && !shouldIgnore(file))
    .sort((a, b) => {
      const aWeight = getSortWeight(a);
      const bWeight = getSortWeight(b);
      return aWeight - bWeight || a.localeCompare(b);
    });
  
  console.log(`处理目录: ${basePath}, 有效项目:`, validFiles);
  
  for (const file of validFiles) {
    const fullPath = path.join(basePath, file);
    const isDir = isDirectory(fullPath);
    
    if (isDir) {
      const childItems = generateSidebarItems(
        fullPath, 
        `${relativePath}/${file}`
      );
      
      if (childItems.length > 0) {
        sidebarItems.push({
          text: formatTitle(file),
          collapsible: true,
          collapsed: true,
          items: childItems,
        });
      }
    } else {
      const suffix = path.extname(file);
      if (suffix !== ".md") continue;
      
      const isIndex = file.toLowerCase() === "index.md";
      const name = file.replace(/\.md$/i, "");
      
      sidebarItems.push({
        text: isIndex ? "概述" : formatTitle(name),
        link: isIndex 
          ? `${relativePath}/` 
          : `${relativePath}/${name}`,
      });
    }
  }
  
  // 添加目录首页
  if (fs.existsSync(path.join(basePath, "index.md"))) {
    sidebarItems.unshift({
      text: "概述",
      link: `${relativePath}/`
    });
  }
  
  return sidebarItems;
}

// 格式化标题
function formatTitle(name) {
  return name
    .replace(/-/g, ' ')
    .replace(/^\w/, c => c.toUpperCase());
}

export const set_sidebar = (pathname = "") => {
  const targetPath = pathname ? pathname : "";
  const dirPath = path.join(DOCS_PATH, targetPath);

  console.log(`开始扫描目录: ${dirPath}`);
  console.log(`目录存在: ${fs.existsSync(dirPath)}`);

  if (!fs.existsSync(dirPath)) {
    console.warn(`目录不存在: ${dirPath}`);
    return [];
  }

  const sidebar = generateSidebarItems(dirPath, `/${targetPath}`);
  console.log(`生成侧边栏: ${JSON.stringify(sidebar, null, 2)}`);
  return sidebar;
};
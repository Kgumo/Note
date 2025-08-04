import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
// import config from './utils/sidebar-config.json' assert { type: 'json' };
// const NO_OVERVIEW_DIRS = config.noOverviewDirs || [];
// 获取当前模块的绝对路径
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 项目根目录
const ROOT_PATH = path.join(__dirname, "../..");
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
  "getting-started.md": 1,
  "installation.md": 2,
  "configuration.md": 3,
  "advanced.md": 4
};

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
function generateSidebarItems(basePath, relativePath, config) {
  const sidebarItems = [];
  
  // 读取目录内容
  let files;
  try {
    files = fs.readdirSync(basePath);
    console.log(`在 ${basePath} 中找到文件:`, files);
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
  
  console.log(`过滤后的文件: ${validFiles}`);
  
  // 检查当前目录是否需要跳过"概述"项
  const skipOverview = (config?.noOverviewDirs || []).some(dir => 
    relativePath.endsWith(dir)
  );
  
  // 检查目录是否有 index.md
  const hasIndex = validFiles.some(file => 
    file.toLowerCase() === "index.md" && 
    !isDirectory(path.join(basePath, file))
  );
  
  // 如果有 index.md 且不需要跳过概述，添加概述项
  if (hasIndex && !skipOverview) {
    sidebarItems.push({
      text: "概述",
      link: `${relativePath}/`
    });
  }
  
  for (const file of validFiles) {
    const fullPath = path.join(basePath, file);
    const isDir = isDirectory(fullPath);
    
    if (isDir) {
      console.log(`处理目录: ${file}`);
      const childItems = generateSidebarItems(
        fullPath, 
        `${relativePath}/${file}`,
        config
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
      if (suffix !== ".md") {
        console.log(`跳过非 Markdown 文件: ${file}`);
        continue;
      }
      
      // 在需要跳过概述的目录中不跳过 index.md
      if (file.toLowerCase() === "index.md" && !skipOverview) continue;
      
      const name = file.replace(/\.md$/i, "");
      
      console.log(`添加文件: ${name}`);
      sidebarItems.push({
        text: formatTitle(name),
        link: `${relativePath}/${name}`,
      });
    }
  }
  
  return sidebarItems;
}

// 格式化标题
function formatTitle(name) {
  return name
    .replace(/-/g, ' ')
    .replace(/^\w/, c => c.toUpperCase());
}

export const set_sidebar = (pathname = "", configPath) => {
  console.log(`生成 ${pathname} 的侧边栏`);
  
  const targetPath = pathname ? pathname : "";
  const dirPath = path.join(DOCS_PATH, targetPath);
  console.log(`目录路径: ${dirPath}`);

  if (!fs.existsSync(dirPath)) {
    console.error(`目录不存在: ${dirPath}`);
    console.log('请创建目录或检查路径');
    return [];
  }

  // 读取配置文件
  let config = { noOverviewDirs: [] };
  try {
    if (fs.existsSync(configPath)) {
      console.log(`加载配置文件: ${configPath}`);
      const configData = fs.readFileSync(configPath, 'utf-8');
      config = JSON.parse(configData);
      console.log(`已加载 ${pathname} 侧边栏配置:`, config);
    } else {
      console.warn(`未找到配置文件: ${configPath}`);
    }
  } catch (error) {
    console.error(`加载配置文件失败: ${configPath}`, error);
  }

  return generateSidebarItems(dirPath, `/${targetPath}`, config);
};
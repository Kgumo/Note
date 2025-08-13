// 浏览器兼容的 createRequire 实现
export default function createRequire() {
  return () => {
    throw new Error('createRequire is not available in browser environment');
  };
}
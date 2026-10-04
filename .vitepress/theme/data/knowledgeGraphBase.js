export const baseNodes = [
  // ===== 核心技术域 =====
  { id: 'c++', name: 'C++', group: 'language', level: 1, link: '/C++/' },
  { id: 'qt', name: 'Qt', group: 'framework', level: 2, link: '/C++/' },
  { id: 'stl', name: 'STL', group: 'library', level: 2, link: '/C++/' },

  { id: 'ai', name: '人工智能', group: 'domain', level: 1, link: '/AI/' },
  { id: 'ml', name: '机器学习', group: 'domain', level: 2, link: '/AI/' },
  { id: 'dl', name: '深度学习', group: 'domain', level: 2, link: '/AI/' },
  { id: 'cv', name: '计算机视觉', group: 'domain', level: 2, link: '/AI/' },
  { id: 'python', name: 'Python', group: 'language', level: 2, link: '/AI/' },
  { id: 'pytorch', name: 'PyTorch', group: 'framework', level: 3, link: '/AI/' },

  // ===== 综合技术 / 部署链路 =====
  { id: 'onnx', name: 'ONNX', group: 'exchange', level: 2, link: '/build/' },
  {
    id: 'onnx-runtime',
    name: 'ONNX Runtime',
    group: 'runtime',
    level: 2,
    link: '/build/'
  },
  {
    id: 'model-export',
    name: '模型转换',
    group: 'technique',
    level: 3,
    link: '/build/'
  },
  {
    id: 'inference',
    name: 'C++ 推理',
    group: 'runtime',
    level: 2,
    link: '/build/'
  },
  { id: 'cmake', name: 'CMake', group: 'tooling', level: 3, link: '/build/' },
  { id: 'cuda', name: 'CUDA', group: 'acceleration', level: 3, link: '/build/' },
  {
    id: 'tensorrt',
    name: 'TensorRT',
    group: 'acceleration',
    level: 3,
    link: '/build/'
  },
  {
    id: 'deployment',
    name: '工程部署',
    group: 'experience',
    level: 2,
    link: '/build/'
  },
  {
    id: 'projects',
    name: '项目实战',
    group: 'experience',
    level: 1,
    link: '/build/'
  },

  // ===== 机器学习方法 =====
  { id: 'supervised', name: '监督学习', group: 'method', level: 3 },
  { id: 'unsupervised', name: '无监督学习', group: 'method', level: 3 },
  { id: 'ensemble', name: '集成学习', group: 'method', level: 3 },

  { id: 'decision-tree', name: '决策树', group: 'algorithm', level: 4 },
  { id: 'linear-reg', name: '线性回归', group: 'algorithm', level: 4 },
  { id: 'bayesian', name: '贝叶斯学习', group: 'algorithm', level: 4 },
  { id: 'svm', name: '支持向量机(SVM)', group: 'algorithm', level: 4 },
  { id: 'knn', name: 'K近邻(KNN)', group: 'algorithm', level: 4 },
  { id: 'kd-tree', name: 'KD-Tree', group: 'algorithm', level: 5 },
  { id: 'kmeans', name: 'K-Means聚类', group: 'algorithm', level: 4 },
  { id: 'kmedoids', name: 'K-Medoids聚类', group: 'algorithm', level: 4 },
  {
    id: 'hierarchical-clust',
    name: '层次聚类',
    group: 'algorithm',
    level: 4
  },

  { id: 'bagging', name: 'Bagging', group: 'algorithm', level: 4 },
  { id: 'boosting', name: 'Boosting', group: 'algorithm', level: 4 },
  {
    id: 'weighted-majority',
    name: '加权多数算法',
    group: 'algorithm',
    level: 4
  },

  { id: 'mlp', name: '多层感知机(MLP)', group: 'model', level: 4 },
  { id: 'cnn', name: '卷积神经网络(CNN)', group: 'model', level: 4 },
  { id: 'rnn', name: '循环神经网络(RNN)', group: 'model', level: 4 },
  { id: 'lstm', name: '长短期记忆(LSTM)', group: 'model', level: 5 },
  { id: 'gru', name: '门控循环单元(GRU)', group: 'model', level: 5 },

  // ===== 实验 / 理论 =====
  { id: 'overfitting', name: '过拟合', group: 'problem', level: 3 },
  {
    id: 'cross-validation',
    name: '交叉验证',
    group: 'technique',
    level: 4
  },
  {
    id: 'bootstrap-sampling',
    name: 'Bootstrap采样',
    group: 'technique',
    level: 4
  },
  {
    id: 'inductive-learning',
    name: '归纳学习假设',
    group: 'theory',
    level: 3
  },
  {
    id: 'bayesian-stats',
    name: '贝叶斯统计',
    group: 'theory',
    level: 4
  },
  { id: 'map', name: '极大后验(MAP)', group: 'concept', level: 4 },
  { id: 'mdl', name: '最小描述长度(MDL)', group: 'principle', level: 4 },
  {
    id: 'ml-estimation',
    name: '极大似然估计(ML)',
    group: 'concept',
    level: 4
  },

  // ===== 计算机基础 / 成长路径 =====
  {
    id: 'postgraduate',
    name: '计算机考研',
    group: 'activity',
    level: 1,
    link: '/Postgraduate/'
  },
  { id: 'ds', name: '数据结构', group: 'subject', level: 2 },
  { id: 'os', name: '操作系统', group: 'subject', level: 2 },
  { id: 'network', name: '计算机网络', group: 'subject', level: 2 },
  {
    id: 'resources',
    name: '学习资源',
    group: 'resource',
    level: 1,
    link: '/resources'
  },
  { id: 'books', name: '推荐书籍', group: 'resource', level: 2 },
  { id: 'courses', name: '在线课程', group: 'resource', level: 2 }
]

export const baseLinks = [
  // C++ / Qt
  { source: 'c++', target: 'qt', value: 9 },
  { source: 'c++', target: 'stl', value: 7 },
  { source: 'c++', target: 'cmake', value: 8 },

  // AI
  { source: 'ai', target: 'ml', value: 10 },
  { source: 'ai', target: 'dl', value: 10 },
  { source: 'ai', target: 'cv', value: 8 },
  { source: 'ml', target: 'python', value: 8 },
  { source: 'dl', target: 'pytorch', value: 9 },

  // 部署链路：训练 -> 转换 -> ONNX -> Runtime -> C++ -> 产品
  { source: 'pytorch', target: 'model-export', value: 10 },
  { source: 'dl', target: 'model-export', value: 8 },
  { source: 'model-export', target: 'onnx', value: 10 },
  { source: 'onnx', target: 'onnx-runtime', value: 10 },
  { source: 'onnx-runtime', target: 'inference', value: 10 },
  { source: 'c++', target: 'inference', value: 10 },
  { source: 'cmake', target: 'inference', value: 8 },
  { source: 'inference', target: 'projects', value: 9 },
  { source: 'qt', target: 'projects', value: 8 },
  { source: 'cv', target: 'projects', value: 8 },
  { source: 'projects', target: 'deployment', value: 10 },
  { source: 'cuda', target: 'onnx-runtime', value: 8 },
  { source: 'tensorrt', target: 'deployment', value: 8 },
  { source: 'dl', target: 'onnx', value: 8 },

  // ML 方法
  { source: 'ml', target: 'supervised', value: 10 },
  { source: 'ml', target: 'unsupervised', value: 9 },
  { source: 'ml', target: 'ensemble', value: 8 },

  { source: 'supervised', target: 'decision-tree', value: 9 },
  { source: 'supervised', target: 'linear-reg', value: 9 },
  { source: 'supervised', target: 'bayesian', value: 9 },
  { source: 'supervised', target: 'svm', value: 9 },
  { source: 'supervised', target: 'knn', value: 8 },

  { source: 'unsupervised', target: 'kmeans', value: 9 },
  { source: 'unsupervised', target: 'kmedoids', value: 8 },
  { source: 'unsupervised', target: 'hierarchical-clust', value: 8 },

  { source: 'ensemble', target: 'weighted-majority', value: 8 },
  { source: 'ensemble', target: 'bagging', value: 9 },
  { source: 'ensemble', target: 'boosting', value: 9 },

  // Deep learning
  { source: 'dl', target: 'mlp', value: 8 },
  { source: 'dl', target: 'cnn', value: 9 },
  { source: 'dl', target: 'rnn', value: 9 },
  { source: 'rnn', target: 'lstm', value: 8 },
  { source: 'rnn', target: 'gru', value: 8 },
  { source: 'cnn', target: 'cv', value: 9 },

  // Experiment / theory
  { source: 'ml', target: 'overfitting', value: 8 },
  { source: 'overfitting', target: 'cross-validation', value: 9 },
  { source: 'overfitting', target: 'bootstrap-sampling', value: 8 },
  { source: 'ml', target: 'inductive-learning', value: 8 },
  { source: 'ml', target: 'bayesian-stats', value: 8 },
  { source: 'bayesian-stats', target: 'map', value: 9 },
  { source: 'bayesian-stats', target: 'ml-estimation', value: 9 },
  { source: 'bayesian-stats', target: 'mdl', value: 8 },

  // CS foundation
  { source: 'postgraduate', target: 'ds', value: 10 },
  { source: 'postgraduate', target: 'os', value: 10 },
  { source: 'postgraduate', target: 'network', value: 9 },
  { source: 'resources', target: 'books', value: 9 },
  { source: 'resources', target: 'courses', value: 9 }
]

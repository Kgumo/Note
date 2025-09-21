# 决策树实战：鸢尾花分类

## 问题概述

**目标**：根据鸢尾花的花萼和花瓣测量数据，自动分类三种鸢尾花（Setosa, Versicolor, Virginica）

**数据集**：著名的Iris数据集，包含150个样本，每个样本有4个特征：

1. 花萼长度（sepal length）
    
2. 花萼宽度（sepal width）
    
3. 花瓣长度（petal length）
    
4. 花瓣宽度（petal width）
    

## 解决思路

1. **数据准备**：加载数据，划分训练集和测试集
    
2. **模型训练**：使用决策树算法学习特征与类别之间的关系
    
3. **模型评估**：在测试集上评估模型性能
    
4. **可视化分析**：可视化决策树，理解模型决策过程

## 先决条件：安装必要的库

如果你是第一次使用Python进行机器学习，需要先安装这些库。在命令行（Windows上是CMD或PowerShell，Mac/Linux上是终端）中输入：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 代码实现与详细讲解

### 1. 导入必要的库

```python
# 基础数学计算库：提供数组、矩阵等数学运算功能
# 就像高级的计算器，能快速处理大量数字
import numpy as np

# 数据处理库：提供表格型数据结构，方便数据清洗和处理
# 类似于Excel，但更强大，可以用代码操作数据
import pandas as pd

# 绘图库：用于创建各种静态图表
# 相当于数字画板，可以把数据变成直观的图形
import matplotlib.pyplot as plt

# 基于matplotlib的高级绘图库：让图表更美观
# 像是matplotlib的"美颜相机"，让图表更好看
import seaborn as sns

# 从sklearn库中导入数据集模块
# sklearn是Python最流行的机器学习库
from sklearn.datasets import load_iris

# 导入数据划分函数：将数据分为训练集和测试集
# 就像把练习题分成学习用的和考试用的
from sklearn.model_selection import train_test_split

# 导入决策树分类器和可视化工具
# 决策树分类器是我们的"学习机器"
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 导入准确率评估函数和混淆矩阵
# 这些是"评分工具"，用来评价学习效果
from sklearn.metrics import accuracy_score, confusion_matrix

# 设置图表风格为seaborn样式，让图表更好看
plt.style.use('seaborn-v0_8')
```

**简单理解**：这些库就像不同的工具——numpy是计算器，pandas是Excel，matplotlib是画板，sklearn是机器学习工具箱。

### 2. 加载和探索数据

```python
# 加载鸢尾花数据集
# 这就像打开一本已经准备好的练习册
iris = load_iris()

# 获取特征数据（花的测量数据）
# 这就像练习册中的题目
X = iris.data

# 获取目标标签（花的种类）
# 这就像练习册的答案
y = iris.target

# 查看数据的基本信息
print("数据集包含的特征:", iris.feature_names)
print("要分类的花的种类:", iris.target_names)
print("数据形状:", X.shape)  # 输出(150, 4)表示150行4列
print("前5个样本的数据:")
print(X[:5])  # 显示前5朵花的测量数据
print("前5个样本的类别:", y[:5])  # 显示前5朵花实际是什么种类
```

**输出结果解释**：
```
数据集包含的特征: ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
要分类的花的种类: ['setosa山鸢尾', 'versicolor变色鸢尾', 'virginica维吉尼亚鸢尾']
数据形状: (150, 4)  # 150朵花，每朵花有4个测量值
前5个样本的数据:
[[5.1 3.5 1.4 0.2]  # 第一朵花：花萼长5.1，宽3.5，花瓣长1.4，宽0.2
 [4.9 3.0 1.4 0.2]  # 第二朵花：花萼长4.9，宽3.0，花瓣长1.4，宽0.2
 ...]
前5个样本的类别: [0 0 0 0 0]  # 前5朵都是setosa山鸢尾（用0表示）
```

### 3. 数据可视化：看看数据长什么样

```python
# 将数据转换为DataFrame格式（类似于Excel表格）
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y  # 添加一列表示花的种类

# 创建2x2的子图来显示4个特征
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 遍历4个特征，分别绘制箱线图
for i, feature in enumerate(iris.feature_names):
    row, col = i // 2, i % 2  # 计算当前特征应该在哪个子图位置
    
    # 绘制箱线图：显示每个种类的该特征分布情况
    sns.boxplot(x='species', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Species')
    
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图表
```

**图表解释**：你会看到4个图表，每个图表显示一个特征（如花瓣长度）在三种花中的分布情况。可以看到setosa花的花瓣明显较小，这有助于我们理解为什么决策树能区分它们。

### 4. 划分训练集和测试集

```python
# 将数据划分为训练集和测试集
# test_size=0.3表示30%的数据作为测试集，70%作为训练集
# random_state=42确保每次运行结果一致（设置随机种子）
# stratify=y确保训练集和测试集中各类别的比例相同
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}")  # 大约105朵花用于训练
print(f"测试集大小: {X_test.shape[0]}")   # 大约45朵花用于测试
```

**为什么要划分**：就像学习时要用一些例题练习（训练集），然后用另一些题测试是否真正学会（测试集）。

### 5. 创建和训练决策树模型

```python
# 创建决策树分类器
dtree = DecisionTreeClassifier(
    criterion='gini',  # 使用基尼系数作为分裂标准（衡量不纯度）
    max_depth=3,       # 限制树的最大深度为3层，防止过拟合
    random_state=42    # 设置随机种子确保结果可重现
)

# 训练模型：让决策树学习训练数据中的规律
# 这就像学生看例题学习解题方法
dtree.fit(X_train, y_train)

# 用训练好的模型进行预测
y_train_pred = dtree.predict(X_train)  # 对训练集预测
y_test_pred = dtree.predict(X_test)    # 对测试集预测

# 计算准确率：预测正确的比例
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"训练集准确率: {train_accuracy:.4f}")  # 模型在训练集上的表现
print(f"测试集准确率: {test_accuracy:.4f}")   # 模型在测试集上的表现
```

**结果解释**：
- 训练集准确率：模型在"练习题"上的得分
- 测试集准确率：模型在"考试题"上的得分
- 如果两者接近，说明模型学习得很好；如果训练集远高于测试集，说明可能过拟合了

### 6. 可视化决策树：看看模型是如何做决定的

```python
# 创建一个大图表来显示决策树
plt.figure(figsize=(16, 10))

# 绘制决策树
plot_tree(
    dtree,
    feature_names=iris.feature_names,  # 使用特征的真实名称
    class_names=iris.target_names,     # 使用类别的真实名称
    filled=True,        # 给节点着色：颜色越深表示纯度越高
    rounded=True,       # 使用圆角矩形，让树更好看
    fontsize=10         # 设置字体大小
)
plt.title("决策树可视化 - 鸢尾花分类")
plt.show()
```

**决策树解读**：
- 每个节点显示分裂条件（如petal width <= 0.8）
- 显示基尼系数（衡量不纯度，越小越好）
- 显示样本数量和类别分布
- 颜色表示主要类别（蓝色=setosa，绿色=versicolor，橙色=virginica）

### 7. 评估模型性能

```python
# 创建混淆矩阵：显示预测结果和真实结果的对比
cm = confusion_matrix(y_test, y_test_pred)

# 绘制混淆矩阵的热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('预测标签')  # x轴是模型预测的种类
plt.ylabel('真实标签')  # y轴是实际的种类
plt.title('混淆矩阵')
plt.show()

# 分析特征重要性：哪些特征对分类最重要
feature_importance = dtree.feature_importances_
feature_names = iris.feature_names

# 创建特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('特征重要性')
plt.xlabel('重要性得分')
plt.tight_layout()
plt.show()

# 打印具体的特征重要性数值
print("\n特征重要性分析:")
for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"{i+1}. {feature}: {importance:.4f}")
```

## 完整代码总结

```python
# 1. 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

plt.style.use('seaborn-v0_8')

# 2. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 3. 探索数据
print("特征:", iris.feature_names)
print("类别:", iris.target_names)
print("数据形状:", X.shape)

# 4. 可视化
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(iris.feature_names):
    row, col = i // 2, i % 2
    sns.boxplot(x='species', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Species')
plt.tight_layout()
plt.show()

# 5. 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. 训练模型
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# 7. 预测和评估
y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试准确率: {accuracy:.4f}")

# 8. 可视化决策树
plt.figure(figsize=(16, 10))
plot_tree(dtree, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# 9. 特征重要性
importance = dtree.feature_importances_
for i, (feature, imp) in enumerate(zip(iris.feature_names, importance)):
    print(f"{i+1}. {feature}: {imp:.4f}")
```

## 学习要点回顾

1. **数据准备**：加载数据、了解数据结构
2. **数据探索**：可视化分析特征分布
3. **数据划分**：分为训练集和测试集
4. **模型训练**：创建并训练决策树模型
5. **模型评估**：评估模型性能并可视化决策过程
6. **结果解释**：理解模型如何做决策以及哪些特征最重要

# 简单线性回归实例：学习时间与考试成绩

## 问题描述

我们想研究学生的学习时间与考试成绩之间的关系，并建立一个预测模型。

## 步骤1：环境准备和库导入

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```

**为什么要这些库？**
- `numpy`：Python的科学计算库，提供高效的数组操作和数学函数
- `matplotlib`：Python的绘图库，用于数据可视化
- `sklearn.linear_model`：Scikit-learn库中的线性模型模块
- `sklearn.metrics`：Scikit-learn库中的评估指标模块

## 步骤2：创建示例数据

```python
# 设置随机种子以确保结果可重现
np.random.seed(42)

# 创建模拟数据
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 学习时间（小时）
exam_scores = np.array([50, 55, 60, 70, 75, 80, 85, 90, 95, 98])  # 考试成绩（分）

# 添加一些随机噪声，使数据更真实
exam_scores = exam_scores + np.random.normal(0, 3, size=exam_scores.shape)
```

**为什么这样做？**
- 我们创建了模拟数据而不是使用真实数据，便于控制和理解
- 添加随机噪声使数据更接近真实世界的情况（真实数据很少完美分布在一条直线上）
- 设置随机种子确保每次运行结果一致，便于教学和调试

## 步骤3：数据可视化

```python
# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, exam_scores, color='blue', alpha=0.7)
plt.title('学习时间与考试成绩关系')
plt.xlabel('学习时间（小时）')
plt.ylabel('考试成绩（分）')
plt.grid(True, alpha=0.3)
plt.show()
```

**为什么这样做？**
- 可视化是数据分析的第一步，帮助我们直观理解数据分布和关系
- 散点图适合展示两个连续变量之间的关系
- 可以初步判断是否存在线性关系，以及是否有异常值

## 步骤4：数据预处理

```python
# 重塑数据形状以适应Scikit-learn的要求
# Scikit-learn要求特征矩阵是二维的，即使只有一个特征
X = study_hours.reshape(-1, 1)  # 从形状(10,)变为(10, 1)
y = exam_scores
```

**为什么这样做？**
- Scikit-learn的API设计要求特征矩阵X是二维的(n_samples, n_features)
- 即使只有一个特征，也需要确保它是二维数组
- `reshape(-1, 1)`表示将数组变为n行1列的二维数组

## 步骤5：创建和训练模型

```python
# 创建线性回归模型实例
model = LinearRegression()

# 训练模型（拟合数据）
model.fit(X, y)

# 获取模型参数
slope = model.coef_[0]  # 斜率（回归系数）
intercept = model.intercept_  # 截距

print(f"回归方程: y = {intercept:.2f} + {slope:.2f}x")
print(f"斜率(系数): {slope:.2f}, 截距: {intercept:.2f}")
```

**模型内部发生了什么？**
- Scikit-learn使用最小二乘法找到最佳拟合线
- 它最小化残差平方和：$\sum_{i=1}^{n} (y_i - \hat{y_i})^2$
- 计算出的斜率和截距定义了最佳拟合线

## 步骤6：模型预测

```python
# 使用模型进行预测
y_pred = model.predict(X)

# 预测一个新样本（学习8.5小时）
new_hours = np.array([[8.5]])  # 注意是二维数组
predicted_score = model.predict(new_hours)
print(f"预测学习8.5小时的考试成绩: {predicted_score[0]:.2f}分")
```

**预测原理**
- 模型使用学习到的参数：$\hat{y} = b_1 + b_2x$
- 对于新输入$x$，计算对应的$\hat{y}$值
- 这就是回归模型的预测能力

## 步骤7：模型评估

```python
# 计算决定系数 R²
r_squared = r2_score(y, y_pred)
print(f"决定系数 R²: {r_squared:.4f}")

# 手动计算 R²
ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
r_squared_manual = 1 - (ss_res / ss_tot)
print(f"手动计算的 R²: {r_squared_manual:.4f}")
```

**R²的意义**
- R²衡量模型对数据变动的解释程度
- 值越接近1，表示模型拟合越好
- 这里R²应该很高，因为我们创建的数据本身就有强线性关系

## 步骤8：结果可视化

```python
# 绘制回归线
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, exam_scores, color='blue', alpha=0.7, label='实际数据')
plt.plot(study_hours, y_pred, color='red', linewidth=2, label='回归线')
plt.scatter(new_hours, predicted_score, color='green', s=100, label='预测点 (8.5小时)')

plt.title('学习时间与考试成绩的线性回归')
plt.xlabel('学习时间（小时）')
plt.ylabel('考试成绩（分）')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加回归方程文本
equation_text = f'y = {intercept:.2f} + {slope:.2f}x\nR² = {r_squared:.4f}'
plt.text(1, 90, equation_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.show()
```

**可视化的重要性**
- 直观展示模型拟合效果
- 可以判断线性假设是否合理
- 帮助发现异常值或非线性模式

## 步骤9：模型解释

```python
# 解释模型参数
print("\n=== 模型解释 ===")
print(f"截距 (b₁ = {intercept:.2f})：理论上，学习时间为0小时时，预计考试成绩为{intercept:.2f}分")
print(f"斜率 (b₂ = {slope:.2f})：学习时间每增加1小时，考试成绩平均提高{slope:.2f}分")

# 计算相关系数
correlation = np.corrcoef(study_hours, exam_scores)[0, 1]
print(f"相关系数 r: {correlation:.4f}")
print(f"r²: {correlation**2:.4f} (应与R²一致)")
```

**模型解释的意义**
- 截距：当x=0时y的值（有时有实际意义，有时只是数学上的基准点）
- 斜率：x变化一个单位时y的平均变化量
- 相关系数r：两个变量线性关系的强度和方向

## 完整代码

```python
# 简单线性回归完整示例
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. 创建数据
np.random.seed(42)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 70, 75, 80, 85, 90, 95, 98])
exam_scores = exam_scores + np.random.normal(0, 3, size=exam_scores.shape)

# 2. 数据预处理
X = study_hours.reshape(-1, 1)
y = exam_scores

# 3. 创建和训练模型
model = LinearRegression()
model.fit(X, y)

# 4. 获取模型参数
slope = model.coef_[0]
intercept = model.intercept_
print(f"回归方程: y = {intercept:.2f} + {slope:.2f}x")

# 5. 预测
y_pred = model.predict(X)
new_hours = np.array([[8.5]])
predicted_score = model.predict(new_hours)
print(f"预测学习8.5小时的考试成绩: {predicted_score[0]:.2f}分")

# 6. 评估
r_squared = r2_score(y, y_pred)
print(f"决定系数 R²: {r_squared:.4f}")

# 7. 可视化
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, exam_scores, color='blue', alpha=0.7, label='实际数据')
plt.plot(study_hours, y_pred, color='red', linewidth=2, label='回归线')
plt.scatter(new_hours, predicted_score, color='green', s=100, label='预测点 (8.5小时)')
plt.title('学习时间与考试成绩的线性回归')
plt.xlabel('学习时间（小时）')
plt.ylabel('考试成绩（分）')
plt.legend()
plt.grid(True, alpha=0.3)

equation_text = f'y = {intercept:.2f} + {slope:.2f}x\nR² = {r_squared:.4f}'
plt.text(1, 90, equation_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.show()

# 8. 解释
print("\n=== 模型解释 ===")
print(f"截距 (b₁ = {intercept:.2f})：理论上，学习时间为0小时时，预计考试成绩为{intercept:.2f}分")
print(f"斜率 (b₂ = {slope:.2f})：学习时间每增加1小时，考试成绩平均提高{slope:.2f}分")

correlation = np.corrcoef(study_hours, exam_scores)[0, 1]
print(f"相关系数 r: {correlation:.4f}")
print(f"r²: {correlation**2:.4f} (应与R²一致)")
```

## 关键理解点

1. **线性回归的本质**：找到一条最佳直线来描述两个变量之间的关系
2. **最小二乘法**：通过最小化预测值与真实值之间的平方误差和来找到最佳参数
3. **模型评估**：R²衡量模型对数据变动的解释程度，值越接近1越好
4. **相关系数r**：衡量两个变量线性关系的强度和方向
5. **模型解释**：斜率和截距都有实际意义，可以解释变量之间的关系


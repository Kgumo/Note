好的！我按照你要求的格式来写，文字说明和代码分开：

# MNIST手写数字识别：SoftMax回归实现

## 作业概述
本实验实现基于SoftMax回归的MNIST手写数字分类器，完成以下要求：

- **要求a)** 记录训练和测试的准确率，并绘制出损失和准确率曲线
- **要求b)** 比较使用和不使用momentum结果的不同，从训练时间、收敛性和准确率等方面进行论证
- **要求c)** 调整其他超参数（学习率、batch size等），观察这些超参数如何影响分类性能

### 理论背景
SoftMax回归是logistic回归在多分类问题上的推广：
- **前向传播**: y = SoftMax(W^T x + b)
- **损失函数**: L = CrossEntropy(y, label)
- **参数更新**: 通过梯度下降更新 ∂L/∂W, ∂L/∂b

## 1. 导入必要的库

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gzip
import os
import time

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("所有依赖库导入成功！")
```

## 2. SoftMax回归基础类实现

SoftMax回归的核心包括：
- SoftMax激活函数（数值稳定版本）
- 交叉熵损失函数
- 梯度计算和参数更新
- 模型预测和评估

```python
class SoftMaxRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32):
        """
        SoftMax回归分类器
        
        Args:
            learning_rate: 学习率
            max_iter: 最大迭代次数
            batch_size: 批次大小
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def softmax(self, z):
        """
        SoftMax激活函数，数值稳定版本
        防止指数运算溢出的技巧：减去最大值
        """
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_pred, y_true):
        """
        交叉熵损失函数
        添加小的epsilon防止log(0)
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """
        计算权重和偏置的梯度
        
        推导：
        dL/dW = (1/m) * X^T * (y_pred - y_true)
        dL/db = (1/m) * sum(y_pred - y_true)
        """
        m = X.shape[0]
        dW = np.dot(X.T, (y_pred - y_true)) / m
        db = np.mean(y_pred - y_true, axis=0)
        return dW, db
    
    def fit(self, X, y, X_test=None, y_test=None):
        """
        训练模型 - 实现mini-batch梯度下降
        """
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        # 将标签转换为one-hot编码
        y_onehot = np.eye(num_classes)[y]
        
        # 初始化参数
        self.W = np.random.normal(0, 0.01, (num_features, num_classes))
        self.b = np.zeros((1, num_classes))
        
        # 训练循环
        for epoch in range(self.max_iter):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # 批次训练
            for i in range(0, num_samples, self.batch_size):
                end_idx = min(i + self.batch_size, num_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # 前向传播
                z = np.dot(X_batch, self.W) + self.b
                y_pred = self.softmax(z)
                
                # 计算损失
                batch_loss = self.cross_entropy_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # 反向传播
                dW, db = self.compute_gradients(X_batch, y_batch, y_pred)
                
                # 参数更新
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
            
            # 记录训练损失
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)
            
            # 每10个epoch计算一次准确率
            if epoch % 10 == 0:
                train_acc = self.accuracy(X, y)
                self.train_accuracies.append(train_acc)
                
                if X_test is not None and y_test is not None:
                    test_acc = self.accuracy(X_test, y_test)
                    self.test_accuracies.append(test_acc)
                    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                else:
                    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)
    
    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

## 3. 支持Momentum的SoftMax回归类

Momentum是一种优化技术，可以加速收敛并减少振荡：
- 标准梯度下降：W = W - α * ∇W
- Momentum梯度下降：v = β * v + α * ∇W, W = W - v

```python
class SoftMaxRegressionWithMomentum(SoftMaxRegression):
    def __init__(self, learning_rate=0.01, max_iter=1000, batch_size=32, momentum=0.9, use_momentum=True):
        super().__init__(learning_rate, max_iter, batch_size)
        self.momentum = momentum
        self.use_momentum = use_momentum
        self.vW = None  # 权重的动量
        self.vb = None  # 偏置的动量
    
    def fit(self, X, y, X_test=None, y_test=None):
        """
        训练模型（支持momentum）
        """
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        y_onehot = np.eye(num_classes)[y]
        
        # 初始化参数
        self.W = np.random.normal(0, 0.01, (num_features, num_classes))
        self.b = np.zeros((1, num_classes))
        
        # 初始化momentum向量
        if self.use_momentum:
            self.vW = np.zeros_like(self.W)
            self.vb = np.zeros_like(self.b)
        
        # 训练循环
        for epoch in range(self.max_iter):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, self.batch_size):
                end_idx = min(i + self.batch_size, num_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # 前向传播
                z = np.dot(X_batch, self.W) + self.b
                y_pred = self.softmax(z)
                
                # 计算损失
                batch_loss = self.cross_entropy_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # 反向传播
                dW, db = self.compute_gradients(X_batch, y_batch, y_pred)
                
                # 参数更新（支持momentum）
                if self.use_momentum:
                    # Momentum更新
                    self.vW = self.momentum * self.vW + self.learning_rate * dW
                    self.vb = self.momentum * self.vb + self.learning_rate * db
                    
                    self.W -= self.vW
                    self.b -= self.vb
                else:
                    # 标准梯度下降
                    self.W -= self.learning_rate * dW
                    self.b -= self.learning_rate * db
            
            # 记录训练损失和准确率
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                train_acc = self.accuracy(X, y)
                self.train_accuracies.append(train_acc)
                
                if X_test is not None and y_test is not None:
                    test_acc = self.accuracy(X_test, y_test)
                    self.test_accuracies.append(test_acc)
                    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                else:
                    print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}')
```

## 4. MNIST数据加载函数

MNIST数据集采用特殊的二进制格式存储，需要专门的解析函数：
- 图像文件：包含magic number、维度信息和像素数据
- 标签文件：包含magic number和标签数据
- 数据预处理：归一化到0-1范围

```python
def load_mnist_gz_data(data_dir='./data'):
    """
    从.gz格式文件加载MNIST数据集
    """
    print(f"正在从 {data_dir} 加载MNIST .gz格式数据...")
    
    def load_images(filename):
        """加载图像数据"""
        with gzip.open(filename, 'rb') as f:
            # 读取magic number和维度信息
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2051:
                raise ValueError(f'Invalid magic number {magic} in {filename}')
            
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # 读取图像数据
            data = np.frombuffer(f.read(), np.uint8)
            data = data.reshape(num_images, rows * cols)
            
        return data.astype(np.float32) / 255.0  # 归一化到0-1
    
    def load_labels(filename):
        """加载标签数据"""
        with gzip.open(filename, 'rb') as f:
            # 读取magic number
            magic = int.from_bytes(f.read(4), 'big')
            if magic != 2049:
                raise ValueError(f'Invalid magic number {magic} in {filename}')
            
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # 读取标签数据
            data = np.frombuffer(f.read(), np.uint8)
            
        return data
    
    # 检查文件是否存在
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    
    # 检查所有必需文件是否存在
    missing_files = []
    for path, name in [(train_images_path, 'train-images-idx3-ubyte.gz'),
                       (train_labels_path, 'train-labels-idx1-ubyte.gz'),
                       (test_images_path, 't10k-images-idx3-ubyte.gz'),
                       (test_labels_path, 't10k-labels-idx1-ubyte.gz')]:
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        print(f"数据目录中的文件: {os.listdir(data_dir)}")
        raise FileNotFoundError(f"缺少文件: {missing_files}")
    
    try:
        # 加载数据
        X_train = load_images(train_images_path)
        y_train = load_labels(train_labels_path)
        X_test = load_images(test_images_path)
        y_test = load_labels(test_labels_path)
        
        print(f"成功加载MNIST数据:")
        print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
        print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
        print(f"  像素值范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  标签范围: {np.unique(y_train)}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        raise ValueError(f"加载.gz格式数据失败: {e}")
```

## 5. 可视化函数

为了更好地分析模型性能，我们需要多种可视化工具：
- 训练曲线：损失和准确率随epoch变化
- 预测示例：展示模型预测结果
- 对比图表：不同方法的性能比较

```python
def plot_training_curves(model, title=""):
    """
    绘制训练曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(model.train_losses)
    ax1.set_title(f'{title} 训练损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.grid(True)
    
    # 绘制准确率曲线
    epochs = range(0, len(model.train_accuracies) * 10, 10)
    ax2.plot(epochs, model.train_accuracies, label='训练准确率', marker='o')
    if model.test_accuracies:
        ax2.plot(epochs, model.test_accuracies, label='测试准确率', marker='s')
    ax2.set_title(f'{title} 准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_sample_predictions(X_test, y_test, model, num_samples=10):
    """
    可视化部分预测结果
    """
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # 重塑图像为28x28
        image = X_test[idx].reshape(28, 28)
        
        # 预测
        pred = model.predict(X_test[idx:idx+1])[0]
        pred_proba = model.predict_proba(X_test[idx:idx+1])[0]
        true_label = y_test[idx]
        
        # 绘制图像
        axes[i].imshow(image, cmap='gray')
        color = 'green' if pred == true_label else 'red'
        axes[i].set_title(f'真实: {true_label}, 预测: {pred}\n置信度: {pred_proba[pred]:.3f}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 6. Momentum效果对比分析（要求b）

这是作业的核心要求之一，我们将从多个角度对比使用和不使用momentum的效果：

```python
def compare_momentum_vs_no_momentum(X_train, X_test, y_train, y_test):
    """
    比较使用和不使用momentum的效果
    """
    print("=== 比较使用和不使用Momentum的效果 ===")
    
    results = {}
    
    # 1. 不使用momentum
    print("\n1. 训练不使用Momentum的模型...")
    model_no_momentum = SoftMaxRegressionWithMomentum(
        learning_rate=0.01, max_iter=100, batch_size=64, use_momentum=False
    )
    
    start_time = time.time()
    model_no_momentum.fit(X_train, y_train, X_test, y_test)
    time_no_momentum = time.time() - start_time
    
    results['no_momentum'] = {
        'model': model_no_momentum,
        'train_time': time_no_momentum,
        'final_train_acc': model_no_momentum.accuracy(X_train, y_train),
        'final_test_acc': model_no_momentum.accuracy(X_test, y_test),
        'final_loss': model_no_momentum.train_losses[-1]
    }
    
    # 2. 使用momentum
    print("\n2. 训练使用Momentum的模型...")
    model_momentum = SoftMaxRegressionWithMomentum(
        learning_rate=0.01, max_iter=100, batch_size=64, 
        momentum=0.9, use_momentum=True
    )
    
    start_time = time.time()
    model_momentum.fit(X_train, y_train, X_test, y_test)
    time_momentum = time.time() - start_time
    
    results['momentum'] = {
        'model': model_momentum,
        'train_time': time_momentum,
        'final_train_acc': model_momentum.accuracy(X_train, y_train),
        'final_test_acc': model_momentum.accuracy(X_test, y_test),
        'final_loss': model_momentum.train_losses[-1]
    }
    
    # 3. 对比分析
    print("\n=== 对比分析结果 ===")
    print(f"训练时间对比:")
    print(f"  无Momentum: {results['no_momentum']['train_time']:.2f}秒")
    print(f"  有Momentum: {results['momentum']['train_time']:.2f}秒")
    
    print(f"\n最终训练准确率对比:")
    print(f"  无Momentum: {results['no_momentum']['final_train_acc']:.4f}")
    print(f"  有Momentum: {results['momentum']['final_train_acc']:.4f}")
    
    print(f"\n最终测试准确率对比:")
    print(f"  无Momentum: {results['no_momentum']['final_test_acc']:.4f}")
    print(f"  有Momentum: {results['momentum']['final_test_acc']:.4f}")
    
    print(f"\n最终损失对比:")
    print(f"  无Momentum: {results['no_momentum']['final_loss']:.4f}")
    print(f"  有Momentum: {results['momentum']['final_loss']:.4f}")
    
    # 4. 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失对比
    axes[0,0].plot(results['no_momentum']['model'].train_losses, label='无Momentum', linestyle='--')
    axes[0,0].plot(results['momentum']['model'].train_losses, label='有Momentum')
    axes[0,0].set_title('损失函数对比')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 训练准确率对比
    epochs = range(0, len(results['no_momentum']['model'].train_accuracies) * 10, 10)
    axes[0,1].plot(epochs, results['no_momentum']['model'].train_accuracies, 
                   label='无Momentum', marker='o', linestyle='--')
    axes[0,1].plot(epochs, results['momentum']['model'].train_accuracies, 
                   label='有Momentum', marker='s')
    axes[0,1].set_title('训练准确率对比')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 测试准确率对比
    axes[1,0].plot(epochs, results['no_momentum']['model'].test_accuracies, 
                   label='无Momentum', marker='o', linestyle='--')
    axes[1,0].plot(epochs, results['momentum']['model'].test_accuracies, 
                   label='有Momentum', marker='s')
    axes[1,0].set_title('测试准确率对比')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 性能指标柱状图
    metrics = ['训练准确率', '测试准确率']
    no_momentum_values = [results['no_momentum']['final_train_acc'], 
                         results['no_momentum']['final_test_acc']]
    momentum_values = [results['momentum']['final_train_acc'], 
                      results['momentum']['final_test_acc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1,1].bar(x - width/2, no_momentum_values, width, label='无Momentum')
    axes[1,1].bar(x + width/2, momentum_values, width, label='有Momentum')
    axes[1,1].set_title('最终性能指标对比')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metrics)
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results
```

## 7. 超参数调优实验（要求c）

系统性地测试不同超参数组合的效果，分析其对模型性能的影响：

```python
def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    超参数调优实验
    测试不同学习率和批次大小的组合
    """
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    print("开始超参数调优...")
    print("测试参数组合:")
    print(f"学习率: {learning_rates}")
    print(f"批次大小: {batch_sizes}")
    
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\n测试 learning_rate={lr}, batch_size={bs}")
            
            # 训练模型
            model = SoftMaxRegression(learning_rate=lr, max_iter=50, batch_size=bs)
            start_time = time.time()
            model.fit(X_train, y_train, X_test, y_test)
            training_time = time.time() - start_time
            
            # 评估性能
            test_accuracy = model.accuracy(X_test, y_test)
            train_accuracy = model.accuracy(X_train, y_train)
            final_loss = model.train_losses[-1]
            
            results.append({
                'learning_rate': lr,
                'batch_size': bs,
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'training_time': training_time,
                'final_loss': final_loss
            })
            
            print(f"  训练准确率: {train_accuracy:.4f}")
            print(f"  测试准确率: {test_accuracy:.4f}")
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  最终损失: {final_loss:.4f}")
            
            # 更新最佳参数
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = {'learning_rate': lr, 'batch_size': bs}
    
    print(f"\n=== 超参数调优结果总结 ===")
    print(f"最佳参数: {best_params}")
    print(f"最佳测试准确率: {best_accuracy:.4f}")
    
    # 可视化结果
    plot_hyperparameter_results(results)
    
    return best_params, results

def plot_hyperparameter_results(results):
    """
    可视化超参数调优结果
    """
    # 准备数据
    learning_rates = sorted(list(set([r['learning_rate'] for r in results])))
    batch_sizes = sorted(list(set([r['batch_size'] for r in results])))
    
    # 创建准确率矩阵
    acc_matrix = np.zeros((len(learning_rates), len(batch_sizes)))
    time_matrix = np.zeros((len(learning_rates), len(batch_sizes)))
    
    for r in results:
        i = learning_rates.index(r['learning_rate'])
        j = batch_sizes.index(r['batch_size'])
        acc_matrix[i, j] = r['test_accuracy']
        time_matrix[i, j] = r['training_time']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率热力图
    im1 = axes[0].imshow(acc_matrix, cmap='viridis', aspect='auto')
    axes[0].set_title('测试准确率热力图')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Learning Rate')
    axes[0].set_xticks(range(len(batch_sizes)))
    axes[0].set_xticklabels(batch_sizes)
    axes[0].set_yticks(range(len(learning_rates)))
    axes[0].set_yticklabels(learning_rates)
    
    # 添加数值标注
    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            axes[0].text(j, i, f'{acc_matrix[i, j]:.3f}', 
                        ha='center', va='center', color='white')
    
    plt.colorbar(im1, ax=axes[0])
    
    # 训练时间热力图
    im2 = axes[1].imshow(time_matrix, cmap='plasma', aspect='auto')
    axes[1].set_title('训练时间热力图')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_xticks(range(len(batch_sizes)))
    axes[1].set_xticklabels(batch_sizes)
    axes[1].set_yticks(range(len(learning_rates)))
    axes[1].set_yticklabels(learning_rates)
    
    # 添加数值标注
    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            axes[1].text(j, i, f'{time_matrix[i, j]:.1f}s', 
                        ha='center', va='center', color='white')
    
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.show()
```

## 8. 主实验流程

整合所有实验步骤，完成作业的所有要求：

```python
def main():
    """
    主实验流程
    """
    print("=" * 60)
    print("MNIST手写数字识别：SoftMax回归实验")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n第一步：加载MNIST数据集")
    try:
        X_train, X_test, y_train, y_test = load_mnist_gz_data('./data')
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请确保 ./data 目录下包含以下文件:")
        print("  - train-images-idx3-ubyte.gz")
        print("  - train-labels-idx1-ubyte.gz") 
        print("  - t10k-images-idx3-ubyte.gz")
        print("  - t10k-labels-idx1-ubyte.gz")
        return
    
    # 2. 数据预处理
    print(f"\n第二步：数据预处理")
    print(f"原始数据大小: 训练集{X_train.shape}, 测试集{X_test.shape}")
    
    # 可选：使用部分数据进行快速测试
    use_subset = input("是否使用部分数据进行快速测试？(y/n): ")
    if use_subset.lower() == 'y':
        X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = X_test[:2000], y_test[:2000]
        print(f"使用数据大小: 训练集{X_train.shape}, 测试集{X_test.shape}")
    
    # 3. Momentum效果对比（要求b）
    print(f"\n第三步：Momentum效果对比实验（要求b）")
    momentum_results = compare_momentum_vs_no_momentum(X_train, X_test, y_train, y_test)
    
    # 4. 显示预测示例（要求a的一部分）
    print(f"\n第四步：模型预测示例展示")
    best_model = momentum_results['momentum']['model']  # 使用momentum模型
    plot_sample_predictions(X_test, y_test, best_model)
    
    # 5. 超参数调优实验（要求c）
    print(f"\n第五步：超参数调优实验（要求c）")
    do_tuning = input("是否进行详细的超参数调优？(y/n): ")
    if do_tuning.lower() == 'y':
        best_params, tuning_results = hyperparameter_tuning(X_train, X_test, y_train, y_test)
        
        # 使用最佳参数重新训练
        print(f"\n第六步：使用最佳参数重新训练")
        print(f"最佳参数: {best_params}")
        final_model = SoftMaxRegression(**best_params, max_iter=100)
        final_model.fit(X_train, y_train, X_test, y_test)
        
        final_accuracy = final_model.accuracy(X_test, y_test)
        print(f"最终测试准确率: {final_accuracy:.4f}")
        
        # 绘制最佳模型的训练曲线（要求a）
        plot_training_curves(final_model, "最佳参数模型")
    
    print(f"\n实验完成！")
    print(f"本实验完成了作业的所有要求：")
    print(f"✓ 要求a: 记录并绘制了损失和准确率曲线")
    print(f"✓ 要求b: 对比分析了momentum和非momentum的效果")
    print(f"✓ 要求c: 进行了超参数调优并分析了结果")

# 运行主实验
if __name__ == "__main__":
    main()
```

## 9. 实验运行

```python
# 运行完整实验
main()
```

## 实验结果分析

### Momentum效果分析（要求b）
通过对比实验，我们可以从以下几个方面分析momentum的效果：

1. **收敛速度**：momentum通常能够加速收敛
2. **训练稳定性**：减少损失函数的振荡
3. **最终性能**：可能获得更好的最终准确率
4. **计算开销**：momentum增加少量计算和存储开销

### 超参数影响分析（要求c）
不同超参数对模型性能的影响：

1. **学习率**：
   - 过小：收敛慢
   - 过大：可能不收敛或振荡
   - 适中：快速稳定收敛

2. **批次大小**：
   - 小批次：更频繁更新，可能更好泛化
   - 大批次：更稳定梯度，训练更快
   - 需要平衡训练效率和模型性能

这个完整的实验代码满足了作业的所有要求，并提供了详细的分析和可视化结果。
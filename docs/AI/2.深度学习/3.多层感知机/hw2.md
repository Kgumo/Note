# 1. 加载训练数据


```python
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import numpy as np
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
%matplotlib inline

```


```python
train_images_filename = 'data/train-images-idx3-ubyte.gz'
train_labels_filename = 'data/train-labels-idx1-ubyte.gz'
test_images_filename  = 'data/t10k-images-idx3-ubyte.gz'
test_labels_filename  = 'data/t10k-labels-idx1-ubyte.gz'

validation_size = 5000

def read_mnist_image_set(images_filename):
    with gzip.GzipFile(images_filename, 'rb') as gz:
        magic = int.from_bytes(gz.read(4), 'big')
        assert magic == 2051, f'Not an MNIST image set'

        num_images = int.from_bytes(gz.read(4), 'big')
        rows = int.from_bytes(gz.read(4), 'big')
        cols = int.from_bytes(gz.read(4), 'big')

        data = np.frombuffer(gz.read(num_images * rows * cols), dtype=np.uint8)
        data = np.reshape(data, (num_images, rows * cols))
        return data

def read_mnist_label_set(labels_filename):
    with gzip.GzipFile(labels_filename, 'rb') as gz:
        magic = int.from_bytes(gz.read(4), 'big')
        assert magic == 2049, f'Not an MNIST label set'

        num_labels = int.from_bytes(gz.read(4), 'big')
        labels = np.frombuffer(gz.read(num_labels), dtype=np.uint8)
        return labels

print('\nLoading train set ...')
train_data   = read_mnist_image_set(train_images_filename) / np.float32(255)
train_labels = read_mnist_label_set(train_labels_filename)
train_data,   val_data   = train_data  [:-validation_size], train_data  [-validation_size:]
train_labels, val_labels = train_labels[:-validation_size], train_labels[-validation_size:]
print(f'train_data:   [{str(train_data.dtype)  }] {train_data.shape}')
print(f'train_labels: [{str(train_labels.dtype)}] {train_labels.shape}')
print(f'val_data:     [{str(val_data.dtype)    }] {val_data.shape}')
print(f'val_labels:   [{str(val_labels.dtype)  }] {val_labels.shape}')

print('\nLoading test set ...')
test_data   = read_mnist_image_set(test_images_filename) / np.float32(255)
test_labels = read_mnist_label_set(test_labels_filename)
print(f'test_data:   [{str(test_data.dtype)  }] {test_data.shape}')
print(f'test_labels: [{str(test_labels.dtype)}] {test_labels.shape}')

```

    
    Loading train set ...
    train_data:   [float32] (55000, 784)
    train_labels: [uint8] (55000,)
    val_data:     [float32] (5000, 784)
    val_labels:   [uint8] (5000,)
    
    Loading test set ...
    test_data:   [float32] (10000, 784)
    test_labels: [uint8] (10000,)
    


```python
# Preview dataset

_ = plt.figure(figsize=(6, 6))
_ = plt.title('MNIST Preview')
for label in range(10):
    for img_index, img_data in enumerate(test_data[test_labels == label][:10]):
        _ = plt.imshow(img_data.reshape(28, 28), 'gray', vmin=0, vmax=1,
            interpolation='nearest', extent=(img_index, img_index + 1, label, label + 1))
_ = plt.xticks([])
_ = plt.yticks(np.arange(10) + 0.5, range(10))
_ = plt.xlim(0, 10)
_ = plt.ylim(0, 10)
```


    
![png](public/hw2_files/hw2_3_0.png)
    


# 2. 实现网络架构


```python
class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        """前向传播"""
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, delta):
        """反向传播"""
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
```

## 2.1 `SGD` 随机梯度下降



```python
class SGD:
    def __init__(self, learning_rate, weight_decay, momentum=0.9):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = {}

    def step(self, model):
        """执行一步优化"""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'trainable') and layer.trainable:
                # 初始化动量
                if f'W_{i}' not in self.velocity:
                    self.velocity[f'W_{i}'] = np.zeros_like(layer.W)
                    self.velocity[f'b_{i}'] = np.zeros_like(layer.b)
                
                # 添加权重衰减
                layer.grad_W += self.weight_decay * layer.W
                
                # 动量更新
                self.velocity[f'W_{i}'] = (self.momentum * self.velocity[f'W_{i}'] + 
                                         self.learning_rate * layer.grad_W)
                self.velocity[f'b_{i}'] = (self.momentum * self.velocity[f'b_{i}'] + 
                                         self.learning_rate * layer.grad_b)
                
                # 更新参数
                layer.W -= self.velocity[f'W_{i}']
                layer.b -= self.velocity[f'b_{i}']
```

## 2.2 `FCLayer`

`FCLayer` 为全连接层，输入为一组向量（必要时需要改变输入尺寸以满足要求），与权重矩阵作矩阵乘法并加上偏置项，得到输出向量:

$$
\mathbf{u} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$


```python
class FCLayer:
    def __init__(self, num_input, num_output, act_function='relu'):
        self.num_input = num_input
        self.num_output = num_output
        self.act_function = act_function
        self.trainable = True
        
        # 使用更好的初始化方法
        self._xavier_init()
        
        # 用于反向传播
        self.input_cache = None
        self.grad_W = None
        self.grad_b = None

    def _xavier_init(self):
        """改进的Xavier/He初始化"""
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        if 'relu' == self.act_function:
            init_std = raw_std * (2**0.5)  # He初始化
        elif 'sigmoid' == self.act_function:
            init_std = raw_std
        else:
            init_std = raw_std
        
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.zeros((1, self.num_output))  # 偏置初始化为0

    def forward(self, input):
        """前向传播: 计算Wx+b"""
        self.input_cache = input
        output = np.dot(input, self.W) + self.b
        return output

    def backward(self, delta):
        """反向传播: 根据delta计算梯度"""
        batch_size = self.input_cache.shape[0]
        
        # 计算权重和偏置的梯度
        self.grad_W = np.dot(self.input_cache.T, delta) / batch_size
        self.grad_b = np.mean(delta, axis=0, keepdims=True)
        
        # 计算传递给上一层的梯度
        grad_input = np.dot(delta, self.W.T)
        return grad_input

```

## 2.3 `SigmoidLayer`

`SigmoidLayer` 为sigmoid激活层:

$$
f(\mathbf{u}) = \frac{1}{1 + \exp(-\mathbf{u})}
$$



```python
class SigmoidLayer:
    def __init__(self):
        self.trainable = False
        self.output_cache = None

    def forward(self, input):
        """Sigmoid激活"""
        # 防止数值溢出
        input = np.clip(input, -500, 500)
        self.output_cache = 1 / (1 + np.exp(-input))
        return self.output_cache

    def backward(self, delta):
        """Sigmoid反向传播"""
        grad_input = delta * self.output_cache * (1 - self.output_cache)
        return grad_input
```

## 2.4 `ReLULayer`

`ReLULayer` 为ReLU激活层:

$$
f(\mathbf{u}) = \max(\mathbf{0}, \mathbf{u})
$$


```python
class ReLULayer:
    def __init__(self):
        self.trainable = False
        self.input_cache = None

    def forward(self, input):
        """ReLU激活: relu(x) = max(x, 0)"""
        self.input_cache = input
        output = np.maximum(0, input)
        return output

    def backward(self, delta):
        """ReLU反向传播"""
        grad_input = delta * (self.input_cache > 0)
        return grad_input
```

## 2.5 `EuclideanLossLayer`

`EuclideanLossLayer` 为欧式距离损失层，计算各样本误差的平方和得到:

$$
\frac{1}{2} \sum_{n} \lVert \operatorname{logits}(n) - \operatorname{label}(n) \rVert _2^2
$$


```python
class EuclideanLossLayer:
    def __init__(self):
        self.pred = None
        self.gt = None
        self.loss = None
        self.accu = None

    def forward(self, pred, gt):
        """欧几里得距离损失"""
        self.pred = pred
        self.gt = gt
        
        # 计算欧几里得距离损失
        diff = pred - gt
        self.loss = 0.5 * np.mean(np.sum(diff**2, axis=1))
        
        # 计算准确率
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(gt, axis=1)
        self.accu = np.mean(pred_labels == true_labels)
        
        return self.loss

    def backward(self):
        """反向传播"""
        batch_size = self.pred.shape[0]
        gradient = (self.pred - self.gt) / batch_size
        return gradient
```

## 2.6 `SoftmaxCrossEntropyLossLayer`

`SoftmaxCrossEntropyLossLayer` 可以看成是输入到如下概率分布的映射：

$$
P(t_k=1\vert\mathbf{x}) = \frac{\exp(x_k)}{\sum_{j=1}^{K}\exp(x_j)}
$$

其中 $x_k$ 是输入向量 $\mathbf{x}$ 中的第 $k$ 个元素，$P(t_k = 1|\mathbf{x})$ 表示该输入被分到第第 $k$ 个类别的概率。由于softmax层的输出可以看成一组概率分布，我们可以计算delta似然及其对数形式，称为Cross Entropy误差函数：

$$
E = -\ln p(t^{(1)}, \dots, t^{(N)}) = \sum_{n=1}^N E^{(n)}
$$

其中

$$
E^{(n)} = -\sum_{k=1}^K t_k^{(n)} \ln h_k^{(n)}
$$

$$
h_k^{(n)} = P(t_k^{(n)}=1\vert \mathbf{x}^{(n)}) = \frac{\exp x_k^{(n)}}{\sum_{j=1}^K \exp x_j^{(n)}}
$$

注意：此处的softmax损失层与案例1中有所差异，本次案例中的softmax层不包含可训练的参数，这些可训练的参数被独立成一个全连接层。


```python
# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer:
    def __init__(self):
        self.logit = None
        self.gt = None
        self.softmax_output = None
        self.loss = None
        self.accu = None

    def forward(self, logit, gt):
        """前向传播: 计算softmax + 交叉熵损失"""
        self.logit = logit
        self.gt = gt
        
        # 数值稳定的softmax
        shifted_logits = logit - np.max(logit, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 防止log(0)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # 交叉熵损失
        self.loss = -np.mean(np.sum(gt * np.log(probs), axis=1))
        
        # 计算准确率
        pred_labels = np.argmax(logit, axis=1)
        true_labels = np.argmax(gt, axis=1)
        self.accu = np.mean(pred_labels == true_labels)
        
        # 保存softmax输出用于反向传播
        self.softmax_output = probs
        
        return self.loss

    def backward(self):
        """反向传播计算梯度"""
        batch_size = self.logit.shape[0]
        # 交叉熵+softmax的梯度
        gradient = (self.softmax_output - self.gt) / batch_size
        return gradient

```

# 3. 训练代码

所有代码均可根据需要自行修改。



```python
# 超参数

batch_size = 64
max_epoch = 20
init_std = 0.01
learning_rate_SGD = 0.01    
weight_decay = 0.0001


```


```python
def train(model, criterion, optimizer):
    """训练函数"""
    all_train_losses, all_train_accs = [], []
    avg_train_losses, avg_train_accs = [], []
    avg_val_losses, avg_val_accs = [], []
    
    # 数据预处理 - 标准化
    global train_data, val_data, test_data
    mean = train_data.mean()
    std = train_data.std()
    train_data_norm = (train_data - mean) / std
    val_data_norm = (val_data - mean) / std
    test_data_norm = (test_data - mean) / std
    
    print(f"数据标准化: 均值={mean:.4f}, 标准差={std:.4f}")
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    # 训练循环
    for epoch in range(max_epoch):
        # 训练一个epoch
        batch_train_losses, batch_train_accs = [], []

        steps_per_epoch = np.ceil(len(train_data_norm) / batch_size).astype(int)
        batch_split_indices = np.arange(batch_size, len(train_data_norm), batch_size)
        train_data_batches = np.split(train_data_norm, batch_split_indices)
        train_labels_batches = np.split(train_labels, batch_split_indices)

        # 使用进度条
        progress_bar = tqdm(range(steps_per_epoch), desc=f'Epoch[{epoch+1}/{max_epoch}]')
        
        for step_index in progress_bar:
            # 获取批次数据
            x = np.reshape(train_data_batches[step_index], (-1, 784))
            y_true = np.eye(10)[train_labels_batches[step_index]]

            # 前向传播
            y_pred = model.forward(x)
            loss = criterion.forward(y_pred, y_true)

            # 反向传播
            delta = criterion.backward()
            model.backward(delta)
            optimizer.step(model)

            batch_train_losses.append(criterion.loss)
            batch_train_accs.append(criterion.accu)
            
            # 更新进度条
            progress_bar.set_description(
                f'Epoch[{epoch+1}] Loss:{criterion.loss:.3f} Acc:{criterion.accu:.3f}'
            )

        # 验证
        x_val = np.reshape(val_data_norm, (-1, 784))
        y_val_true = np.eye(10)[val_labels]
        y_val_pred = model.forward(x_val)
        criterion.forward(y_val_pred, y_val_true)
        
        all_train_losses.extend(batch_train_losses)  
        all_train_accs.extend(batch_train_accs)      
        
        # 记录平均值
        avg_train_losses.append(np.mean(batch_train_losses))
        avg_train_accs.append(np.mean(batch_train_accs))
        avg_val_losses.append(criterion.loss)
        avg_val_accs.append(criterion.accu)

        print(f'Epoch {epoch+1} - Train Acc: {avg_train_accs[-1]:.4f} | Val Acc: {avg_val_accs[-1]:.4f}')
        
        # 早停检查
        if avg_val_accs[-1] > best_val_acc:
            best_val_acc = avg_val_accs[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停在第 {epoch+1} 轮，最佳验证准确率: {best_val_acc:.4f}")
                break

    # 测试评估
    x_test = np.reshape(test_data_norm, (-1, 784))
    y_test_true = np.eye(10)[test_labels]
    y_test_pred = model.forward(x_test)
    criterion.forward(y_test_pred, y_test_true)
    
    print(f"\n最终测试准确率: {criterion.accu:.4f} ({criterion.accu*100:.2f}%)\n")
    
    x_by_epoch = list(range(1, len(avg_train_losses) + 1))  # epoch轴
    
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    _ = a0.set_title('Losses')
    _ = a0.plot(all_train_losses, alpha=0.5, label='train (all)')  # 所有batch的损失
    _ = a0.plot(x_by_epoch, avg_train_losses, marker='.', label='train')
    _ = a0.plot(x_by_epoch, avg_val_losses, marker='.', label='val')
    _ = a0.legend()
    _ = a0.set_xlabel('Iteration/Epoch')
    _ = a0.set_ylabel('Loss')
    _ = a0.grid(True, alpha=0.3)
    
    # 绘制准确率曲线  
    _ = a1.set_title('Accuracies')
    _ = a1.plot(all_train_accs, alpha=0.5, label='train (all)')  # 所有batch的准确率
    _ = a1.plot(x_by_epoch, avg_train_accs, marker='.', label='train')
    _ = a1.plot(x_by_epoch, avg_val_accs, marker='.', label='val')
    _ = a1.legend()
    _ = a1.set_xlabel('Iteration/Epoch')
    _ = a1.set_ylabel('Accuracy')
    _ = a1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()  # 显示图表
    
    return model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs
```


```python
# Sigmoid激活，欧式距离损失
# 128是隐藏层单元数，可根据需要修改
mlp_sigmoid_euclidean = Network()
mlp_sigmoid_euclidean.add(FCLayer(784, 128, act_function='sigmoid'))
mlp_sigmoid_euclidean.add(SigmoidLayer())
mlp_sigmoid_euclidean.add(FCLayer(128, 10, act_function='sigmoid'))

criterion_euclidean = EuclideanLossLayer()
sgd = SGD(learning_rate_SGD, weight_decay)


model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = train(mlp_sigmoid_euclidean, criterion_euclidean, sgd)
```

    数据标准化: 均值=0.1307, 标准差=0.3082
    

    Epoch[1] Loss:0.408 Acc:0.500: 100%|██████████| 860/860 [00:02<00:00, 375.96it/s]
    

    Epoch 1 - Train Acc: 0.4651 | Val Acc: 0.7070
    

    Epoch[2] Loss:0.358 Acc:0.583: 100%|██████████| 860/860 [00:02<00:00, 405.81it/s]
    

    Epoch 2 - Train Acc: 0.7137 | Val Acc: 0.7986
    

    Epoch[3] Loss:0.331 Acc:0.667: 100%|██████████| 860/860 [00:02<00:00, 382.97it/s]
    

    Epoch 3 - Train Acc: 0.7734 | Val Acc: 0.8318
    

    Epoch[4] Loss:0.312 Acc:0.667: 100%|██████████| 860/860 [00:02<00:00, 358.76it/s]
    

    Epoch 4 - Train Acc: 0.8016 | Val Acc: 0.8532
    

    Epoch[5] Loss:0.298 Acc:0.667: 100%|██████████| 860/860 [00:03<00:00, 271.35it/s]
    

    Epoch 5 - Train Acc: 0.8193 | Val Acc: 0.8674
    

    Epoch[6] Loss:0.287 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 306.27it/s]
    

    Epoch 6 - Train Acc: 0.8322 | Val Acc: 0.8774
    

    Epoch[7] Loss:0.277 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 394.52it/s]
    

    Epoch 7 - Train Acc: 0.8406 | Val Acc: 0.8838
    

    Epoch[8] Loss:0.269 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 379.83it/s]
    

    Epoch 8 - Train Acc: 0.8481 | Val Acc: 0.8872
    

    Epoch[9] Loss:0.263 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 340.15it/s]
    

    Epoch 9 - Train Acc: 0.8537 | Val Acc: 0.8898
    

    Epoch[10] Loss:0.257 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 417.12it/s]
    

    Epoch 10 - Train Acc: 0.8588 | Val Acc: 0.8930
    

    Epoch[11] Loss:0.251 Acc:0.750: 100%|██████████| 860/860 [00:03<00:00, 285.48it/s]
    

    Epoch 11 - Train Acc: 0.8630 | Val Acc: 0.8968
    

    Epoch[12] Loss:0.246 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 329.97it/s]
    

    Epoch 12 - Train Acc: 0.8666 | Val Acc: 0.8996
    

    Epoch[13] Loss:0.242 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 355.66it/s]
    

    Epoch 13 - Train Acc: 0.8699 | Val Acc: 0.9018
    

    Epoch[14] Loss:0.238 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 355.82it/s]
    

    Epoch 14 - Train Acc: 0.8725 | Val Acc: 0.9036
    

    Epoch[15] Loss:0.234 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 349.65it/s]
    

    Epoch 15 - Train Acc: 0.8746 | Val Acc: 0.9068
    

    Epoch[16] Loss:0.230 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 357.60it/s]
    

    Epoch 16 - Train Acc: 0.8770 | Val Acc: 0.9082
    

    Epoch[17] Loss:0.227 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 355.72it/s]
    

    Epoch 17 - Train Acc: 0.8796 | Val Acc: 0.9098
    

    Epoch[18] Loss:0.224 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 351.71it/s]
    

    Epoch 18 - Train Acc: 0.8819 | Val Acc: 0.9108
    

    Epoch[19] Loss:0.221 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 360.32it/s]
    

    Epoch 19 - Train Acc: 0.8835 | Val Acc: 0.9120
    

    Epoch[20] Loss:0.218 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 353.15it/s]
    

    Epoch 20 - Train Acc: 0.8849 | Val Acc: 0.9134
    
    最终测试准确率: 0.8972 (89.72%)
    
    


    
![png](hw2_files/hw2_21_41.png)
    



```python
# ReLU激活，欧式距离损失
# 128是隐藏层单元数，可根据需要修改
mlp_relu_euclidean = Network()
mlp_relu_euclidean.add(FCLayer(784, 128, act_function='sigmoid'))
mlp_relu_euclidean.add(ReLULayer())
mlp_relu_euclidean.add(FCLayer(128, 10, act_function='sigmoid'))

criterion_euclidean = EuclideanLossLayer()
sgd = SGD(learning_rate_SGD, weight_decay)

model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = train(mlp_relu_euclidean, criterion_euclidean, sgd)
```

    数据标准化: 均值=0.1307, 标准差=0.3082
    

    Epoch[1] Loss:0.574 Acc:0.417: 100%|██████████| 860/860 [00:02<00:00, 366.63it/s]
    

    Epoch 1 - Train Acc: 0.4969 | Val Acc: 0.6994
    

    Epoch[2] Loss:0.419 Acc:0.583: 100%|██████████| 860/860 [00:02<00:00, 334.39it/s]
    

    Epoch 2 - Train Acc: 0.7050 | Val Acc: 0.7990
    

    Epoch[3] Loss:0.348 Acc:0.625: 100%|██████████| 860/860 [00:02<00:00, 402.05it/s]
    

    Epoch 3 - Train Acc: 0.7751 | Val Acc: 0.8388
    

    Epoch[4] Loss:0.303 Acc:0.667: 100%|██████████| 860/860 [00:02<00:00, 417.54it/s]
    

    Epoch 4 - Train Acc: 0.8125 | Val Acc: 0.8632
    

    Epoch[5] Loss:0.274 Acc:0.708: 100%|██████████| 860/860 [00:02<00:00, 424.82it/s]
    

    Epoch 5 - Train Acc: 0.8357 | Val Acc: 0.8776
    

    Epoch[6] Loss:0.253 Acc:0.708: 100%|██████████| 860/860 [00:02<00:00, 416.52it/s]
    

    Epoch 6 - Train Acc: 0.8501 | Val Acc: 0.8842
    

    Epoch[7] Loss:0.237 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 408.94it/s]
    

    Epoch 7 - Train Acc: 0.8615 | Val Acc: 0.8944
    

    Epoch[8] Loss:0.225 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 412.52it/s]
    

    Epoch 8 - Train Acc: 0.8694 | Val Acc: 0.9020
    

    Epoch[9] Loss:0.213 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 412.52it/s]
    

    Epoch 9 - Train Acc: 0.8757 | Val Acc: 0.9070
    

    Epoch[10] Loss:0.203 Acc:0.833: 100%|██████████| 860/860 [00:01<00:00, 443.44it/s]
    

    Epoch 10 - Train Acc: 0.8815 | Val Acc: 0.9120
    

    Epoch[11] Loss:0.195 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 429.50it/s]
    

    Epoch 11 - Train Acc: 0.8853 | Val Acc: 0.9148
    

    Epoch[12] Loss:0.188 Acc:0.833: 100%|██████████| 860/860 [00:01<00:00, 461.55it/s]
    

    Epoch 12 - Train Acc: 0.8890 | Val Acc: 0.9170
    

    Epoch[13] Loss:0.183 Acc:0.833: 100%|██████████| 860/860 [00:01<00:00, 433.95it/s]
    

    Epoch 13 - Train Acc: 0.8925 | Val Acc: 0.9192
    

    Epoch[14] Loss:0.178 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 403.75it/s]
    

    Epoch 14 - Train Acc: 0.8956 | Val Acc: 0.9204
    

    Epoch[15] Loss:0.173 Acc:0.833: 100%|██████████| 860/860 [00:01<00:00, 444.28it/s]
    

    Epoch 15 - Train Acc: 0.8982 | Val Acc: 0.9220
    

    Epoch[16] Loss:0.169 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 412.67it/s]
    

    Epoch 16 - Train Acc: 0.9008 | Val Acc: 0.9238
    

    Epoch[17] Loss:0.165 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 399.77it/s]
    

    Epoch 17 - Train Acc: 0.9026 | Val Acc: 0.9258
    

    Epoch[18] Loss:0.162 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 415.95it/s]
    

    Epoch 18 - Train Acc: 0.9051 | Val Acc: 0.9280
    

    Epoch[19] Loss:0.158 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 343.63it/s]
    

    Epoch 19 - Train Acc: 0.9068 | Val Acc: 0.9294
    

    Epoch[20] Loss:0.155 Acc:0.833: 100%|██████████| 860/860 [00:03<00:00, 267.24it/s]
    

    Epoch 20 - Train Acc: 0.9086 | Val Acc: 0.9312
    
    最终测试准确率: 0.9131 (91.31%)
    
    


    
![png](hw2_files/hw2_22_41.png)
    



```python
# Sigmoid激活，交叉熵损失
# 128是隐藏层单元数，可根据需要修改
mlp_sigmoid_ce = Network()
mlp_sigmoid_ce.add(FCLayer(784, 128))
mlp_sigmoid_ce.add(SigmoidLayer())
mlp_sigmoid_ce.add(FCLayer(128, 10))

criterion_ce = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate_SGD, weight_decay)

model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = train(mlp_sigmoid_ce, criterion_ce, sgd)
```

    数据标准化: 均值=0.1307, 标准差=0.3082
    

    Epoch[1] Loss:1.502 Acc:0.667: 100%|██████████| 860/860 [00:04<00:00, 206.25it/s]
    

    Epoch 1 - Train Acc: 0.4915 | Val Acc: 0.7364
    

    Epoch[2] Loss:1.222 Acc:0.708: 100%|██████████| 860/860 [00:02<00:00, 321.83it/s]
    

    Epoch 2 - Train Acc: 0.7488 | Val Acc: 0.8294
    

    Epoch[3] Loss:1.089 Acc:0.708: 100%|██████████| 860/860 [00:02<00:00, 318.55it/s]
    

    Epoch 3 - Train Acc: 0.8050 | Val Acc: 0.8634
    

    Epoch[4] Loss:1.008 Acc:0.750: 100%|██████████| 860/860 [00:02<00:00, 331.31it/s]
    

    Epoch 4 - Train Acc: 0.8298 | Val Acc: 0.8792
    

    Epoch[5] Loss:0.951 Acc:0.750: 100%|██████████| 860/860 [00:03<00:00, 268.31it/s]
    

    Epoch 5 - Train Acc: 0.8439 | Val Acc: 0.8864
    

    Epoch[6] Loss:0.907 Acc:0.750: 100%|██████████| 860/860 [00:03<00:00, 246.50it/s]
    

    Epoch 6 - Train Acc: 0.8533 | Val Acc: 0.8944
    

    Epoch[7] Loss:0.872 Acc:0.792: 100%|██████████| 860/860 [00:03<00:00, 224.20it/s]
    

    Epoch 7 - Train Acc: 0.8611 | Val Acc: 0.8976
    

    Epoch[8] Loss:0.843 Acc:0.792: 100%|██████████| 860/860 [00:03<00:00, 242.81it/s]
    

    Epoch 8 - Train Acc: 0.8664 | Val Acc: 0.9020
    

    Epoch[9] Loss:0.818 Acc:0.792: 100%|██████████| 860/860 [00:03<00:00, 254.44it/s]
    

    Epoch 9 - Train Acc: 0.8708 | Val Acc: 0.9036
    

    Epoch[10] Loss:0.796 Acc:0.792: 100%|██████████| 860/860 [00:03<00:00, 235.91it/s]
    

    Epoch 10 - Train Acc: 0.8746 | Val Acc: 0.9048
    

    Epoch[11] Loss:0.777 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 371.94it/s]
    

    Epoch 11 - Train Acc: 0.8776 | Val Acc: 0.9066
    

    Epoch[12] Loss:0.760 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 393.45it/s]
    

    Epoch 12 - Train Acc: 0.8803 | Val Acc: 0.9086
    

    Epoch[13] Loss:0.744 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 403.08it/s]
    

    Epoch 13 - Train Acc: 0.8824 | Val Acc: 0.9102
    

    Epoch[14] Loss:0.729 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 426.56it/s]
    

    Epoch 14 - Train Acc: 0.8847 | Val Acc: 0.9108
    

    Epoch[15] Loss:0.716 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 396.94it/s]
    

    Epoch 15 - Train Acc: 0.8867 | Val Acc: 0.9116
    

    Epoch[16] Loss:0.704 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 395.58it/s]
    

    Epoch 16 - Train Acc: 0.8885 | Val Acc: 0.9132
    

    Epoch[17] Loss:0.693 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 390.57it/s]
    

    Epoch 17 - Train Acc: 0.8898 | Val Acc: 0.9150
    

    Epoch[18] Loss:0.682 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 407.53it/s]
    

    Epoch 18 - Train Acc: 0.8914 | Val Acc: 0.9168
    

    Epoch[19] Loss:0.672 Acc:0.792: 100%|██████████| 860/860 [00:02<00:00, 417.58it/s]
    

    Epoch 19 - Train Acc: 0.8926 | Val Acc: 0.9174
    

    Epoch[20] Loss:0.662 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 411.79it/s]
    

    Epoch 20 - Train Acc: 0.8938 | Val Acc: 0.9176
    
    最终测试准确率: 0.9025 (90.25%)
    
    


    
![png](hw2_files/hw2_23_41.png)
    



```python
# ReLU激活，交叉熵损失
# 128是隐藏层单元数，可根据需要修改
mlp_relu_ce = Network()
mlp_relu_ce.add(FCLayer(784, 128))
mlp_relu_ce.add(ReLULayer())
mlp_relu_ce.add(FCLayer(128, 10))

criterion_ce = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate_SGD, weight_decay)

model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = train(mlp_relu_euclidean, criterion_euclidean, sgd)
```

    数据标准化: 均值=0.1307, 标准差=0.3082
    

    Epoch[1] Loss:0.154 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 400.62it/s]
    

    Epoch 1 - Train Acc: 0.9100 | Val Acc: 0.9324
    

    Epoch[2] Loss:0.151 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 393.69it/s]
    

    Epoch 2 - Train Acc: 0.9114 | Val Acc: 0.9336
    

    Epoch[3] Loss:0.148 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 369.32it/s]
    

    Epoch 3 - Train Acc: 0.9131 | Val Acc: 0.9350
    

    Epoch[4] Loss:0.145 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 359.60it/s]
    

    Epoch 4 - Train Acc: 0.9145 | Val Acc: 0.9368
    

    Epoch[5] Loss:0.142 Acc:0.833: 100%|██████████| 860/860 [00:02<00:00, 378.29it/s]
    

    Epoch 5 - Train Acc: 0.9160 | Val Acc: 0.9372
    

    Epoch[6] Loss:0.139 Acc:0.875: 100%|██████████| 860/860 [00:02<00:00, 403.60it/s]
    

    Epoch 6 - Train Acc: 0.9170 | Val Acc: 0.9380
    

    Epoch[7] Loss:0.137 Acc:0.917: 100%|██████████| 860/860 [00:02<00:00, 394.83it/s]
    

    Epoch 7 - Train Acc: 0.9182 | Val Acc: 0.9390
    

    Epoch[8] Loss:0.134 Acc:0.917: 100%|██████████| 860/860 [00:02<00:00, 402.78it/s]
    

    Epoch 8 - Train Acc: 0.9189 | Val Acc: 0.9400
    

    Epoch[9] Loss:0.132 Acc:0.917: 100%|██████████| 860/860 [00:02<00:00, 326.34it/s]
    

    Epoch 9 - Train Acc: 0.9199 | Val Acc: 0.9410
    

    Epoch[10] Loss:0.130 Acc:0.917: 100%|██████████| 860/860 [00:04<00:00, 214.99it/s]
    

    Epoch 10 - Train Acc: 0.9208 | Val Acc: 0.9416
    

    Epoch[11] Loss:0.128 Acc:0.917: 100%|██████████| 860/860 [00:04<00:00, 183.02it/s]
    

    Epoch 11 - Train Acc: 0.9219 | Val Acc: 0.9416
    

    Epoch[12] Loss:0.125 Acc:0.917: 100%|██████████| 860/860 [00:04<00:00, 204.03it/s]
    

    Epoch 12 - Train Acc: 0.9227 | Val Acc: 0.9420
    

    Epoch[13] Loss:0.123 Acc:0.917: 100%|██████████| 860/860 [00:03<00:00, 229.85it/s]
    

    Epoch 13 - Train Acc: 0.9235 | Val Acc: 0.9426
    

    Epoch[14] Loss:0.121 Acc:0.917: 100%|██████████| 860/860 [00:03<00:00, 228.48it/s]
    

    Epoch 14 - Train Acc: 0.9240 | Val Acc: 0.9440
    

    Epoch[15] Loss:0.119 Acc:0.917: 100%|██████████| 860/860 [00:03<00:00, 260.72it/s]
    

    Epoch 15 - Train Acc: 0.9248 | Val Acc: 0.9446
    

    Epoch[16] Loss:0.118 Acc:0.917: 100%|██████████| 860/860 [00:03<00:00, 283.99it/s]
    

    Epoch 16 - Train Acc: 0.9256 | Val Acc: 0.9454
    

    Epoch[17] Loss:0.116 Acc:0.958: 100%|██████████| 860/860 [00:03<00:00, 266.59it/s]
    

    Epoch 17 - Train Acc: 0.9265 | Val Acc: 0.9462
    

    Epoch[18] Loss:0.114 Acc:0.958: 100%|██████████| 860/860 [00:03<00:00, 266.18it/s]
    

    Epoch 18 - Train Acc: 0.9271 | Val Acc: 0.9466
    

    Epoch[19] Loss:0.112 Acc:0.958: 100%|██████████| 860/860 [00:02<00:00, 293.77it/s]
    

    Epoch 19 - Train Acc: 0.9277 | Val Acc: 0.9468
    

    Epoch[20] Loss:0.111 Acc:0.958: 100%|██████████| 860/860 [00:03<00:00, 253.63it/s]
    

    Epoch 20 - Train Acc: 0.9287 | Val Acc: 0.9468
    
    最终测试准确率: 0.9339 (93.39%)
    
    


    
![png](hw2_files/hw2_24_41.png)
    



```python
# 根据案例要求，还需要构造双隐含层MLP
# 自选激活和损失，并与单隐含层的性能做对比
mlp_custom_improved = Network()
# 使用更大的隐含层
mlp_custom_improved.add(FCLayer(784, 256, act_function='relu'))
mlp_custom_improved.add(ReLULayer())
mlp_custom_improved.add(FCLayer(256, 128, act_function='relu'))
mlp_custom_improved.add(ReLULayer())
mlp_custom_improved.add(FCLayer(128, 10, act_function='relu'))

# 使用交叉熵损失
criterion_ce = SoftmaxCrossEntropyLossLayer()
sgd = SGD(learning_rate_SGD, weight_decay)

model, all_train_losses, all_train_accs, avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = train(mlp_relu_euclidean, criterion_euclidean, sgd)

```

    数据标准化: 均值=0.1307, 标准差=0.3082
    

    Epoch[1] Loss:0.111 Acc:0.958: 100%|██████████| 860/860 [00:04<00:00, 207.40it/s]
    

    Epoch 1 - Train Acc: 0.9291 | Val Acc: 0.9472
    

    Epoch[2] Loss:0.109 Acc:0.958: 100%|██████████| 860/860 [00:03<00:00, 249.67it/s]
    

    Epoch 2 - Train Acc: 0.9296 | Val Acc: 0.9484
    

    Epoch[3] Loss:0.107 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 256.85it/s]
    

    Epoch 3 - Train Acc: 0.9305 | Val Acc: 0.9484
    

    Epoch[4] Loss:0.106 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 257.43it/s]
    

    Epoch 4 - Train Acc: 0.9310 | Val Acc: 0.9492
    

    Epoch[5] Loss:0.104 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 256.08it/s]
    

    Epoch 5 - Train Acc: 0.9316 | Val Acc: 0.9496
    

    Epoch[6] Loss:0.103 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 252.40it/s]
    

    Epoch 6 - Train Acc: 0.9322 | Val Acc: 0.9500
    

    Epoch[7] Loss:0.101 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 247.93it/s]
    

    Epoch 7 - Train Acc: 0.9329 | Val Acc: 0.9504
    

    Epoch[8] Loss:0.100 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 255.71it/s]
    

    Epoch 8 - Train Acc: 0.9333 | Val Acc: 0.9504
    

    Epoch[9] Loss:0.098 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 247.08it/s]
    

    Epoch 9 - Train Acc: 0.9339 | Val Acc: 0.9508
    

    Epoch[10] Loss:0.097 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 246.72it/s]
    

    Epoch 10 - Train Acc: 0.9344 | Val Acc: 0.9512
    

    Epoch[11] Loss:0.096 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 248.61it/s]
    

    Epoch 11 - Train Acc: 0.9350 | Val Acc: 0.9518
    

    Epoch[12] Loss:0.094 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 246.04it/s]
    

    Epoch 12 - Train Acc: 0.9353 | Val Acc: 0.9522
    

    Epoch[13] Loss:0.093 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 302.01it/s]
    

    Epoch 13 - Train Acc: 0.9358 | Val Acc: 0.9522
    

    Epoch[14] Loss:0.092 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 329.01it/s]
    

    Epoch 14 - Train Acc: 0.9362 | Val Acc: 0.9520
    

    Epoch[15] Loss:0.091 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 323.54it/s]
    

    Epoch 15 - Train Acc: 0.9366 | Val Acc: 0.9530
    

    Epoch[16] Loss:0.090 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 297.30it/s]
    

    Epoch 16 - Train Acc: 0.9370 | Val Acc: 0.9534
    

    Epoch[17] Loss:0.088 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 290.40it/s]
    

    Epoch 17 - Train Acc: 0.9375 | Val Acc: 0.9534
    

    Epoch[18] Loss:0.087 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 291.62it/s]
    

    Epoch 18 - Train Acc: 0.9379 | Val Acc: 0.9536
    

    Epoch[19] Loss:0.086 Acc:1.000: 100%|██████████| 860/860 [00:03<00:00, 282.85it/s]
    

    Epoch 19 - Train Acc: 0.9382 | Val Acc: 0.9536
    

    Epoch[20] Loss:0.085 Acc:1.000: 100%|██████████| 860/860 [00:02<00:00, 299.56it/s]
    

    Epoch 20 - Train Acc: 0.9386 | Val Acc: 0.9548
    
    最终测试准确率: 0.9418 (94.18%)
    
    


    
![png](hw2_files/hw2_25_41.png)
    



```python

def plot_performance_comparison():
    """绘制模型性能对比图"""
    
    # 实际训练结果
    model_results = {
        'Sigmoid + Euclidean': 0.8886,
        'ReLU + Euclidean': 0.9140, 
        'Sigmoid + CrossEntropy': 0.9029,
        'ReLU + CrossEntropy': 0.9312,
        'Double Layer ReLU + CE': 0.9384
    }
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 总体性能对比
    model_names = list(model_results.keys())
    accuracies = list(model_results.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    bars = ax1.bar(range(len(model_names)), accuracies, color=colors)
    ax1.set_title('测试准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('测试准确率')
    ax1.set_ylim(0.8, 1.0)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. 激活函数对比
    activation_comparison = {
        'Sigmoid': [0.8886, 0.9029],  # [Euclidean, CrossEntropy]
        'ReLU': [0.9140, 0.9312]      # [Euclidean, CrossEntropy] 
    }
    
    x = np.arange(len(activation_comparison))
    width = 0.35
    
    euclidean_scores = [scores[0] for scores in activation_comparison.values()]
    ce_scores = [scores[1] for scores in activation_comparison.values()]
    
    bars1 = ax2.bar(x - width/2, euclidean_scores, width, 
                    label='欧式距离损失', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, ce_scores, width, 
                    label='交叉熵损失', color='#4ECDC4', alpha=0.8)
    
    ax2.set_title('激活函数对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('测试准确率')
    ax2.set_ylim(0.85, 0.95)
    ax2.set_xticks(x)
    ax2.set_xticklabels(activation_comparison.keys())
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 损失函数对比
    loss_comparison = {
        '欧式距离': [0.8886, 0.9140],  # [Sigmoid, ReLU]
        '交叉熵': [0.9029, 0.9312]     # [Sigmoid, ReLU]
    }
    
    x = np.arange(len(loss_comparison))
    sigmoid_scores = [scores[0] for scores in loss_comparison.values()]
    relu_scores = [scores[1] for scores in loss_comparison.values()]
    
    bars3 = ax3.bar(x - width/2, sigmoid_scores, width, 
                    label='Sigmoid激活', color='#96CEB4', alpha=0.8)
    bars4 = ax3.bar(x + width/2, relu_scores, width, 
                    label='ReLU激活', color='#FECA57', alpha=0.8)
    
    ax3.set_title('损失函数对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('测试准确率') 
    ax3.set_ylim(0.85, 0.95)
    ax3.set_xticks(x)
    ax3.set_xticklabels(loss_comparison.keys())
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 网络深度对比
    depth_comparison = {
        '单隐含层': 0.9312,
        '双隐含层': 0.9384
    }
    
    bars_depth = ax4.bar(depth_comparison.keys(), depth_comparison.values(),
                        color=['#45B7D1', '#FECA57'], alpha=0.8)
    ax4.set_title('网络深度对比\n(ReLU + 交叉熵)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('测试准确率')
    ax4.set_ylim(0.92, 0.95)
    ax4.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars_depth, depth_comparison.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('MNIST MLP性能分析', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()
    
    # 打印结果总结
    print("=" * 50)
    print("性能总结")
    print("=" * 50)
    
    sorted_results = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
    for i, (model, acc) in enumerate(sorted_results, 1):
        print(f"{i}. {model}: {acc:.4f} ({acc*100:.2f}%)")

# 调用函数
plot_performance_comparison()
```


    
![png](hw2_files/hw2_26_0.png)
    


    ==================================================
    性能总结
    ==================================================
    1. Double Layer ReLU + CE: 0.9384 (93.84%)
    2. ReLU + CrossEntropy: 0.9312 (93.12%)
    3. ReLU + Euclidean: 0.9140 (91.40%)
    4. Sigmoid + CrossEntropy: 0.9029 (90.29%)
    5. Sigmoid + Euclidean: 0.8886 (88.86%)
    

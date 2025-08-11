不建议大家一直学最基础的 一直在学python的数据类型、运算和条件语句 简单看下这部分内容，马上学习使用python的各种包 基础的东西太多了，用到的时候再学和问AI吧

学习使用[Jupyter notebook](https://obsidiannote.netlify.app/%E7%AC%94%E8%AE%B0/%F0%9F%8F%AB%E6%B8%85%E5%8D%8E%E9%A9%AD%E9%A3%8E%E8%AE%A1%E5%88%92/%F0%9F%A7%B1%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%A8%8B/python%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9F%A5%E8%AF%86/Jupyter%20notebook.html)，长得像笔记的一个编译器，界面更通人性一点😂

数据分析三剑客 [Numpy](https://obsidiannote.netlify.app/%E7%AC%94%E8%AE%B0/%F0%9F%8F%AB%E6%B8%85%E5%8D%8E%E9%A9%AD%E9%A3%8E%E8%AE%A1%E5%88%92/%F0%9F%A7%B1%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%A8%8B/python%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9F%A5%E8%AF%86/Numpy.html)提供一种矩阵，构造，运算，访问，赋值的操作 [Pandas](https://obsidiannote.netlify.app/%E7%AC%94%E8%AE%B0/%F0%9F%8F%AB%E6%B8%85%E5%8D%8E%E9%A9%AD%E9%A3%8E%E8%AE%A1%E5%88%92/%F0%9F%A7%B1%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%A8%8B/python%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9F%A5%E8%AF%86/Pandas.html)更实用，一维数据和二维表格的处理，去重、查找、替换、分组、聚合等操作 [Matplotlib](https://obsidiannote.netlify.app/%E7%AC%94%E8%AE%B0/%F0%9F%8F%AB%E6%B8%85%E5%8D%8E%E9%A9%AD%E9%A3%8E%E8%AE%A1%E5%88%92/%F0%9F%A7%B1%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%A8%8B/python%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9F%A5%E8%AF%86/Matplotlib.html)数据的可视化包，画布，绘图的所有元素，丰富的调色功能，2D和3D图像的绘制

这三个家伙，融合了数据的运算、处理和可视化 撑起来了数据分析的一片天

更深入一点，数据挖掘的话，用到sklearn，机器学习的一个包

## 官方手册

[The Python Tutorial — Python 3.13.3 documentation](https://docs.python.org/3/tutorial/)

## 安装

### 环境变量

> 以win11 为例

1. 下载适合自己的版本 www.python.org
2. 环境配置：
    1. 按下 `Win + S` 搜索 **"环境变量"**，选择 **"编辑系统环境变量"**。在弹出的窗口中，点击右下角的 **"环境变量"** 按钮
    2. Path配置

#### 成功的标志

打开新的命令提示符（或 PowerShell），输入：

bash

```
python --version
pip --version
```

如果能正常输出版本号，则配置成功。

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250527192222.DvIWs6EK.png)

### Pycharm 编译器

1. 下载 [PyCharm: The only Python IDE you need](https://www.jetbrains.com/pycharm/) community即可
2. 新建项目
3. New——package

## 基础

了解一下数据类型、基本运算、条件语句

//是除的时候向下取整，多的小数就不要了 两个星号是幂运算 一个等号是赋值 两个等号是判断

### 函数

抽象出一个功能，定义一个函数 组织好的、可重复使用的代码块，用于执行单一或相关联的任务。函数能提高代码的模块化程度和复用性

python

```
def 函数名(参数1, 参数2, ...):
    """函数文档字符串（可选）"""
    函数体
    return 返回值  # 可选
```

python

```
def add(a, b):
    """计算两个数的和"""
    return a + b

result = add(3, 5)  # 调用函数
print(result)  # 输出: 8
```

### 面向对象编程
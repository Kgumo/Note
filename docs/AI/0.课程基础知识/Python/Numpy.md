## 官方手册

[What is NumPy? — NumPy v2.2 Manual](https://numpy.org/doc/stable/user/whatisnumpy.html)

## 菜鸟

[NumPy 教程 | 菜鸟教程](https://www.runoob.com/numpy/numpy-tutorial.html)

主要是做==数据处理== 数据分析的基础

核心数据是==数组==

1. 有序
2. 数据类型一致

高维数据，简单运算

首先在[Jupyter notebook](https://obsidiannote.netlify.app/%E7%AC%94%E8%AE%B0/%F0%9F%8F%AB%E6%B8%85%E5%8D%8E%E9%A9%AD%E9%A3%8E%E8%AE%A1%E5%88%92/%F0%9F%A7%B1%E5%9F%BA%E7%A1%80%E8%AF%BE%E7%A8%8B/python%E6%9C%80%E5%B0%8F%E5%8F%AF%E7%94%A8%E7%9F%A5%E8%AF%86/Jupyter%20notebook.html)里面对Numpy进行导入 大家装anaconda的时候，这些都自动有的，直接导就行

```python
import numpy as np
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601101547.BKb_gtvO.png)numpy太长了，我们导入时将它 重命名 np（牛皮）

查看当前np版本(注意这里是双下划线)

```python
np.__version__
```

## 操作

### 构造数组

目标

快速构造自己想要的数组

1. 普通数组

```python
np.array([1,2,3])
```

传入列表的形式

一个二维的

```python
np.array([[1,2,3],[3,4,5]])
```

2. 更便捷的方式

```python
np.ones(shape=(3,2))
# 以1填充一个3行2列的数组


<NolebasePageProperties />
```

忘记怎么用了怎么办

Shift Tab可以查看当前方法的说明文档![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601103648.BBO6lFXZ.png)

shape指定形状，比如3行2列

指定数填充

```python
np.full(shape=(4,3),fill_value=6)
```

随机数填充

```python
np.random.randint(0,10,size=(2,3))
# 2行3列的0到10的随机数数组
```

> size的作用类似于shape

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601104835.DalC8eo1.png)

0到1的浮点数数组

```python
np.random.random(size=(3,4))
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601104822.-zPAR61M.png)

等差数列 起始，终止，取多少个数

```python
np.linspace(0,10,11)
```

指定步长(不包含结束值)

```python
np.arange(0,10,1)
```

### 访问数组

目标

得多练习，精准定位的能力

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601110131.BCAYnZXj.png)

```python
array[你想访问到的数字]
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250601110335.DOPAEPBI.png)

Numpy数组的优势(相比python原生的数组来说) 可以用列表当成index访问

```python
array[[0,1,2]]
```

用列表对数组重新排列和组合，

```python
array[[2,3,1,5]]
```

用布尔列表的形式访问

```python
arr1=np.array([1,2,3])
arr1[[True,False,True]]
```

true就拿到，false就忽略

广播运算

> 快速找到一个数组中大于2的数

```python
arr1 > 2
# 广播运算
arr1[arr1 > 2]
# 此处结合了布尔运算
```

高维的怎么处理？ 对于二维以上的

```python
arr2=np.random.randint(0,10,size=(5,3))
```

可以通过以下方式

```python
arr2[0][0]
```

```python
arr2[0,1]
```

很灵活的访问方式

索引的方式很灵活，可以是整数、列表、条件表达式 array[dim1_index,dim2_index,dim3_index,......dimn_index]

### 运算

比如广播运算

```python
arr2 + 10
# 对每一个元素进行+10的操作
```

会自动扩展的相加运算

```python
arr1=np.array([1,2,3])

arr2=np.array([[1],[2],[3]])
# arr2有行和列两个维度，是二维数组
```

用到了广播机制 arr1会扩展到和arr2行数相同 arr2会扩展到和arr1列数相同 比如对于arr1来说，会变成 1 2 3 1 2 3 1 2 3 对于arr2来说，会变成 1 1 1 2 2 2 3 3 3

arr1+10 10会变成 10 10 10

- 缺失维度自动补充

#### 不能扩展相加的情况

```python
arr1 = np.array([1,2,3])
arr2 = np.array([[1,2],[3,4]])
```

- 能否相加取决于最后一个轴的长度是否相等或其中一个为 1

运算不止相加，还有关系运算、逻辑运算、赋值运算

### 排序方法

```python
data.sort()
# 在原数据结构进行的从小到大的排序
```

```python
np.sort(data)
# 不会破坏原始的结构
```

建议拷贝后再进行排序等操作

```python
c_data=data.copy()
```

```python
c_data[0]=9
# 重新写第一个元素，不会影响原结构
```

### 级联

形状需要是一样的 3行n列可以和3行k列级联 n行2列可以和k行2列级联 主要用于数据汇总

```python
arr1=np.random.randint(0,10,size=(3,2))
arr2=np.random.randint(10,20,size=(3,3))
np.concatenate((arr1,arr2),axis=1)
# 行数相同，只能横向级联，axis=1
```
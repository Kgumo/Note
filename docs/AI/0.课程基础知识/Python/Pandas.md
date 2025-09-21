## 官方手册

[10 minutes to pandas — pandas 2.2.3 documentation](https://pandas.pydata.org/docs/user_guide/10min.html)

## 菜鸟

[Pandas 教程 | 菜鸟教程](https://www.runoob.com/pandas/pandas-tutorial.html)

业务处理

使用前先导入

```
import pandas as pd
import numpy as np
```

### 基础数据类型

```
from pandas import Series,DataFrame
```

Series是一维表格 DataFrame是二维表格

#### Series

```python
# 用series构建一维列表


<NolebasePageProperties />




Series([1,2,3])
# 左边是索引，右边是值
```

```python
0    1
1    2
2    3
dtype: int64
```

甚至能用字符串来当索引，还是显式索引，这点比numpy强

```python
s=Series([1,2,3],index=['tom','jack','lucy'])
```

```python
# pandas特有的显式访问是loc
s.loc['tom']
# 在数据写入的时候，推荐使用loc方法
s.loc['tom']=10
```

显式访问的方式比较直观

```python
s.loc[['tom','lucy']]
s.loc[[True,False,True]]
s>5
# 类似于广播的形式
s.loc[s>5]
```

也支持隐式访问

```python
print(s.iloc[0])
```

#### DataFrame

二维结构 引入了行标签和列标签的概念

```python
DataFrame(data=np.random.randint(0,10,size=(3,5)))
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602161509.BzvAqxXj.png)

```python
df=DataFrame(data=np.random.randint(0,10,size=(3,5)),index=['tom','lucy','jack'],columns=['语文','英语','数学','物理','化学'])
df
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602163806.any0Xdhs.png)

```python
# 显式访问
df.loc['tom','物理']
```

支持多个访问

```python
df.loc[['tom','lucy'],'数学']
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602163850.BOE3VaSj.png)

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602163859.CVOoZzXg.png)

## 运算

```
# 广播 pandas与np.array，需要遵守广播原则
# 索引对齐 pandas的对象之间，比如series与series，series与dataframe之间
```

```python
score=df.loc['tom']
score
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602163959.BNwANtDz.png)

```python
df+score
# 会自动索引对齐
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602164016.Hse6AqjL.png)

### 聚合运算

```python
score.sum(),score.mean(),score.var(),score.std()
```

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602164050.C9YFost2.png)

![](https://obsidiannote.netlify.app/assets/Pasted%20image%2020250602164121.bAampB3q.png)

#### any与all

```python
# 非常好用的any与all
score.isnull().any()
# 帮助查看某个值是否为true

(score>5).any()
# 在整个score中是否存在一个大于5的值
```
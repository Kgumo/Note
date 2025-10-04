我理解您想要一套完整的Pandas练习题，从入门到进阶。我将为您提供一套结构化的Pandas闯关100题，按难度递进。

# Pandas 闯关 100 题：从入门到进阶

## 第一关：基础入门（1-20题）

### 数据创建与基本操作

1. **创建Series**：创建一个包含1-10的Series，索引为a-j
2. **创建DataFrame**：创建一个3x4的DataFrame，包含随机数
3. **查看数据信息**：显示DataFrame的基本信息（shape, dtypes, info）
4. **数据选择**：选择DataFrame的前3行和后2列
5. **条件筛选**：筛选DataFrame中大于某个值的行

### 数据读取与保存

6. **读取CSV**：读取一个CSV文件到DataFrame
7. **保存CSV**：将DataFrame保存为CSV文件
8. **读取Excel**：读取Excel文件的特定sheet
9. **JSON操作**：读取和保存JSON格式数据
10. **设置索引**：将某一列设置为索引

### 基本统计

11. **描述性统计**：计算DataFrame的基本统计信息
12. **计算均值**：计算每列的平均值
13. **计算中位数**：找出每列的中位数
14. **计算标准差**：计算数值列的标准差
15. **计数统计**：统计非空值的数量

### 数据查看

16. **查看头尾**：显示前5行和后5行数据
17. **随机采样**：随机选择10行数据
18. **数据类型**：检查和转换数据类型
19. **唯一值**：找出某列的唯一值
20. **重复值**：检测和处理重复行

## 第二关：数据清洗（21-40题）

### 缺失值处理

21. **检测缺失值**：找出DataFrame中的缺失值
22. **删除缺失值**：删除包含缺失值的行或列
23. **填充缺失值**：用均值填充数值列的缺失值
24. **前向填充**：使用前一个有效值填充缺失值
25. **插值填充**：使用线性插值填充缺失值

### 数据类型转换

26. **字符串转数字**：将字符串列转换为数值类型
27. **日期转换**：将字符串转换为日期时间格式
28. **分类数据**：创建和使用分类数据类型
29. **布尔转换**：将字符串转换为布尔值
30. **类型检查**：检查列的数据类型并批量转换

### 异常值处理

31. **异常值检测**：使用IQR方法检测异常值
32. **异常值处理**：删除或替换异常值
33. **数据标准化**：对数值列进行标准化处理
34. **数据归一化**：将数据缩放到0-1范围
35. **分箱操作**：将连续变量转换为分类变量

### 字符串处理

36. **字符串分割**：分割字符串列
37. **字符串替换**：替换字符串中的特定内容
38. **大小写转换**：转换字符串的大小写
39. **字符串匹配**：使用正则表达式匹配字符串
40. **字符串提取**：从字符串中提取特定部分

## 第三关：数据操作（41-60题）

### 数据选择与过滤

41. **多条件筛选**：使用多个条件筛选数据
42. **isin操作**：使用isin方法筛选数据
43. **字符串筛选**：基于字符串内容筛选行
44. **正则表达式筛选**：使用正则表达式筛选数据
45. **索引选择**：使用loc和iloc进行数据选择

### 数据排序

46. **单列排序**：按单列对DataFrame排序
47. **多列排序**：按多列排序，指定升序降序
48. **索引排序**：按索引对DataFrame排序
49. **值排序**：对Series进行值排序
50. **自定义排序**：使用自定义键进行排序

### 数据重塑

51. **数据透视**：创建数据透视表
52. **长宽转换**：在长格式和宽格式之间转换
53. **melt操作**：将宽格式数据转换为长格式
54. **pivot操作**：将长格式数据转换为宽格式
55. **堆叠操作**：使用stack和unstack重塑数据

### 数据合并

56. **concat合并**：使用concat连接多个DataFrame
57. **merge合并**：使用merge进行数据库式合并
58. **join操作**：使用join方法合并数据
59. **多表合并**：合并三个或更多表
60. **合并类型**：掌握不同类型的合并（inner, outer, left, right）

## 第四关：分组与聚合（61-80题）

### 基础分组

61. **简单分组**：按单列分组并计算统计量
62. **多列分组**：按多列进行分组操作
63. **分组计数**：统计每组的数量
64. **分组求和**：计算每组的总和
65. **分组平均值**：计算每组的平均值

### 高级分组

66. **自定义聚合**：使用agg进行自定义聚合
67. **多重聚合**：对同一列应用多个聚合函数
68. **分组应用函数**：使用apply对分组应用自定义函数
69. **分组转换**：使用transform进行分组转换
70. **分组过滤**：使用filter过滤分组

### 时间序列分组

71. **按时间分组**：按年、月、日分组
72. **重采样**：对时间序列数据进行重采样
73. **滚动窗口**：计算滚动平均值
74. **时间窗口聚合**：在时间窗口内进行聚合
75. **时间序列插值**：对时间序列进行插值

### 复杂聚合

76. **分位数计算**：计算各组的分位数
77. **累积统计**：计算累积和、累积均值
78. **组内排名**：在每组内进行排名
79. **组间比较**：计算组间差异
80. **分组标准化**：在组内进行标准化

## 第五关：高级应用（81-100题）

### 数据可视化

81. **基础绘图**：使用pandas绘制线图和柱状图
82. **分组绘图**：按分组绘制多个子图
83. **直方图**：绘制数值分布直方图
84. **箱线图**：绘制箱线图检测异常值
85. **散点图**：绘制相关性散点图

### 性能优化

86. **内存优化**：优化DataFrame的内存使用
87. **数据类型优化**：选择合适的数据类型
88. **分块处理**：处理大型数据集
89. **向量化操作**：使用向量化提高性能
90. **避免循环**：用pandas操作替代Python循环

### 高级技巧

91. **多级索引**：创建和操作多级索引
92. **自定义函数**：编写自定义聚合函数
93. **数据验证**：验证数据质量和完整性
94. **管道操作**：使用pipe创建数据处理管道
95. **样式设置**：设置DataFrame的显示样式

### 实战应用

96. **数据报告**：生成自动化数据报告
97. **时间序列分析**：进行时间序列分析
98. **相关性分析**：计算和可视化相关性矩阵
99. **数据建模准备**：为机器学习准备数据
100. **综合项目**：完成一个端到端的数据分析项目

## 学习建议

1. **循序渐进**：按顺序完成题目，每个阶段都要完全掌握
2. **实践为主**：每道题都要亲自编写代码实现
3. **查阅文档**：遇到问题时查阅pandas官方文档
4. **总结笔记**：记录重要的方法和技巧
5. **项目实战**：将学到的知识应用到实际项目中

---
## 第一关：基础入门（1-20题）

### 数据创建与基本操作

**1. 创建Series：创建一个包含1-10的Series，索引为a-j**

```python
import pandas as pd
import numpy as np

# 创建Series
s = pd.Series(range(1, 11), index=list('abcdefghij'))
print(s)
```

**2. 创建DataFrame：创建一个3x4的DataFrame，包含随机数**

```python
# 创建DataFrame
df = pd.DataFrame(np.random.randn(3, 4), 
                  columns=['A', 'B', 'C', 'D'],
                  index=['行1', '行2', '行3'])
print(df)
```

**3. 查看数据信息：显示DataFrame的基本信息**

```python
# 查看基本信息
print("形状:", df.shape)
print("数据类型:\n", df.dtypes)
print("基本信息:")
df.info()
```

**4. 数据选择：选择DataFrame的前3行和后2列**

```python
# 选择前3行和后2列
print(df.iloc[:3, -2:])  # 使用iloc按位置选择
# 或者
print(df[['C', 'D']])    # 使用列名选择
```

**5. 条件筛选：筛选DataFrame中大于某个值的行**

```python
# 筛选A列大于0的行
filtered_df = df[df['A'] > 0]
print(filtered_df)
```

### 数据读取与保存

**6. 读取CSV：读取一个CSV文件到DataFrame**

```python
# 读取CSV文件
# df = pd.read_csv('data.csv')
# 示例：从字符串读取（模拟文件）
data = """name,age,salary
Alice,25,50000
Bob,30,60000
Charlie,35,70000"""
df = pd.read_csv(pd.compat.StringIO(data))
print(df)
```

**7. 保存CSV：将DataFrame保存为CSV文件**

```python
# 保存为CSV
df.to_csv('output.csv', index=False)
```

**8. 读取Excel：读取Excel文件的特定sheet**

```python
# 读取Excel（需要安装openpyxl或xlrd）
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

**9. JSON操作：读取和保存JSON格式数据**

```python
# 保存为JSON
df.to_json('data.json', orient='records')

# 读取JSON
# df_json = pd.read_json('data.json')
```

**10. 设置索引：将某一列设置为索引**

```python
# 将name列设置为索引
df_indexed = df.set_index('name')
print(df_indexed)
```

### 基本统计

**11. 描述性统计：计算DataFrame的基本统计信息**

```python
print(df.describe())
```

**12. 计算均值：计算每列的平均值**

```python
print("各列均值:")
print(df.mean())
```

**13. 计算中位数：找出每列的中位数**

```python
print("各列中位数:")
print(df.median())
```

**14. 计算标准差：计算数值列的标准差**

```python
print("各列标准差:")
print(df.std())
```

**15. 计数统计：统计非空值的数量**

```python
print("非空值计数:")
print(df.count())
```

### 数据查看

**16. 查看头尾：显示前5行和后5行数据**

```python
print("前2行:")
print(df.head(2))
print("后2行:")
print(df.tail(2))
```

**17. 随机采样：随机选择10行数据**

```python
# 如果有更多数据，可以随机采样
print("随机采样2行:")
print(df.sample(2, random_state=42))
```

**18. 数据类型：检查和转换数据类型**

```python
print("数据类型:")
print(df.dtypes)

# 转换数据类型
df['age'] = df['age'].astype('float64')
print("转换后的数据类型:")
print(df.dtypes)
```

**19. 唯一值：找出某列的唯一值**

```python
print("name列的唯一值:")
print(df['name'].unique())
```

**20. 重复值：检测和处理重复行**

```python
# 添加重复行进行演示
df_with_duplicates = pd.concat([df, df.iloc[[0]]], ignore_index=True)
print("包含重复行的数据:")
print(df_with_duplicates)

print("重复行:")
print(df_with_duplicates[df_with_duplicates.duplicated()])

# 删除重复行
df_no_duplicates = df_with_duplicates.drop_duplicates()
print("删除重复行后:")
print(df_no_duplicates)
```

## 完整示例代码

```python
import pandas as pd
import numpy as np
from io import StringIO

def first_level_practice():
    print("=== 第一关：Pandas基础入门 ===\n")
    
    # 1. 创建Series
    print("1. 创建Series:")
    s = pd.Series(range(1, 11), index=list('abcdefghij'))
    print(s)
    print()
    
    # 2. 创建DataFrame
    print("2. 创建DataFrame:")
    df = pd.DataFrame(np.random.randn(3, 4), 
                      columns=['A', 'B', 'C', 'D'],
                      index=['行1', '行2', '行3'])
    print(df)
    print()
    
    # 3. 查看数据信息
    print("3. 数据信息:")
    print("形状:", df.shape)
    print("数据类型:\n", df.dtypes)
    print()
    
    # 创建示例数据用于后续练习
    data = """name,age,salary
Alice,25,50000
Bob,30,60000
Charlie,35,70000"""
    df = pd.read_csv(StringIO(data))
    
    # 4. 数据选择
    print("4. 数据选择:")
    print(df[['name', 'salary']])
    print()
    
    # 5. 条件筛选
    print("5. 条件筛选:")
    print(df[df['age'] > 28])
    print()
    
    # 11-15. 基本统计
    print("11. 描述性统计:")
    print(df.describe())
    print()
    
    print("16. 查看头尾:")
    print("头部:")
    print(df.head(2))
    print("尾部:")
    print(df.tail(2))
    print()
    
    print("18. 数据类型:")
    print(df.dtypes)
    print()
    
    print("19. 唯一值:")
    print(df['name'].unique())

# 运行练习
first_level_practice()
```

## 练习建议

1. **逐题练习**：每道题都要亲自输入代码并运行
2. **理解输出**：仔细观察每个操作的结果，理解其含义
3. **尝试变体**：对每道题尝试不同的参数和变体
4. **查阅文档**：遇到不理解的函数时查看官方文档

## 第二关：数据清洗（21-40题）

### 缺失值处理

**21. 检测缺失值：找出DataFrame中的缺失值**

```python
import pandas as pd
import numpy as np

# 创建包含缺失值的示例数据
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5],
    'D': ['a', 'b', np.nan, 'd', 'e']
})

print("原始数据:")
print(df)

print("\n缺失值检测:")
print(df.isnull())

print("\n每列缺失值数量:")
print(df.isnull().sum())

print("\n缺失值比例:")
print(df.isnull().mean())
```

**22. 删除缺失值：删除包含缺失值的行或列**

```python
# 删除包含缺失值的行
df_dropped_rows = df.dropna()
print("删除缺失值行后:")
print(df_dropped_rows)

# 删除包含缺失值的列
df_dropped_cols = df.dropna(axis=1)
print("删除缺失值列后:")
print(df_dropped_cols)

# 只删除全为缺失值的行
df_dropped_all_na = df.dropna(how='all')
print("删除全为缺失值的行后:")
print(df_dropped_all_na)
```

**23. 填充缺失值：用均值填充数值列的缺失值**

```python
# 用每列的均值填充缺失值
df_filled_mean = df.copy()
df_filled_mean['A'] = df_filled_mean['A'].fillna(df_filled_mean['A'].mean())
df_filled_mean['B'] = df_filled_mean['B'].fillna(df_filled_mean['B'].mean())
print("用均值填充后:")
print(df_filled_mean)

# 或者使用fillna一次性填充所有数值列
df_filled = df.copy()
numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
print("一次性填充所有数值列:")
print(df_filled)
```

**24. 前向填充：使用前一个有效值填充缺失值**

```python
# 前向填充
df_ffill = df.fillna(method='ffill')
print("前向填充后:")
print(df_ffill)

# 后向填充
df_bfill = df.fillna(method='bfill')
print("后向填充后:")
print(df_bfill)
```

**25. 插值填充：使用线性插值填充缺失值**

```python
# 线性插值
df_interpolated = df.interpolate()
print("线性插值后:")
print(df_interpolated)

# 时间序列插值（如果有时间索引）
df_time = pd.DataFrame({
    'value': [1, np.nan, np.nan, 4, 5]
}, index=pd.date_range('2023-01-01', periods=5))
df_time_interpolated = df_time.interpolate()
print("时间序列插值:")
print(df_time_interpolated)
```

### 数据类型转换

**26. 字符串转数字：将字符串列转换为数值类型**

```python
# 创建包含字符串数字的DataFrame
df_str = pd.DataFrame({
    'A': ['1', '2', '3', '4'],
    'B': ['5.1', '6.2', '7.3', '8.4'],
    'C': ['1000', '2,000', '3,000', '4,000']  # 包含千位分隔符
})

print("原始字符串数据:")
print(df_str)
print("数据类型:")
print(df_str.dtypes)

# 转换为数值类型
df_str['A'] = pd.to_numeric(df_str['A'])
df_str['B'] = pd.to_numeric(df_str['B'])
df_str['C'] = pd.to_numeric(df_str['C'].str.replace(',', ''))  # 先去除逗号

print("\n转换后:")
print(df_str)
print("数据类型:")
print(df_str.dtypes)
```

**27. 日期转换：将字符串转换为日期时间格式**

```python
# 创建包含日期字符串的DataFrame
df_date = pd.DataFrame({
    'date_str': ['2023-01-01', '2023-02-01', '2023-03-01', '2023/04/01'],
    'datetime_str': ['2023-01-01 10:30:00', '2023-02-01 14:45:00', 
                    '2023-03-01 09:15:00', '2023-04-01 16:20:00']
})

print("原始日期数据:")
print(df_date)

# 转换为日期时间
df_date['date'] = pd.to_datetime(df_date['date_str'])
df_date['datetime'] = pd.to_datetime(df_date['datetime_str'])

print("\n转换后:")
print(df_date)
print("数据类型:")
print(df_date.dtypes)
```

**28. 分类数据：创建和使用分类数据类型**

```python
# 创建包含分类数据的DataFrame
df_cat = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'value': [10, 20, 30, 40, 50, 60, 70]
})

print("原始数据:")
print(df_cat)
print("数据类型:")
print(df_cat.dtypes)

# 转换为分类类型
df_cat['category'] = pd.Categorical(df_cat['category'])
print("\n转换后数据类型:")
print(df_cat.dtypes)

# 分类数据的优势
print("分类的类别:")
print(df_cat['category'].cat.categories)
print("内存使用优化:")
print(f"原始内存: {df_cat['category'].astype('object').memory_usage(deep=True)}")
print(f"分类内存: {df_cat['category'].memory_usage(deep=True)}")
```

**29. 布尔转换：将字符串转换为布尔值**

```python
# 创建包含布尔字符串的DataFrame
df_bool = pd.DataFrame({
    'bool_str': ['True', 'False', 'true', 'false', 'YES', 'NO'],
    'active': ['1', '0', '1', '0', '1', '0']
})

print("原始布尔字符串:")
print(df_bool)

# 转换为布尔值
df_bool['bool'] = df_bool['bool_str'].map({'True': True, 'False': False, 
                                         'true': True, 'false': False,
                                         'YES': True, 'NO': False})
df_bool['is_active'] = df_bool['active'] == '1'

print("\n转换后:")
print(df_bool)
```

**30. 类型检查：检查列的数据类型并批量转换**

```python
# 检查数据类型
print("数据类型:")
print(df.dtypes)

# 批量转换数据类型
df_types = df.copy()
type_conversions = {
    'A': 'int64',    # 转换为整数
    'B': 'float32',  # 转换为32位浮点数
    'D': 'category'  # 转换为分类
}

for col, new_type in type_conversions.items():
    if col in df_types.columns:
        try:
            df_types[col] = df_types[col].astype(new_type)
        except Exception as e:
            print(f"转换列 {col} 时出错: {e}")

print("\n转换后数据类型:")
print(df_types.dtypes)
```

### 异常值处理

**31. 异常值检测：使用IQR方法检测异常值**

```python
# 创建包含异常值的示例数据
df_outlier = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 100]  # 100是异常值
})

print("数据:")
print(df_outlier)

# IQR方法检测异常值
Q1 = df_outlier['value'].quantile(0.25)
Q3 = df_outlier['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"下限: {lower_bound}, 上限: {upper_bound}")

outliers = df_outlier[(df_outlier['value'] < lower_bound) | (df_outlier['value'] > upper_bound)]
print("\n异常值:")
print(outliers)
```

**32. 异常值处理：删除或替换异常值**

```python
# 删除异常值
df_no_outlier = df_outlier[(df_outlier['value'] >= lower_bound) & (df_outlier['value'] <= upper_bound)]
print("删除异常值后:")
print(df_no_outlier)

# 替换异常值为边界值
df_capped = df_outlier.copy()
df_capped['value'] = df_capped['value'].clip(lower=lower_bound, upper=upper_bound)
print("替换异常值为边界值后:")
print(df_capped)

# 替换异常值为中位数
median_value = df_outlier['value'].median()
df_median = df_outlier.copy()
df_median.loc[(df_median['value'] < lower_bound) | (df_median['value'] > upper_bound), 'value'] = median_value
print("替换异常值为中位数后:")
print(df_median)
```

**33. 数据标准化：对数值列进行标准化处理**

```python
from sklearn.preprocessing import StandardScaler

# 标准化（均值为0，标准差为1）
scaler = StandardScaler()
df_standardized = df_outlier.copy()
df_standardized['value_standardized'] = scaler.fit_transform(df_standardized[['value']])
print("标准化后:")
print(df_standardized)
print(f"均值: {df_standardized['value_standardized'].mean():.2f}, 标准差: {df_standardized['value_standardized'].std():.2f}")
```

**34. 数据归一化：将数据缩放到0-1范围**

```python
from sklearn.preprocessing import MinMaxScaler

# 归一化（0-1范围）
scaler = MinMaxScaler()
df_normalized = df_outlier.copy()
df_normalized['value_normalized'] = scaler.fit_transform(df_normalized[['value']])
print("归一化后:")
print(df_normalized)
print(f"最小值: {df_normalized['value_normalized'].min():.2f}, 最大值: {df_normalized['value_normalized'].max():.2f}")
```

**35. 分箱操作：将连续变量转换为分类变量**

```python
# 分箱（离散化）
df_binned = df_outlier.copy()

# 等宽分箱
df_binned['value_bin_equal_width'] = pd.cut(df_binned['value'], bins=3, labels=['低', '中', '高'])

# 等频分箱
df_binned['value_bin_equal_freq'] = pd.qcut(df_binned['value'], q=3, labels=['低', '中', '高'])

# 自定义分箱
bins = [0, 2, 4, 100]  # 自定义边界
labels = ['小', '中', '大']
df_binned['value_bin_custom'] = pd.cut(df_binned['value'], bins=bins, labels=labels)

print("分箱后:")
print(df_binned)
```

### 字符串处理

**36. 字符串分割：分割字符串列**

```python
# 创建包含字符串的DataFrame
df_str = pd.DataFrame({
    'full_name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com']
})

print("原始字符串:")
print(df_str)

# 分割字符串
df_str[['first_name', 'last_name']] = df_str['full_name'].str.split(' ', expand=True)
df_str[['username', 'domain']] = df_str['email'].str.split('@', expand=True)

print("\n分割后:")
print(df_str)
```

**37. 字符串替换：替换字符串中的特定内容**

```python
# 替换
df_str_rep = df_str.copy()
df_str_rep['email'] = df_str_rep['email'].str.replace('email.com', 'company.com')
df_str_rep['full_name'] = df_str_rep['full_name'].str.replace(' ', '_')

print("替换后:")
print(df_str_rep)
```

**38. 大小写转换：转换字符串的大小写**

```python
# 大小写转换
df_str_case = df_str.copy()
df_str_case['first_name_upper'] = df_str_case['first_name'].str.upper()
df_str_case['last_name_lower'] = df_str_case['last_name'].str.lower()
df_str_case['full_name_title'] = df_str_case['full_name'].str.title()

print("大小写转换后:")
print(df_str_case)
```

**39. 字符串匹配：使用正则表达式匹配字符串**

```python
# 正则匹配
df_str_match = df_str.copy()

# 检查是否包含特定模式
df_str_match['has_li'] = df_str_match['first_name'].str.contains('li', case=False)
df_str_match['starts_with_c'] = df_str_match['first_name'].str.match('^C', case=False)

# 提取匹配的部分
df_str_match['vowels'] = df_str_match['first_name'].str.extract('([aeiou]+)', expand=False)

print("字符串匹配结果:")
print(df_str_match)
```

**40. 字符串提取：从字符串中提取特定部分**

```python
# 字符串提取
df_str_extract = df_str.copy()

# 提取前n个字符
df_str_extract['first_3_chars'] = df_str_extract['first_name'].str[:3]

# 提取最后n个字符
df_str_extract['last_2_chars'] = df_str_extract['last_name'].str[-2:]

# 使用正则表达式提取
df_str_extract['domain_ext'] = df_str_extract['email'].str.extract(r'@(.+)$')

print("字符串提取结果:")
print(df_str_extract)
```

## 完整示例代码

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def second_level_practice():
    print("=== 第二关：数据清洗 ===\n")
    
    # 创建示例数据
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': ['a', 'b', np.nan, 'd', 'e']
    })
    
    # 21. 检测缺失值
    print("21. 缺失值检测:")
    print("缺失值数量:\n", df.isnull().sum())
    print()
    
    # 23. 填充缺失值
    print("23. 用均值填充数值列:")
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    print(df_filled)
    print()
    
    # 31. 异常值检测
    print("31. 异常值检测:")
    df_outlier = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
    Q1, Q3 = df_outlier['value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df_outlier[(df_outlier['value'] < Q1-1.5*IQR) | (df_outlier['value'] > Q3+1.5*IQR)]
    print("异常值:\n", outliers)
    print()
    
    # 36. 字符串分割
    print("36. 字符串分割:")
    df_str = pd.DataFrame({'full_name': ['Alice Smith', 'Bob Johnson']})
    df_str[['first', 'last']] = df_str['full_name'].str.split(' ', expand=True)
    print(df_str)

# 运行练习
second_level_practice()
```

## 学习要点

1. **缺失值处理策略**：根据数据特点选择删除、填充或插值
2. **数据类型优化**：合理的数据类型可以节省内存和提高性能
3. **异常值识别**：理解业务背景，判断异常值的处理方式
4. **字符串操作**：掌握常用的字符串处理方法

## 第三关：数据操作（41-60题）

### 数据选择与过滤

**41. 多条件筛选：使用多个条件筛选数据**

```python
import pandas as pd
import numpy as np

# 创建示例数据
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 70000, 55000, 65000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
})

print("原始数据:")
print(df)

# 多条件筛选：年龄大于28且部门为IT
condition = (df['age'] > 28) & (df['department'] == 'IT')
filtered_df = df[condition]
print("\n41. 多条件筛选（年龄>28且部门=IT）:")
print(filtered_df)

# 使用|进行或操作
condition_or = (df['age'] < 28) | (df['salary'] > 60000)
filtered_df_or = df[condition_or]
print("\n或条件筛选（年龄<28或工资>60000）:")
print(filtered_df_or)
```

**42. isin操作：使用isin方法筛选数据**

```python
# 使用isin筛选特定值
departments = ['IT', 'HR']
filtered_isin = df[df['department'].isin(departments)]
print("42. isin筛选（部门为IT或HR）:")
print(filtered_isin)

# 排除某些值
filtered_not_in = df[~df['department'].isin(['Finance'])]
print("\n排除Finance部门:")
print(filtered_not_in)
```

**43. 字符串筛选：基于字符串内容筛选行**

```python
# 字符串包含特定内容
filtered_contains = df[df['name'].str.contains('a', case=False)]
print("43. 包含字母'a'的名字:")
print(filtered_contains)

# 字符串以特定内容开头
filtered_starts = df[df['name'].str.startswith('A')]
print("\n名字以'A'开头:")
print(filtered_starts)

# 字符串以特定内容结尾
filtered_ends = df[df['name'].str.endswith('e')]
print("\n名字以'e'结尾:")
print(filtered_ends)
```

**44. 正则表达式筛选：使用正则表达式筛选数据**

```python
# 使用正则表达式
filtered_regex = df[df['name'].str.contains('^[A-C]', regex=True)]
print("44. 名字以A-C开头（正则表达式）:")
print(filtered_regex)

# 更复杂的正则匹配
filtered_regex2 = df[df['name'].str.contains('(a|e)$', case=False, regex=True)]
print("\n名字以a或e结尾:")
print(filtered_regex2)
```

**45. 索引选择：使用loc和iloc进行数据选择**

```python
# 使用loc按标签选择
print("45. loc选择:")
print("选择前3行:")
print(df.loc[0:2])  # 包含结束位置

print("\n选择特定行和列:")
print(df.loc[1:3, ['name', 'salary']])

# 使用iloc按位置选择
print("\niloc选择:")
print("选择第1-3行（不含第3行）:")
print(df.iloc[0:2])  # 不包含结束位置

print("\n选择特定位置:")
print(df.iloc[[0, 2, 4], [0, 2]])  # 第0,2,4行的第0,2列

# 布尔索引与loc结合
print("\nloc布尔索引:")
print(df.loc[df['age'] > 30, ['name', 'department']])
```

### 数据排序

**46. 单列排序：按单列对DataFrame排序**

```python
# 按单列排序
df_sorted_age = df.sort_values('age')
print("46. 按年龄升序排序:")
print(df_sorted_age)

# 降序排序
df_sorted_age_desc = df.sort_values('age', ascending=False)
print("\n按年龄降序排序:")
print(df_sorted_age_desc)
```

**47. 多列排序：按多列排序，指定升序降序**

```python
# 多列排序：先按部门升序，再按工资降序
df_sorted_multi = df.sort_values(['department', 'salary'], ascending=[True, False])
print("47. 多列排序（部门↑, 工资↓）:")
print(df_sorted_multi)
```

**48. 索引排序：按索引对DataFrame排序**

```python
# 打乱索引进行演示
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("48. 打乱后的数据:")
print(df_shuffled)

# 按索引排序
df_sorted_index = df_shuffled.sort_index()
print("\n按索引排序后:")
print(df_sorted_index)
```

**49. 值排序：对Series进行值排序**

```python
# Series排序
age_series = df['age']
age_sorted = age_series.sort_values()
print("49. Series值排序:")
print(age_sorted)

# 获取排序后的索引
sorted_index = age_series.sort_values().index
print("\n排序后的索引:", sorted_index.tolist())
```

**50. 自定义排序：使用自定义键进行排序**

```python
# 自定义排序顺序
custom_order = ['HR', 'IT', 'Finance']
df_custom_sorted = df.copy()
df_custom_sorted['department'] = pd.Categorical(df_custom_sorted['department'], 
                                              categories=custom_order, 
                                              ordered=True)
df_custom_sorted = df_custom_sorted.sort_values('department')
print("50. 自定义部门排序顺序:")
print(df_custom_sorted)

# 使用key参数进行复杂排序（pandas 1.1.0+）
df_name_length = df.copy()
df_name_length = df_name_length.sort_values('name', key=lambda x: x.str.len())
print("\n按名字长度排序:")
print(df_name_length)
```

### 数据重塑

**51. 数据透视：创建数据透视表**

```python
# 创建更丰富的数据用于透视
df_pivot = pd.DataFrame({
    'department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance'],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'salary': [70000, 65000, 60000, 55000, 80000, 75000],
    'age': [35, 32, 30, 28, 40, 38]
})

print("51. 透视表数据:")
print(df_pivot)

# 创建透视表
pivot_table = df_pivot.pivot_table(
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean'
)
print("\n部门-性别的平均工资透视表:")
print(pivot_table)

# 多重聚合
pivot_multi = df_pivot.pivot_table(
    values=['salary', 'age'],
    index='department',
    aggfunc={'salary': ['mean', 'max'], 'age': 'mean'}
)
print("\n多重聚合透视表:")
print(pivot_multi)
```

**52. 长宽转换：在长格式和宽格式之间转换**

```python
# 宽格式转长格式
df_wide = pd.DataFrame({
    'id': [1, 2],
    'name': ['Alice', 'Bob'],
    '2020_salary': [50000, 60000],
    '2021_salary': [55000, 65000],
    '2022_salary': [60000, 70000]
})

print("52. 宽格式数据:")
print(df_wide)

# 转换为长格式
df_long = pd.wide_to_long(
    df_wide,
    stubnames='salary',
    i=['id', 'name'],
    j='year',
    sep='_',
    suffix='\\d+'
).reset_index()

print("\n转换为长格式:")
print(df_long)
```

**53. melt操作：将宽格式数据转换为长格式**

```python
# 使用melt进行长宽转换
df_melted = df_wide.melt(
    id_vars=['id', 'name'],
    value_vars=['2020_salary', '2021_salary', '2022_salary'],
    var_name='year',
    value_name='salary'
)

# 清理年份数据
df_melted['year'] = df_melted['year'].str.replace('_salary', '').astype(int)

print("53. melt转换后的长格式:")
print(df_melted)
```

**54. pivot操作：将长格式数据转换为宽格式**

```python
# 使用pivot将长格式转回宽格式
df_pivoted = df_melted.pivot(
    index=['id', 'name'],
    columns='year',
    values='salary'
).reset_index()

df_pivoted.columns.name = None  # 删除列名
df_pivoted.columns = ['id', 'name', '2020', '2021', '2022']

print("54. pivot转换回的宽格式:")
print(df_pivoted)
```

**55. 堆叠操作：使用stack和unstack重塑数据**

```python
# 创建多索引数据
df_multi = df_pivot.set_index(['department', 'gender'])[['salary', 'age']]
print("55. 多索引数据:")
print(df_multi)

# 堆叠操作
stacked = df_multi.stack()
print("\nstack操作后:")
print(stacked)

# 取消堆叠
unstacked = stacked.unstack()
print("\nunstack操作后:")
print(unstacked)

# 指定取消堆叠的级别
unstacked_gender = df_multi.unstack(level='gender')
print("\n按gender取消堆叠:")
print(unstacked_gender)
```

### 数据合并

**56. concat合并：使用concat连接多个DataFrame**

```python
# 创建多个DataFrame
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})

print("56. 原始DataFrames:")
print("df1:\n", df1)
print("df2:\n", df2)
print("df3:\n", df3)

# 纵向连接
df_concat_vertical = pd.concat([df1, df2, df3], ignore_index=True)
print("\n纵向连接:")
print(df_concat_vertical)

# 横向连接
df4 = pd.DataFrame({'C': [13, 14], 'D': [15, 16]})
df_concat_horizontal = pd.concat([df1, df4], axis=1)
print("\n横向连接:")
print(df_concat_horizontal)
```

**57. merge合并：使用merge进行数据库式合并**

```python
# 创建两个有关联的DataFrame
df_left = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'dept_id': [101, 102, 101, 103]
})

df_right = pd.DataFrame({
    'dept_id': [101, 102, 104],
    'dept_name': ['IT', 'HR', 'Finance']
})

print("57. 合并数据:")
print("左表:\n", df_left)
print("右表:\n", df_right)

# 内连接
df_inner = pd.merge(df_left, df_right, on='dept_id', how='inner')
print("\n内连接结果:")
print(df_inner)
```

**58. join操作：使用join方法合并数据**

```python
# 设置索引后进行join
df_left_indexed = df_left.set_index('dept_id')
df_right_indexed = df_right.set_index('dept_id')

# 左连接
df_join_left = df_left_indexed.join(df_right_indexed, how='left')
print("58. 左连接结果:")
print(df_join_left)

# 右连接
df_join_right = df_left_indexed.join(df_right_indexed, how='right')
print("\n右连接结果:")
print(df_join_right)
```

**59. 多表合并：合并三个或更多表**

```python
# 创建第三个表
df_salary = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'salary': [50000, 60000, 70000, 55000]
})

print("59. 多表合并:")
print("工资表:\n", df_salary)

# 合并三个表
df_three_way = pd.merge(df_left, df_right, on='dept_id', how='left')
df_three_way = pd.merge(df_three_way, df_salary, on='id', how='left')
print("\n三表合并结果:")
print(df_three_way)
```

**60. 合并类型：掌握不同类型的合并（inner, outer, left, right）**

```python
print("60. 不同合并类型对比:")

# 内连接
df_inner = pd.merge(df_left, df_right, on='dept_id', how='inner')
print("内连接（只保留两边都有的键）:")
print(df_inner)

# 左连接
df_left_join = pd.merge(df_left, df_right, on='dept_id', how='left')
print("\n左连接（保留左表所有记录）:")
print(df_left_join)

# 右连接
df_right_join = pd.merge(df_left, df_right, on='dept_id', how='right')
print("\n右连接（保留右表所有记录）:")
print(df_right_join)

# 外连接
df_outer = pd.merge(df_left, df_right, on='dept_id', how='outer')
print("\n外连接（保留所有记录）:")
print(df_outer)

# 合并冲突处理
df_right_dup = pd.DataFrame({
    'dept_id': [101, 101, 102],
    'dept_name': ['IT', 'Technology', 'HR']
})

df_conflict = pd.merge(df_left, df_right_dup, on='dept_id', how='left')
print("\n合并冲突处理:")
print(df_conflict)
```

## 完整示例代码

```python
import pandas as pd
import numpy as np

def third_level_practice():
    print("=== 第三关：数据操作 ===\n")
    
    # 创建示例数据
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    })
    
    # 41. 多条件筛选
    print("41. 多条件筛选:")
    condition = (df['age'] > 28) & (df['department'] == 'IT')
    print(df[condition])
    print()
    
    # 46. 排序
    print("46. 多列排序:")
    df_sorted = df.sort_values(['department', 'salary'], ascending=[True, False])
    print(df_sorted)
    print()
    
    # 51. 透视表
    df_pivot = pd.DataFrame({
        'department': ['IT', 'IT', 'HR', 'HR'],
        'gender': ['M', 'F', 'M', 'F'],
        'salary': [70000, 65000, 60000, 55000]
    })
    pivot_table = df_pivot.pivot_table(values='salary', index='department', columns='gender')
    print("51. 透视表:")
    print(pivot_table)
    print()
    
    # 56. 合并
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    df_concat = pd.concat([df1, df2], ignore_index=True)
    print("56. 数据合并:")
    print(df_concat)

# 运行练习
third_level_practice()
```

## 学习要点

1. **灵活的数据选择**：掌握loc/iloc、布尔索引、字符串筛选等多种选择方式
2. **多种排序方式**：单列、多列、自定义排序等
3. **数据重塑技巧**：透视表、长宽转换、堆叠操作
4. **数据合并策略**：concat、merge、join的不同应用场景

## 第四关：分组与聚合（61-80题）

### 基础分组

**61. 简单分组：按单列分组并计算统计量**

```python
import pandas as pd
import numpy as np

# 创建示例数据
df = pd.DataFrame({
    'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance', 'IT', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry'],
    'Salary': [70000, 80000, 60000, 65000, 75000, 85000, 72000, 62000],
    'Age': [25, 30, 28, 35, 40, 45, 32, 29],
    'Experience': [2, 5, 3, 8, 12, 15, 4, 4]
})

print("原始数据:")
print(df)

# 按部门分组并计算平均工资
grouped = df.groupby('Department')['Salary'].mean()
print("\n61. 各部门平均工资:")
print(grouped)

# 多个统计量
grouped_multi = df.groupby('Department').agg({
    'Salary': ['mean', 'min', 'max', 'std'],
    'Age': 'mean',
    'Experience': 'sum'
})
print("\n各部门多维度统计:")
print(grouped_multi)
```

**62. 多列分组：按多列进行分组操作**

```python
# 添加性别列
df['Gender'] = ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M']

# 按部门和性别分组
grouped_multi = df.groupby(['Department', 'Gender']).agg({
    'Salary': 'mean',
    'Age': ['mean', 'count']
})
print("62. 按部门和性别分组:")
print(grouped_multi)

# 重置索引使结果更易读
grouped_reset = grouped_multi.reset_index()
print("\n重置索引后:")
print(grouped_reset)
```

**63. 分组计数：统计每组的数量**

```python
# 统计每个部门的员工数量
count_by_dept = df.groupby('Department').size()
print("63. 各部门员工数量:")
print(count_by_dept)

# 使用count()统计非空值
count_salary = df.groupby('Department')['Salary'].count()
print("\n各部门有工资记录的员工数:")
print(count_salary)

# 多列分组计数
count_multi = df.groupby(['Department', 'Gender']).size().reset_index(name='Count')
print("\n各部门各性别人数:")
print(count_multi)
```

**64. 分组求和：计算每组的总和**

```python
# 各部门工资总额
salary_sum = df.groupby('Department')['Salary'].sum()
print("64. 各部门工资总额:")
print(salary_sum)

# 多列分组求和
sum_multi = df.groupby(['Department', 'Gender'])['Salary'].sum()
print("\n各部门各性别工资总额:")
print(sum_multi)
```

**65. 分组平均值：计算每组的平均值**

```python
# 各部门平均工资
salary_mean = df.groupby('Department')['Salary'].mean()
print("65. 各部门平均工资:")
print(salary_mean)

# 多列平均值
mean_multi = df.groupby(['Department', 'Gender']).agg({
    'Salary': 'mean',
    'Age': 'mean',
    'Experience': 'mean'
})
print("\n各部门各性别多指标平均值:")
print(mean_multi)
```

### 高级分组

**66. 自定义聚合：使用agg进行自定义聚合**

```python
# 自定义聚合函数
def salary_range(x):
    return x.max() - x.min()

def q90(x):
    return x.quantile(0.9)

# 使用自定义函数
custom_agg = df.groupby('Department').agg({
    'Salary': [salary_range, q90, 'mean'],
    'Age': ['min', 'max', lambda x: x.max() - x.min()]  # 使用lambda
})
print("66. 自定义聚合:")
print(custom_agg)

# 重命名聚合列
custom_agg_named = df.groupby('Department').agg(
    avg_salary=('Salary', 'mean'),
    salary_range=('Salary', salary_range),
    total_experience=('Experience', 'sum')
)
print("\n命名聚合结果:")
print(custom_agg_named)
```

**67. 多重聚合：对同一列应用多个聚合函数**

```python
# 对工资列应用多个聚合函数
multi_agg_salary = df.groupby('Department')['Salary'].agg([
    'mean',
    'std',
    'min',
    'max',
    lambda x: x.quantile(0.75) - x.quantile(0.25),  # IQR
    'count'
]).round(2)

# 重命名列
multi_agg_salary.columns = ['平均工资', '标准差', '最低工资', '最高工资', '四分位距', '人数']
print("67. 工资多重聚合:")
print(multi_agg_salary)
```

**68. 分组应用函数：使用apply对分组应用自定义函数**

```python
# 自定义函数：为每个员工计算相对于部门平均工资的差异
def salary_analysis(group):
    avg_salary = group['Salary'].mean()
    group['SalaryVsAvg'] = group['Salary'] - avg_salary
    group['SalaryRatio'] = group['Salary'] / avg_salary
    return group

df_analyzed = df.groupby('Department').apply(salary_analysis)
print("68. 应用自定义函数分析工资:")
print(df_analyzed)

# 返回汇总统计的自定义函数
def department_summary(group):
    return pd.Series({
        '员工数': len(group),
        '总工资': group['Salary'].sum(),
        '平均工资': group['Salary'].mean(),
        '平均年龄': group['Age'].mean(),
        '最大经验': group['Experience'].max()
    })

summary = df.groupby('Department').apply(department_summary)
print("\n部门汇总:")
print(summary)
```

**69. 分组转换：使用transform进行分组转换**

```python
# 计算每个员工工资与部门平均工资的差异
df['DeptAvgSalary'] = df.groupby('Department')['Salary'].transform('mean')
df['SalaryDiffFromAvg'] = df['Salary'] - df['DeptAvgSalary']

print("69. 分组转换 - 工资与部门平均比较:")
print(df[['Employee', 'Department', 'Salary', 'DeptAvgSalary', 'SalaryDiffFromAvg']])

# Z-score标准化（组内）
df['SalaryZScore'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("\nZ-score标准化:")
print(df[['Employee', 'Department', 'Salary', 'SalaryZScore']])
```

**70. 分组过滤：使用filter过滤分组**

```python
# 过滤出员工数大于2的部门
filtered_dept = df.groupby('Department').filter(lambda x: len(x) > 2)
print("70. 员工数大于2的部门:")
print(filtered_dept)

# 过滤出平均工资大于65000的部门
filtered_high_salary = df.groupby('Department').filter(lambda x: x['Salary'].mean() > 65000)
print("\n平均工资大于65000的部门:")
print(filtered_high_salary)

# 过滤出有女性员工的部门
filtered_has_female = df.groupby('Department').filter(lambda x: (x['Gender'] == 'F').any())
print("\n有女性员工的部门:")
print(filtered_has_female)
```

### 时间序列分组

**71. 按时间分组：按年、月、日分组**

```python
# 创建时间序列数据
dates = pd.date_range('2023-01-01', periods=100, freq='D')
time_df = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(100, 1000, 100),
    'Product': np.random.choice(['A', 'B', 'C'], 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
})

print("71. 时间序列数据（前10行）:")
print(time_df.head(10))

# 按月份分组
monthly_sales = time_df.groupby(time_df['Date'].dt.month)['Sales'].sum()
print("\n各月销售总额:")
print(monthly_sales)

# 按年份和月份分组
time_df['Year'] = time_df['Date'].dt.year
time_df['Month'] = time_df['Date'].dt.month
monthly_sales_detailed = time_df.groupby(['Year', 'Month']).agg({
    'Sales': ['sum', 'mean', 'count'],
    'Product': lambda x: x.mode()[0]  # 最常销售的产品
})
print("\n按月详细统计:")
print(monthly_sales_detailed)
```

**72. 重采样：对时间序列数据进行重采样**

```python
# 设置日期为索引
time_series = time_df.set_index('Date')

# 按周重采样（计算每周销售总额）
weekly_sales = time_series['Sales'].resample('W').sum()
print("72. 每周销售总额:")
print(weekly_sales.head())

# 多重重采样聚合
weekly_agg = time_series.resample('W').agg({
    'Sales': ['sum', 'mean', 'std'],
    'Product': 'count'
})
print("\n每周多重聚合:")
print(weekly_agg.head())
```

**73. 滚动窗口：计算滚动平均值**

```python
# 计算7天滚动平均值
time_series['RollingMean7'] = time_series['Sales'].rolling(window=7).mean()

# 计算7天滚动总和
time_series['RollingSum7'] = time_series['Sales'].rolling(window=7).sum()

print("73. 滚动窗口计算（前15行）:")
print(time_series[['Sales', 'RollingMean7', 'RollingSum7']].head(15))

# 最小周期数为1的滚动计算（避免NaN）
time_series['RollingMean7_min1'] = time_series['Sales'].rolling(window=7, min_periods=1).mean()
print("\n最小周期数为1的滚动平均值:")
print(time_series[['Sales', 'RollingMean7', 'RollingMean7_min1']].head(10))
```

**74. 时间窗口聚合：在时间窗口内进行聚合**

```python
# 使用rolling进行多个聚合
rolling_agg = time_series['Sales'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])
print("74. 7天滚动窗口多重聚合:")
print(rolling_agg.head(10))

# 扩展窗口（累积但带衰减）
expanding_mean = time_series['Sales'].expanding().mean()
print("\n扩展窗口均值:")
print(expanding_mean.head(10))
```

**75. 时间序列插值：对时间序列进行插值**

```python
# 创建有缺失值的时间序列
ts_with_gaps = time_series.copy()
ts_with_gaps.loc[ts_with_gaps.sample(20).index, 'Sales'] = np.nan

print("75. 含缺失值的时间序列:")
print(f"缺失值数量: {ts_with_gaps['Sales'].isnull().sum()}")

# 前向填充
ts_ffill = ts_with_gaps['Sales'].fillna(method='ffill')

# 线性插值
ts_interpolated = ts_with_gaps['Sales'].interpolate(method='linear')

print("\n插值结果对比（片段）:")
comparison = pd.DataFrame({
    'Original': ts_with_gaps['Sales'].head(30),
    'ForwardFill': ts_ffill.head(30),
    'LinearInterp': ts_interpolated.head(30)
})
print(comparison[comparison['Original'].isnull()].head())
```

### 复杂聚合

**76. 分位数计算：计算各组的分位数**

```python
# 计算各部门工资的分位数
quantiles = df.groupby('Department')['Salary'].quantile([0.25, 0.5, 0.75, 0.9])
print("76. 各部门工资分位数:")
print(quantiles)

# 分位数展开形式
quantiles_unstack = df.groupby('Department')['Salary'].quantile([0.25, 0.5, 0.75]).unstack()
quantiles_unstack.columns = ['Q25', 'Median', 'Q75']
print("\n分位数展开:")
print(quantiles_unstack)

# 自定义分位数计算
def q10(x):
    return x.quantile(0.1)

def q90(x):
    return x.quantile(0.9)

custom_quantiles = df.groupby('Department').agg({
    'Salary': [q10, 'median', q90, 'mean']
})
print("\n自定义分位数:")
print(custom_quantiles)
```

**77. 累积统计：计算累积和、累积均值**

```python
# 按部门排序后计算累积和
df_sorted = df.sort_values(['Department', 'Salary'])
df_sorted['CumulativeSalary'] = df_sorted.groupby('Department')['Salary'].cumsum()
df_sorted['CumulativeCount'] = df_sorted.groupby('Department').cumcount() + 1
df_sorted['CumulativeAvg'] = df_sorted['CumulativeSalary'] / df_sorted['CumulativeCount']

print("77. 累积统计:")
print(df_sorted[['Department', 'Employee', 'Salary', 'CumulativeSalary', 'CumulativeAvg']])

# 扩展窗口均值
df_sorted['ExpandingMean'] = df_sorted.groupby('Department')['Salary'].expanding().mean().values
print("\n扩展窗口均值:")
print(df_sorted[['Department', 'Employee', 'Salary', 'ExpandingMean']])
```

**78. 组内排名：在每组内进行排名**

```python
# 按部门分组，按工资排名
df['SalaryRankInDept'] = df.groupby('Department')['Salary'].rank(ascending=False, method='dense')
df['AgeRankInDept'] = df.groupby('Department')['Age'].rank(ascending=True, method='dense')

print("78. 组内排名:")
print(df[['Department', 'Employee', 'Salary', 'SalaryRankInDept', 'Age', 'AgeRankInDept']])

# 多种排名方法
ranking_methods = ['average', 'min', 'max', 'first', 'dense']
for method in ranking_methods:
    df[f'Rank_{method}'] = df.groupby('Department')['Salary'].rank(method=method, ascending=False)

print("\n不同排名方法对比:")
print(df[['Employee', 'Department', 'Salary'] + [f'Rank_{method}' for method in ranking_methods]])
```

**79. 组间比较：计算组间差异**

```python
# 计算每个部门平均工资与总体平均工资的差异
dept_means = df.groupby('Department')['Salary'].mean()
overall_mean = df['Salary'].mean()
dept_vs_overall = dept_means - overall_mean

print("79. 部门平均工资与总体平均的差异:")
print(dept_vs_overall)

# 计算每个部门与最高部门平均工资的差异
max_dept_mean = dept_means.max()
dept_vs_max = dept_means - max_dept_mean

print("\n部门平均工资与最高部门的差异:")
print(dept_vs_max)

# 在原始数据中添加比较列
df['VsDeptAvg'] = df['Salary'] - df.groupby('Department')['Salary'].transform('mean')
df['VsOverallAvg'] = df['Salary'] - overall_mean

print("\n个体与平均值的比较:")
print(df[['Employee', 'Department', 'Salary', 'VsDeptAvg', 'VsOverallAvg']])
```

**80. 分组标准化：在组内进行标准化**

```python
# Z-score标准化（组内）
df['SalaryZScore'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Min-Max标准化（组内，缩放到0-1）
df['SalaryMinMax'] = df.groupby('Department')['Salary'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# 百分位数排名（组内）
df['SalaryPercentile'] = df.groupby('Department')['Salary'].rank(pct=True)

print("80. 分组标准化结果:")
standardized_cols = ['Employee', 'Department', 'Salary', 'SalaryZScore', 'SalaryMinMax', 'SalaryPercentile']
print(df[standardized_cols].round(3))
```

## 完整示例代码

```python
import pandas as pd
import numpy as np

def fourth_level_practice():
    print("=== 第四关：分组与聚合 ===\n")
    
    # 创建示例数据
    df = pd.DataFrame({
        'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance', 'IT', 'HR'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry'],
        'Salary': [70000, 80000, 60000, 65000, 75000, 85000, 72000, 62000],
        'Age': [25, 30, 28, 35, 40, 45, 32, 29]
    })
    
    # 61. 简单分组
    print("61. 各部门平均工资:")
    grouped = df.groupby('Department')['Salary'].mean()
    print(grouped)
    print()
    
    # 66. 自定义聚合
    def salary_range(x):
        return x.max() - x.min()
    
    custom_agg = df.groupby('Department').agg({
        'Salary': [salary_range, 'mean'],
        'Age': ['min', 'max']
    })
    print("66. 自定义聚合:")
    print(custom_agg)
    print()
    
    # 69. 分组转换
    df['DeptAvgSalary'] = df.groupby('Department')['Salary'].transform('mean')
    print("69. 分组转换 - 部门平均工资:")
    print(df[['Employee', 'Department', 'Salary', 'DeptAvgSalary']])
    print()
    
    # 78. 组内排名
    df['SalaryRank'] = df.groupby('Department')['Salary'].rank(ascending=False)
    print("78. 组内工资排名:")
    print(df[['Employee', 'Department', 'Salary', 'SalaryRank']])

# 运行练习
fourth_level_practice()
```

## 学习要点

1. **分组策略**：掌握单列、多列、时间序列等不同分组方式
2. **聚合函数**：熟练使用内置函数和自定义函数进行聚合
3. **转换技巧**：transform的巧妙应用，避免循环操作
4. **高级分析**：排名、标准化、累积统计等复杂分析
5. **时间序列**：时间分组、重采样、滚动窗口等时间相关操作

## 第五关：高级应用（81-100题）

### 数据可视化

**81. 基础绘图：使用pandas绘制线图和柱状图**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
df = pd.DataFrame({
    'Year': [2010, 2011, 2012, 2013, 2014, 2015],
    'Sales': [100, 120, 140, 160, 180, 200],
    'Profit': [10, 15, 20, 25, 30, 35]
})

# 设置Year为索引
df.set_index('Year', inplace=True)

# 绘制线图
plt.figure(figsize=(10, 6))
df['Sales'].plot(kind='line', title='Sales Over Years', marker='o')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# 绘制柱状图
plt.figure(figsize=(10, 6))
df['Profit'].plot(kind='bar', title='Profit Over Years', color='skyblue')
plt.ylabel('Profit')
plt.show()

# 多系列线图
plt.figure(figsize=(10, 6))
df[['Sales', 'Profit']].plot(kind='line', title='Sales and Profit Over Years')
plt.ylabel('Amount')
plt.grid(True)
plt.show()
```

**82. 分组绘图：按分组绘制多个子图**

```python
# 创建分组数据
df_group = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'] * 10,
    'Value': np.random.randn(60) + np.repeat([1, 2, 3], 20),
    'Type': ['X', 'Y'] * 30
})

# 按Category分组绘制箱线图
plt.figure(figsize=(12, 6))
df_group.boxplot(column='Value', by='Category')
plt.title('Value Distribution by Category')
plt.suptitle('')  # 移除自动标题
plt.show()

# 使用子图绘制每个类别的直方图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
categories = df_group['Category'].unique()

for i, category in enumerate(categories):
    df_group[df_group['Category'] == category]['Value'].hist(
        ax=axes[i], alpha=0.7, bins=10
    )
    axes[i].set_title(f'Category {category}')
    axes[i].set_xlabel('Value')

plt.tight_layout()
plt.show()
```

**83. 直方图：绘制数值分布直方图**

```python
# 生成正态分布数据
np.random.seed(42)
data = np.random.normal(100, 15, 1000)
series = pd.Series(data, name='IQ Scores')

# 绘制直方图
plt.figure(figsize=(10, 6))
series.hist(bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(series.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {series.mean():.1f}')
plt.axvline(series.median(), color='green', linestyle='dashed', linewidth=1, label=f'Median: {series.median():.1f}')
plt.title('Distribution of IQ Scores')
plt.xlabel('IQ Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 密度图
plt.figure(figsize=(10, 6))
series.plot(kind='density', title='Density Plot of IQ Scores')
plt.xlabel('IQ Score')
plt.show()
```

**84. 箱线图：绘制箱线图检测异常值**

```python
# 创建有异常值的数据
df_box = pd.DataFrame({
    'Normal': np.random.normal(0, 1, 100),
    'With Outliers': np.concatenate([np.random.normal(0, 1, 95), [10, -8, 9, -7, 8]]),
    'Group A': np.random.normal(5, 2, 100),
    'Group B': np.random.normal(8, 1.5, 100)
})

# 绘制箱线图
plt.figure(figsize=(12, 6))
df_box[['Normal', 'With Outliers']].boxplot()
plt.title('Boxplot: Normal vs With Outliers')
plt.ylabel('Values')
plt.show()

# 分组箱线图
plt.figure(figsize=(10, 6))
df_box[['Group A', 'Group B']].boxplot()
plt.title('Boxplot by Group')
plt.ylabel('Values')
plt.show()

# 使用seaborn绘制更漂亮的箱线图（可选）
try:
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_box[['Group A', 'Group B']])
    plt.title('Seaborn Boxplot')
    plt.show()
except ImportError:
    print("Seaborn未安装，使用matplotlib版本")
```

**85. 散点图：绘制相关性散点图**

```python
# 创建相关数据
np.random.seed(42)
n = 100
x = np.random.normal(0, 1, n)
y = 2 * x + np.random.normal(0, 0.5, n)  # y与x相关
z = np.random.normal(0, 1, n)  # z与x不相关

df_scatter = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

# 绘制散点图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df_scatter['X'], df_scatter['Y'], alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Correlation: {df_scatter[["X", "Y"]].corr().iloc[0,1]:.2f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df_scatter['X'], df_scatter['Z'], alpha=0.6, color='red')
plt.xlabel('X')
plt.ylabel('Z')
plt.title(f'Correlation: {df_scatter[["X", "Z"]].corr().iloc[0,1]:.2f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 使用pandas绘制
df_scatter.plot(kind='scatter', x='X', y='Y', title='X vs Y Correlation', alpha=0.6)
plt.show()
```

### 性能优化

**86. 内存优化：优化DataFrame的内存使用**

```python
# 创建大型DataFrame
large_df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 1000000),
    'float_col': np.random.randn(1000000),
    'category_col': np.random.choice(['A', 'B', 'C', 'D'], 1000000),
    'bool_col': np.random.choice([True, False], 1000000)
})

print("86. 原始内存使用:")
print(large_df.info(memory_usage='deep'))

# 优化数据类型
optimized_df = large_df.copy()

# 整数列优化
optimized_df['int_col'] = pd.to_numeric(optimized_df['int_col'], downcast='integer')

# 浮点数列优化
optimized_df['float_col'] = pd.to_numeric(optimized_df['float_col'], downcast='float')

# 分类列优化
optimized_df['category_col'] = optimized_df['category_col'].astype('category')

# 布尔列优化（已经是bool类型，无需优化）

print("\n优化后内存使用:")
print(optimized_df.info(memory_usage='deep'))

# 计算节省的内存
orig_memory = large_df.memory_usage(deep=True).sum()
opt_memory = optimized_df.memory_usage(deep=True).sum()
savings = (orig_memory - opt_memory) / orig_memory * 100

print(f"\n内存节省: {savings:.1f}%")
print(f"原始内存: {orig_memory / 1024**2:.2f} MB")
print(f"优化后内存: {opt_memory / 1024**2:.2f} MB")
```

**87. 数据类型优化：选择合适的数据类型**

```python
# 分析每列的最佳数据类型
def analyze_column_dtype(series):
    name = series.name
    dtype = series.dtype
    memory = series.memory_usage(deep=True)
    unique_ratio = series.nunique() / len(series)
    
    # 建议的数据类型
    if dtype == 'object':
        if unique_ratio < 0.5:  # 低基数字符串适合category
            suggested = 'category'
        else:
            suggested = 'object'
    elif 'int' in str(dtype):
        suggested = f"int{pd.to_numeric(series, downcast='integer').dtype.itemsize * 8}"
    elif 'float' in str(dtype):
        suggested = f"float{pd.to_numeric(series, downcast='float').dtype.itemsize * 8}"
    else:
        suggested = str(dtype)
    
    return {
        'Column': name,
        'Current_Dtype': str(dtype),
        'Suggested_Dtype': suggested,
        'Memory_MB': memory / 1024**2,
        'Unique_Ratio': unique_ratio
    }

# 分析所有列
analysis = [analyze_column_dtype(large_df[col]) for col in large_df.columns]
analysis_df = pd.DataFrame(analysis)
print("87. 数据类型分析:")
print(analysis_df)
```

**88. 分块处理：处理大型数据集**

```python
# 模拟处理大型CSV文件（分块读取和处理）
def process_large_file_chunked(file_path, chunk_size=10000):
    """分块处理大型文件"""
    chunks = []
    total_rows = 0
    
    # 分块读取
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        # 处理每个块（示例：过滤和转换）
        processed_chunk = chunk[chunk['int_col'] > 50]  # 过滤条件
        processed_chunk['processed_col'] = processed_chunk['float_col'] * 2  # 转换
        
        chunks.append(processed_chunk)
        total_rows += len(processed_chunk)
        
        print(f"处理第{i+1}个块，当前总行数: {total_rows}")
    
    # 合并所有块
    result = pd.concat(chunks, ignore_index=True)
    return result

# 由于我们没有实际的大文件，创建一个模拟版本
def demo_chunk_processing():
    """演示分块处理"""
    # 创建模拟的大DataFrame
    large_data = pd.DataFrame({
        'int_col': range(100000),
        'float_col': np.random.randn(100000),
        'text_col': ['text'] * 100000
    })
    
    # 模拟分块处理
    chunk_size = 10000
    results = []
    
    for start in range(0, len(large_data), chunk_size):
        end = start + chunk_size
        chunk = large_data.iloc[start:end]
        
        # 处理逻辑
        processed_chunk = chunk[chunk['int_col'] % 2 == 0]  # 只保留偶数
        processed_chunk['squared'] = processed_chunk['int_col'] ** 2
        
        results.append(processed_chunk)
    
    final_result = pd.concat(results, ignore_index=True)
    print(f"原始数据行数: {len(large_data)}")
    print(f"处理後数据行数: {len(final_result)}")
    return final_result

demo_result = demo_chunk_processing()
```

**89. 向量化操作：使用向量化提高性能**

```python
import time

# 创建测试数据
df_perf = pd.DataFrame({
    'A': np.random.rand(10000),
    'B': np.random.rand(10000),
    'C': np.random.rand(10000)
})

# 非向量化操作（慢）
def non_vectorized_operation(df):
    result = []
    for i in range(len(df)):
        if df.iloc[i]['A'] > 0.5:
            result.append(df.iloc[i]['B'] * df.iloc[i]['C'])
        else:
            result.append(df.iloc[i]['B'] + df.iloc[i]['C'])
    return pd.Series(result)

# 向量化操作（快）
def vectorized_operation(df):
    condition = df['A'] > 0.5
    return np.where(condition, df['B'] * df['C'], df['B'] + df['C'])

# 性能比较
print("89. 向量化操作性能比较:")

start_time = time.time()
result_non_vec = non_vectorized_operation(df_perf)
non_vec_time = time.time() - start_time

start_time = time.time()
result_vec = vectorized_operation(df_perf)
vec_time = time.time() - start_time

print(f"非向量化时间: {non_vec_time:.4f}秒")
print(f"向量化时间: {vec_time:.4f}秒")
print(f"速度提升: {non_vec_time/vec_time:.1f}倍")

# 验证结果一致性
print(f"结果一致: {np.allclose(result_non_vec, result_vec)}")
```

**90. 避免循环：用pandas操作替代Python循环**

```python
# 创建示例数据
df_loop = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value1': np.random.randn(1000),
    'value2': np.random.randn(1000)
})

print("90. 避免循环的优化技巧:")

# 不好的做法：使用iterrows()
def slow_group_operations(df):
    results = []
    for category in df['category'].unique():
        group = df[df['category'] == category]
        results.append({
            'category': category,
            'mean_val1': group['value1'].mean(),
            'mean_val2': group['value2'].mean()
        })
    return pd.DataFrame(results)

# 好的做法：使用groupby
def fast_group_operations(df):
    return df.groupby('category').agg({
        'value1': 'mean',
        'value2': 'mean'
    }).reset_index()

# 性能比较
import time

start = time.time()
slow_result = slow_group_operations(df_loop)
slow_time = time.time() - start

start = time.time()
fast_result = fast_group_operations(df_loop)
fast_time = time.time() - start

print(f"循环方法时间: {slow_time:.4f}秒")
print(f"向量化方法时间: {fast_time:.4f}秒")
print(f"性能提升: {slow_time/fast_time:.1f}倍")

# 更多避免循环的技巧
print("\n更多向量化操作示例:")

# 条件赋值 - 不好的做法
df_loop['category_type'] = ''
for i in range(len(df_loop)):
    if df_loop.iloc[i]['value1'] > 0:
        df_loop.iloc[i, df_loop.columns.get_loc('category_type')] = 'High'
    else:
        df_loop.iloc[i, df_loop.columns.get_loc('category_type')] = 'Low'

# 条件赋值 - 好的做法
df_loop['category_type_better'] = np.where(df_loop['value1'] > 0, 'High', 'Low')

print("条件赋值结果一致:", (df_loop['category_type'] == df_loop['category_type_better']).all())
```

### 高级技巧

**91. 多级索引：创建和操作多级索引**

```python
# 创建多级索引DataFrame
arrays = [
    ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    [1, 2, 3, 1, 2, 3, 1, 2, 3]
]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df_multi = pd.DataFrame({
    'value1': np.random.randn(9),
    'value2': np.random.randn(9),
    'value3': np.random.randn(9)
}, index=index)

print("91. 多级索引DataFrame:")
print(df_multi)

# 选择数据
print("\n选择first为A的所有数据:")
print(df_multi.loc['A'])

print("\n选择first为B且second为2的数据:")
print(df_multi.loc[('B', 2)])

print("\n使用xs方法选择second为1的所有数据:")
print(df_multi.xs(1, level='second'))

# 多级索引的聚合操作
print("\n按first级别聚合:")
print(df_multi.groupby(level='first').mean())

# 堆叠和取消堆叠
print("\n取消堆叠second级别:")
print(df_multi.unstack('second'))

# 从列创建多级索引
df_flat = pd.DataFrame({
    'A_value1': [1, 2], 'A_value2': [3, 4],
    'B_value1': [5, 6], 'B_value2': [7, 8]
})

# 设置多级列索引
df_flat.columns = pd.MultiIndex.from_tuples(
    [('A', 'value1'), ('A', 'value2'), ('B', 'value1'), ('B', 'value2')]
)
print("\n多级列索引:")
print(df_flat)
```

**92. 自定义函数：编写自定义聚合函数**

```python
# 自定义聚合函数
def weighted_mean(group):
    """计算加权平均值"""
    return np.average(group['value'], weights=group['weight'])

def geometric_mean(x):
    """计算几何平均数"""
    return np.exp(np.mean(np.log(x)))

def outlier_ratio(x, threshold=2):
    """计算异常值比例"""
    z_scores = (x - x.mean()) / x.std()
    return (np.abs(z_scores) > threshold).mean()

def trend_direction(x):
    """判断趋势方向（基于最后两个值）"""
    if len(x) < 2:
        return 'Unknown'
    diff = x.iloc[-1] - x.iloc[-2]
    return 'Up' if diff > 0 else 'Down' if diff < 0 else 'Flat'

# 使用自定义函数
df_custom = pd.DataFrame({
    'group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'value': [10, 20, 30, 15, 25, 35, 5, 100, 15],  # 100是异常值
    'weight': [1, 2, 1, 1, 1, 2, 1, 1, 2],
    'sequence': [1, 2, 3, 1, 2, 3, 1, 2, 3]
})

print("92. 自定义聚合函数:")

# 应用自定义函数
result = df_custom.groupby('group').agg({
    'value': [
        ('mean', 'mean'),
        ('geometric_mean', geometric_mean),
        ('outlier_ratio', lambda x: outlier_ratio(x)),
        ('trend', trend_direction)
    ],
    'weight': [('weighted_mean', lambda g: weighted_mean(df_custom[df_custom['group'] == g.name]))]
})

print(result)
```

**93. 数据验证：验证数据质量和完整性**

```python
def validate_dataframe(df):
    """综合数据验证函数"""
    validation_results = {}
    
    # 1. 缺失值检查
    missing_info = df.isnull().sum()
    validation_results['missing_values'] = missing_info[missing_info > 0].to_dict()
    
    # 2. 数据类型检查
    dtype_info = df.dtypes.to_dict()
    validation_results['data_types'] = dtype_info
    
    # 3. 重复行检查
    duplicate_count = df.duplicated().sum()
    validation_results['duplicate_rows'] = duplicate_count
    
    # 4. 数值范围检查
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    range_violations = {}
    
    for col in numeric_cols:
        col_min, col_max = df[col].min(), df[col].max()
        # 检查是否有可能的异常值（超出4个标准差）
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outlier_count = (z_scores > 4).sum()
        
        range_violations[col] = {
            'min': col_min,
            'max': col_max,
            'outliers_4std': outlier_count
        }
    
    validation_results['numeric_ranges'] = range_violations
    
    # 5. 分类数据检查
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_info = {}
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else None
        
        categorical_info[col] = {
            'unique_values': unique_count,
            'most_frequent': most_frequent,
            'sample_values': df[col].unique()[:5].tolist()  # 前5个唯一值
        }
    
    validation_results['categorical_info'] = categorical_info
    
    return validation_results

# 创建测试数据（包含一些数据质量问题）
df_validate = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['Alice', 'Bob', 'Charlie', None, 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'age': [25, 30, 35, 40, 45, 200, 55, 60, 65, 70],  # 200是异常值
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'IT', 'Finance', 'IT', 'HR'],
    'join_date': pd.date_range('2020-01-01', periods=10, freq='M')
})

# 添加重复行
df_validate = pd.concat([df_validate, df_validate.iloc[[0, 1]]], ignore_index=True)

print("93. 数据验证结果:")
validation_results = validate_dataframe(df_validate)

for check_name, result in validation_results.items():
    print(f"\n{check_name.upper().replace('_', ' ')}:")
    if isinstance(result, dict) and result:
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {result}")
```

**94. 管道操作：使用pipe创建数据处理管道**

```python
# 定义管道函数
def remove_duplicates(df):
    """移除重复行"""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """填充缺失值"""
    df_filled = df.copy()
    
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if df_filled[col].dtype in [np.number]:
                if strategy == 'mean':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif strategy == 'median':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            else:
                df_filled[col] = df_filled[col].fillna('Unknown')
    
    return df_filled

def remove_outliers(df, columns=None, threshold=3):
    """移除异常值"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores <= threshold]
    
    return df_clean

def add_derived_features(df):
    """添加衍生特征"""
    df_enhanced = df.copy()
    
    # 添加年龄分组
    if 'age' in df_enhanced.columns:
        df_enhanced['age_group'] = pd.cut(
            df_enhanced['age'], 
            bins=[0, 30, 40, 50, 100], 
            labels=['<30', '30-40', '40-50', '50+']
        )
    
    # 添加工资百分位数
    if 'salary' in df_enhanced.columns:
        df_enhanced['salary_percentile'] = df_enhanced['salary'].rank(pct=True)
    
    return df_enhanced

def create_pipeline_report(df):
    """创建管道处理报告"""
    report = {
        'original_shape': df.shape,
        'processed_shape': None,
        'removed_duplicates': None,
        'filled_missing': df.isnull().sum().sum(),
        'removed_outliers': None
    }
    return report

# 创建数据处理管道
def data_processing_pipeline(df):
    """完整的数据处理管道"""
    
    # 初始报告
    report = create_pipeline_report(df)
    original_rows = len(df)
    
    # 应用管道操作
    processed_df = (df
                   .pipe(remove_duplicates)
                   .pipe(fill_missing_values, strategy='mean')
                   .pipe(remove_outliers, threshold=3)
                   .pipe(add_derived_features))
    
    # 更新报告
    report['processed_shape'] = processed_df.shape
    report['removed_duplicates'] = original_rows - len(processed_df)
    report['removed_outliers'] = len(df) - len(processed_df) - report['removed_duplicates']
    
    return processed_df, report

print("94. 管道操作演示:")

# 应用管道
final_df, pipeline_report = data_processing_pipeline(df_validate)

print("管道处理报告:")
for key, value in pipeline_report.items():
    print(f"{key}: {value}")

print("\n处理后的数据前5行:")
print(final_df.head())
```

**95. 样式设置：设置DataFrame的显示样式**

```python
# 创建样式化的DataFrame
df_style = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'],
    'Sales': [150000, 200000, 75000, 50000, 25000],
    'Profit': [30000, 40000, 15000, 10000, 5000],
    'Growth': [0.15, 0.25, -0.05, 0.10, 0.30],
    'Market Share': [0.25, 0.35, 0.15, 0.10, 0.05]
})

print("95. DataFrame样式设置:")

# 基本样式设置
styled_df = (df_style.style
             .format({
                 'Sales': '${:,.0f}',
                 'Profit': '${:,.0f}', 
                 'Growth': '{:.1%}',
                 'Market Share': '{:.1%}'
             })
             .highlight_max(subset=['Sales', 'Profit', 'Growth', 'Market Share'], color='lightgreen')
             .highlight_min(subset=['Sales', 'Profit'], color='lightcoral')
             .highlight_min(subset=['Growth', 'Market Share'], color='lightyellow')
             .bar(subset=['Sales', 'Profit'], color='lightblue')
             .set_caption('产品销售业绩报表')
             .set_table_styles([{
                 'selector': 'caption',
                 'props': [('font-size', '16px'), ('font-weight', 'bold')]
             }]))

# 显示样式化的DataFrame
styled_df

# 条件格式函数
def color_negative_red(val):
    """负值显示为红色"""
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'black'
        return f'color: {color}'
    return ''

def highlight_high_growth(val):
    """高增长显示背景色"""
    if isinstance(val, (int, float)) and val > 0.2:
        return 'background-color: lightgreen'
    return ''

# 应用条件格式
styled_conditional = (df_style.style
                      .format({
                          'Sales': '${:,.0f}',
                          'Profit': '${:,.0f}',
                          'Growth': '{:.1%}',
                          'Market Share': '{:.1%}'
                      })
                      .applymap(color_negative_red, subset=['Growth'])
                      .applymap(highlight_high_growth, subset=['Growth'])
                      .set_properties(**{'text-align': 'center'}))

print("条件格式化的DataFrame:")
styled_conditional

# 渐变颜色映射
styled_gradient = (df_style.style
                   .background_gradient(subset=['Sales', 'Profit'], cmap='Blues')
                   .background_gradient(subset=['Growth'], cmap='RdYlGn', vmin=-0.1, vmax=0.3)
                   .format({
                       'Sales': '${:,.0f}',
                       'Profit': '${:,.0f}',
                       'Growth': '{:.1%}',
                       'Market Share': '{:.1%}'
                   }))

print("渐变颜色映射:")
styled_gradient
```

### 实战应用

**96. 数据报告：生成自动化数据报告**

```python
def generate_data_report(df, title="数据报告"):
    """生成完整的数据报告"""
    
    report_parts = []
    
    # 1. 报告标题
    report_parts.append(f"=== {title} ===\n")
    
    # 2. 基本信息
    report_parts.append("1. 数据集基本信息:")
    report_parts.append(f"   形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
    report_parts.append(f"   内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report_parts.append("")
    
    # 3. 数据类型统计
    report_parts.append("2. 数据类型分布:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        report_parts.append(f"   {dtype}: {count} 列")
    report_parts.append("")
    
    # 4. 数据质量评估
    report_parts.append("3. 数据质量评估:")
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    report_parts.append(f"   缺失值比例: {missing_cells}/{total_cells} ({missing_cells/total_cells:.1%})")
    report_parts.append(f"   重复行数: {duplicate_rows} ({duplicate_rows/df.shape[0]:.1%})")
    report_parts.append("")
    
    # 5. 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        report_parts.append("4. 数值列统计摘要:")
        numeric_stats = df[numeric_cols].describe()
        for col in numeric_cols:
            stats = numeric_stats[col]
            report_parts.append(f"   {col}:")
            report_parts.append(f"     均值: {stats['mean']:.2f}, 标准差: {stats['std']:.2f}")
            report_parts.append(f"     最小值: {stats['min']:.2f}, 最大值: {stats['max']:.2f}")
            report_parts.append(f"     25%/50%/75%分位数: {stats['25%']:.2f}/{stats['50%']:.2f}/{stats['75%']:.2f}")
        report_parts.append("")
    
    # 6. 分类列统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        report_parts.append("5. 分类列统计:")
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            report_parts.append(f"   {col} (唯一值: {df[col].nunique()}):")
            for i, (value, count) in enumerate(value_counts.head().items()):
                if i < 3:  # 只显示前3个最常见的值
                    report_parts.append(f"     {value}: {count} ({count/len(df):.1%})")
            if len(value_counts) > 3:
                report_parts.append(f"     其他 {len(value_counts) - 3} 个值...")
        report_parts.append("")
    
    # 7. 相关性分析（如果有多个数值列）
    if len(numeric_cols) > 1:
        report_parts.append("6. 数值列相关性:")
        correlation_matrix = df[numeric_cols].corr()
        # 找出高度相关的列对（|r| > 0.7）
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr))
        
        if high_corr_pairs:
            for col1, col2, corr in high_corr_pairs:
                report_parts.append(f"   {col1} 和 {col2}: {corr:.3f}")
        else:
            report_parts.append("   没有发现高度相关的列对 (|r| > 0.7)")
        report_parts.append("")
    
    # 8. 数据质量建议
    report_parts.append("7. 数据质量建议:")
    if missing_cells > 0:
        report_parts.append("   ⚠️ 存在缺失值，建议进行填充或删除处理")
    if duplicate_rows > 0:
        report_parts.append("   ⚠️ 存在重复行，建议进行去重处理")
    
    numeric_outliers = 0
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        numeric_outliers += (z_scores > 3).sum()
    
    if numeric_outliers > 0:
        report_parts.append(f"   ⚠️ 数值列中存在 {numeric_outliers} 个潜在异常值 (|z-score| > 3)")
    
    if len(categorical_cols) > 0:
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality_cols:
            report_parts.append(f"   ⚠️ 以下分类列基数较高: {', '.join(high_cardinality_cols)}")
    
    report_parts.append("")
    report_parts.append("=== 报告结束 ===")
    
    return "\n".join(report_parts)

# 生成报告
print("96. 自动化数据报告:")
report = generate_data_report(df_validate, "员工数据质量报告")
print(report)
```

**97. 时间序列分析：进行时间序列分析**

```python
# 创建时间序列数据
dates = pd.date_range('2023-01-01', periods=365, freq='D')
time_series = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, 365).cumsum() + 10000,  # 带有趋势的销售数据
    'temperature': np.sin(np.arange(365) * 2 * np.pi / 365) * 15 + 20,  # 季节性温度
    'events': np.random.choice([0, 1], 365, p=[0.95, 0.05])  # 随机事件
})

# 添加一些特殊日期效应
time_series.loc[time_series['date'].dt.month.isin([12]), 'sales'] *= 1.2  # 12月销售增加
time_series.loc[time_series['date'].dt.dayofweek == 0, 'sales'] *= 0.9    # 周一销售减少

time_series.set_index('date', inplace=True)

print("97. 时间序列分析:")

# 基本时间序列分析
print("时间序列基本信息:")
print(f"时间范围: {time_series.index.min()} 到 {time_series.index.max()}")
print(f"总天数: {len(time_series)}")
print(f"销售总额: {time_series['sales'].sum():,.0f}")
print(f"平均日销售: {time_series['sales'].mean():,.0f}")

# 重采样分析
print("\n月度销售分析:")
monthly_sales = time_series['sales'].resample('M').agg(['sum', 'mean', 'std'])
print(monthly_sales.head())

# 滚动统计
time_series['7d_rolling_avg'] = time_series['sales'].rolling(window=7).mean()
time_series['30d_rolling_avg'] = time_series['sales'].rolling(window=30).mean()

# 同比分析（需要两年数据，这里模拟）
time_series['sales_shifted'] = time_series['sales'].shift(365)  # 模拟去年数据
time_series['yoy_growth'] = (time_series['sales'] - time_series['sales_shifted']) / time_series['sales_shifted']

# 季节性分解（简化版）
def simple_seasonal_decomposition(series, period=30):
    """简单的季节性分解"""
    # 趋势成分（移动平均）
    trend = series.rolling(window=period, center=True).mean()
    
    # 季节性成分（去趋势后的平均值）
    detrended = series - trend
    seasonal = detrended.groupby(detrended.index.dayofyear).mean()
    
    # 残差成分
    residual = series - trend - seasonal.reindex(series.index, method='nearest')
    
    return trend, seasonal, residual

# 应用分解
trend, seasonal, residual = simple_seasonal_decomposition(time_series['sales'], period=30)

print("\n季节性分解统计:")
print(f"趋势标准差: {trend.std():.2f}")
print(f"季节性标准差: {seasonal.std():.2f}")
print(f"残差标准差: {residual.std():.2f}")

# 时间序列可视化
plt.figure(figsize=(15, 10))

# 原始销售数据
plt.subplot(3, 2, 1)
time_series['sales'].plot(title='原始销售数据', color='blue')
plt.ylabel('Sales')

# 滚动平均
plt.subplot(3, 2, 2)
time_series[['sales', '7d_rolling_avg', '30d_rolling_avg']].plot(title='滚动平均', alpha=0.7)
plt.ylabel('Sales')

# 月度聚合
plt.subplot(3, 2, 3)
monthly_sales['sum'].plot(kind='bar', title='月度销售总额')
plt.ylabel('Sales')

# 季节性模式
plt.subplot(3, 2, 4)
pd.Series(seasonal).plot(title='季节性模式')
plt.ylabel('Seasonal Component')

# 温度与销售的关系
plt.subplot(3, 2, 5)
plt.scatter(time_series['temperature'], time_series['sales'], alpha=0.5)
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.title('温度与销售关系')

# 残差分析
plt.subplot(3, 2, 6)
residual.hist(bins=30, alpha=0.7)
plt.title('残差分布')

plt.tight_layout()
plt.show()
```

**98. 相关性分析：计算和可视化相关性矩阵**

```python
# 创建多变量数据集
np.random.seed(42)
n = 1000

df_corr = pd.DataFrame({
    'age': np.random.normal(35, 10, n),
    'income': np.random.normal(50000, 15000, n),
    'education_years': np.random.normal(16, 3, n),
    'work_experience': np.random.normal(10, 5, n),
    'house_price': np.random.normal(300000, 100000, n),
    'satisfaction': np.random.normal(7, 2, n)
})

# 添加一些相关性
df_corr['income'] = df_corr['income'] + df_corr['education_years'] * 2000  # 教育与收入正相关
df_corr['house_price'] = df_corr['house_price'] + df_corr['income'] * 2    # 收入与房价正相关
df_corr['satisfaction'] = df_corr['satisfaction'] + df_corr['income'] / 10000  # 收入与满意度正相关

# 添加一些分类变量
df_corr['gender'] = np.random.choice(['Male', 'Female'], n)
df_corr['region'] = np.random.choice(['North', 'South', 'East', 'West'], n)

print("98. 相关性分析:")

# 计算数值列的相关性矩阵
numeric_cols = df_corr.select_dtypes(include=[np.number]).columns
correlation_matrix = df_corr[numeric_cols].corr()

print("相关性矩阵:")
print(correlation_matrix.round(3))

# 找出高度相关的变量对
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.5:  # 高度相关阈值
            col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
            high_corr_pairs.append((col1, col2, corr))

print("\n高度相关的变量对 (|r| > 0.5):")
for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
    print(f"  {col1} - {col2}: {corr:.3f}")

# 相关性可视化
plt.figure(figsize=(12, 10))

# 热力图
plt.subplot(2, 2, 1)
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('相关性热力图')

# 散点图矩阵（前4个变量）
from pandas.plotting import scatter_matrix
plt.subplot(2, 2, 2)
scatter_matrix(df_corr[numeric_cols[:4]], alpha=0.5, figsize=(8, 8), diagonal='hist')
plt.suptitle('散点图矩阵', y=0.95)

# 最强相关性的详细散点图
if high_corr_pairs:
    strongest_pair = max(high_corr_pairs, key=lambda x: abs(x[2]))
    col1, col2, corr = strongest_pair
    
    plt.subplot(2, 2, 3)
    plt.scatter(df_corr[col1], df_corr[col2], alpha=0.5)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'{col1} vs {col2} (r = {corr:.3f})')
    
    # 添加趋势线
    z = np.polyfit(df_corr[col1], df_corr[col2], 1)
    p = np.poly1d(z)
    plt.plot(df_corr[col1], p(df_corr[col1]), "r--", alpha=0.8)

# 分类变量的相关性（点二列相关）
plt.subplot(2, 2, 4)
# 将性别转换为数值
df_corr['gender_numeric'] = df_corr['gender'].map({'Male': 0, 'Female': 1})

# 计算分类变量与数值变量的相关性
categorical_correlations = {}
for num_col in numeric_cols:
    corr = df_corr['gender_numeric'].corr(df_corr[num_col])
    categorical_correlations[num_col] = corr

# 绘制条形图
plt.bar(range(len(categorical_correlations)), list(categorical_correlations.values()))
plt.xticks(range(len(categorical_correlations)), list(categorical_correlations.keys()), rotation=45)
plt.title('性别与数值变量的相关性')
plt.ylabel('相关系数')

plt.tight_layout()
plt.show()

# 高级相关性分析：偏相关
def partial_correlation(df, x, y, control_vars):
    """计算偏相关系数"""
    from sklearn.linear_model import LinearRegression
    
    # 控制变量对x的回归残差
    lr_x = LinearRegression()
    lr_x.fit(df[control_vars], df[x])
    resid_x = df[x] - lr_x.predict(df[control_vars])
    
    # 控制变量对y的回归残差
    lr_y = LinearRegression()
    lr_y.fit(df[control_vars], df[y])
    resid_y = df[y] - lr_y.predict(df[control_vars])
    
    # 残差的相关性就是偏相关
    return np.corrcoef(resid_x, resid_y)[0, 1]

# 计算偏相关示例
if 'age' in df_corr.columns and 'income' in df_corr.columns and 'education_years' in df_corr.columns:
    partial_corr = partial_correlation(df_corr, 'income', 'house_price', ['age', 'education_years'])
    print(f"\n偏相关系数 (收入 vs 房价 | 控制年龄和教育): {partial_corr:.3f}")
```

**99. 数据建模准备：为机器学习准备数据**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 创建机器学习数据集
df_ml = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'loan_amount': np.random.normal(10000, 5000, 1000),
    'default': np.random.choice([0, 1], 1000, p=[0.9, 0.1])  # 目标变量
})

# 添加一些缺失值
df_ml.loc[df_ml.sample(50).index, 'income'] = np.nan
df_ml.loc[df_ml.sample(30).index, 'credit_score'] = np.nan

print("99. 机器学习数据准备:")

# 1. 数据探索
print("数据形状:", df_ml.shape)
print("目标变量分布:")
print(df_ml['default'].value_counts(normalize=True))

# 2. 特征和目标分离
X = df_ml.drop('default', axis=1)
y = df_ml['default']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 4. 定义预处理管道
# 数值特征管道
numeric_features = ['age', 'income', 'credit_score', 'loan_amount']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征管道
categorical_features = ['education', 'gender', 'city']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 应用预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"预处理后训练集形状: {X_train_processed.shape}")
print(f"预处理后测试集形状: {X_test_processed.shape}")

# 6. 获取特征名称（用于模型解释）
feature_names = (numeric_features + 
                list(preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)))

print(f"特征数量: {len(feature_names)}")
print("前10个特征名称:", feature_names[:10])

# 7. 创建完整的建模管道（示例：逻辑回归）
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 创建完整管道
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# 预测
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# 评估模型
print("\n模型性能评估:")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

# 8. 特征重要性分析
if hasattr(model_pipeline.named_steps['classifier'], 'coef_'):
    # 逻辑回归的特征重要性
    importance = model_pipeline.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', key=abs, ascending=False)
    
    print("\n特征重要性 (逻辑回归系数):")
    print(feature_importance.head(10))

# 9. 使用随机森林比较
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n随机森林性能:")
print(classification_report(y_test, y_pred_rf))
print(f"随机森林 AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.3f}")

# 随机森林特征重要性
if hasattr(rf_pipeline.named_steps['classifier'], 'feature_importances_'):
    rf_importance = rf_pipeline.named_steps['classifier'].feature_importances_
    rf_feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    print("\n随机森林特征重要性:")
    print(rf_feature_importance.head(10))
```

**100. 综合项目：完成一个端到端的数据分析项目**

```python
def complete_data_analysis_project():
    """完整的端到端数据分析项目"""
    
    print("="*60)
    print("100. 端到端数据分析项目: 销售数据分析")
    print("="*60)
    
    # 1. 数据生成（模拟真实业务数据）
    np.random.seed(42)
    n_customers = 1000
    n_transactions = 5000
    
    # 客户数据
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 70, n_customers),
        'income': np.random.normal(50000, 20000, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                               n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'member_since': pd.to_datetime(np.random.choice(
            pd.date_range('2018-01-01', '2023-12-31', freq='D'), n_customers))
    })
    
    # 交易数据
    transactions = pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 
                                           n_transactions, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'amount': np.random.lognormal(4, 1, n_transactions),  # 对数正态分布，更符合真实交易
        'transaction_date': pd.to_datetime(np.random.choice(
            pd.date_range('2023-01-01', '2023-12-31', freq='D'), n_transactions))
    })
    
    # 添加一些业务逻辑
    # 高收入客户消费更多
    high_income_customers = customers[customers['income'] > 70000]['customer_id']
    transactions.loc[transactions['customer_id'].isin(high_income_customers), 'amount'] *= 1.5
    
    # 季节性模式
    transactions.loc[transactions['transaction_date'].dt.month.isin([11, 12]), 'amount'] *= 1.3  # 假日季
    
    print("1. 数据概览:")
    print(f"客户数量: {len(customers)}")
    print(f"交易数量: {len(transactions)}")
    print(f"时间范围: {transactions['transaction_date'].min()} 到 {transactions['transaction_date'].max()}")
    print(f"总交易额: ${transactions['amount'].sum():,.2f}")
    
    # 2. 数据合并与丰富
    merged_data = transactions.merge(customers, on='customer_id', how='left')
    
    # 添加衍生特征
    merged_data['transaction_month'] = merged_data['transaction_date'].dt.to_period('M')
    merged_data['transaction_dayofweek'] = merged_data['transaction_date'].dt.dayofweek
    merged_data['transaction_season'] = merged_data['transaction_date'].dt.quarter
    
    # 客户级别的聚合特征
    customer_stats = transactions.groupby('customer_id').agg({
        'amount': ['count', 'sum', 'mean', 'std'],
        'transaction_date': 'max'
    }).round(2)
    
    customer_stats.columns = ['transaction_count', 'total_spent', 'avg_transaction', 'std_transaction', 'last_purchase']
    customer_stats['days_since_last_purchase'] = (pd.Timestamp('2023-12-31') - customer_stats['last_purchase']).dt.days
    
    # 合并回客户数据
    customers_enriched = customers.merge(customer_stats, on='customer_id', how='left')
    
    print("\n2. 数据丰富完成")
    print(f"合并后数据形状: {merged_data.shape}")
    
    # 3. 探索性数据分析
    print("\n3. 探索性数据分析:")
    
    # 基本统计
    print("交易金额统计:")
    print(transactions['amount'].describe())
    
    # 月度趋势
    monthly_sales = merged_data.groupby('transaction_month')['amount'].sum()
    print(f"\n最佳销售月份: {monthly_sales.idxmax()} (${monthly_sales.max():,.2f})")
    
    # 产品类别分析
    category_analysis = merged_data.groupby('product_category').agg({
        'amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    
    category_analysis.columns = ['total_sales', 'avg_transaction', 'transaction_count', 'unique_customers']
    category_analysis['sales_per_customer'] = category_analysis['total_sales'] / category_analysis['unique_customers']
    
    print("\n产品类别分析:")
    print(category_analysis.sort_values('total_sales', ascending=False))
    
    # 4. 客户细分分析
    print("\n4. 客户细分分析:")
    
    # RFM分析 (Recency, Frequency, Monetary)
    rfm = customers_enriched[['customer_id', 'days_since_last_purchase', 'transaction_count', 'total_spent']].copy()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # RFM评分
    rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])  # 最近购买得分高
    rfm['f_score'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4])  # 高频购买得分高
    rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])  # 高消费得分高
    
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    rfm['rfm_total'] = rfm[['r_score', 'f_score', 'm_score']].astype(int).sum(axis=1)
    
    # RFM细分
    def rfm_segment(row):
        if row['rfm_total'] >= 10:
            return 'Champions'
        elif row['rfm_total'] >= 8:
            return 'Loyal Customers'
        elif row['rfm_total'] >= 6:
            return 'Potential Loyalists'
        elif row['rfm_total'] >= 4:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    rfm['segment'] = rfm.apply(rfm_segment, axis=1)
    
    segment_summary = rfm.groupby('segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    }).round(2)
    
    segment_summary['pct_customers'] = (segment_summary['customer_id'] / len(rfm) * 100).round(1)
    
    print("客户细分分析:")
    print(segment_summary)
    
    # 5. 高级分析：预测客户价值
    print("\n5. 高级分析: 客户终身价值预测")
    
    # 简单CLV计算（历史方法）
    avg_purchase_value = transactions['amount'].mean()
    purchase_frequency = len(transactions) / len(customers)
    customer_value = avg_purchase_value * purchase_frequency
    
    # 假设平均客户寿命为3年
    avg_customer_lifespan = 3
    clv = customer_value * avg_customer_lifespan
    
    print(f"平均购买价值: ${avg_purchase_value:.2f}")
    print(f"购买频率: {purchase_frequency:.2f} 次/年")
    print(f"客户价值: ${customer_value:.2f}/年")
    print(f"预估客户终身价值: ${clv:.2f}")
    
    # 6. 数据可视化
    print("\n6. 生成分析图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 月度销售趋势
    plt.subplot(2, 3, 1)
    monthly_sales.plot(kind='line', marker='o')
    plt.title('月度销售趋势')
    plt.xticks(rotation=45)
    
    # 产品类别销售分布
    plt.subplot(2, 3, 2)
    category_analysis['total_sales'].sort_values(ascending=False).plot(kind='bar')
    plt.title('产品类别销售分布')
    plt.xticks(rotation=45)
    
    # 客户细分分布
    plt.subplot(2, 3, 3)
    segment_summary['pct_customers'].plot(kind='pie', autopct='%1.1f%%')
    plt.title('客户细分分布')
    
    # 交易金额分布
    plt.subplot(2, 3, 4)
    transactions['amount'].hist(bins=30, alpha=0.7)
    plt.title('交易金额分布')
    plt.xlabel('交易金额')
    
    # 城市销售分析
    plt.subplot(2, 3, 5)
    merged_data.groupby('city')['amount'].sum().sort_values(ascending=False).plot(kind='bar')
    plt.title('各城市销售总额')
    plt.xticks(rotation=45)
    
    # 收入与消费关系
    plt.subplot(2, 3, 6)
    plt.scatter(customers_enriched['income'], customers_enriched['total_spent'], alpha=0.5)
    plt.xlabel('收入')
    plt.ylabel('总消费')
    plt.title('收入与消费关系')
    
    plt.tight_layout()
    plt.show()
    
    # 7. 业务洞察和建议
    print("\n7. 业务洞察和建议:")
    print("="*50)
    
    insights = [
        f"• 最佳销售月份: {monthly_sales.idxmax()}，建议加大该时段营销投入",
        f"• 最畅销品类: {category_analysis['total_sales'].idxmax()}，占总销售额{category_analysis['total_sales'].max()/category_analysis['total_sales'].sum()*100:.1f}%",
        f"• 高价值客户占比: {segment_summary.loc['Champions', 'pct_customers'] if 'Champions' in segment_summary.index else 0:.1f}%",
        f"• 风险客户占比: {segment_summary.loc['At Risk', 'pct_customers'] if 'At Risk' in segment_summary.index else 0:.1f}%，需要制定保留策略",
        f"• 平均客户终身价值: ${clv:.2f}，可作为客户获取成本参考"
    ]
    
    for insight in insights:
        print(insight)
    
    print("\n8. 项目总结:")
    print("="*50)
    print("本项目完成了从数据生成、清洗、探索分析到业务洞察的全流程。")
    print("关键成果包括客户细分、销售趋势分析和客户价值预测。")
    print("这些分析可为市场营销、产品策略和客户关系管理提供数据支持。")
    
    return {
        'customers': customers,
        'transactions': transactions,
        'merged_data': merged_data,
        'rfm_analysis': rfm,
        'category_analysis': category_analysis,
        'monthly_trends': monthly_sales
    }

# 运行完整项目
print("开始执行端到端数据分析项目...")
project_results = complete_data_analysis_project()
print("\n项目执行完成！")
```

## 完整示例代码

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fifth_level_practice():
    print("=== 第五关：高级应用 ===\n")
    
    # 81. 基础绘图
    df = pd.DataFrame({
        'Year': [2010, 2011, 2012, 2013, 2014, 2015],
        'Sales': [100, 120, 140, 160, 180, 200]
    })
    df.set_index('Year', inplace=True)
    
    print("81. 基础绘图示例:")
    df['Sales'].plot(kind='line', title='Sales Over Years')
    plt.show()
    
    # 86. 内存优化
    large_df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 10000),
        'category_col': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    print("\n86. 内存优化:")
    print(f"优化前: {large_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    large_df['int_col'] = pd.to_numeric(large_df['int_col'], downcast='integer')
    large_df['category_col'] = large_df['category_col'].astype('category')
    
    print(f"优化后: {large_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # 91. 多级索引
    arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
    index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
    df_multi = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)
    
    print("\n91. 多级索引:")
    print(df_multi)

# 运行练习
fifth_level_practice()
```

## 学习要点

1. **数据可视化**：掌握各种图表类型和可视化技巧
2. **性能优化**：学会处理大型数据集和优化计算效率
3. **高级技巧**：多级索引、自定义函数、数据验证等高级功能
4. **实战应用**：端到端的数据分析项目流程

## 恭喜完成Pandas闯关100题！

通过这100个题目的系统学习，您已经：

✅ **掌握了Pandas的核心功能** - 从基础操作到高级应用  
✅ **具备了数据处理能力** - 清洗、转换、分析、可视化  
✅ **学会了性能优化** - 处理大型数据集和优化计算  
✅ **完成了实战项目** - 端到端的数据分析流程  

**下一步学习建议：**
1. **实践项目**：将学到的知识应用到真实项目中
2. **学习相关库**：NumPy、Matplotlib、Seaborn、Scikit-learn
3. **深入学习**：时间序列分析、机器学习、大数据处理
4. **参与社区**：阅读开源代码，参与数据科学社区

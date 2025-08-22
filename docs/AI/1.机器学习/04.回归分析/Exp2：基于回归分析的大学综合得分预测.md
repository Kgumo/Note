
通过网盘分享的文件：基于回归分析的大学综合得分预测 (1).zip
链接: https://pan.baidu.com/s/1Jdu24uRYY4wDDqvfIsDQAg?pwd=nt29 提取码: nt29 

------

## 一、案例简介

大学排名是一个非常重要同时也极富挑战性与争议性的问题，一所大学的综合实力涉及科研、师资、学生等方方面面。目前全球有上百家评估机构会评估大学的综合得分进行排序，而这些机构的打分也往往并不一致。在这些评分机构中，世界大学排名中心（Center for World University Rankings，缩写CWUR）以评估教育质量、校友就业、研究成果和引用，而非依赖于调查和大学所提交的数据著称，是非常有影响力的一个。

本任务中我们将根据 CWUR 所提供的世界各地知名大学各方面的排名（师资、科研等），一方面通过数据可视化的方式观察不同大学的特点，另一方面希望构建机器学习模型（线性回归）预测一所大学的综合得分。

## 二、作业说明

使用来自 Kaggle 的[数据](https://www.kaggle.com/mylesoneill/world-university-rankings?select=cwurData.csv)，构建「线性回归」模型，根据大学各项指标的排名预测综合得分。

**基本要求：**

- 按照 8:2 随机划分训练集测试集，用 RMSE 作为评价指标，得到测试集上线性回归模型的 RMSE 值；
- 对线性回归模型的系数进行分析。

**扩展要求：**

- 对数据进行观察与可视化，展示数据特点；
- 尝试其他的回归模型，对比效果；
- 尝试将离散的地区特征融入线性回归模型，并对结果进行对比。

**注意事项：**

- 基本输入特征有 8 个：`quality_of_education`, `alumni_employment`, `quality_of_faculty`, `publications`, `influence`, `citations`, `broad_impact`, `patents`；
- 预测目标为`score`；
- 可以使用 sklearn 等第三方库，不要求自己实现线性回归；
- 需要保留所有数据集生成、模型训练测试的代码；

## 三、数据概览

假设数据文件位于当前文件夹，我们用 pandas 读入标准 csv 格式文件的函数`read_csv()`将数据转换为`DataFrame`的形式。观察前几条数据记录：

In [1]:

```
import pandas as pd
import numpy as np

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
data_df.head(3).T  # 观察前几列并转置方便观察
```

Out[1]:

|                      |                  0 |                                     1 |                   2 |
| -------------------: | -----------------: | ------------------------------------: | ------------------: |
|           world_rank |                  1 |                                     2 |                   3 |
|          institution | Harvard University | Massachusetts Institute of Technology | Stanford University |
|               region |                USA |                                   USA |                 USA |
|        national_rank |                  1 |                                     2 |                   3 |
| quality_of_education |                  7 |                                     9 |                  17 |
|    alumni_employment |                  9 |                                    17 |                  11 |
|   quality_of_faculty |                  1 |                                     3 |                   5 |
|         publications |                  1 |                                    12 |                   4 |
|            influence |                  1 |                                     4 |                   2 |
|            citations |                  1 |                                     4 |                   2 |
|         broad_impact |                NaN |                                   NaN |                 NaN |
|              patents |                  5 |                                     1 |                  15 |
|                score |                100 |                                 91.67 |                89.5 |
|                 year |               2012 |                                  2012 |                2012 |

去除其中包含 NaN 的数据，保留 2000 条有效记录。

In [2]:

```
data_df = data_df.dropna()  # 舍去包含 NaN 的 row
len(data_df)
```

Out[2]:

```
2000
```

取出对应自变量以及因变量的列，之后就可以基于此切分训练集和测试集，并进行模型构建与分析。

In [3]:

```
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment', 
                'influence', 'quality_of_education', 'broad_impact', 'patents']
X = data_df[feature_cols]
Y = data_df['score']
X
```

Out[3]:

|      | quality_of_faculty | publications | citations | alumni_employment | influence | quality_of_education | broad_impact | patents |
| ---: | -----------------: | -----------: | --------: | ----------------: | --------: | -------------------: | -----------: | ------: |
|  200 |                  1 |            1 |         1 |                 1 |         1 |                    1 |          1.0 |       2 |
|  201 |                  4 |            5 |         3 |                 2 |         3 |                   11 |          4.0 |       6 |
|  202 |                  2 |           15 |         2 |                11 |         2 |                    3 |          2.0 |       1 |
|  203 |                  5 |           10 |        12 |                10 |         9 |                    2 |         13.0 |      48 |
|  204 |                 10 |           11 |        11 |                12 |        12 |                    7 |         12.0 |      16 |
|  ... |                ... |          ... |       ... |               ... |       ... |                  ... |          ... |     ... |
| 2195 |                218 |          926 |       812 |               567 |       845 |                  367 |        969.0 |     816 |
| 2196 |                218 |          997 |       645 |               566 |       908 |                  236 |        981.0 |     871 |
| 2197 |                218 |          830 |       812 |               549 |       823 |                  367 |        975.0 |     824 |
| 2198 |                218 |          886 |       812 |               567 |       974 |                  367 |        975.0 |     651 |
| 2199 |                218 |          861 |       812 |               567 |       991 |                  367 |        981.0 |     547 |

2000 rows × 8 columns

## 四、模型构建

（待完成）
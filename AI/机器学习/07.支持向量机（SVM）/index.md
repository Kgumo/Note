### 支持向量机（SVM）第一部分

---

#### **1. 背景与核心概念**
**支持向量机（SVM）** 是由 Vapnik 团队在 AT&T 贝尔实验室开发的监督学习算法：
- **核心目标**：寻找最优分类超平面，最大化两类数据点的间隔（Margin）。
- **关键特性**：
  - 最大间隔分类器（Max Margin Classifier）
  - 最初用于分类，后扩展至回归和时间序列预测
  - 曾作为文本处理的强基准模型（Strong Baseline）

---

#### **2. 问题定义**
给定训练集：
$$(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N), \quad x_i \in \mathbb{R}^n, \quad y_i \in \{-1, 1\}$$
需找到分类函数：
$$
f(x, \theta) = \langle w, x \rangle + b \quad \text{满足} \quad 
\begin{cases} 
f(x_i) > 0 & \text{若 } y_i = +1 \\ 
f(x_i) < 0 & \text{若 } y_i = -1 
\end{cases}
$$
其中：
- **分类超平面**：$f(x) = \langle w, x \rangle + b = 0$
- **决策规则**：预测标签为 $\text{sign}(f(x))$

---

#### **3. 线性可分与最大间隔**
##### **3.1 线性分类器**
- 超平面方程：$\langle w, x \rangle + b = 0$
- **问题**：线性可分时存在无限多超平面，需选择最优解


##### **3.2 间隔（Margin）定义**
- **关键思想**：在分类超平面两侧构建平行边界超平面：
  $$
  \langle w, x \rangle + b = +1 \quad \text{和} \quad \langle w, x \rangle + b = -1
  $$
- **间隔宽度**：两边界超平面之间的距离（最大化此距离即提升泛化能力）

##### **3.3 间隔计算**
- 边界超平面到原点的距离：
  $$
  \rho_1 = \frac{|1 - b|}{\|w\|_2}, \quad \rho_2 = \frac{|-1 - b|}{\|w\|_2}
  $$
- **间隔公式**：
  $$
  \text{Margin} = |\rho_1 - \rho_2| = \frac{2}{\|w\|_2}
  $$
  > **推导**：由 $\langle \rho_1 \frac{w}{\|w\|_2}, w \rangle + b = 1$ 代入可得。

---

#### **4. 优化问题：原始形式**
最大化间隔等价于最小化 $\|w\|_2^2$：
$$
\min_{w,b} \frac{1}{2} \|w\|_2^2 \quad \text{subject to} \quad y_i(\langle w, x_i \rangle + b) \geq 1 \quad \forall i
$$
- **约束条件**：确保所有样本被正确分类且位于边界外
- **几何意义**：最小化 $w$ 的模长 → 最大化间隔 $\frac{2}{\|w\|_2}$

---

#### **5. 对偶问题与拉格朗日乘子法**
##### **5.1 拉格朗日函数**
引入乘子 $\alpha_i \geq 0$：
$$
L(w,b,\alpha) = \frac{1}{2} \|w\|_2^2 - \sum_{i=1}^N \alpha_i \left[ y_i(\langle w, x_i \rangle + b) - 1 \right]
$$

##### **5.2 KKT 条件**
最优解需满足：
$$
\begin{cases}
\frac{\partial L}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \\
\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0 \\
\alpha_i \left[ y_i(\langle w, x_i \rangle + b) - 1 \right] = 0 \\
\alpha_i \geq 0
\end{cases}
$$

##### **5.3 对偶问题**
代入 KKT 条件化简得：
$$
\min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle - \sum_i \alpha_i
$$
$$
\text{subject to} \quad \sum_i \alpha_i y_i = 0, \quad \alpha_i \geq 0 \quad \forall i
$$
- **优势**：
  - 复杂度取决于样本数而非特征维数
  - 显式引入内积 $\langle x_i, x_j \rangle$ → 为核函数铺垫

---

#### **6. 支持向量（Support Vectors）**
##### **6.1 定义与性质**
- **支持向量**：满足 $\alpha_i > 0$ 的样本点
- **几何意义**：位于边界超平面上的点（即 $y_i(\langle w, x_i \rangle + b) = 1$）
- **关键性质**：
  - 解 $w$ 的表达式：$w = \sum_{i} \alpha_i y_i x_i$（仅由支持向量决定）
  - 稀疏性：多数 $\alpha_i = 0$，仅支持向量影响模型

##### **6.2 分类决策函数**
$$
f(x) = \sum_{i \in SV} \alpha_i y_i \langle x_i, x \rangle + b
$$
- $b$ 通过支持向量求解：$b = y_k - \langle w, x_k \rangle$（$x_k$ 为任意支持向量）

---

#### **7. 算法流程总结**
1. **输入**：训练集 $\{(x_i, y_i)\}_{i=1}^N$
2. **求解对偶问题**：
   $$
   \min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle - \sum_i \alpha_i \quad \text{s.t.} \quad \sum_i \alpha_i y_i = 0, \alpha_i \geq 0
   $$
3. **计算参数**：
   - $w = \sum_i \alpha_i y_i x_i$
   - $b = y_k - \langle w, x_k \rangle$（$x_k$ 为支持向量）
4. **预测**：$\hat{y} = \text{sign} \left( \langle w, x \rangle + b \right)$

---

#### **8. 重要说明**
- **线性不可分情况**：需引入松弛变量（后续讲解）
- **核方法**：通过核函数 $K(x_i, x_j)$ 替换内积 $\langle x_i, x_j \rangle$ 处理非线性问题（后续讲解）
- **命名由来**：解仅由支持向量决定，模型复杂度与支持向量数量相关

### 支持向量机（SVM）第二部分：线性不可分与核方法详解

---

#### **1. 线性不可分问题与软间隔（Soft Margin）**

##### **1.1 问题背景**
- **线性可分SVM的局限性**：当数据存在噪声或重叠时，严格满足 $y_i(\langle w, x_i \rangle + b) \geq 1$ 不可能。
- **解决方案**：允许部分样本违反间隔边界，但惩罚这些违规行为。

##### **1.2 损失函数选择**
- **0/1损失**（难优化）：
  $$
  \ell_{0/1}(z_i) = \begin{cases} 
  1 & \text{if } z_i < 0 \\ 
  0 & \text{otherwise}
  \end{cases}, \quad z_i = y_i(\langle w, x_i \rangle + b)
  $$
- **Hinge损失**（SVM采用）：
  $$
  \ell_{\text{hinge}}(z_i) = \max(0, 1 - z_i)
  $$
  > **几何意义**：
  > - $z_i \geq 1$：无惩罚（样本在间隔外）
  > - $0 \leq z_i < 1$：惩罚 $1 - z_i$（样本在间隔内）
  > - $z_i < 0$：惩罚 $1 - z_i > 1$（样本被错误分类）

##### **1.3 优化问题形式化**
引入**松弛变量（Slack Variables）** $\varepsilon_i \geq 0$：
$$
\min_{w,b,\varepsilon} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^N \varepsilon_i
$$
$$
\text{s.t.} \quad y_i(\langle w, x_i \rangle + b) \geq 1 - \varepsilon_i, \quad \varepsilon_i \geq 0, \quad \forall i
$$
- **参数 $C$**：平衡间隔最大化与分类错误的权重：
  - $C \to \infty$：严格分类（退化为线性可分SVM）
  - $C \to 0$：忽略错误（间隔无限大）

##### **1.4 松弛变量的解释**
- $\varepsilon_i = 0$：样本满足约束（位于间隔外）
- $0 < \varepsilon_i \leq 1$：样本在间隔内但分类正确
- $\varepsilon_i > 1$：样本被错误分类

---

#### **2. 软间隔SVM的对偶问题**

##### **2.1 拉格朗日函数**
$$
L(w,b,\varepsilon,\alpha,\mu) = \frac{1}{2} \|w\|_2^2 + C \sum_i \varepsilon_i - \sum_i \alpha_i [y_i(\langle w, x_i \rangle + b) - 1 + \varepsilon_i] - \sum_i \mu_i \varepsilon_i
$$
其中 $\alpha_i \geq 0, \mu_i \geq 0$ 为拉格朗日乘子。

##### **2.2 KKT条件**
$$
\begin{cases}
\frac{\partial L}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \\
\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0 \\
\frac{\partial L}{\partial \varepsilon_i} = C - \alpha_i - \mu_i = 0 \\
\alpha_i [y_i(\langle w, x_i \rangle + b) - 1 + \varepsilon_i] = 0 \\
\mu_i \varepsilon_i = 0 \\
y_i(\langle w, x_i \rangle + b) - 1 + \varepsilon_i \geq 0 \\
\varepsilon_i \geq 0, \alpha_i \geq 0, \mu_i \geq 0
\end{cases}
$$

##### **2.3 对偶问题**
代入KKT条件化简得：
$$
\min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle - \sum_i \alpha_i
$$
$$
\text{s.t.} \quad \sum_i \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C, \quad \forall i
$$
- **关键变化**：乘子 $\alpha_i$ 被约束在 $[0, C]$ 区间内
- **支持向量**：
  - $\alpha_i > 0$ 的样本为支持向量
  - $\alpha_i < C$：样本恰在间隔边界上（$\varepsilon_i=0$）
  - $\alpha_i = C$：样本在间隔内或被错误分类（$\varepsilon_i > 0$）

---

#### **3. 核方法（Kernel SVM）**

##### **3.1 动机：非线性可分问题**
- **输入空间线性不可分** → 通过映射 $\Phi: \mathbb{R}^n \to \mathcal{F}$ 将数据升维到**特征空间** $\mathcal{F}$，使其线性可分
  > **示例**：二维不可分数据 $\xrightarrow{\Phi(x)=(x_1^2, x_2^2, \sqrt{2}x_1x_2)}$ 三维空间线性可分

##### **3.2 核技巧（Kernel Trick）**
- **核心思想**：避免显式计算高维映射 $\Phi(x)$，直接定义**核函数**：
  $$
  k(x_i, x_j) = \langle \Phi(x_i), \Phi(x_j) \rangle
  $$
- **对偶问题中的核替换**：
  $$
  \min_{\alpha} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \color{red}{k(x_i, x_j)} - \sum_i \alpha_i
  $$
  $$
  \text{s.t.} \quad \sum_i \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C
  $$
- **决策函数**：
  $$
  f(x) = \sum_{i \in SV} \alpha_i y_i \color{red}{k(x_i, x)} + b
  $$

##### **3.3 常用核函数**
| 核函数 | 表达式 | 特点 |
|--------|--------|------|
| **线性核** | $k(x,y) = \langle x, y \rangle$ | 退化为线性SVM |
| **多项式核** | $k(x,y) = (\gamma \langle x, y \rangle + r)^d$ | $\gamma$: 缩放因子, $d$: 次数 |
| **高斯核（RBF）** | $k(x,y) = \exp(-\gamma \|x - y\|^2)$ | $\gamma = \frac{1}{2\sigma^2}$, 无限维特征空间 |
| **Sigmoid核** | $k(x,y) = \tanh(\gamma \langle x, y \rangle + r)$ | 等价于两层神经网络 |

##### **3.4 核函数的合法性（Mercer条件）**
- **定理**：$k(x,y)$ 是合法核函数 $\iff$ 对任意样本，**Gram矩阵** $K = [k(x_i, x_j)]$ 对称半正定

##### **3.5 核函数构造方法**
若 $k_1, k_2$ 为合法核，则以下函数也是合法核：
1. $k(x,y) = c \cdot k_1(x,y) \quad (c>0)$
2. $k(x,y) = f(x) k_1(x,y) f(y) \quad (f: \text{任意函数})$
3. $k(x,y) = \exp(k_1(x,y))$
4. $k(x,y) = k_1(x,y) + k_2(x,y)$
5. $k(x,y) = k_1(x,y) \cdot k_2(x,y)$

> **高斯核构造示例**：
> $$
> \begin{aligned}
> k_{\text{Gauss}}(x,y) &= \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right) \\
> &= \exp\left(-\frac{\color{blue}{\langle x,x\rangle + \langle y,y\rangle - 2\langle x,y\rangle}}{2\sigma^2}\right) \\
> &= \underbrace{\exp\left(-\frac{\langle x,x\rangle}{2\sigma^2}\right) \exp\left(-\frac{\langle y,y\rangle}{2\sigma^2}\right)}_{\text{函数 } f(\cdot)} \cdot \underbrace{\exp\left(\frac{\langle x,y\rangle}{\sigma^2}\right)}_{\text{核 } k_1(x,y)}
> \end{aligned}
> $$

---

#### **4. SVM的优缺点总结**

##### **优点**：
1. **理论完备**：最大化间隔提供强泛化保证
2. **核技巧灵活**：高效处理非线性问题
3. **稀疏解**：模型仅依赖支持向量

##### **缺点**：
1. **核函数选择依赖经验**：无理论指导最优核
2. **大规模训练慢**：对偶问题求解复杂度 $O(N^3)$
3. **概率输出缺失**：需额外步骤（如Platt Scaling）

---

#### **5. 实战工具推荐**
- **Libsvm**：通用核SVM [[链接]](https://www.csie.ntu.edu.tw/~cjlin/libsvm)
- **Liblinear**：大规模线性SVM [[链接]](https://www.csie.ntu.edu.tw/~cjlin/liblinear)
- **Scikit-learn**：Python集成接口（`sklearn.svm.SVC`）

> 下一部分将深入无监督学习（聚类、降维等）。 
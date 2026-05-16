# Chapter 3-2 Principal Component Analysis (PCA)

# 1 Motivation and main idea
## 1.1 降维 (Dimensionality Reduction) 简介
**定义**：降维是将数据从高维空间映射到一个维度小得多的新空间的过程。

**主要动机 (Motivation)**：
* **克服计算挑战**：高维数据会带来极大的计算负担（例如：矩阵-向量乘法带来的“维度灾难” / curse of dimensionality）。
* **提升泛化能力**：高维数据容易导致学习算法的泛化能力变差 (poor generalization abilities)。
* **增加可解释性与可视化**：降维有助于解释数据、发现数据中有意义的结构，并方便进行数据的可视化展示 (illustration purposes)。

## 1.2 主成分分析 (PCA) 的核心要素
假设原始数据集为 $\{x_1, \cdots, x_n\} \subset \mathbb{R}^d$，其中原始维度 $d \gg 1$ （维度极高）。

PCA 的核心过程可以分为两步：
1.  **压缩 (Compression)**：应用一个线性变换矩阵 $\mathbb{W} \in \mathbb{R}^{k \times d}$（其中目标维度 $k \ll d$），将原始数据映射到低维空间：
    $$x_i \mapsto \mathbb{W}x_i$$
2.  **恢复 (Recovery)**：应用另一个线性变换矩阵 $\mathbb{U} \in \mathbb{R}^{d \times k}$，试图从低维表示中重构/近似原始数据：
    $$x_i \approx \mathbb{U}(\mathbb{W}x_i)$$

**优化问题 (Optimization Problem)**：
PCA 的根本目标是使得原始向量与恢复向量之间的**总平方距离最小化**。这可以被形式化为如下优化问题：
$$
\arg \min_{\substack{\mathbb{W} \in \mathbb{R}^{k \times d} \\ \mathbb{U} \in \mathbb{R}^{d \times k}}} \sum_{i=1}^{n} \|x_i - \mathbb{U}\mathbb{W}x_i\|_2^2
$$

---

# 2 Interpretation of PCA as Variance Maximization

## 2.1 PCA 的数学直觉 (Mathematical Intuition)

* **核心目标 (Goal)**：将 $d$ 维空间中的向量集合投影到低维子空间中进行总结压缩，同时保证原始数据中的信息在降维后被“本质上”地保留下来。
* **基本原则 (Moral)**：寻找主成分最简单、最核心的方法，就是去寻找那些能够**最大化方差 (maximize the variance)** 的投影方向。
* **重要等价性质 (Remark)**：
    * 直觉上，我们希望寻找的投影能让原始向量与其在主成分上的投影之间的**均方距离最小 (smallest mean-squared distance)**（即上面提到的重构误差最小化）。
    * **结论**：在数学上，**最小化重构的均方误差** 与 **最大化投影后的方差** 是完全等价的 (equivalent to maximizing the variance)。

在深入机器学习算法（如 PCA）的数学推导前，需要掌握多元随机变量的基本统计概念。

### 多元随机变量 / 随机向量 (Multivariate Random Variable / Random Vector)
**定义**：多元随机变量是一个列向量 $\mathbf{x} = (x_1, \dots, x_n)^T$ （或者其转置构成的行向量），它的每一个分量（$x_1$ 到 $x_n$）都是定义在同一个概率空间上的标量随机变量 (scalar-valued random variables)。


## 2.2 均值 / 期望 (Mean / Expected Value)
**定义**：随机向量 $\mathbf{x}$ 的期望值（或均值）是一个固定向量 $\mathbb{E}[\mathbf{x}]$，它的每一个元素就是对应位置上随机变量的期望值：
$$
\mathbb{E}[\mathbf{x}] := (\mathbb{E}[x_1], \dots, \mathbb{E}[x_n])^T
$$

* **零均值 (Zero Mean)**：如果一组向量的均值是零向量，我们称这组向量具有零均值。
* **核心技巧 (Comment)**：在理论推导和实际计算中，**通常只考虑零均值的向量就足够了**。如果原始数据不是零均值，我们可以通过让每个向量减去均值向量来调整它们（这在机器学习中通常被称为数据的**中心化 / Centering**）。


## 2.3 协方差矩阵 (Covariance Matrix)
**定义**：一个 $n \times 1$ 随机向量的协方差矩阵（也称为二阶中心矩或方差-协方差矩阵）是一个 $n \times n$ 的矩阵。该矩阵的第 $(i,j)$ 个元素，代表着第 $i$ 个随机变量和第 $j$ 个随机变量之间的协方差。

**公式推导**：
协方差矩阵 $\text{Var}[\mathbf{x}]$ 的标准定义如下：
$$
\text{Var}[\mathbf{x}] := \mathbb{E}\left[(\mathbf{x} - \mathbb{E}[\mathbf{x}])(\mathbf{x} - \mathbb{E}[\mathbf{x}])^T\right]
$$
将上述式子展开并利用期望的线性性质，可以得到一个极其常用的等价计算公式：
$$
\text{Var}[\mathbf{x}] = \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T
$$
*(注：结合上一节的“零均值”概念，如果数据已经过中心化使得 $\mathbb{E}[\mathbf{x}] = \mathbf{0}$，那么协方差矩阵就可以极大简化为 $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$，这在 PCA 推导中非常关键。)*


### 课后思考练习 (Exercise)
**问题**：证明任意随机向量的协方差矩阵都是**对称的 (Symmetric)** 且 **半正定的 (Positive Semidefinite)** 矩阵。

**复习提示 (Hint)**：
* **证明对称性**：只需证明 $\text{Var}[\mathbf{x}] = \text{Var}[\mathbf{x}]^T$。利用转置的性质 $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T\mathbf{A}^T$ 展开定义式即可。
* **证明半正定性**：需要证明对于任意非零列向量 $\mathbf{v} \in \mathbb{R}^n$，都有二次型 $\mathbf{v}^T \text{Var}[\mathbf{x}] \mathbf{v} \ge 0$。你可以尝试构造一个标量随机变量 $y = \mathbf{v}^T(\mathbf{x} - \mathbb{E}[\mathbf{x}])$，然后利用期望 $\mathbb{E}[y^2] \ge 0$ 的必然性质来完成证明。

## 2.4 数据假设与协方差化简 (Data Setup & Covariance)
假设有 $d$ 维向量集合 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n \in \mathbb{R}^d$。定义随机向量 $\mathbf{x}$ 在该集合上服从均匀分布：
$$
\mathbb{P}(\mathbf{x} = \mathbf{x}_i) = \frac{1}{n} \quad \text{for all } i = 1, \cdots, n
$$

**零均值假设 (Vanishing Mean)**：
不失一般性，我们假设数据已经中心化，即均值为零：
$$
\mathbb{E}[\mathbf{x}] = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i = 0
$$

基于零均值假设，随机向量 $\mathbf{x}$ 的协方差矩阵可以大幅化简为：
$$
\text{Var}[\mathbf{x}] = \mathbb{E}[\mathbf{x}\mathbf{x}^T] = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^T
$$

## 2.5 一维投影的目标 (One-dimensional Projection: Objective)
为了降维到一维，我们需要寻找一个**单位向量 (unit vector)** $\mathbf{w} \in \mathbb{R}^d$ 作为投影方向。

**核心等价目标**：
1.  **最小化残差**：最小化随机变量 $\mathbf{x}$ 与其在 $\mathbf{w}$ 决定直线上的投影之间的均方误差。
2.  **最大化方差**：这完全等价于**最大化**投影后随机变量 $\langle \mathbf{w}, \mathbf{x} \rangle$ 的方差。

**投影后的方差计算**：
由于原始数据零均值，投影后的变量 $\langle \mathbf{w}, \mathbf{x} \rangle$ 也具有零均值（即 $\mathbb{E}[\langle \mathbf{w}, \mathbf{x} \rangle] = \langle \mathbf{w}, \mathbb{E}[\mathbf{x}] \rangle = 0$）。因此，它的方差等于其二阶矩：
$$
\text{Var}(\langle \mathbf{w}, \mathbf{x} \rangle) = \mathbb{E}[|\langle \mathbf{w}, \mathbf{x} \rangle|^2]
$$

## 2.6 投影残差的推导 (Minimizing Projection Residuals)
要理解为什么“最小化残差”等价于“最大化方差”，我们需要计算具体的投影误差。

* **标量投影 (Scalar projection)**：数据点 $\mathbf{x}_i$ 在方向 $\mathbf{w}$ 上的投影长度为 $\langle \mathbf{w}, \mathbf{x}_i \rangle$。
* **向量投影 (Vector projection)**：投影点在 $d$ 维空间中的实际坐标向量为 $\mathcal{P}_1(\mathbf{x}_i) := \langle \mathbf{w}, \mathbf{x}_i \rangle \mathbf{w}$。

**残差 (Residual) 计算推导**：
如果我们用投影向量来代替原始向量，对于任意向量 $\mathbf{x}_i$，其产生的平方残差为：
$$
\|\mathbf{x}_i - \mathcal{P}_1(\mathbf{x}_i)\|^2 = \langle \mathbf{x}_i - \langle \mathbf{x}_i, \mathbf{w} \rangle \mathbf{w}, \mathbf{x}_i - \langle \mathbf{x}_i, \mathbf{w} \rangle \mathbf{w} \rangle
$$
将其展开：
$$
= \langle \mathbf{x}_i, \mathbf{x}_i \rangle - \langle \mathbf{x}_i, \langle \mathbf{x}_i, \mathbf{w} \rangle \mathbf{w} \rangle - \langle \langle \mathbf{w}, \mathbf{x}_i \rangle \mathbf{x}_i, \mathbf{w} \rangle + \langle \langle \mathbf{w}, \mathbf{x}_i \rangle \mathbf{w}, \langle \mathbf{w}, \mathbf{x}_i \rangle \mathbf{w} \rangle
$$
利用内积的性质 $\langle a\mathbf{u}, b\mathbf{v} \rangle = ab\langle \mathbf{u}, \mathbf{v} \rangle$ 进一步化简：
$$
= \|\mathbf{x}_i\|^2 - 2|\langle \mathbf{w}, \mathbf{x}_i \rangle|^2 + |\langle \mathbf{w}, \mathbf{x}_i \rangle|^2 \langle \mathbf{w}, \mathbf{w} \rangle
$$
因为 $\mathbf{w}$ 是单位向量，所以 $\langle \mathbf{w}, \mathbf{w} \rangle = 1$。最终残差化简为勾股定理的形式：
$$
\|\mathbf{x}_i - \mathcal{P}_1(\mathbf{x}_i)\|^2 = \|\mathbf{x}_i\|^2 - |\langle \mathbf{w}, \mathbf{x}_i \rangle|^2
$$

**结论**：
从上式可以看出，由于 $\|\mathbf{x}_i\|^2$ 是固定不变的常数（原始数据的范数），要使得残差 $\|\mathbf{x}_i - \mathcal{P}_1(\mathbf{x}_i)\|^2$ **最小**，就必须使得 $|\langle \mathbf{w}, \mathbf{x}_i \rangle|^2$（即投影后的方差相关项）**最大**。这就是 PCA 两种直觉（最小重构误差 vs 最大化方差）在数学上完全等价的根本原因。


## 2.7 均方误差与优化目标 (Mean-Squared Error & Optimization Target)
**均方误差 (MSE)** 定义为预测器误差平方的平均值。结合上一节的单点残差公式，所有数据点在投影方向 $\mathbf{w}$ 上的总均方误差可以展开为：
$$
\text{MSE}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \left( \|\mathbf{x}_i\|^2 - |\langle \mathbf{w}, \mathbf{x}_i \rangle|^2 \right) = \frac{1}{n} \left( \sum_{i=1}^{n} \|\mathbf{x}_i\|^2 - \sum_{i=1}^{n} |\langle \mathbf{w}, \mathbf{x}_i \rangle|^2 \right)
$$

**关键观察 (Observation 1)**：
公式中的第一项求和 $\sum_{i=1}^{n} \|\mathbf{x}_i\|^2$ 是原始数据自身的能量，它是一个与投影方向 $\mathbf{w}$ 完全无关的常数。因此，**为了让总 MSE 尽可能小，我们必须让第二项（即减数部分）尽可能大**。

## 2.8 求解第一主成分 (The First Principal Component)
**核心目标 (Goal)**：
基于上述观察，我们的优化问题正式转化为：在所有可能的单位向量中，寻找一个方向 $\mathbf{w}$，使得数据在该方向上投影的平方和（期望）最大化。形式化表达如下：
$$
\mathbf{w}^* := \arg \max_{\substack{\mathbf{w} \in \mathbb{R}^d \\ \|\mathbf{w}\|=1}} \mathbb{E} \left[ |\langle \mathbf{w}, \mathbf{x} \rangle|^2 \right]
$$
求解得到的这个最优方向向量 $\mathbf{w}^*$，就被称为**第一主成分 (first principal component)**。

**总结 (Observation 2)**：
由于我们已经假设数据是零均值的（投影的均值也为零），这里的推导再次印证了之前的直觉：**最小化残差平方和 (minimizing the residual sum of squares) 在数学上完全等价于最大化投影的方差 (maximizing the variance of the projections)**。


## 2.9 多主成分的推广 (Multiple Principal Components)
在实际的降维任务中，通常只投影到一个维度是不够的，我们需要投影到多个主成分构成的低维子空间中。

假设我们已经找到了一组 $s$ 个**正交单位向量 (orthogonal unit vectors)** $\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_s$ 作为主成分。那么数据点 $\mathbf{x}_i$ 在由这些向量张成的空间中的投影 $\mathcal{P}_s(\mathbf{x}_i)$，等于它在各个基向量上投影的线性叠加：
$$
\mathcal{P}_s(\mathbf{x}_i) := \sum_{j=1}^{s} \langle \mathbf{x}_i, \mathbf{w}_j \rangle \mathbf{w}_j
$$

**矩阵形式的统一**：
结合我们在第一节“降维简介”中提到的压缩 ($\mathbb{W}$) 与恢复 ($\mathbb{U}$) 矩阵，多维投影过程可以被优雅地写为矩阵乘法：
$$
\mathcal{P}_s(\mathbf{x}_i) = \mathbb{U}\mathbb{W}\mathbf{x}_i
$$
其中：
* **恢复矩阵 (基矩阵)**：$\mathbb{U} = [\mathbf{w}_1, \cdots, \mathbf{w}_s]$ （由主成分列向量组成）
* **压缩矩阵 (投影矩阵)**：$\mathbb{W} = \mathbb{U}^T$
*(注：这揭示了在标准 PCA 中，用于降维映射的矩阵恰好就是主成分基矩阵的转置。)*

---

# 3 Implementation of PCA

## 3.1 术语与矩阵表示 (Terminology & Matrix Formulation)
为了方便进行全局计算，我们将 $n$ 个数据向量 $\{\mathbf{x}_1, \cdots, \mathbf{x}_n\}$ 堆叠成一个数据矩阵 $\mathbb{X}$：
$$
\mathbb{X} := [\mathbf{x}_1, \cdots, \mathbf{x}_n]^T \in \mathbb{R}^{n \times d}
$$
那么，所有数据在方向 $\mathbf{w}$ 上的投影可以统一表示为 $n \times 1$ 的矩阵（向量） $\mathbb{X}\mathbf{w}$。

**方差的矩阵推导**：
投影后的方差 $\sigma_{\mathbf{w}}^2$ 可以通过矩阵形式优雅地推导出来：
$$
\sigma_{\mathbf{w}}^2 := \mathbb{E}[|\langle \mathbf{w}, \mathbf{x} \rangle|^2] = \frac{1}{n} \sum_{i=1}^{n} |\langle \mathbf{w}, \mathbf{x}_i \rangle|^2
$$
转化为矩阵内积的形式：
$$
= \frac{1}{n} \langle \mathbb{X}\mathbf{w}, \mathbb{X}\mathbf{w} \rangle = \frac{1}{n} \mathbf{w}^T \mathbb{X}^T \mathbb{X} \mathbf{w}
$$
将常数项移到中间：
$$
= \mathbf{w}^T \left( \frac{\mathbb{X}^T \mathbb{X}}{n} \right) \mathbf{w} := \mathbf{w}^T \mathbb{V} \mathbf{w}
$$
其中，$\mathbb{V} = \frac{\mathbb{X}^T \mathbb{X}}{n}$ 正是这组数据的**协方差矩阵 (Covariance Matrix)**（假设数据已零均值化）。


## 3.2 约束最大化 (Constrained Maximization)
我们的目标是选择一个**单位向量** $\mathbf{w}$ 来最大化方差 $\sigma_{\mathbf{w}}^2$。
由于必须是单位向量，这就引入了一个约束条件 (constraint)：
$$
\|\mathbf{w}\|^2 = \mathbf{w}^T\mathbf{w} = 1
$$
因此，这是一个**带约束的优化问题**。

## 3.3 拉格朗日乘子法求解 (Method of Lagrange Multiplier)
为了求解上述约束优化问题，我们引入拉格朗日乘子 $\lambda$，构造拉格朗日函数 $\mathcal{L}$：
$$
\mathcal{L}(\mathbf{w}, \lambda) = \sigma_{\mathbf{w}}^2 - \lambda(\mathbf{w}^T \mathbf{w} - 1) = \mathbf{w}^T \mathbb{V} \mathbf{w} - \lambda(\mathbf{w}^T \mathbf{w} - 1)
$$

**寻找驻点 (Stationary Point)**：
对拉格朗日函数分别求 $\lambda$ 和 $\mathbf{w}$ 的偏导数：
1.  对 $\lambda$ 求偏导：
    $$\frac{\partial \mathcal{L}}{\partial \lambda} = \mathbf{w}^T \mathbf{w} - 1$$
2.  对 $\mathbf{w}$ 求偏导（这里用到了矩阵求导法则 $\frac{\partial (\mathbf{w}^T \mathbb{V} \mathbf{w})}{\partial \mathbf{w}} = 2\mathbb{V}\mathbf{w}$）：
    $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbb{V}\mathbf{w} - 2\lambda\mathbf{w}$$

令偏导数等于零，我们得到最优化条件：
$$
\begin{cases}
\mathbf{w}^T \mathbf{w} = 1 \\
\mathbb{V}\mathbf{w} = \lambda\mathbf{w}
\end{cases}
$$

## 3.4 核心结论 (Conclusion)
观察等式 $\mathbb{V}\mathbf{w} = \lambda\mathbf{w}$，这正是线性代数中标准的**特征值方程**！

* **性质界定**：我们所寻找的投影方向向量 $\mathbf{w}$，必须是协方差矩阵 $\mathbb{V}$ 的**特征向量 (eigenvector)**。
* **最大化条件**：将 $\mathbb{V}\mathbf{w} = \lambda\mathbf{w}$ 代回方差公式，得到最大化的方差 $\sigma_{\mathbf{w}}^2 = \mathbf{w}^T(\lambda\mathbf{w}) = \lambda\mathbf{w}^T\mathbf{w} = \lambda$。这意味着投影后的方差恰好等于特征值 $\lambda$。
* **最终结论**：为了使方差最大化，**最优的主成分向量 $\mathbf{w}$ 必须是协方差矩阵 $\mathbb{V}$ 对应于最大特征值 $\lambda$ 的特征向量**。

## 3.5 协方差矩阵 $\mathbb{V}$ 的核心性质 (Properties of $\mathbb{V}$)
根据前文推导，协方差矩阵 $\mathbb{V} = \frac{1}{n}\mathbb{X}^T\mathbb{X}$ 具有以下重要线性代数性质：
* **维度与特征向量**：$\mathbb{V}$ 是一个 $d \times d$ 的矩阵，因此它将有 $d$ 个不同的特征向量。
* **对称性与正交性**：因为 $\mathbb{V}$ 是协方差矩阵，所以它是**对称矩阵 (Symmetric)**。线性代数定理表明，它的特征向量必定**相互正交 (orthogonal to one another)**。
* **半正定性与特征值**：由于对任意向量 $\mathbf{x}$ 都有 $\mathbf{x}^T\mathbb{V}\mathbf{x} \ge 0$，所以 $\mathbb{V}$ 是一个半正定矩阵 (Positive matrix)。这意味着它的**所有特征值都必须是非负的 (non-negative)**。

**最终结论 (Conclusion)**：协方差矩阵 $\mathbb{V}$ 的特征向量构成了数据的**主成分 (principal components)**。


## 3.6 计算复杂度与 $n \ll d$ 时的降维技巧 (Computational Complexity & Trick)
* **维度灾难**：在某些场景下，原始数据的维度 $d$ 远远大于样本数量 $n$ ($n \ll d$)。此时直接计算 PCA 非常昂贵：求解 $\mathbb{V}$ 的特征值复杂度为 $\mathcal{O}(d^3)$，构造矩阵 $\mathbb{V}$ 的复杂度为 $\mathcal{O}(nd^2)$。
* **对偶 / Gram 矩阵技巧**：
    * 当 $n \ll d$ 时，我们可以使用一个技巧：不去直接求 $d \times d$ 协方差矩阵的特征值，而是去求解 $n \times n$ 矩阵的特征值问题。
    * *(注：课件图片中“eigenvalue problem of $\mathbb{V}^T$” 疑为笔误，因为 $\mathbb{V}$ 是对称矩阵 $\mathbb{V}^T = \mathbb{V}$。这里的标准做法是计算 Gram 矩阵 $\mathbb{X}\mathbb{X}^T$ 的特征值。假设 $(\lambda_i, \mathbf{u}_i)_{i=1}^n$ 是该 $n \times n$ 矩阵的特征对，那么 $(\lambda_i, \mathbb{X}^T\mathbf{u}_i)_{i=1}^n$ 就是原矩阵的特征对。)*
    * **优化后的复杂度**：运用此技巧后，求解特征值的复杂度大幅降至 $\mathcal{O}(n^3)$，构造矩阵的复杂度降至 $\mathcal{O}(dn^2)$。

## 3.7 PCA 三大核心等价问题 (Connection to our main question)
我们可以从数学上证明，以下三个问题的求解本质上是**完全等价的**：

1.  **最小化向量投影残差**：
    $$[\mathbf{w}_1^*, \cdots, \mathbf{w}_s^*] := \arg \min_{\substack{\mathbf{w}_1, \cdots, \mathbf{w}_s \in \mathbb{R}^d \\ \|\mathbf{w}_1\|, \cdots, \|\mathbf{w}_s\| = 1}} \sum_{i=1}^{n} \left\| \mathbf{x}_i - \sum_{j=1}^{s} \langle \mathbf{x}_i, \mathbf{w}_j \rangle \mathbf{w}_j \right\|_2^2$$
2.  **最小化矩阵重构误差**：
    $$[\mathbb{W}^*, \mathbb{U}^*] = \arg \min_{\substack{\mathbb{W} \in \mathbb{R}^{s \times d} \\ \mathbb{U} \in \mathbb{R}^{d \times s}}} \sum_{i=1}^{n} \| \mathbf{x}_i - \mathbb{U}\mathbb{W}\mathbf{x}_i \|_2^2$$
3.  **最大化特征值 (特征分解)**：
    寻找协方差矩阵 $\mathbb{V}$ 的前 $s$ 个最大特征值。
    * 在上述等价关系中，最优解满足 $\mathbb{U}^* = [\mathbf{w}_1^*, \cdots, \mathbf{w}_s^*]$，且压缩矩阵 $\mathbb{W}^* = (\mathbb{U}^*)^T$。
    * 每一个最优方向向量 $\mathbf{w}_j^*$ 正好对应于协方差矩阵的第 $j$ 大特征值的特征向量。

## 3.8 核心总结与降维性质 (Remark)
* **空间张成**：$\mathbb{V}$ 的所有特征向量相互正交，它们共同张成了完整的 $d$ 维空间。
* **最大方差解释**：
    * **第一主成分**（对应最大特征值的特征向量）是数据拥有**最大方差 (largest variance)** 的投影方向。
    * **第二主成分**是在与第一主成分正交的所有方向中，拥有最大方差的方向。
* **去相关性 (Uncorrelated)**：因为主成分之间相互正交，所以数据在这些主成分上的投影是**互不相关的 (uncorrelated)**。事实上，数据在所有主成分上的投影彼此都不相关。
* **保留的总方差**：如果我们使用前 $s$ 个主成分进行降维（使得权重矩阵 $\mathbb{W}$ 是一个 $d \times s$ 的矩阵），那么投影到这 $s$ 个主成分上所保留的**总方差**，恰好等于前 $s$ 个特征值的总和：
    $$\sum_{i=1}^{s} \lambda_i$$

---

# 4 核主成分分析 (Kernel PCA)

## 4.1 动机：线性降维的局限性 (Motivation)
标准 PCA 假设数据的潜在结构是**线性 (linear)** 的低维流形。
* **局限性示例**：如果数据分布呈现一个圆环状（非线性结构），与完全随机散布的数据相比，标准 PCA 可能无法区分它们（因为它们在各个正交方向上的线性方差可能非常相似，导致求出的主成分轴完全一样）。
* **核心问题**：我们是否有办法去寻找和提取数据中潜在的**非线性 (nonlinear)** 低维流形结构？


## 4.2 引入核方法使 PCA 非线性化 (How to make PCA non-linear?)
为了处理非线性结构，我们利用“核技巧 (Kernel Trick)”，将原始数据映射到一个极高维（甚至无限维）的空间，在那个高维空间中，原本非线性的结构可能会变得线性可分/可提取。

* **特征映射 (Feature Mapping)**：
  假设原始数据集 $S := \{\mathbf{x}_1, \cdots, \mathbf{x}_n\} \subset \mathbb{R}^d$，维度为 $d$。
  定义一个特征映射 $\psi: \mathcal{X} \rightarrow \mathcal{H}$，将原始实例空间 $\mathcal{X}$ 中的数据点映射到一个高维的**希尔伯特空间 (Hilbert space)** $\mathcal{H}$ 中。
* **核函数 (Kernel Function)**：
  定义核函数 $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$。它的本质是计算映射到高维空间后，两个向量的内积：
  $$K(\mathbf{x}, \mathbf{x}') = \langle \psi(\mathbf{x}), \psi(\mathbf{x}') \rangle_{\mathcal{H}}$$
* **Kernel PCA 的定义**：
  Kernel PCA 的过程就是：首先使用特征映射 $\psi$ 将集合 $S$ 中的元素映射到高维空间 $\mathcal{H}$ 中，然后在映射后的数据集 $\{\psi(\mathbf{x}_1), \cdots, \psi(\mathbf{x}_n)\}$ 上**执行标准的 PCA**。

## 4.3 Kernel PCA 的数学推导 (Mathematical Formulation)

在特征空间 $\mathcal{H}$ 中执行 PCA 的步骤与标准 PCA 类似，只是所有的数据点都替换为了映射后的 $\psi(\mathbf{x}_i)$。

* **零均值假设 (Zero Mean Assumption)**：
  假设数据在特征空间 $\mathcal{H}$ 中已经中心化（均值为零）：
  $$\sum_{i=1}^{n} \psi(\mathbf{x}_i) = 0$$
  *(注：如果均值不为零，我们需要在特征空间中先对数据减去均值进行中心化处理。)*

* **协方差矩阵 (Covariance Matrix)**：
  在特征空间 $\mathcal{H}$ 中，协方差矩阵 $\mathbb{V}$ 定义为：
  $$\mathbb{V} = \frac{1}{n} \sum_{i=1}^{n} \psi(\mathbf{x}_i)\psi(\mathbf{x}_i)^T$$

* **特征值问题 (Eigenvalue Problem)**：
  寻找主成分即为求解协方差矩阵 $\mathbb{V}$ 的特征值和特征向量。我们需要找到特征对 $(\lambda_j, v_j) \in \mathbb{R} \times \mathcal{H}$，满足以下条件：
  $$
  \begin{cases}
  \mathbb{V}v_j = \lambda_j v_j, \quad j = 1, \cdots \\
  v_j^T v_j = 1
  \end{cases}
  $$
  *(注：这里的 $v_j^T v_j = 1$ 表示在希尔伯特空间中特征向量是单位向量，即 $\langle v_j, v_j \rangle_{\mathcal{H}} = 1$。)*

## 4.4 协方差矩阵与方差最大化视角 (Covariance Matrix & Variance Maximization)
在特征空间 $\mathcal{H}$ 中，协方差矩阵定义为：
$$
\mathbb{V} = \frac{1}{n} \sum_{i=1}^{n} \psi(\mathbf{x}_i)\psi(\mathbf{x}_i)^T
$$

**方差最大化解释**：
我们可以引入一个在样本空间 $\Omega := \{\psi(\mathbf{x}_1), \cdots, \psi(\mathbf{x}_n)\}$ 上服从均匀分布的随机向量 $\xi$。我们的目标是寻找一个单位范数的方向向量 $\omega \in \mathcal{H}$，使得投影后的随机变量 $\langle \omega, \xi \rangle_{\mathcal{H}}$ 的方差最大。

优化问题表示为：
$$
\arg \max_{\substack{\omega \in \mathcal{H}: \\ \|\omega\|_{\mathcal{H}}=1}} \text{Var}[\langle \omega, \xi \rangle_{\mathcal{H}}] = \arg \max_{\substack{\omega \in \mathcal{H}: \\ \|\omega\|_{\mathcal{H}}=1}} \frac{1}{n} \sum_{i=1}^{n} |\langle \omega, \psi(\mathbf{x}_i) \rangle_{\mathcal{H}}|^2
$$
* **关键推论**：利用核技巧可以证明，我们只需要在由映射后数据 $\{\psi(\mathbf{x}_1), \cdots, \psi(\mathbf{x}_n)\}$ 所张成的**有限维子空间**中去寻找最优的 $\omega$ 即可。找到的 $\omega$ 即为第一主成分；后续的主成分则是在与前面主成分正交的条件下，继续寻找使方差最大的方向。

## 4.5 核技巧与 Gram 矩阵的推导 (Gram Matrix & Mathematical Trick)
为了避免直接在可能无限维的特征空间 $\mathcal{H}$ 中进行计算，我们引入 **Gram 矩阵** $\mathbb{K} \in \mathbb{R}^{n \times n}$，其元素定义为：
$$
\mathbb{K}_{i,k} := K(\mathbf{x}_i, \mathbf{x}_k) = \langle \psi(\mathbf{x}_i), \psi(\mathbf{x}_k) \rangle_{\mathcal{H}}
$$

**特征函数的线性组合**：
由于主成分存在于样本张成的空间中，每个特征函数（主成分方向） $v_j$ 都可以写成映射后特征的线性组合：
$$
v_j = \sum_{i=1}^{n} a_{ji}\psi(\mathbf{x}_i)
$$
因此，寻找特征函数 $v_j$ 的问题，**等价于寻找对应的组合系数 $a_{ji}$**。

**代入特征值方程并化简**：
将 $v_j$ 代入原特征值方程 $\mathbb{V}v_j = \lambda_j v_j$，并利用核函数的定义，我们得到：
$$
\frac{1}{n} \sum_{i=1}^{n} \psi(\mathbf{x}_i) \left( \sum_{l=1}^{n} a_{jl} K(\mathbf{x}_i, \mathbf{x}_l) \right) = \lambda_j \left( \sum_{i=1}^{n} a_{ji}\psi(\mathbf{x}_i) \right)
$$
🔥 **核心技巧 (A small trick)**：在等式两边同时左乘 $\psi(\mathbf{x}_k)^T$，利用内积生成核函数，我们就可以完全消除特征映射 $\psi$，得到只包含核函数的方程：
$$
\frac{1}{n} \sum_{i=1}^{n} K(\mathbf{x}_k, \mathbf{x}_i) \left( \sum_{l=1}^{n} a_{jl} K(\mathbf{x}_i, \mathbf{x}_l) \right) = \lambda_j \left( \sum_{i=1}^{n} a_{ji} K(\mathbf{x}_k, \mathbf{x}_i) \right)
$$


## 4.6 求解系数矩阵与新数据投影 (Matrix Form & Projection)
将上述化简后的等式写成紧凑的矩阵形式：
$$
\mathbb{K}^2 \mathbf{a}_j = n\lambda_j \mathbb{K}\mathbf{a}_j
$$
两边同时约去一个 $\mathbb{K}$（这只会影响特征值为 0 的非主成分项），得到最终的特征值问题：
$$
\mathbb{K}\mathbf{a}_j = n\lambda_j \mathbf{a}_j
$$

**归一化条件 (Normalization)**：
由 $v_j^T v_j = 1$，推导得出系数向量的归一化条件为 $\mathbf{a}_j^T \mathbb{K} \mathbf{a}_j = 1$。
将其代入特征值方程 $\mathbb{K}\mathbf{a}_j = n\lambda_j \mathbf{a}_j$ 中，可以得到：
$$
\lambda_j n \mathbf{a}_j^T \mathbf{a}_j = 1 \quad \text{for all } j
$$

**对新数据点进行降维投影**：
对于一个全新的数据点 $\mathbf{x} \in \mathcal{X}$，它在第 $j$ 个主成分上的投影坐标可以直接通过核函数计算，完全不需要知道显式的映射 $\psi$：
$$
\psi(\mathbf{x})^T v_j = \sum_{i=1}^{n} a_{ji} K(\mathbf{x}, \mathbf{x}_i)
$$


## 4.7 Kernel PCA 算法流程总结 (Summary of Kernel PCA)
在实际应用中，Kernel PCA 的步骤非常清晰：
1. **选择一个核函数 (Pick a kernel)**：例如高斯核 (RBF)、多项式核等。
2. **构建核矩阵 (Construct Matrix)**：计算原始数据在核函数下的 $n \times n$ 核矩阵 $\mathbb{K}$，并对其进行中心化处理 (Normalized kernel matrix)。
3. **求解特征值问题 (Find eigenpairs)**：计算该核矩阵的特征对 $(\lambda_j, \mathbf{a}_j)$。
4. **特征表示 (Representation)**：对于任意数据点 $\mathbf{x}$，它可以表示为一组新的特征：
   $$
   y_j = \sum_{i=1}^{n} a_{ji} K(\mathbf{x}, \mathbf{x}_i), \quad j = 1, \cdots, n
   $$
5. **截断与降维 (Dimensionality Reduction)**：为了获得更紧凑的表示，我们可以只挑选对应于**最大特征值**的前 $s$ 个分量（$s \ll n$），从而完成非线性降维。


# 5 PCA 的实际应用
## 5.1 PCA 的广泛应用 (Wide Applications of PCA)
主成分分析不仅是一个数学工具，在实际业务和科学研究中有着极为广泛的应用：

* **构建发展指数 (Development indexes)**：例如“城市发展指数 (City Development Index)”。起初可能有多达 200 个反映城市状况的指标，通过 PCA 降维，最终提取出约 15 个核心指标，这些核心指标依然能很好地预测和代表众多原始变量。
* **群体遗传学 (Population genetics)**：遗传变异在很大程度上与地理位置相关。研究表明，基因数据的**前两个主成分**实际上就能直接反映出人口在空间上的地理分布 (spatial distribution)。
* **市场调研与态度指数 (Market research and indexes of attitude)**：在商业分析中，PCA 常被用来将多项复杂的问卷反馈，降维整合成单一的“客户满意度”或“客户忠诚度”得分。


## 5.2 金融背景知识：债券分析实例预备 (Analysis of US Zero-Coupon Rates)
为了展示 PCA 在金融时间序列上的应用，课件引入了美国零息债券利率的分析。在此之前，需要掌握相关的基础概念。

### 债券的基础定义 (Definition of Bond)
* **概念**：债券是企业或政府实体筹集资金的“入口 (portal)”。
* **机制**：当债券发行时，投资者购买债券，实际上扮演了向发行实体**借款人 (lenders)** 的角色。
* **回报**：投资者在债券的整个生命周期内，通常会获得半年度或年度的**票息支付 (coupon payments)** 作为回报。

### 零息债券 (Definition of Zero-Coupon Bond)
* **概念**：零息债券（也称为应计债券 / accrual bond）是一种**不支付利息 (does not pay interest)** 的债务证券。
* **盈利模式**：它在发行时以**极深的折扣价 (deep discount)** 交易。投资者购买后无需在此期间领取利息，而是在债券**到期 (maturity)** 时，按照其**全额面值 (full face value)** 赎回，从而获得利润。

## 5.3 零息债券的收益计算 (Interest Earned on a Zero-Coupon Bond)

* **推算利息 (Imputed Interest)**：由于零息债券不支付实际利息，其赚取的利息被称为“推算利息”。这是一种**估算 (estimated)** 出来的利率，而不是确定的明文票息率。

**计算案例 (Example)**：
假设有一张面值 $20,000$ 美元、期限为 20 年的零息债券。
1.  **购买**：如果以 $5.5\%$ 的收益率 (yield) 计算，现在的购买价格大约仅为 $6,855$ 美元。
    计算公式如下（复利增长）：
    $$6855(1 + 5.5\%)^{20} \approx 20000$$
2.  **到期**：20 年期满后，投资者将收到完整的 $20,000$ 美元。
3.  **收益本质**：面值与购买价之间的差额（$20,000 - $6,855）就代表了在债券到期前**自动复利 (compounds automatically)** 的利息收益。
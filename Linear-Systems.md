# Chapter 2 Numerical methods for solving linear systems

# 1 Direct Method
## 1.1 前/后向替换 （Forward / backward substitution）

### 1.1.1 线性方程组概述

一般形式：
\[
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\quad\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
\]

- 未知数：\(x_1, x_2, \dots, x_n\)
- 超定系统（overdetermined）
当 m > n 时，方程的数量多于未知量的数量。这类方程组通常没有精确满足所有方程的解，一般需要通过最小二乘法等方法寻找 “最优近似解”。
- 欠定系统（underdetermined）
当 m < n 时，方程的数量少于未知量的数量。这类方程组通常有无穷多组解，需要通过约束条件或正则化方法确定特解。
- 本节仅考虑**方阵系统**，
当 m = n 时，方程数等于未知量数，这也是本节重点讨论的情况。


**利用矩阵可将上述系统简洁地表示为：**
\[
A \mathbf{x} = \mathbf{b},
\]
其中
\[
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & \cdots & a_{1n} \\
a_{21} & a_{22} & a_{23} & \cdots & a_{2n} \\
a_{31} & a_{32} & a_{33} & \cdots & a_{3n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & a_{n3} & \cdots & a_{nn}
\end{bmatrix},\quad
\mathbf{x} = \begin{bmatrix}x_1\\ x_2\\ x_3\\ \vdots\\ x_n\end{bmatrix},\quad
\mathbf{b} = \begin{bmatrix}b_1\\ b_2\\ b_3\\ \vdots\\ b_n\end{bmatrix}.
\]

- 我们通常假设 \(A\) 是 **可逆（非奇异）** 矩阵，此时系统存在唯一解。
- 由于计算机采用有限精度浮点数，当 \(A\) 接近奇异时，某些算法可能产生错误的结果。
- 因此，我们的目标是设计高效且可靠的计算机算法来求解此类问题。

### 1.1.2 初等变换

**三种基本操作，用于简化线性方程组：**

1. **交换两行**：\(E_i \leftrightarrow E_j\)
2. **某行乘以非零常数**：\(\lambda E_i \to E_i\)，其中 \(\lambda \neq 0\)
3. **将一行乘以一个倍数加到另一行**：\(E_i + \lambda E_j \to E_i\)

**定理**：若一个方程组由另一个通过有限次初等行变换得到，则两个方程组等价。

**三角矩阵**

- **下三角矩阵** \(L\)：主对角线以上元素全为零。
- **上三角矩阵** \(U\)：主对角线以下元素全为零。
- **对角矩阵**：既是上三角又是下三角。

### 1.1.3 前/后向替换

对于下三角系统 \(L \mathbf{x} = \mathbf{b}\)：
\[
\begin{aligned}
\ell_{11}x_1 &= b_1 \\
\ell_{21}x_1 + \ell_{22}x_2 &= b_2 \\
\vdots \\
\ell_{n1}x_1 + \ell_{n2}x_2 + \cdots + \ell_{nn}x_n &= b_n
\end{aligned}
\]

解的过程：
\[
x_1 = \frac{b_1}{\ell_{11}},\quad
x_2 = \frac{b_2 - \ell_{21}x_1}{\ell_{22}},\quad
\dots,\quad
x_n = \frac{b_n - \sum_{i=1}^{n-1}\ell_{ni}x_i}{\ell_{nn}}
\]

**计算复杂度**
- 加减法：\(\frac{n(n-1)}{2}\)
- 乘法：\(\frac{n(n-1)}{2}\)
- 除法：\(n\)
- 总运算量：\(n^2\)（忽略低阶项）



对于上三角系统 \(U \mathbf{x} = \mathbf{b}\)：
\[
\begin{aligned}
u_{11}x_1 + u_{12}x_2 + \cdots + u_{1n}x_n &= b_1 \\
u_{22}x_2 + \cdots + u_{2n}x_n &= b_2 \\
&\vdots \\
u_{nn}x_n &= b_n
\end{aligned}
\]

解的过程（从最后一个方程开始）：
\[
x_n = \frac{b_n}{u_{nn}},\quad
x_{n-1} = \frac{b_{n-1} - u_{n-1,n}x_n}{u_{n-1,n-1}},\quad
\dots,\quad
x_1 = \frac{b_1 - \sum_{i=2}^{n}u_{1i}x_i}{u_{11}}
\]

**复杂度**：与前向替换相同，也是 \(n^2\)。

### 总结
- 三角系统的求解（前向/后向替换）是线性代数中基本且高效的算法，复杂度 \(O(n^2)\)。
- 对于一般方阵，可通过高斯消元法化为三角系统再求解，这是后续课程的重点。

## 1.2 高斯消元法（Gaussian Elimination）

### 1.2.1 前向消元算法
- **核心问题**：线性方程组在系数矩阵为三角矩阵时可高效求解，因此需要将一般矩阵转化为三角矩阵。
- **前向消元（Forward Elimination）算法**：
该步骤通过初等行变换，逐步将矩阵转化为上三角形式。

```
For i = 1, ..., n-1
    For j = i+1, ..., n
        Multiply i-th row by -a_ji / a_ii and add to j-th row
```


- 高斯消元可通过构造增广矩阵序列 \(\tilde{A}^{(1)}, \tilde{A}^{(2)}, \dots, \tilde{A}^{(n)}\) 精确描述，其中 \(\tilde{A}^{(1)}\) 为初始增广矩阵，\(\tilde{A}^{(k)}\) 的元素 \(a_{ij}^{(k)}\) 更新规则为：
\[
  a_{ij}^{(k)} = 
  \begin{cases}
  a_{ij}^{(k-1)} & \text{当 } i=1,2,\dots,k-1 \text{ 且 } j=1,2,\dots,n+1 \\
  0 & \text{当 } i=k,k+1,\dots,n \text{ 且 } j=1,2,\dots,k-1 \\
  a_{ij}^{(k-1)} - \frac{a_{i,k-1}^{(k-1)}}{a_{k-1,k-1}^{(k-1)}} a_{k-1,j}^{(k-1)} & \text{当 } i=k,k+1,\dots,n \text{ 且 } j=k,k+1,\dots,n+1
  \end{cases}
  \]
- 经过 \(k\) 步消元后，矩阵 \(\tilde{A}^{(k)}\) 呈现为分块上三角结构：
\[
  \tilde{A}^{(k)} = 
  \begin{bmatrix}
  a_{11}^{(1)} & a_{12}^{(1)} & \dots & a_{1k}^{(1)} & \dots & a_{1n}^{(1)} \\
  0 & a_{22}^{(2)} & \dots & a_{2k}^{(2)} & \dots & a_{2n}^{(2)} \\
  \vdots & & \ddots & & & \vdots \\
  0 & \dots & 0 & a_{kk}^{(k)} & \dots & a_{kn}^{(k)} \\
  \vdots & & & & \ddots & \vdots \\
  0 & \dots & 0 & a_{nk}^{(k)} & \dots & a_{nn}^{(k)}
  \end{bmatrix}
  \]


### 1.2.2 计算复杂度分析
- 前向消元的操作数按列累加：
- 第1列（第2行到第n行）：除法 \(n-1\) 次，乘法 \((n-1)^2\) 次，加减 \((n-1)^2\) 次
- 第2列（第3行到第n行）：除法 \(n-2\) 次，乘法 \((n-2)^2\) 次，加减 \((n-2)^2\) 次
- ...
- 总操作数求和公式：
\[
  \sum_{i=1}^{n-1} \left( \sum_{j=i+1}^n 1 + 2(n-i) \right) = \frac{2n^3}{3} - \frac{n^2}{2} - \frac{n}{6}
  \]
- 因此，前向消元的计算复杂度为 **\(O(n^3)\)**。


### 1.2.3 线性系统求解与示例
**求解步骤**
1. 对增广矩阵 \([A \mid b]\) 应用初等行变换，转化为上三角矩阵 \([U \mid c]\)
2. 通过回代（Back Substitution）求解上三角方程组 \(Ux = c\)

**命名由来**
- 消元法求解线性方程组的思想可追溯至古代中国，在高斯之前欧亚大陆的数学家已在使用，但因缺乏详细记载，该方法被命名为“高斯消元法”，是复杂历史的简称。

**示例**
给定矩阵 \(A = \begin{bmatrix} 2 & 1 & -1 \\ -3 & -1 & 2 \\ -2 & 1 & 2 \end{bmatrix}\) 和向量 \(b = \begin{bmatrix} 8 \\ -11 \\ -3 \end{bmatrix}\)，求解 \(Ax = b\)：
1. 消元过程：
 \[
   \begin{bmatrix}
   2 & 1 & -1 & 8 \\
   -3 & -1 & 2 & -11 \\
   -2 & 1 & 2 & -3
   \end{bmatrix}
   \rightarrow
   \begin{bmatrix}
   2 & 1 & -1 & 8 \\
   0 & \frac{1}{2} & \frac{1}{2} & 1 \\
   0 & 2 & 1 & 5
   \end{bmatrix}
   \rightarrow
   \begin{bmatrix}
   2 & 1 & -1 & 8 \\
   0 & \frac{1}{2} & \frac{1}{2} & 1 \\
   0 & 0 & -1 & 1
   \end{bmatrix}
   \]
2. 回代求解得：\(x = [2, 3, -1]^T\)

### 1.2.4 潜在问题

在高斯消元过程中，若第 \(i\) 步的主元 \(a_{ii} = 0\)，会直接导致无法计算乘数 \(m_{ji} = a_{ji}/a_{ii}\)，使消元过程中断；即使 \(a_{ii} \neq 0\) 但绝对值极小，也会使乘数 \(m_{ji}\) 过大，放大浮点舍入误差，严重影响解的精度。


- **前向消元复杂度**：\( \frac{2n^3}{3} + \frac{n^2}{2} - \frac{n}{6} \)
- **回代复杂度**：\( n^2 \)
- **总复杂度**：\( \frac{2n^3}{3} - \frac{n^2}{2} - \frac{n}{6} \)
- **关键问题**：若第 \(i\) 步的主元 \(a_{ii} = 0\)，直接消元会失败，需采用**带主元选择的高斯消元法**（Gaussian elimination with pivoting）。

#### 核心解决方案：带选主元的高斯消元

- **部分选主元（Partial pivoting）**：在第 \(i\) 步消元前，交换行，选择当前列 \(i\) 中从第 \(i\) 行到第 \(n\) 行里绝对值最大的元素作为新主元，避免主元过小或为零，是工程计算中最常用的策略。
- **全选主元（Complete pivoting）**：不仅交换行，还交换列，选择当前子矩阵中绝对值最大的元素作为主元，数值稳定性更高，但会增加额外的计算和内存开销。

选主元策略是保证高斯消元法数值稳定性的关键，在实际数值计算中几乎是必须的步骤。



#### 原始问题（1位精度计算机）
\[
\begin{cases}
0.001x_1 + x_2 = 3 \\
x_1 + 2x_2 = 5
\end{cases}
\]
- 直接前向消元：\(-998x_2 = -2995\)，舍入后 \(x_2 = 3\)
- 回代得 \(x_1 = 0\)，与精确解 \(x_1 = -1.002, x_2 = 3.001\) 相比：
  - \(x_2\) 相对误差：\(0.3 \times 10^{-3}\)
  - \(x_1\) 相对误差：\(1\)（完全失真）

#### 交换方程顺序（主元选择）
\[
\begin{cases}
x_1 + 2x_2 = 5 \\
0.001x_1 + x_2 = 3
\end{cases}
\]
- 前向消元：\(998x_2 = 2995\)，舍入后 \(x_2 = 3\)
- 回代得 \(x_1 = -1\)，误差显著改善：
  - \(x_2\) 相对误差：\(0.3 \times 10^{-3}\)
  - \(x_1\) 相对误差：\(2 \times 10^{-3}\)

### 1.2.5 部分主元选择（Partial Pivoting）策略
何时需要主元选择
1. 当主元 \(a_{kk}^{(k)} = 0\) 时，必须进行行交换。
2. 当 \(a_{kk}^{(k)}\) 非零但绝对值远小于下方元素 \(a_{jk}^{(k)}\) 时，乘子 \(m_{jk} = \frac{a_{jk}^{(k)}}{a_{kk}^{(k)}}\) 会远大于1，导致舍入误差被严重放大。

部分主元选择步骤
1. 在第 \(k\) 列中，找到主对角线下方绝对值最大的元素，记其行索引为 \(p\)：
   \[
   |a_{pk}^{(k)}| = \max_{k \le i \le n} |a_{ik}^{(k)}|
   \]
2. 交换第 \(k\) 行和第 \(p\) 行，使最大绝对值元素成为新的主元，再进行消元。


## 1.3 LU Factorization

### 1.3.1 定义（Definition）
- **核心概念**：对于一个方阵 $\mathbb{A}$，其 **LU 分解** 是将其分解为两个三角矩阵的乘积：
  $$\mathbb{A} = \mathbb{L}\mathbb{U}$$
  其中：
  - $\mathbb{L}$ 是**下三角矩阵**（Lower triangular matrix），主对角线下方元素非零，主对角线及上方元素可含零。
  - $\mathbb{U}$ 是**上三角矩阵**（Upper triangular matrix），主对角线及上方元素非零，主对角线下方元素为零。

- **一般形式（$n \times n$ 矩阵）**：
  $$
  \mathbb{L} = \begin{bmatrix}
  \ell_{1,1} & 0 & 0 & \dots & 0 \\
  \ell_{2,1} & \ell_{2,2} & 0 & \dots & 0 \\
  \ell_{3,1} & \ell_{3,2} & \ddots & \dots & 0 \\
  \vdots & \vdots & \ddots & \ddots & 0 \\
  \ell_{n,1} & \ell_{n,2} & \dots & \ell_{n,n-1} & \ell_{n,n}
  \end{bmatrix}, \quad
  \mathbb{U} = \begin{bmatrix}
  u_{1,1} & u_{1,2} & u_{1,3} & \dots & u_{1,n} \\
  0 & u_{2,2} & u_{2,3} & \dots & u_{2,n} \\
  0 & 0 & \ddots & \ddots & \vdots \\
  \vdots & \vdots & \ddots & \ddots & u_{n-1,n} \\
  0 & 0 & \dots & 0 & u_{n,n}
  \end{bmatrix}
  $$

#### 动机（Motivation）
- **数值分析中的作用**：LU 分解（"LU" 代表 "lower upper"）是将矩阵分解为下三角矩阵与上三角矩阵乘积的技术。
- **历史背景**：由波兰天文学家 **Tadeusz Banachiewicz** 于 1938 年提出。
- **工程与计算应用**：
  - 计算机求解线性方程组 $\mathbb{A}\boldsymbol{x} = \boldsymbol{b}$ 的核心方法。
  - 矩阵求逆、计算矩阵行列式的关键步骤。



### 1.3.2 存在性问题（Does LU factorization exist?）

![1771679324085](image/Linear-Systems/1771679324085.png)

**分解失败的原因**：

  - 若矩阵 $\mathbb{A}$ 未进行适当的行排序或置换，LU 分解可能无法实现。
  - 例如：$a_{11} = l_{11}u_{11}$，若 $a_{11} = 0$，则至少有一个 $l_{11}$ 或 $u_{11}$ 为零，导致 $\mathbb{L}$ 或 $\mathbb{U}$ 奇异；但当 $\mathbb{A}$ 非奇异时，这是不可能的。
  - 这是一个**过程性问题**，可通过对 $\mathbb{A}$ 的行进行**重排（reordering）或主元选择（pivoting）**解决，使置换后矩阵的首元素非零。


**存在性与唯一性定理（Existence and Uniqueness of LU Decomposition）**
- **存在性条件**：
  设 $\mathbb{A}$ 是 $n$ 阶方阵，若其前 $(n-1)$ 个**顺序主子式（leading principal minors）**均非零，则 $\mathbb{A}$ 存在 LU 分解。

- **唯一性条件**：
  若 $\det(\mathbb{A}) \neq 0$，且 $\mathbb{L}$ 为**单位下三角矩阵**（即主对角线元素 $\ell_{ii} = 1$），则 LU 分解是唯一的。

- **顺序主子式定义**：
  $n \times n$ 矩阵的 $k$ 阶顺序主子矩阵，是删除最后 $n-k$ 行和列后得到的子矩阵；该子矩阵的行列式称为**顺序主子式**。

### 1.3.3 Doolittle Algorithm

#### 算法背景与设定
- 目标：对 $n \times n$ 矩阵 $\mathbb{A}$ 进行 LU 分解 $\mathbb{A} = \mathbb{L}\mathbb{U}$，其中：
  $$
  \mathbb{L} = \begin{bmatrix}
  \ell_{1,1} & 0 & 0 & \dots & 0 \\
  \ell_{2,1} & \ell_{2,2} & 0 & \dots & 0 \\
  \ell_{3,1} & \ell_{3,2} & \ddots & \dots & 0 \\
  \vdots & \vdots & \ddots & \ddots & 0 \\
  \ell_{n,1} & \ell_{n,2} & \dots & \ell_{n,n-1} & \ell_{n,n}
  \end{bmatrix}, \quad
  \mathbb{U} = \begin{bmatrix}
  u_{1,1} & u_{1,2} & u_{1,3} & \dots & u_{1,n} \\
  0 & u_{2,2} & u_{2,3} & \dots & u_{2,n} \\
  0 & 0 & \ddots & \ddots & \vdots \\
  \vdots & \vdots & \ddots & \ddots & u_{n-1,n} \\
  0 & 0 & \dots & 0 & u_{n,n}
  \end{bmatrix}
  $$
- 唯一性处理：
  - 仅由 $\mathbb{A} = \mathbb{L}\mathbb{U}$ 无法唯一确定 $\mathbb{L}$ 和 $\mathbb{U}$。
  - 常用约束：令 $\mathbb{L}$ 为**单位下三角矩阵**（即 $\ell_{i,i} = 1$，对所有 $1 \le i \le n$），从而唯一确定分解。

#### 特殊情形：Cholesky 分解
- 当满足 $\mathbb{U} = \mathbb{L}^T$（即 $\ell_{i,i} = u_{i,i}$）时，该分解称为 **Cholesky 分解**。
- 适用条件：矩阵 $\mathbb{A}$ 必须是**实对称正定矩阵**。
- 应用领域：在**概率与统计**中有广泛应用。

#### Doolittle 算法伪代码
![1771679404881](image/Linear-Systems/1771679404881.png)

#### 计算复杂度分析
问题：Doolittle 算法的计算复杂度是多少？
- 复杂度求和式：
  $$
  \sum_{k=1}^n \left[(n - k + 1)2(k - 1) + (n - k)(2k - 1)\right]
  $$
- 展开结果：
  $$
  = \frac{1}{3}n^3 + \frac{1}{2}n^2 - \frac{5n}{6}
  $$

**与高斯消元法的复杂度比较**
- 与高斯消元法相比，Doolittle 算法的**最高阶项系数更小**（二者均为 $O(n^3)$ 复杂度，但 Doolittle 的高阶项占比更低）。

### 1.3.4 Doolittle算法示例：4×4矩阵分解

**矩阵与存在性条件**
给定矩阵：
$$
\mathbb{A} = \begin{pmatrix}
1 & 1 & 1 & 1 \\
2 & 3 & 1 & 5 \\
-1 & 1 & -5 & 3 \\
3 & 1 & 7 & -2
\end{pmatrix}
$$
其顺序主子式为 $1, 1, -2, 2$，均非零，因此存在LU分解。

**分解形式（单位下三角L）**
$$
\mathbb{A} = \mathbb{L}\mathbb{U} = 
\begin{pmatrix}
1 & 0 & 0 & 0 \\
\ell_{21} & 1 & 0 & 0 \\
\ell_{31} & \ell_{32} & 1 & 0 \\
\ell_{41} & \ell_{42} & \ell_{43} & 1
\end{pmatrix}
\begin{pmatrix}
u_{11} & u_{12} & u_{13} & u_{14} \\
0 & u_{22} & u_{23} & u_{24} \\
0 & 0 & u_{33} & u_{34} \\
0 & 0 & 0 & u_{44}
\end{pmatrix}
$$

**分步计算过程**

1.  **初始化**
    - U的第一行与A的第一行相同：$u_{1j} = a_{1j}$
    - L的第一列：$\ell_{i1} = a_{i1} / u_{11} = a_{i1}$
    $$
    \mathbb{L} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    2 & 1 & 0 & 0 \\
    -1 & \ell_{32} & 1 & 0 \\
    3 & \ell_{42} & \ell_{43} & 1
    \end{pmatrix}, \quad
    \mathbb{U} = \begin{pmatrix}
    1 & 1 & 1 & 1 \\
    0 & u_{22} & u_{23} & u_{24} \\
    0 & 0 & u_{33} & u_{34} \\
    0 & 0 & 0 & u_{44}
    \end{pmatrix}
    $$

2.  **计算U的第二行与L的第二列**
    - U的第二行：
      $$
      \begin{align*}
      u_{22} &= 3 - (2)(1) = 1 \\
      u_{23} &= 1 - (2)(1) = -1 \\
      u_{24} &= 5 - (2)(1) = 3
      \end{align*}
      $$
    - L的第二列：
      $$
      \begin{align*}
      \ell_{32} &= \frac{1 - (-1)(1)}{u_{22}} = \frac{2}{1} = 2 \\
      \ell_{42} &= \frac{1 - (3)(1)}{u_{22}} = \frac{-2}{1} = -2
      \end{align*}
      $$
    $$
    \mathbb{L} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    2 & 1 & 0 & 0 \\
    -1 & 2 & 1 & 0 \\
    3 & -2 & \ell_{43} & 1
    \end{pmatrix}, \quad
    \mathbb{U} = \begin{pmatrix}
    1 & 1 & 1 & 1 \\
    0 & 1 & -1 & 3 \\
    0 & 0 & u_{33} & u_{34} \\
    0 & 0 & 0 & u_{44}
    \end{pmatrix}
    $$

3.  **计算U的第三行与L的第三列**
    - U的第三行：
      $$
      \begin{align*}
      u_{33} &= -5 - (-1)(1) - (2)(-1) = -2 \\
      u_{34} &= 3 - (-1)(1) - (2)(3) = -2
      \end{align*}
      $$
    - L的第三列：
      $$
      \ell_{43} = \frac{7 - (3)(1) - (-2)(-1)}{u_{33}} = \frac{2}{-2} = -1
      $$
    $$
    \mathbb{L} = \begin{pmatrix}
    1 & 0 & 0 & 0 \\
    2 & 1 & 0 & 0 \\
    -1 & 2 & 1 & 0 \\
    3 & -2 & -1 & 1
    \end{pmatrix}, \quad
    \mathbb{U} = \begin{pmatrix}
    1 & 1 & 1 & 1 \\
    0 & 1 & -1 & 3 \\
    0 & 0 & -2 & -2 \\
    0 & 0 & 0 & u_{44}
    \end{pmatrix}
    $$

4.  **计算U的第四行**
    - U的第四行：
      $$
      u_{44} = -2 - (3)(1) - (-2)(3) - (-1)(-2) = -1
      $$

**最终分解结果**
$$
\mathbb{A} = 
\begin{pmatrix}
1 & 0 & 0 & 0 \\
2 & 1 & 0 & 0 \\
-1 & 2 & 1 & 0 \\
3 & -2 & -1 & 1
\end{pmatrix}
\begin{pmatrix}
1 & 1 & 1 & 1 \\
0 & 1 & -1 & 3 \\
0 & 0 & -2 & -2 \\
0 & 0 & 0 & -1
\end{pmatrix}
$$


### 1.3.5 LU分解求解线性方程组 $\mathbb{A}\boldsymbol{x} = \boldsymbol{b}$

**求解步骤**
1.  **分解**：对系数矩阵 $\mathbb{A}$ 进行LU分解，得到 $\mathbb{A} = \mathbb{L}\mathbb{U}$。
2.  **前向代入**：求解下三角方程组 $\mathbb{L}\boldsymbol{y} = \boldsymbol{b}$，得到 $\boldsymbol{y}$。
3.  **后向代入**：求解上三角方程组 $\mathbb{U}\boldsymbol{x} = \boldsymbol{y}$，得到解 $\boldsymbol{x}$。

**数学关系**
$$
\boldsymbol{x} = \mathbb{U}^{-1}\boldsymbol{y} = \mathbb{U}^{-1}\mathbb{L}^{-1}\boldsymbol{b} = (\mathbb{L}\mathbb{U})^{-1}\boldsymbol{b} = \mathbb{A}^{-1}\boldsymbol{b}
$$

**计算复杂度**
- **步骤1（LU分解）**：$O(\frac{1}{3}n^3 + \frac{1}{2}n^2 - \frac{5n}{6})$
- **步骤2和3（前向/后向代入）**：$O(2n^2)$

**优势分析**
- 若仅求解一次 $\mathbb{A}\boldsymbol{x} = \boldsymbol{b}$，与直接高斯消元相比节省有限。
- 若需对**不同的 $\boldsymbol{b}$ 多次求解**，由于只需进行一次LU分解，后续求解仅需两次代入，计算节省巨大。


### 1.3.6 通过LU分解计算行列式

**核心公式**
若 $\mathbb{A} = \mathbb{L}\mathbb{U}$，则行列式满足：
$$
\begin{align*}
\det(\mathbb{A}) &= \det(\mathbb{L}\mathbb{U}) \\
&= \det(\mathbb{L})\det(\mathbb{U}) \\
&= \left( \prod_{i=1}^n \mathbb{L}_{ii} \right) \left( \prod_{i=1}^n \mathbb{U}_{ii} \right)
\end{align*}
$$
- 对于Doolittle分解（$\mathbb{L}$ 为单位下三角矩阵），$\det(\mathbb{L}) = 1$，因此 $\det(\mathbb{A}) = \prod_{i=1}^n \mathbb{U}_{ii}$。

**计算复杂度**
- 浮点运算次数约为 $O(2n)$，远低于直接计算行列式。
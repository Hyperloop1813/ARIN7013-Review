# Midterm Review
# 1 Convolutional Neural Network
## 填充
| 填充类型 | 效果 | 输出尺寸示例 |
|----------|------|--------------|
| **Valid Padding（无填充）** | 仅对图像有效区域进行卷积，输出尺寸小于输入 | $5 \times 5$ 输入 → $3 \times 3$ 输出 |
| **Same Padding（补零填充）** | 对输入图像边缘补零，保持输出尺寸与输入一致 | $5 \times 5$ 输入 → $5 \times 5$ 输出（需先扩充为 $7 \times 7$） |

![1772096366231](image/Midterm-Review/1772096366231.png)

## 常见池化类型
- **最大池化（Max Pooling）**：取卷积核覆盖区域的最大值，保留最显著的局部特征。
- **平均池化（Average Pooling）**：取卷积核覆盖区域的平均值，平滑局部特征，减少噪声干扰。

## 参数计算
![1772096509316](image/Midterm-Review/1772096509316.png)
![1772096522091](image/Midterm-Review/1772096522091.png)
![1772096534486](image/Midterm-Review/1772096534486.png)
![1772096546410](image/Midterm-Review/1772096546410.png)

---

# 2 Kernel Method
## 2.1 正定核的基本概念
![1772096850250](image/Midterm-Review/1772096850250.png)
![1772096877060](image/Midterm-Review/1772096877060.png)
![1772096954239](image/Midterm-Review/1772096954239.png)
![1772096964603](image/Midterm-Review/1772096964603.png)

## 2.2 基础的正定核
### 向量空间上最简单的正定核
![1772097095529](image/Midterm-Review/1772097095529.png)
![1772097122499](image/Midterm-Review/1772097122499.png)

### 更具普适性的正定核构造方法
![1772196241583](image/Midterm-Review/1772196241583.png)
![1772097155901](image/Midterm-Review/1772097155901.png)
![1772097192953](image/Midterm-Review/1772097192953.png)
![1772097218672](image/Midterm-Review/1772097218672.png)
![1772097243512](image/Midterm-Review/1772097243512.png)
![1772097306908](image/Midterm-Review/1772097306908.png)

## 2.3 核函数的组合定理
![1772097379654](image/Midterm-Review/1772097379654.png)
![1772097367243](image/Midterm-Review/1772097367243.png)
![1772195893951](image/Midterm-Review/1772195893951.png)
![1772097453351](image/Midterm-Review/1772097453351.png)
![1772104561797](image/Midterm-Review/1772104561797.png)
![1772195841279](image/Midterm-Review/1772195841279.png)

## 2.4 练习题
![1772104900933](image/Midterm-Review/1772104900933.png)
![1772197931801](image/Midterm-Exercise/1772197931801.png)
![1772197957435](image/Midterm-Exercise/1772197957435.png)
![1772197966656](image/Midterm-Exercise/1772197966656.png)
![1772197979663](image/Midterm-Exercise/1772197979663.png)
![1772197979663](image/Midterm-Exercise/1772197979663.png)
![1772200187491](image/Midterm-Exercise/1772200187491.png)
![1772200771193](image/Midterm-Exercise/1772200771193.png)
![1772200795511](image/Midterm-Exercise/1772200795511.png)
![1772200811366](image/Midterm-Exercise/1772200811366.png)


## 2.5 Aronszajn 定理
![1772108189369](image/Midterm-Review/1772108189369.png)
![1772108214970](image/Midterm-Review/1772108214970.png)

---

# 3 Supervised Learning with Kernels
## 3.1 核技巧（Kernel Trick）
![1772170996054](image/Midterm-Review/1772170996054.png)
![1772171027356](image/Midterm-Review/1772171027356.png)
![1772171054688](image/Midterm-Review/1772171054688.png)
![1772171077133](image/Midterm-Review/1772171077133.png)
![1772171094160](image/Midterm-Review/1772171094160.png)

## 3.2 表示定理（Representer Theorem）
![1772171143398](image/Midterm-Review/1772171143398.png)
![1772171172474](image/Midterm-Review/1772171172474.png)
![1772171197834](image/Midterm-Review/1772171197834.png)
**表示定理（Representer Theorem）的正式陈述**
![1772171299383](image/Midterm-Review/1772171299383.png)
**表示定理的核心意义**
![1772171317953](image/Midterm-Review/1772171317953.png)

## 3.3 核岭回归（Kernel Ridge Regression, KRR）
![1772176145704](image/Midterm-Review/1772176145704.png)
![1772176181510](image/Midterm-Review/1772176181510.png)
![1772176210941](image/Midterm-Review/1772176210941.png)
![1772176251285](image/Midterm-Review/1772176251285.png)
![1772176265880](image/Midterm-Review/1772176265880.png)
![1772177298924](image/Midterm-Review/1772177298924.png)
![1772177451838](image/Midterm-Review/1772177451838.png)
![1772177836682](image/Midterm-Review/1772177836682.png)
![1772177984649](image/Midterm-Review/1772177984649.png)
![1771579758342](image/Kernel-Operators/1771579758342.png)
![1771579770551](image/Kernel-Operators/1771579770551.png)
![1771579795170](image/Kernel-Operators/1771579795170.png)
![1771579806566](image/Kernel-Operators/1771579806566.png)
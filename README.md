<!-- <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script> -->
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# PINN 学习记录
## PINN 概述
#### PINN 是什么、能解决什么问题
2019年，来自布朗大学应用数学的研究团队首次提出了[PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125/pdfft?md5=089ff261bda4a8795ae8b0cd73dcd9d4&pid=1-s2.0-S0021999118307125-main.pdf)并发表在了计算物理学领域权威杂志《计算物理学期刊》(Journal of Computational Physics) 上。

PINN的解释为 neural networks that are trained to solve supervised learning tasks while respecting any given laws of physics described by general nonlinear partial differential equations，含义为在给定的、由非线性偏微分方程描述的物理规律的监督下完成学习任务的一种神经网络。可以看到，PINN的提出目的是求解微分方程，这为微分方程求解方法开辟了新思路。

根据笔者的调研，发现 PINN 不但可以求解微分方程 (Data-driven solutions of partial differential equations) ，还可以求解微分方程的逆问题 (Data-driven discovery of partial differential equations) ，即根据已有信息求解微分方程的系数。目前比较成熟的 PINN 开源代码为 [deeponet](https://github.com/lululxvi/deepxde) ，里面提供了很多微分方程的求解案例，安装和应用也很简单。相关代码的下载和使用教程可以参考其中的说明文档。此外还有 [FNO](https://github.com/neuraloperator)(Fourier Neural Operator) ，这个相关代码还未学习，期待能有更好的表现。

#### PINN 基本原理
经过调研和学习，笔者认为 PINN 的关键特点在于损失函数的定义。这里直接引用上面提到的论文中的例子：对于下面一个初值-边值问题

$N_{\boldsymbol{x},t}[u(\boldsymbol{x},t)] = f(\boldsymbol{x},t) \qquad \boldsymbol{x}\in\Omega \quad t\in[0, T]\\
B_{\boldsymbol{x},t}[u(\boldsymbol{x},t)] = g(\boldsymbol{x},t) \qquad \boldsymbol{x}\in\partial\Omega \quad t\in[0, T]\\
u(\boldsymbol{x},t)|_{t=0} = h(\boldsymbol{x})  \qquad\qquad \boldsymbol{x}\in\bar{\Omega}
$

经典的 PINN 损失函数本质上仍然是最小均方误差(MSE)，但具体的表达式与其他神经网络不同: $L(\boldsymbol{w}) = L_s(\boldsymbol{w}) + L_r(\boldsymbol{w}) + L_b(\boldsymbol{w}) + L_0(\boldsymbol{w})$。 式中
- $L_s(\boldsymbol{w})$ 为模型求解与精确解之间的误差带来的损失
- $L_r(\boldsymbol{w})$ 为微分方程带来的损失
- $L_b(\boldsymbol{w})$ 为边界条件带来的损失
- $L_0(\boldsymbol{w})$ 为初始条件带来的损失

此外与传统数值求解方法不同的是，PINN 求解微分方程不再划分网格，而是在求解区域内的边界上随机取点。取点的类型也与上面四个部分对应
- $\boldsymbol{x_s}$ 已知精确解的点
- $\boldsymbol{x_r}$ 求解区域内随机取点
- $\boldsymbol{x_b}$ 边界条件上随机取点
- $\boldsymbol{x_0}$ 初始时刻随机取点

因此损失函数四部分的具体表达式为
$L_s(\boldsymbol{w}) = \sum_{i=1}^{N_s}\left | u(\boldsymbol{x_s^i},t_s^i;\boldsymbol{w}) - y_{s}^{i}\right |^2     \\
L_r(\boldsymbol{w}) = \sum_{i=1}^{N_r}\left | N_{\boldsymbol{x},t}[u(\boldsymbol{x_r^i},t_r^i;\boldsymbol{w})] - f(\boldsymbol{x_r^i},t_r^i) \right |^2    \\
L_b(\boldsymbol{w}) = \sum_{i=1}^{N_b}\left | B_{\boldsymbol{x},t}[u(\boldsymbol{x_b^i},t_b^i;\boldsymbol{w})] - g(\boldsymbol{x_b^i},t_b^i) \right |^2     \\
L_0(\boldsymbol{w}) = \sum_{i=1}^{N_0}\left | u(\boldsymbol{x_0^i},0;\boldsymbol{w}) - h(\boldsymbol{x_0^i})\right |^2\\$

## 简单案例
## PINN 优化方法
## 目前比较成熟的网络结构
## 求解中子扩散方程
# RankNet
RankNet是2005年微软提出的一种pairwise的Learning to Rank算法，它从概率的角度来解决排序问题。RankNet的核心是提出了一种概率损失函数来学习Ranking Function，并应用Ranking Function对文档进行排序。这里的Ranking Function可以是任意对参数可微的模型，也就是说，该概率损失函数并不依赖于特定的机器学习模型，在论文中，RankNet是基于神经网络实现的。除此之外，GDBT等模型也可以应用于该框架。
## 相关性概率
我们先定义两个概率：预测相关性概率、真实相关性概率。
* 预测相关性概率

* 真实相关性概率

## 加速训练


# LambdaRank
上面我们介绍了以错误pair最少为优化目标的RankNet算法，然而许多时候仅以错误pair数来评价排序的好坏是不够的，像NDCG或者ERR等评价指标就只关注top-k个结果的排序，当我们采用RankNet算法时，往往无法以这些指标为优化目标进行迭代，所以RankNet的优化目标和IR评价指标之间还是存在gap的。

LambdaRank是一个经验算法，它不是通过显示定义损失函数再求梯度的方式对排序问题进行求解，而是分析排序问题需要的梯度的物理意义，直接定义梯度，即Lambda梯度。

损失函数的梯度代表了文档下一次迭代优化的方向和强度，由于引入了IR评价指标，Lambda梯度更关注位置靠前的优质文档的排序位置的提升。有效的避免了下调位置靠前优质文档的位置这种情况的发生。LambdaRank相比RankNet的优势在于分解因式后训练速度变快，同时考虑了评价指标，直接对问题求解，效果更明显。

## 问题：
    乘以|deltaNDCG|，为什么少了前面那项？

# LambdaMART
- Mart定义了一个框架，缺少一个梯度。
- LambdaRank重新定义了梯度，赋予了梯度新的物理意义。

因此，所有可以使用梯度下降法求解的模型都可以使用这个梯度，MART就是其中一种，将梯度Lambda和MART结合就是大名鼎鼎的LambdaMART。

# Latex
![](http://latex.codecogs.com/gif.latex?\\frac{1}{1+sin(x)})
## 代码说明
### 参数
|参数|含义|
|:--:|:--:|
|x|数据集|
|k|聚类个数|
|meanVectors|储存均值向量|
|clusters|储存每个簇包含的点|
|resolustion|绘图时的网格密度|
|plot|是否需要绘制过程图|

### 函数及功能
|函数|功能|
|:--:|:--:|
|train|训练|
|predict|预测|
|select_k|通过手肘法挑选k|
|sumDistance|计算所有点到对应聚点的距离之和|
|plot_decision_boundaries|绘制决策边界|

## K-Means迭代过程

![kmeans](/figure/kmeans_0.png)
![kmeans](/figure/kmeans_1.png)
![kmeans](/figure/kmeans_2.png)
![kmeans](/figure/kmeans_3.png)

## 通过手肘法确定k的取值

![手肘法](/figure/kmeans_手肘法.png)

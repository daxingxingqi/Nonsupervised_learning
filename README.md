# Nonsupervised_learning

Hierarchical Clustering Example

### 1. [Using Hierarchical Clustering of Secreted Protein Families to Classify and Rank Candidate Effectors of Rust Fungi](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029847)



## [K-means](https://github.com/daxingxingqi/Nonsupervised_learning/tree/master/movie_re)
下图是K-means对于不同数据集的表现。
<div align=center><img src=resources/1.png></div>

层次聚类和密度聚类的区别
<div align=center><img src=resources/2.png></div>

### [Hierrchical clustering](https://github.com/daxingxingqi/Nonsupervised_learning/blob/master/Hierarchical%20Clustering%20Lab-zh.ipynb)
#### single-link clustering 
- SLC测量的是簇之间最小的距离，然后对比簇与簇之间最小的距离，选择最小的聚合。
<div align=center><img src=resources/3.png></div>
- SLC有时候会导致狭长的类
<div align=center><img src=resources/4.png></div>
- 下图的SLC对于不同数据集的表现
<div align=center><img src=resources/5.png></div>

#### complete-link clustering
- CLC测量的是簇之间最大的距离，然后对比簇与簇之间最大的距离，选择最小的聚合。
<div align=center><img src=resources/6.png></div>
- CLC有时会忽略其他的点
<div align=center><img src=resources/7.png></div>

#### average-link clustering
- 计算所有距离的平均值
<div align=center><img src=resources/8.png></div>

>ward method clustering
- 计算所有距离的平均值到中间值的平方，加在一起然后减去内部点到内部中间值的平方
<div align=center><img src=resources/9.png></div>

```  python
from sklearn.cluster import AgglomerativeClustering
# Hierarchical clustering
# Ward is the default linkage algorithm, so we'll start with that
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(iris.data)

#可视化
# Import scipy's linkage function to conduct the clustering
from scipy.cluster.hierarchy import linkage
# Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
# Pick the one that resulted in the highest Adjusted Rand Score
linkage_type = 'ward'
linkage_matrix = linkage(normalized_X, linkage_type)
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
plt.figure(figsize=(22,18))
# plot using 'dendrogram()'
dendrogram(linkage_matrix)
plt.show()
import seaborn as sns
sns.clustermap(normalized_X, figsize=(12,18), method=linkage_type, cmap='viridis')
# Expand figsize to a value like (18, 50) if you want the sample labels to be readable
# Draw back is that you'll need more scrolling to observe the dendrogram
plt.show()
```


## Density clustering
#### [DNSCAN](https://github.com/daxingxingqi/Nonsupervised_learning/blob/master/DBSCAN%20Notebook-zh.ipynb) [可视化](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
- 聚合一部分，把其他的作为噪音
<div align=center><img src=resources/10.png></div>
- 具体步骤如下
<div align=center><img src=resources/11.png></div>
- K-means 对比 DBSCAN
<div align=center><img src=resources/12.png></div>

- 优点
  - 不用指定簇的个数
  - 形状和大小不固定
  - 可以处理噪音
  - 可以处理离群值
  
- 缺点
  - Border points 可以被两个簇同时涉及
  - 不同密度需要用HDBSCAN
  
### DBSCAN example
### 1. [Traffic Classification Using Clustering Algorithms](https://pages.cpsc.ucalgary.ca/~mahanti/papers/clustering.pdf) [pdf]

### 2. [Anomaly detection in temperature data using dbscan algorithm](https://ieeexplore.ieee.org/abstract/document/5946052/)
----
## Gaussian Mixture Models
- 假定每个簇都遵循特定的统计分布（高斯分布）
<div align=center><img src=resources/13.png></div>
  - step 1 Initialize Gaussian Distribution 使用数据的均值初始化或者使用K-means找到几个簇
  <div align=center><img src=resources/14.png></div>
  - step 2 Soft-cluster the data points -"Expection step" 一个数据点的计算，分别计算点A属于A和B的置信虑
  <div align=center><img src=resources/15.png></div>
  - step 3 Re-estimate parameters of gaussians - "Maximization"step
  <div align=center><img src=resources/16.png></div>
  - step 4 Evaluate log-likelihood
  <div align=center><img src=resources/17.png></div>
  
``` python
from sklearn import datasets, mixture

X = datasets.load_iris().data[:10]
gmm = mixture.GaussianMixture(n_components = 3)
gmm.fit(X)
clustering = gmm.predict

# results[1000120100]
```

- 优点
  - soft-clustering
  - 簇的形状和大小多样
  
  
- 缺点
  - 对于初始值敏感
  - 有可能收敛到local minimum
  - 收敛缓慢
 
### GMM example
### paper: Nonparametric discovery of human routines from sensor data [PDf]-http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.681.3152&rep=rep1&type=pdf

### Paper: Application of the Gaussian mixture model in pulsar astronomy [PDF]-https://arxiv.org/abs/1205.6221

### Paper: Speaker Verification Using Adapted Gaussian Mixture Models [PDF]-http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf

### Paper: Adaptive background mixture models for real-time tracking [PDF]-http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf

### Video: https://www.youtube.com/watch?v=lLt9H6RFO6A

## Cluster validation
[基于密度的聚类验证](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/10.1.1.707.9034.pdf)
- External indices(有label的情况下）
<div align=center><img src=resources/19.png></div>
<div align=center><img src=resources/20.png></div>
- Internial indices(relative indices)
- silhouette coefficient-不能很好的评价环形（DBSCAN 不能使用SC，验证使用上面提到的基于密度的聚类验证）
<div align=center><img src=resources/21.png></div>
- Relative indices
<div align=center><img src=resources/18.png></div>

## 特征缩放
注意输入data需要是float
``` python
>>> from sklearn.preprocessing import MinMaxScaler
>>>
>>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
>>> scaler = MinMaxScaler()
>>> print(scaler.fit(data))
MinMaxScaler(copy=True, feature_range=(0, 1))
>>> print(scaler.data_max_)
[ 1. 18.]
>>> print(scaler.transform(data))
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [1.   1.  ]]
>>> print(scaler.transform([[2, 2]]))
[[1.5 0. ]]
```
## 哪些机器学习算法会受到特征缩放的影响？

- 决策树 无
- 使用 RBF 核函数的 SVM 大
- 线性回归 wx + b 无
- K-均值聚类 大

## [PCA](https://github.com/daxingxingqi/Nonsupervised_learning/blob/master/PCA%20Mini-Project-zh.ipynb)
**例子： 图像数据集，1800个图像10000个特征（100*100），做PCA时选择，min（1800，10000）。这时把1800个图像映射到前200个图像中（根据varience排序），最后通过机器学习算法计算这200个图像。

- 将输入特征转化为主成份
- 使用主成份作为新特征
- 主成份是数据方差最大的方向（损失信息最小，在压缩的时候）
- 方差越大，主成份的比重越大
- 主成份相互垂直

### 何时使用PCA
- 探查隐藏特征
- 减少维度
  - 查看高维度特征
  - 减少噪音
  - PCA 作为预处理

### 为什么PCA在人脸识别中有不错的应用呐？
- 人脸照片通常有很高的输入维度（很多像素）
- 人脸具有一些一般性形态，这些形态可以以较小维数的方式捕捉，比如人一般都有两只眼睛，眼睛基本都位于接近脸的顶部的位置
- 使用机器学习技术，人脸识别是非常容易的（因为人类可以轻易做到）

## 数据降维 
- random projection

**随机投影可以在高维度是更快的减少维度（不用像PCA先计算varience，节省资源），但是效果不如PCA。在选择特征数量时，可以有下面的公式自动选择。

<div align=center><img src=resources/22.png></div>

``` python
from sklearn import random_projection
rp = random_projection.SparseRandomProjection()#效果最好
new_X = rp.fit_trnsform(X)
```

- [Independent component analysis](https://github.com/daxingxingqi/Nonsupervised_learning/blob/master/Independent%20Component%20Analysis%20Lab-zh.ipynb)

**PCA旨在增大varience，ICA假设所有成份独立,并且数据不遵循高斯分布

具体例子参考，[独立成份分析](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/10.1.1.322.679.pdf)

``` python
from sklearn.decomposition import FastICA
#zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
#如果需要了解 Pyhton3 的应用，可以参考 Python3 zip()。
X = list(zip(signal_1, signal_2, signal_3))
ica = FastICA(n_components=3)
components = ica.fit_transform(X)
```
zip 
``` python
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 打包为元组的列表
[(1, 4), (2, 5), (3, 6)]
>>> zip(a,c)              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
>>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
[(1, 2, 3), (4, 5, 6)]
```

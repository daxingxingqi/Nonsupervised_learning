# Nonsupervised_learning

Hierarchical Clustering Example

### 1. [Using Hierarchical Clustering of Secreted Protein Families to Classify and Rank Candidate Effectors of Rust Fungi](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029847)



## K-means
下图是K-means对于不同数据集的表现。
<div align=center><img src=resources/1.png></div>

层次聚类和密度聚类的区别
<div align=center><img src=resources/2.png></div>

### Hierrchical clustering
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

``` 
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
#### DNSCAN [可视化](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)
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



# Nonsupervised_learning

Example

## 1. [Using Hierarchical Clustering of Secreted Protein Families to Classify and Rank Candidate Effectors of Rust Fungi](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029847)

## 2. [Traffic Classification Using Clustering Algorithms](https://pages.cpsc.ucalgary.ca/~mahanti/papers/clustering.pdf) [pdf]

## 3. [Anomaly detection in temperature data using dbscan algorithm](https://ieeexplore.ieee.org/abstract/document/5946052/)

K-means
下图是K-means对于不同数据集的表现。
<div align=center><img src=resources/1.png></div>

层次聚类和密度聚类的区别
<div align=center><img src=resources/2.png></div>

Hierrchical clustering
>single-link clustering 
- SLC测量的是簇之间最小的距离，然后对比簇与簇之间最小的距离，选择最小的聚合。
<div align=center><img src=resources/3.png></div>
- SLC有时候会导致狭长的类
<div align=center><img src=resources/4.png></div>
- 下图的SLC对于不同数据集的表现
<div align=center><img src=resources/5.png></div>

>complete-link clustering
- CLC测量的是簇之间最大的距离，然后对比簇与簇之间最大的距离，选择最小的聚合。
<div align=center><img src=resources/6.png></div>
- CLC有时会忽略其他的点
<div align=center><img src=resources/7.png></div>

>average-link clustering
- 计算所有距离的平均值
<div align=center><img src=resources/8.png></div>

>ward method clustering
- 计算所有距离的平均值到中间值的平方，加在一起然后减去内部点到内部中间值的平方
<div align=center><img src=resources/9.png></div>
```python
from sklearn.cluster import AgglomerativeClustering

# Hierarchical clustering
# Ward is the default linkage algorithm, so we'll start with that
ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(iris.data)
```

```python
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


Density clustering
>

DNSCAN

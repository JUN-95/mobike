import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import seaborn
from boto import sns
from sklearn import cluster, metrics
from sklearn.neighbors import NearestNeighbors

dataLoc = pd.read_csv("data/locDisDF.csv")
nbrs = NearestNeighbors(n_neighbors=200).fit(dataLoc)  # 对每个点距离最近的200个点分为一组
distances, indices = nbrs.kneighbors(dataLoc)  # 计算每个点最近的200个点的距离，并返回索引

# arr = dataLoc.values
# 画出每个点附近距离第200个点的距离
dist = distances[:, 199]
dist_ = np.sort(dist)
# outdis = len(dist_[dist_ > 0.01])  # 获取距离超过集群距离的值

# plt.plot(dist_)  # 出现拐点的标志是说偏离集群点到最远集群点的距离
# plt.show()

# eps: 一个样本的两个样本之间的最大距离应视为另一个样本的邻域。这不是群集中点的距离的最大界限。这是为数据集和距离函数适当选择的最重要的DBSCAN参数。
# min_samples: 一个点被视为核心点的邻域中的样本数量（或总重量）。这包括点本身

# DBSCAN算法将数据点分为三类：
# 核心点：在半径Eps内含有超过MinPts数目的点
# 边界点：在半径Eps内点的数量小于MinPts，但是落在核心点的邻域内
# 噪音点：既不是核心点也不是边界点的点
mdl_dbscan = cluster.DBSCAN(eps=0.0031, min_samples=200).fit(dataLoc)  # eps 看上图的拐点的大概值，可以查看模型的评分进行调整,min_samples=200并不是固定的
count = pd.Series(mdl_dbscan.labels_).value_counts()  # 展示有多少类，每类有多少个样本   70个样本类，即70个单车停放点
# plt.figure()
# plt.scatter(x='longitude', y='latitude', data=dataLoc, c=mdl_dbscan.labels_)
# plt.show()
# davies_bouldin_score：
# 分数定义为每个群集与其最相似群集的平均相似性度量，其中相似度是群集内距离与群集间距离之比。
# 因此，距离较远且分散程度较低的群集将获得更好的分数。最低分数为零，值越低表示聚类越好。
metrics.davies_bouldin_score(dataLoc, mdl_dbscan.labels_)  # 1.5311 模型评分

# 计算相邻的数据点的欧式距离
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)


# 如果一个点的 eps 邻域内的点的总数小于阈值，那么该点就是低密度点。如果大于阈值，就是高密度点
# 取领域的最大半径Eps = 0.， 阈值minPts = 200
E = 0.62
minPts = 200

arr = dataLoc.values

# 找出核心点
other_points = []
core_points = []
plotted_points = []
arrList = arr.tolist()
for point in arrList:
    point.append(0)  # assign initial level 0
    total = 0
    for otherPoint in arrList:
        distance1 = dist(otherPoint, point)
        if distance1 <= E:
            total += 1

    if total > minPts:
        core_points.append(point)
        plotted_points.append(point)
    else:
        other_points.append(point)

# 找出边界点
# borderPoints = np.nan


border_points = []
for core in core_points:
    for other in other_points:
        if dist(core, other) <= E:
            border_points.append(other)
            plotted_points.append(other)


print(border_points)

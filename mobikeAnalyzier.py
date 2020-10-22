import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, metrics
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('data/mobike_shanghai_sample_updated.csv')

tobestr = ['orderid', 'bikeid', 'userid']
data[tobestr] = data[tobestr].astype('str')
data['start_time'] = pd.to_datetime(data['start_time'])
data['end_time'] = pd.to_datetime(data['end_time'])
data.info()

data['duration'] = data.end_time - data.start_time
data['dur_day'] = data.duration.apply(lambda x: str(x).split(' ')[0])
data['dur_hr'] = data.duration.apply(lambda x: str(x).split(' ')[-1][:2])
data['dur_min'] = data.duration.apply(lambda x: str(x).split(':')[-2])
data['dur_sec'] = data.duration.apply(lambda x: str(x).split(':')[-1])
tobeint = ['dur_day', 'dur_hr', 'dur_min', 'dur_sec']
data[tobeint] = data[tobeint].astype('int')
data['ttl_min'] = data.dur_day * 24 * 60 + data.dur_hr * 60 + data.dur_min + data.dur_sec / 60

# 骑行时间单车数量分布散点
gbtime = data.groupby(by="ttl_min")
gbtimeList = []
for g in gbtime:
    gbtimeList.append((g[0], len(g[1])))
gbtimeListToDF = pd.DataFrame.from_records(gbtimeList, columns=["time", "timeNum"]).sort_values(by="time",
                                                                                                ascending=True)
gbtimeListToDF.to_csv("data/gbtimeListToDF.csv", index=False)

# datetime.datetime.isoweekday（）返回的1-7代表周一--周日；
data['dayId'] = data.start_time.apply(lambda x: x.isoweekday())
# dayType 工作日
data['dayType'] = data.dayId.apply(lambda x: 'weekends' if x == 6 or x == 7 else 'weekdays')

data['hourId'] = data.start_time.apply(lambda x: x.utctimetuple().tm_hour)
# rush hours：7-8，17-20 上班时间 其余时间为non-rush hours
data['hourType'] = data.hourId.apply(lambda x: 'rush hours' if (7 <= x <= 8) or (17 <= x <= 20) else 'non-rush hours')

# 一天24小时骑行频率分布图

data["hourId"] = data["hourId"].apply(lambda x: x + 1)
hour_group = data.groupby("hourId")
hour_num_df = hour_group.agg({"orderid": "count"}).reset_index()  # 计算分组后的单车数
# print(hour_num_df)
hour_num_df.to_csv("data/hour_num_df.csv", index=None)

# 骑行距离数量统计
# 按每条记录的起点位置，作为发起订单所处位置的数据依据
from math import radians, cos, sin, asin, sqrt


# 自定义函数，通过两点的经纬度计算两点之间的直线距离
def geodistance(lng1, lat1, lng2, lat2):
    lng1_r, lat1_r, lng2_r, lat2_r = map(radians, [lng1, lat1, lng2, lat2])  # 经纬度转换成弧度
    dlon = lng1_r - lng2_r
    dlat = lat1_r - lat2_r
    dis = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(dis)) * 6371 * 1000  # 地球平均半径为6371km
    distance = round(distance / 1000, 3)
    return distance


data['distance'] = np.nan
data['disToCenter'] = np.nan


# 自定义函数，通过调用geodistance获取每条记录骑行始末点和起点距中心点的直线距离
def get_dis(item):
    # distance：点和起点距中心点的直线距离
    item['distance'] = geodistance(item['start_location_x'], item['start_location_y'],
                                   item['end_location_x'], item['end_location_y'])  # 计算骑行始末点经纬度的直线距离
    # 国际饭店一般被认为是上海地理中心坐标点，计算骑行起始点经纬度和国际饭店经纬度的直线距离
    # disToCenter：国际饭店的距离
    item['disToCenter'] = geodistance(item['start_location_x'], item['start_location_y'], 121.471632, 31.233705)
    return item

data1 = data.apply(get_dis, axis=1)

# 骑行距离分组数量统计
gbDis = data1.groupby(by="distance")
gbDisList = []
for g in gbDis:
    gbDisList.append((g[0], len(g[1])))
gbDisListToDF = pd.DataFrame.from_records(gbDisList, columns=["distance", "distanceNum"]).sort_values(by="distanceNum",
                                                                                                      ascending=True)
gbDisListToDF.to_csv("data/gbDisListToDF.csv", index=False)

# data.isna().sum()

# data["orderid"].unique().size


# ========================================================================================================================

'''
    1、knn分类获取拐点
    2、kmeans通过与均值算距离在拐点的之间的范围进行筛选出空闲单车
'''

# 确定空闲单车位置
locList = []  # 经纬度
for l in zip(data["end_location_x"], data["end_location_y"]):
    locList.append((l[0], l[1]))

locDisDF = pd.DataFrame(locList, columns=["longitude", "latitude"])
locDisDF.to_csv("data/locDisDF.csv", index=False)

dataLoc = pd.read_csv("data/locDisDF.csv")
nbrs = NearestNeighbors(n_neighbors=200).fit(dataLoc)  # 对每个点距离最近的200个点分为一组
distances, indices = nbrs.kneighbors(dataLoc)  # 计算每个点最近的200个点的距离，并返回索引

dataLocList = np.array(dataLoc).tolist()

# arr = dataLoc.values
# 画出每个点附近距离第200个点的距离

dist = distances[:, 199]
dist_ = np.sort(dist)

plt.plot(dist_)  # 出现拐点的标志是说偏离集群点到最远集群点的距离
plt.show()


# 计算相邻的数据点的欧式距离
def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)


from sklearn.cluster import KMeans

mdl_kmeans = KMeans(n_clusters=70).fit(dataLoc)

count2 = pd.Series(mdl_kmeans.labels_).value_counts()

km_centor = pd.DataFrame(mdl_kmeans.cluster_centers_, columns=['longitude', 'latitude'])  # 获取70个单车停放点的经纬度

km_centorList = np.array(km_centor).tolist()

# 找出边界点
# borderPoints = np.nan


border_points = []
for core in km_centorList:
    for other in dataLocList:
        if 0.0029 <= dist(core, other) <= 0.0031:
            border_points.append(other)

# 去重
temp = []
for item in border_points:
    if not item in temp:
        temp.append(item)

border_pointsDF = pd.DataFrame(temp, columns=["longitude", "latitude"])
border_pointsDF.to_csv("data/borderPointsDF.csv", index=False)

# res_border_points = list(set(border_points))
# =====================================================================================



# DBSCAN算法将数据点分为三类：
# 核心点：在半径Eps内含有超过MinPts数目的点
# 边界点：在半径Eps内点的数量小于MinPts，但是落在核心点的邻域内
# 噪音点：既不是核心点也不是边界点的点

# eps:两个样本之间的最大距离，即扫描半径
#  min_samples ：作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身)。
mdl_dbscan = cluster.DBSCAN(eps=0.0031, min_samples=200).fit(
    dataLoc)  # eps 看上图的拐点的大概值，可以查看模型的评分进行调整,min_samples=200并不是固定的
# davies_bouldin_score：
# 分数定义为每个群集与其最相似群集的平均相似性度量，其中相似度是群集内距离与群集间距离之比。
# 因此，距离较远且分散程度较低的群集将获得更好的分数。最低分数为零，值越低表示聚类越好。
metrics.davies_bouldin_score(dataLoc, mdl_dbscan.labels_)  # 1.5311 模型评分

plt.figure()
plt.scatter(x='longitude', y='latitude', data=dataLoc, c=mdl_dbscan.labels_)
plt.show()
count = pd.Series(mdl_dbscan.labels_).value_counts()  # 展示有多少类，每类有多少个样本   70个样本类，即70个单车停放点




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

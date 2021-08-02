#This code helps clustering the buildings. Using the elbow method 2 possible clusters
#are considered: 3 groups and 5 groups

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('D:/ThesisExperiments/OriginalData/Geometry/Geomdata.csv',sep=";")

X=data[data.columns[15:21]]
metric = []
for i in range(1, 40):
    clusters = KMeans(n_clusters=i)
    clusters.fit(X)
    metric.append(clusters.inertia_)

plt.plot(range(1, 11), metric, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('metric')
plt.show()
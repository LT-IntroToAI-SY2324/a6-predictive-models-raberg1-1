import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#imports the data
data = pd.read_csv("part5-unsupervised-learning/customer_data.csv") 
x = data[["Annual Income", "Age"]].values
y = data["Spending Score"].values

#standardize the data
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)
#the value of k has been defined for you
k = 5

#apply the kmeans algorithm
km = KMeans(n_clusters=k).fit(x_scaled, y)

#get the centroid and label values
centroids = km.cluster_centers_
labels = km.labels_

# print(centroids)
# print(labels)
#sets the size of the graph
plt.figure(figsize=(5,4))

#use a for loop to plot the data points in each cluster


#plot the centroids

            
#shows the graph
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
# plt.show()
from KMeans import KMeans
from sklearn.datasets import make_blobs
centers=[[-1,1],[2,-2],[-2,-3]]
x,y = make_blobs(n_samples=100, centers=centers, cluster_std=0.60,random_state=0)

model = KMeans(x,3,resolustion=100)
print(model.meanVectors)
print(model.train())
model.select_k()

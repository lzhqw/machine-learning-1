import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

class KMeans:
    def __init__(self, x, k, resolustion=1000, plot=True):
        self.x = np.array(x)
        self.k = k
        self.m = len(x)
        self.meanVectors = self.x[np.random.choice(self.m,k,replace=False)]
        self.clusters = [[]for i in range(self.k)]
        self.resolustion = resolustion
        self.plot = plot

    def train(self):
        cnt = 0
        while True:
            # 初始化所有簇
            self.clusters = [[]for i in range(self.k)]
            # 计算最近簇
            for i in range(self.m):
                d = float('inf')
                index = 0
                for j in range(self.k):
                    d_ij = norm(self.x[i]-self.meanVectors[j])
                    if d_ij < d:
                        d = d_ij
                        index = j
                self.clusters[index].append(self.x[i])
            # 更新均值向量
            done = True
            for i in range(self.k):
                newMeanVector = np.average(np.array(self.clusters[i]),axis=0)
                if not np.array_equal(newMeanVector,self.meanVectors[i]):
                    self.meanVectors[i] = newMeanVector
                    done = False
            print(self.meanVectors)
            if self.plot:
                self.plot_decision_boundaries(iteration=cnt)
            cnt += 1
            if done:
                return self.meanVectors
    def predict(self, x):
        d = float('inf')
        index = 0
        for i in range(self.k):
            d_ij = norm(x-self.meanVectors[i])
            if d_ij < d:
                d = d_ij
                index = i
        return index


    def plot_decision_boundaries(self, resolustion=1000, iteration=0):
        # 取这个用来画网格的
        mins = self.x.min(axis=0) - 0.1
        maxs = self.x.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolustion),
                             np.linspace(mins[1], maxs[1], resolustion))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predict = []
        for i in tqdm(grid):
            predict.append(self.predict(i))
        predict = np.array(predict)
        predict = predict.reshape(xx.shape)
        print(predict)

        plt.contourf(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap='Pastel2')

        plt.contour(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')

        plt.scatter(self.x[:,0],self.x[:,1])
        plt.scatter(self.meanVectors[:,0],self.meanVectors[:,1],s=20,c='r')
        # plt.show()
        plt.savefig('iteration_{}.png'.format(iteration))
        plt.close()
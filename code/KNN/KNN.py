from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class KNN():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]

    def predict(self, x):
        dists = []
        for i in range(self.m):
            dists.append([self.dist(x, self.X[i, :]), self.y[i]])
        dists = np.array(dists)
        idx = np.argsort(dists[:, 0])
        points = dists[idx[:self.k]]
        counts = np.bincount(np.array(points[:, 1], dtype=np.int32))
        counts = np.argmax(counts)
        return counts

    def dist(self, x, y):
        return norm(x - y, ord=2)

    def plot_decision_boundaries(self, resolustion=1000):
        '''
        绘制决策边界
        :param resolustion: 网格密度
        :param iteration: 迭代次数
        :return:
        '''
        mins = self.X.min(axis=0) - 0.1
        maxs = self.X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolustion),
                             np.linspace(mins[1], maxs[1], resolustion))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predict = []
        for i in tqdm(grid):
            predict.append(self.predict(i))
        predict = np.array(predict)
        predict = predict.reshape(xx.shape)
        # print(predict)

        plt.contourf(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap='Pastel2')

        plt.contour(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')

        plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.savefig('KNN.png')
        plt.close()

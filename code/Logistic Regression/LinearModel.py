import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
class LogisticRegression:
    def __init__(self, X, y, lr):
        '''
        :param X: shape ——> (m,n) m个数据 n个参数
        :param y: shape ——> m
        '''
        self.y = y
        self.m = np.shape(X)[0]
        self.n = np.shape(X)[1] + 1
        self.X = np.append(X, np.ones(self.m).reshape(self.m,1), axis=1)
        self.w = np.zeros(self.n)
        self.lr = lr
        self.history = []

    def loss(self):
        return np.sum(np.log(1+np.exp(-self.y * self.X.dot(self.w))))/self.m

    def train(self):
        while True:
            gd = np.zeros(self.n)
            for i in range(self.m):
                gd += self.sigmoid(-self.y[i]*self.X[i].dot(self.w))*(-self.y[i]*self.X[i])
            gd = gd / self.m
            self.w -= self.lr*gd
            loss = self.loss()
            print(loss,gd)
            self.history.append(loss)
            print('# ---------------------------------------------------- #')
            if norm(gd, 2) < 1e-3:
                print(self.w)
                print(self.sigmoid(self.X.dot(self.w)))
                break
        return self.w
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def plot_decision_boundaries(self, resolustion=1000):

        # 取这个用来画网格的
        mins = self.X.min(axis=0) - 0.1
        maxs = self.X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolustion),
                             np.linspace(mins[1], maxs[1], resolustion))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predict = []
        for i in grid:
            predict.append(self.predict(i))
        predict = np.array(predict)
        predict = predict.reshape(xx.shape)
        # print(predict[:,0])
        plt.contourf(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap='Pastel2')

        plt.contour(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')
        plt.scatter(self.X[:50, 0], self.X[:50, 1], c='b')
        plt.scatter(self.X[50:, 0],self.X[50:, 1], c='r')
        plt.show()

    def predict(self, x):
        return np.sign(self.sigmoid(np.hstack([x, 1]).dot(self.w)) - 0.5)

    def acc(self, x,y):
        cnt = 0
        for i in range(x.shape[0]):
            predict = self.predict(x[i,:])
            if predict == y[i]:
                cnt+=1
        return cnt/x.shape[0]



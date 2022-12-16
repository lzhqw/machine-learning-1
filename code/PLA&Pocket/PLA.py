import numpy as np
import matplotlib.pyplot as plt

class PLA:
    def __init__(self, X, y):
        '''
        :param X: shape ——> (m,n) m个数据 n个参数
        :param y: shape ——> m
        '''
        self.X = X
        self.y = y
        self.m = np.shape(X)[0]
        self.n = np.shape(X)[1]
        self.w = np.zeros(self.n)
        self.b = 0

    def loss(self):
        '''
        以分类错误的点的个数作为loss
        :return:
        '''
        res = self.y*(np.sign(self.X.dot(self.w) + self.b))
        mistake = np.where(res <= 0)[0]
        loss = len(mistake)/self.m
        return loss

    def dloss(self):
        '''
        以分类错误的点到超平面的距离作为loss
        :return:
        '''
        res = self.y * (np.sign(self.X.dot(self.w) + self.b))
        mistake = np.where(res <= 0)[0]
        temp = self.X.dot(self.w)+self.b
        for i in mistake:
            temp[i] = self.y[i]*temp[i]
        loss = -np.sum(temp[mistake])
        loss = loss/(len(mistake)+1e-7)
        return loss

    def train(self):
        while self.loss() > 0:
            res = self.y * (np.sign(self.X.dot(self.w) + self.b))
            mistake = np.where(res <= 0)[0]
            temp = self.y[mistake[0]] * self.X[mistake[0]]
            self.w += temp
            self.b += self.y[mistake[0]]
            print(f'loss1:{self.loss()}, loss:2{self.dloss()}, w:{self.w}, b:{self.b}')
            # self.plot_decision_boundaries()
    def plot_decision_boundaries(self, resolustion=1000):

        # 取这个用来画网格的
        mins = self.X.min(axis=0) - 0.1
        maxs = self.X.max(axis=0) + 0.1
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolustion),
                             np.linspace(mins[1], maxs[1], resolustion))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predict = []
        for i in grid:
            predict.append(np.sign(i.dot(self.w) + self.b))
        predict = np.array(predict)
        predict = predict.reshape(xx.shape)
        plt.contourf(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap='Pastel2')

        plt.contour(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')
        plt.scatter(self.X[:50, 0], self.X[:50, 1], c='b')
        plt.scatter(self.X[50:, 0],self.X[50:, 1], c='r')
        plt.show()
class Pocket:
    def __init__(self, X, y):
        '''
        :param X: shape ——> (m,n) m个数据 n个参数
        :param y: shape ——> m
        '''
        self.X = X
        self.y = y
        self.m = np.shape(X)[0]
        self.n = np.shape(X)[1]
        self.w = np.zeros(self.n)
        self.b = 0
        self.best_loss = float('inf')
        self.step = 0

    def loss(self):
        res = self.y*(np.sign(self.X.dot(self.w) + self.b))
        mistake = np.where(res <= 0)[0]
        loss = len(mistake)/self.m
        return loss

    def train(self):
        while self.loss() > 0:
            res = self.y * (np.sign(self.X.dot(self.w) + self.b))
            mistake = np.where(res <= 0)[0]
            temp = self.y[mistake[0]] * self.X[mistake[0]]
            self.w += temp
            self.b += self.y[mistake[0]]
            curr_loss = self.loss()
            self.step += 1
            if curr_loss < self.best_loss:
                self.best_loss = curr_loss
                print(self.loss(), self.w, self.b)
            if self.step > 1000:
                break



import numpy as np
import matplotlib.pyplot as plt


class line_model():
    """
    基学习器，以垂直于某条坐标轴的超平面作为分类面
    对于数据中所有点在该坐标轴的值作为决策节点，对于每个决策节点有两种情况
    >=value的为正类或者 <value的为正类
    基学习器训练的时候接受权重参数，用于训练在D_t分布下的最优基学习器
    基学习器学得3个参数：dim(在哪个维度)，value(用什么值作为决策值)，sign(符号是>=还是<)
    同时会返回最小的误分类误差(0最优，1最差，理论上二分类问题应当小于0.5)
    接受的y应当为1和-1
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = X.shape[0]
        self.n = X.shape[1]

    def train(self, w):
        res = []
        for i in range(self.n):
            for j in set(self.X[:, i]):
                X1 = self.X[:, i].copy()
                X1[self.X[:, i] >= j] = 1
                X1[self.X[:, i] < j] = -1
                X2 = self.X[:, i].copy()
                X2[self.X[:, i] < j] = 1
                X2[self.X[:, i] >= j] = -1

                eps1 = np.sum(w[np.where(X1 != self.y)])
                eps2 = np.sum(w[np.where(X2 != self.y)])

                if eps1 <= eps2:
                    res.append([eps1, i, j, 0])
                else:
                    res.append([eps2, i, j, 1])
        res = np.array(res)
        # print(res)
        res = res[np.argmin(res[:, 0])]
        self.res = res
        return res[0]

    def predict(self, x):
        """
        接受两种模式，一种是预测单个值，用于绘制决策边界
        一种是批量预测， 更新D_t时比较方便
        """
        dim = int(self.res[1])
        value = self.res[2]
        sign = self.res[3]
        if x.ndim == 1:
            if sign == 0:
                if x[dim] >= value:
                    return 1
                return -1
            if sign == 1:
                if x[dim] < value:
                    return 1
                return -1
        elif x.ndim == 2:
            if sign == 0:
                predict = np.zeros(self.m)
                predict[x[:, dim] >= value] = 1
                predict[x[:, dim] < value] = -1
            if sign == 1:
                predict = np.zeros(self.m)
                predict[x[:, dim] < value] = 1
                predict[x[:, dim] >= value] = -1
            return predict


class adaboost():
    """
    adaboost
    input : X, y, 训练轮数T, 基学习器base_model（应当为一个类，没有实例化）
    """

    def __init__(self, X, y, T, base_model):
        """
        G -- 基学习器
        w -- 数据集的参数权重，初始时为1/m
        alpha -- alpha_t的列表，表示每个基学习器的权重
        G_list -- 基学习器的列表，存放每个训练好的基学习器
        """
        self.X = X
        self.y = y
        self.T = T
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.G = base_model
        self.w = np.ones(self.m) / self.m
        self.alpha = []
        self.G_list = []

    def train(self):
        for i in range(self.T):
            # 训练
            G = self.G(self.X, self.y)
            self.G_list.append(G)
            epsilon_t = G.train(self.w)
            print(epsilon_t)
            # 计算alpha
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            self.alpha.append(alpha_t)
            # 计算Z_t(用于归一化)和更新w
            predict = G.predict(self.X)
            Z_t = np.sum(self.w * np.exp(-alpha_t * self.y * predict))
            self.w = self.w * np.exp(-alpha_t * self.y * predict) / Z_t

    def predict(self, x):
        """
        预测 加权投票，大于等于0为正类，小于0为负类
        """
        res = []
        for G in self.G_list:
            y = G.predict(x)
            res.append(y)
        res = np.array(res) * np.array(self.alpha)
        if np.sum(res) >= 0:
            return 1
        else:
            return -1

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
        plt.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], c='b')
        plt.scatter(self.X[self.y == -1, 0], self.X[self.y == -1, 1], c='r')
        plt.show()

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

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
        '''
        KMeans 迭代训练
        :return: 均值向量
        '''
        cnt = 0
        while True:
            # ------------------------------------------- #
            # 初始化所有簇
            # ------------------------------------------- #
            self.clusters = [[]for i in range(self.k)]
            # ------------------------------------------- #
            # 计算最近簇
            # ------------------------------------------- #
            for i in range(self.m):
                d = float('inf')
                index = 0
                for j in range(self.k):
                    d_ij = norm(self.x[i]-self.meanVectors[j])
                    if d_ij < d:
                        d = d_ij
                        index = j
                self.clusters[index].append(self.x[i])
            # ------------------------------------------- #
            # 更新均值向量
            # ------------------------------------------- #
            done = True
            for i in range(self.k):
                if len(self.clusters[i])!=0:
                    newMeanVector = np.average(np.array(self.clusters[i]),axis=0)
                else:
                    newMeanVector = self.meanVectors[i]
                if not np.array_equal(newMeanVector,self.meanVectors[i]):
                    self.meanVectors[i] = newMeanVector
                    done = False
            # print('# ------------------------------------ #')
            # print(self.meanVectors)
            # ------------------------------------------- #
            # 绘制决策边界
            # ------------------------------------------- #
            if self.plot:
                self.plot_decision_boundaries(iteration=cnt,resolustion=self.resolustion)
            cnt += 1
            # ------------------------------------------- #
            # 判断结束条件 退出循环
            # ------------------------------------------- #
            if done:
                return self.meanVectors

    def predict(self, x):
        '''
        预测
        :param x: 某个给定向量
        :return: 距离x最近的均指向量的编号
        '''
        d = float('inf')
        index = 0
        for i in range(self.k):
            d_ij = norm(x-self.meanVectors[i])
            if d_ij < d:
                d = d_ij
                index = i
        return index

    def select_k(self, start=1, end=20, plot=False):
        '''
        手肘法确定k的取值
        :return:
        '''
        all_distance = []
        for k in range(start,end+1):
            self.__init__(self.x, k, plot=plot)
            self.train()
            all_distance.append(self.sumDistance())
        plt.plot(range(1,len(all_distance)+1),all_distance)
        plt.ylabel('sum of distance')
        plt.xlabel('K')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()


    def sumDistance(self):
        '''
        计算点到聚点的距离之和
        :return:
        '''
        sumdistance = 0
        for i in range(self.m):
            d = float('inf')
            for j in range(self.k):
                d_ij = norm(self.x[i]-self.meanVectors[j])
                if d_ij<d:
                    d = d_ij
            sumdistance += d
        return sumdistance

    def plot_decision_boundaries(self, resolustion=1000, iteration=0):
        '''
        绘制决策边界
        :param resolustion: 网格密度
        :param iteration: 迭代次数
        :return:
        '''
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
        # print(predict)

        plt.contourf(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                     cmap='Pastel2')

        plt.contour(predict, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                    linewidths=1, colors='k')

        plt.scatter(self.x[:,0],self.x[:,1])
        plt.scatter(self.meanVectors[:,0],self.meanVectors[:,1],s=20,c='r')
        # plt.show()
        plt.savefig('kmeans_{}.png'.format(iteration))
        plt.close()

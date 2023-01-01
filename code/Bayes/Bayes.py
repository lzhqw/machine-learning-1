import numpy as np


class NaiveBayes:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X = X
        self.y = y
        self.y_count = self.count_y()
        self.num_classes = len(self.y_count)
        self.X_count = np.ones(shape=(self.X.shape[1], self.num_classes))
        self.count_X()

    def predict(self, x):
        idxs = np.where(x > 0)[0]
        p = []
        for i in range(self.num_classes):
            pi = 1
            for idx in idxs:
                pi *= self.X_count[idx, i]
            p.append(pi)
        return np.argmax(p)

    def count_y(self):
        count = [0 for i in range(np.max(self.y) + 1)]
        for i in range(self.X.shape[0]):
            count[self.y[i]] += np.sum(self.X[i, :])
        count = np.array(count)
        count = count + self.X.shape[1]
        return count

    def count_X(self):
        for i in range(self.X.shape[0]):
            self.X_count[:, self.y[i]] += self.X[i, :]
        for i in range(self.X_count.shape[1]):
            self.X_count[:, i] = self.X_count[:, i] / self.y_count[i]

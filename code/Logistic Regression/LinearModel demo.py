import pandas as pd
import numpy as np
from LinearModel import LogisticRegression
import matplotlib.pyplot as plt
path = r'iris.data'
iris = pd.read_csv(path,header=None)
iris.loc[:50, 4] = -1
iris.loc[50:, 4] = 1

iris = iris.to_numpy(dtype=np.float64)
X = iris[:, :2]
y = iris[:, 4]

lg = LogisticRegression(X,y,lr=10)
lg.train()
# 计算精度
acc = lg.acc(X,y)
# 绘制决策边界
lg.plot_decision_boundaries()
# 绘制loss
plt.plot(range(len(lg.history)),lg.history)
plt.show()
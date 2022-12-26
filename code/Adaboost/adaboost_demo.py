import pandas as pd
import numpy as np
from adaboost import adaboost, line_model
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_moons(n_samples=200, random_state=42, noise=0.02)
y[y == 0] = -1
print(y)
# plt.scatter(X[:,0],X[:,1])
# plt.show()

ada = adaboost(X, y, 100, line_model)
ada.train()
ada.plot_decision_boundaries(resolustion=100)

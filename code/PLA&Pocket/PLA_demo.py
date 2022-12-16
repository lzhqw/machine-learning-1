from PLA import PLA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
path = r'iris.data'
iris = pd.read_csv(path,header=None)
iris.loc[:50, 4] = -1
iris.loc[50:, 4] = 1

iris = iris.to_numpy(dtype=np.float64)
X = iris[:, 1:3]
y = iris[:, 4]

pla = PLA(X,y)
pla.train()
pla.plot_decision_boundaries()
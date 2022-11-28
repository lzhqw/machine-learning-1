import pandas as pd
import numpy as np
import graphviz as gz
from DecisionTree import Decision_Tree

data = pd.read_csv('watermalen.txt')
print(data)
data = data.loc[:,['编号','脐部', '色泽', '根蒂', '敲声', '纹理', '触感','好瓜']]
print(data)
attrs = list(data.columns)[1:-1]
data = data.iloc[:, 1:].to_numpy()
dt = Decision_Tree(data[[0,1,2,5,6,9,13,14,15,16],:],
                   data[[3,4,7,8,10,11,12],:], attrs,plot=True)

node = dt.train(prune='pre')
dt.predict(['乌黑', '稍蜷', '浊响', '清晰', '凹陷', '硬滑'])

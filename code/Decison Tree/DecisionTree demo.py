import pandas as pd
import numpy as np
import graphviz as gz
from DecisionTree import Decision_Tree

data = pd.read_csv('watermalen.txt')
print(data)
attrs = list(data.columns)[1:-1]
data = data.iloc[:, 1:].to_numpy()
print(data)
dt = Decision_Tree(data, attrs)
node = dt.train(data, attrs)
# graph = gz.Graph()
# draw_DT(graph,node,0)
# graph.view()
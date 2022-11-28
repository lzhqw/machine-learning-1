import pandas as pd
import numpy as np
import graphviz as gz


class Node:
    def __init__(self, class_=None, type=None, attr=None, attr_name=None):
        self.children = []          # 子节点
        self.class_ = class_        # 叶节点类别
        self.type = type            # 节点类型
        self.attr = attr            # 父节点到该节点的属性
        self.class_dict = {}        # 节点类别字典
        self.attrName = attr_name
    def copy(self, node):
        node.children = self.children
        node.class_ = self.class_
        node.type = self.type
        node.attr = self.attr
        node.class_dict = self.class_dict
        node.attrName = self.attrName
    def print(self):
        print(self.class_,self.class_dict,self.attrName,self.type,self.children)


class Decision_Tree:
    def __init__(self, data, val, A, plot=True):
        self.data = data
        self.val = val
        self.attr_num = {}
        self.plot = plot
        self.node = None
        for i in range(len(A)):
            self.attr_num[A[i]]=i

    def train(self, prune = None):
        if prune is None:
            self.node = self.TreeGenerate(self.data,list(self.attr_num.keys()),
                                          attr=None,root=True)
        elif prune == 'post':
            self.node = self.TreeGenerate(self.data,list(self.attr_num.keys()),
                                          attr=None,root=True)
            self.postPruning()
        elif prune == 'pre':
            self.node = self.TreeGenerate(self.data,list(self.attr_num.keys()),
                                          attr=None,root=True,pre=True)

        if self.plot:
            graph = gz.Graph()
            self.draw_DT(graph, self.node, 0)
            graph.view()

    def predict_node(self, node, x):
        attrName = node.attrName
        if len(node.children)==0:
            return node.class_
        for child in node.children:
            if child.attr == x[self.attr_num[attrName]]:
                class_ = self.predict_node(child,x)
                return class_
    def predict(self, x):
        class_ = self.predict_node(self.node, x)
        return class_

    def accuracy(self):
        cnt = 0
        for i in range(len(self.val)):
            if self.predict(self.val[i,:-1]) == self.val[i,-1]:
                cnt+=1
        return cnt/len(self.val)

    def postPruning(self):
        divideList = self.getDivideNode(self.node,0)
        print(divideList)
        while len(divideList)>0:
            depth = 0
            index = 0
            for i in range(len(divideList)):
                if divideList[i][1]>depth:
                    depth = divideList[i][1]
                    index = i
            acc1 = self.accuracy()
            node = divideList[index][0]
            node_copy = Node()
            node.copy(node_copy)
            # ---------------------------------------------------- #
            # 剪枝
            # ---------------------------------------------------- #
            node.type = 'leaf'
            class_dict = node.class_dict
            class_ = max(class_dict, key=class_dict.get)
            node.class_ = class_
            node.children = []
            # ---------------------------------------------------- #
            # 比较剪枝前后的准确率,如果效果没有变好，则恢复
            # ---------------------------------------------------- #
            acc2 = self.accuracy()
            if acc2 <= acc1:
                node_copy.copy(node)
                print(f'acc:{acc1},不剪枝,节点信息：字段 {node.attr},深度 {divideList[index][1]}')
            else:
                print(f'acc:{acc2},剪枝,节点信息：字段 {node.attr},深度 {divideList[index][1]}')
            divideList.pop(index)
        # graph = gz.Graph()
        # self.draw_DT(graph,self.node,0)
        # graph.view()
        return self.node


    def getDivideNode(self, node, depth):
        devideNodeList = []
        if len(node.children)>0:
            for child in node.children:
                devideNodeList.extend(self.getDivideNode(child, depth+1))
        if node.type == 'divide':
            devideNodeList.append((node,depth))
            return devideNodeList
        return devideNodeList

    def TreeGenerate(self, D, A, attr, root, pre=False):
        node = Node()
        if root:
            self.node = node
        node.attr = attr
        # ---------------------------------------------------- #
        # 判断是否都是同一个类，如果都是同一个类，分类完成，return
        # ---------------------------------------------------- #
        if self.isSame(D):
            node.class_ = D[0][-1]
            node.type = 'leaf'
            return node
        # ---------------------------------------------------- #
        # 判断是否可以再分，如果不能再分，分类完成，return
        # ---------------------------------------------------- #
        if len(A)==0 or self.sameValue(D):
            node.type = 'leaf'
            count = self.countClass(D)
            class_ = max(count, key=count.get)
            node.class_ = class_
            return node
        # ---------------------------------------------------- #
        # 计算类别的分布 用于后剪枝的时候判断divide节点
        # 变成叶节点的时候应该归属到哪一个类
        # ---------------------------------------------------- #
        node.class_dict = self.countClass(D)
        # ---------------------------------------------------- #
        # 计算样本D上的信息熵
        # ---------------------------------------------------- #
        ifd = self.IFD(D)
        # ---------------------------------------------------- #
        # 挑选信息增益最大的属性a*
        # ---------------------------------------------------- #
        best_attr = ['',0]
        for attr in A:
            gain = ifd - self.CE(D,self.attr_num[attr])
            if gain > best_attr[1]:
                best_attr = [attr,gain]
        # ---------------------------------------------------- #
        # 统计D中a*每个类别的数量
        # 注：a*的类别应该从整个数据集中统计，而非D中，但是数量要在D中统计
        # ---------------------------------------------------- #
        attr_dict = self.countAttr(D,self.attr_num[best_attr[0]])
        print(attr_dict)
        # ---------------------------------------------------- #
        # 设置node的属性（分类节点，按照什么属性分类）
        # ---------------------------------------------------- #
        node.type = 'divide'
        node.attrName = best_attr[0]
        # ---------------------------------------------------- #
        # 对于分类属性的每一个类，判断是否为空
        # 如果为空则新分支标记为D中最多的类，否则重复上述步骤
        # ---------------------------------------------------- #
        ifContinue = True
        if pre:
            ifContinue = self.prePruning(node,attr_dict,best_attr,D)
        if ifContinue:
            for attr in attr_dict.keys():
                if attr_dict[attr] == 0:
                    count = self.countClass(D)
                    class_ = max(count, key=count.get)
                    node.children.append(Node(class_=class_,type='leaf',attr=attr))
                else:
                    Dv = D[np.where(D[:,self.attr_num[best_attr[0]]]==attr),:][0]
                    A2 = A.copy()
                    A2.remove(best_attr[0])
                    print(best_attr[0],attr)
                    print(A2)
                    print('# ---------------------------------------------------- #')
                    node.children.append(self.TreeGenerate(Dv,A2,attr,False))
        return node

    def prePruning(self, node, attr_dict, best_attr, D):
        if node.class_ is None:
            node.class_ = max(node.class_dict, key=node.class_dict.get)
        acc1 = self.accuracy()
        for attr in attr_dict.keys():
            if attr_dict[attr] == 0:
                count = self.countClass(D)
                class_ = max(count, key=count.get)
                node.children.append(Node(class_=class_,type='leaf',attr=attr))
            else:
                Dv = D[np.where(D[:,self.attr_num[best_attr[0]]]==attr),:][0]
                class_dict = self.countClass(Dv)
                class_ = max(class_dict, key=class_dict.get)
                child = Node(type='leaf',class_=class_)
                node.children.append(child)
        acc2 = self.accuracy()
        node.children = []
        if acc2>acc1:
            return True
        else:
            return False

    def isSame(self, D):
        '''
        判断D中的数据是否是同一个类别
        :param D: 数据集的一个子集
        :return: bool
        '''
        class_ = D[0][-1]
        for i in range(len(D)):
            if D[i][-1] != class_:
                return False
        return True

    def countClass(self, D):
        '''
        统计D中每个类别的个数
        :param D: 数据集的一个子集
        :return: dict{class:cnt}
        '''
        class_dict = {}
        for i in range(len(D)):
            class_ = D[i][-1]
            class_dict.setdefault(class_,0)
            class_dict[class_]+=1
        return class_dict

    def sameValue(self, D):
        '''
        判断D中是否所有的样本都相同
        :param D: 数据集的一个子集
        :return: bool
        '''
        v = D[0]
        for i in range(D.shape[0]):
            if (v != D[i]).any():
                return False
        return True

    def countAttr(self, D, attr_idx):
        '''
        统计D中指定分类属性的各个属性的个数
        :param D: 数据集的一个子集
        :param attr_idx: 属性编号，输入时生成
        :return: dict{attr:cnt}
        '''
        attr_dict = {}
        for i in range(len(self.data)):
            attr = self.data[i][attr_idx]
            attr_dict.setdefault(attr,0)
        for i in range(len(D)):
            attr = D[i][attr_idx]
            # attr_dict.setdefault(attr, 0)
            attr_dict[attr] += 1
        return attr_dict

    def IF(self, x):
        '''
        计算信息熵
        :param x: ndarray
        :return: 信息熵
        '''
        x[np.where(x==0)] = 1
        return -np.sum(x*np.log2(x))

    def CE(self, D, attr_idx):
        '''
        计算每个属性的信息增益
        :param D: 数据集的一个子集
        :param attr_idx: 属性编号，输入时生成
        :return: 信息增益
        '''
        attr_dict = self.countAttr(D,attr_idx)
        EF = 0
        for attr in attr_dict.keys():
            Dv = D[np.where(D[:,attr_idx]==attr),:][0]
            ifd = self.IFD(Dv)
            EF += len(Dv)/len(D)*ifd
        return EF
    def IFD(self, D):
        '''
        计算D上的信息熵
        :param D: 数据集的一个子集
        :return: 信息熵
        '''
        class_dict = self.countClass(D)
        class_dict = np.array(list(class_dict.values()))
        class_dict = class_dict / np.sum(class_dict)
        return self.IF(class_dict)

    def draw_DT(self, graph, node, nodeid):
        '''
            通过graphviz进行决策树可视化（递归）
            :param graph: 图
            :param node: 上一级节点（决策树的节点）
            :param nodeid: 上一级节点编号（graph的节点编号）
            :return: nodeid
            '''
        if len(node.children)==0:
            graph.node(str(nodeid), node.class_, fontname='SimSun')
        else:
            graph.node(str(nodeid), node.attrName + '=?', fontname='SimSun')
        curr_nodeid = nodeid
        # print(node.class_,nodeid)
        if node.children:
            for child in node.children:
                child_id = nodeid + 1
                nodeid = self.draw_DT(graph, child, child_id)
                graph.edge(str(curr_nodeid), str(child_id), label=child.attr, fontname='FangSong')
        return nodeid

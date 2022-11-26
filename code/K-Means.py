def load_data():
    Watermalen_data=[[0.697,0.460],[0.774,0.376],[0.634,0.264],
                     [0.608,0.318],[0.556,0.215],[0.403,0.237],
                     [0.481,0.149],[0.437,0.211],[0.666,0.091],
                     [0.243,0.267],[0.245,0.057],[0.343,0.099],
                     [0.639,0.161],[0.657,0.198],[0.360,0.370],
                     [0.593,0.042],[0.719,0.103],[0.359,0.188],
                     [0.339,0.241],[0.282,0.257],[0.748,0.232],
                     [0.714,0.346],[0.483,0.312],[0.478,0.437],
                     [0.525,0.369],[0.751,0.489],[0.532,0.472],
                     [0.473,0.376],[0.725,0.445],[0.446,0.459]
                     ]
    return Watermalen_data

def show_data(data,meanVector,n):
    import matplotlib.pyplot as plt
    plt.figure()
    color=['r','g','b']
    for i in range(len(data)):
        x = []
        y = []
        vectors=data[i]
        for vector in vectors:
            x.append(vector[0])
            y.append(vector[1])
        plt.scatter(x,y,c=color[i])
    x=[]
    y=[]
    for vector in meanVector:
        x.append(vector[0])
        y.append(vector[1])
    plt.scatter(x,y,marker='+',c='r',s=200)
    plt.savefig('迭代次数：{}.png'.format(n))

def dist(x,y):
    from numpy.linalg import norm
    return norm(x-y)

def KMeans(X,n_clusters=3):
    import numpy as np
    from numpy.linalg import norm
    X=np.array(X)
    # step 1. 从D中随机选择k个样本作为初始均值向量
    np.random.shuffle(X)
    meanVector=np.array([X[i] for i in range(n_clusters)])

    # step 2. 循环 计算每个点到均值向量的距离，并分为k组
    cnt=0
    while True:
        cnt+=1
        C=[]
        for i in range(n_clusters):
            C.append([])
        for i in range(len(X)):
            d_min=float('inf')
            for j in range(n_clusters):
                d=dist(X[i],meanVector[j])
                if d<d_min:
                    d_min=d
                    index=j
            C[index].append(X[i])
        print(C)
        show_data(C,meanVector,cnt)

        # step 3. 更新均值向量
        flag=0
        for i in range(n_clusters):
            sum_vector=0
            for j in range(len(C[i])):
                sum_vector+=C[i][j]
            mu_i=sum_vector/(len(C[i]))
            if not np.array_equal(meanVector[i],mu_i):
                meanVector[i]=mu_i
                flag=1
        print(meanVector)
        if flag==0:
            break


if __name__ == '__main__':
    X=load_data()
    KMeans(X,n_clusters=2)

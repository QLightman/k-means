import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
k = input("input number of centers:")
def test():
    plt.figure(figsize=(12, 12))
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples,centers=k, random_state=random_state)

    plt.subplot(221)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title("results from the data")
    pre_center=np.empty((k,2))
    for i in range (k):
        pre_center[i]=X[i]
    y_pred=distance(X,pre_center)

    his_y_pred=np.empty(len(X))

    while ((his_y_pred==y_pred).all())!=True:
        his_y_pred=copy.copy(y_pred)
        y_pred=distance(X,pre_center)

    plt.subplot(222)
    plt.scatter(X[:,0],X[:,1],c=y_pred)
    plt.title("Result obtained from my k-means")

    y_classical_pred = KMeans(n_clusters=k, random_state=random_state).fit_predict(X)
    plt.subplot(223)
    plt.scatter(X[:, 0], X[:, 1], c=y_classical_pred)
    plt.title("results obtained from sklearn k means")

    plt.show()

def distance(X,pre_center):
    result=np.empty(len(X))
    for i in range(len(X)):
        tmp=[]
        for j in range (k):
            tmp.append(((X[i]-pre_center[j])**2).sum())
            result[i]=tmp.index(min(tmp))
    for i in range(k):
        pre_center[i]=X[result==i].mean(0)

    return result

if __name__ == '__main__':
    test()

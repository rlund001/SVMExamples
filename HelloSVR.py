'''
Created on Mar 7, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SimulateModelData import simulateModelData
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    n_samples = 200
    n_features = 2
    nld = simulateModelData(n_features, n_samples)
    nld.simulateNonLinear(sigma=0.2)
    ss = StandardScaler()
    ss.fit(nld.X)
    XNew = ss.transform(nld.X)

    clf = SVR(C=1.0, epsilon=0.1)
    clf.fit(XNew, nld.y)
    y1 = clf.predict(XNew)
    print np.max(y1-nld.y), np.average(y1-nld.y)
    
    nld2 = simulateModelData(n_features, n_samples)
    nld2.simulateNonLinear(sigma=0.2)
    XNew2 = ss.transform(nld2.X)
    y2 = clf.predict(XNew2) 
    print np.max(y2-nld2.y), np.average(y2-nld2.y)
    
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(nld.X[:,0], nld.X[:,1], nld.y)
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(nld.X[:,0], nld.X[:,1], y1)
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(XNew2[:,0], XNew2[:,1], y2)
    plt.show()

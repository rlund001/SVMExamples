'''
Created on May 22, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SimulateModelData import simulateModelData
from sklearn.preprocessing import StandardScaler
import sklearn.neural_network 

if __name__ == '__main__':
    n_samples = 10
    nFeatures = 2
    numClasses = 2
    nld = simulateModelData(nFeatures, n_samples)
    nld.simulateClassification(numClasses)
    ss = StandardScaler()
    ss.fit(nld.X)
    XNew = ss.transform(nld.X)
    print nld.X
    print nld.y

    clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(XNew, nld.y)
    y1 = clf.predict(XNew)
    print np.max(y1-nld.y), np.average(y1-nld.y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(nld.X[:,0], nld.X[:,1])
    plt.show()

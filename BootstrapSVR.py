'''
Created on Mar 3, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from SimulateModelData import simulateModelData
from sklearn.svm import SVR


if __name__ == '__main__':
    n_samples = 500
    n_features = 2
    n_iterations = 1000
    nld = simulateModelData(n_features,n_samples)
    nld.simulateNonLinear(sigma=0.2)
    trainingPerc = 0.8
    
    results = np.zeros(n_iterations)
    for i in range(0,n_iterations):
        (train, validate) = nld.chooseBootstrapSamples(trainingPerc)
        ss = StandardScaler()
        ss.fit(train.X)
        X = ss.transform(train.X)
        y = train.y
        clf = SVR(C=1.0, epsilon=0.1)
        clf.fit(X, y)
        X1 = ss.transform(validate.X)  
        y1 = clf.predict(X1)
        results[i] = np.average(y1-validate.y)
        
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    print np.average(results)    
    #plt.subplots_adjust(left=.1, right=.98, bottom=.10, top=.95, hspace=.99)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.hist(results)
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(nld.X[:,0], nld.X[:,1], nld.y)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(X[:,0], X[:,1], y)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(X1[:,0], X1[:,1], y1)

    plt.show()
    
    
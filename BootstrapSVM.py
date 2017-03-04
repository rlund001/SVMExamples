'''
Created on Mar 3, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from SimulateModelData import simulateModelData

def chooseBootstrapSamples(cd, trainingPerc):
    tot = len(cd.y)
    nTrain = int(np.round(trainingPerc*tot,0))
    nValidate = tot - nTrain
    perm = np.random.permutation(tot)
    iTrain = perm[0:nTrain]
    iValidate = perm[nTrain:tot]
    train = simulateModelData(cd.numFeatures, cd.numClasses, nTrain)
    train.X = cd.X[iTrain,:]
    train.y = cd.y[iTrain]
    validate = simulateModelData(cd.numFeatures, cd.numClasses, nValidate)
    validate.X = cd.X[iValidate,:]
    validate.y = cd.y[iValidate]
    return (train, validate)

if __name__ == '__main__':
    n_samples = 500
    n_features = 2
    n_classes = 4
    meanVectors = np.random.uniform(0.0,10.0,size=[n_classes,n_features])
    sigmaVectors = 2.0*np.ones([n_classes, n_features])
    cd = simulateModelData(n_features,n_classes,n_samples)
    cd.simulate( mu=meanVectors, sigma=sigmaVectors)
    trainingPerc = 0.8
    
    results = np.zeros(1000)
    for i in range(0,1000):
        (train, validate) = chooseBootstrapSamples(cd, trainingPerc)
        ss = StandardScaler()
        ss.fit(train.X)
        X = ss.transform(train.X)
        y = train.y
        clf = svm.SVC()
        clf.fit(X, y)
        X1 = ss.transform(validate.X)  
        y1 = clf.predict(X1)
        w = np.where(y1==validate.y)[0]
        results[i] = 100.0*len(w)/validate.numSamples
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        
    print results[99]
    print np.average(results)    
    plt.subplot(411)
    plt.hist(results)
    plt.subplot(412)
    plt.scatter(train.X[:, 0], train.X[:, 1], color=colors[train.y].tolist(), s=10)
    plt.subplot(413)
    plt.scatter(validate.X[:, 0], validate.X[:, 1], color=colors[validate.y].tolist(), s=10)
    plt.subplot(414)
    plt.scatter(validate.X[:, 0], validate.X[:, 1], color=colors[y1].tolist(), s=10)
    plt.show()
    
    
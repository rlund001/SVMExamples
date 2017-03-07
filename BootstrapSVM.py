'''
Created on Mar 3, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from SimulateModelData import simulateModelData


if __name__ == '__main__':
    n_samples = 500
    n_features = 2
    n_classes = 4
    n_iterations = 1000
    meanVectors = np.random.uniform(0.0,10.0,size=[n_classes,n_features])
    sigmaVectors = 2.0*np.ones([n_classes, n_features])
    cd = simulateModelData(n_features,n_samples)
    cd.simulateClassification(n_classes, mu=meanVectors, sigma=sigmaVectors)
    trainingPerc = 0.8
    
    results = np.zeros(n_iterations)
    for i in range(0,n_iterations):
        (train, validate) = cd.chooseBootstrapSamples(trainingPerc)
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
    plt.subplots_adjust(left=.1, right=.98, bottom=.10, top=.95, hspace=.99)

    plt.subplot(411)
    plt.hist(results)
    plt.title('Accuracy for ' + str(n_iterations) + ' iterations')
    plt.xlabel('Classification accuracy')
    plt.text(min(results) + .01, 250, r'Average percent: %s' % str(np.average(results)) )
    
    plt.subplot(412)
    plt.scatter(train.X[:, 0], train.X[:, 1], color=colors[train.y].tolist(), s=10)
    plt.title('Training Data')

    plt.subplot(413)
    plt.scatter(validate.X[:, 0], validate.X[:, 1], color=colors[validate.y].tolist(), s=10)
    plt.title('Validation Data Actual - last iteration')
    
    plt.subplot(414)
    plt.scatter(validate.X[:, 0], validate.X[:, 1], color=colors[y1].tolist(), s=10)
    plt.title('Validation Data Predicted - last iteration')
    plt.text(np.min(validate.X[:, 0]), np.max(validate.X[:, 1]), r'Accuracy: %s' % str(results[99]))
    plt.show()
    
    
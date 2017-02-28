import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import svm

class simulateClassifierData(object):
    '''
    classdocs
    '''
    def __init__(self, numFeatures, numClasses, numSamples):
        self.X = np.random.normal(size=[numSamples,numFeatures])/2.0
        self.y = np.random.randint(0,numClasses,size=numSamples)
        self.meanVectors = np.random.uniform(low=0.0,high=10.0,size=[numClasses,numFeatures])
        for i in range(0,numSamples):
            self.X[i,:] = self.X[i,:] + self.meanVectors[self.y[i],:]    
        return 


np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
cd = simulateClassifierData(10,4,100)
normalClusters = (cd.X, cd.y)

no_structure = np.random.rand(n_samples, 2), None



colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
#colors = np.hstack([colors] * 20)



datasets = [noisy_circles, noisy_moons, blobs, normalClusters]

plt.figure(figsize=(len(datasets), 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

for dataset in datasets:
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    clf = svm.SVC()
    clf.fit(X, y)  
    y1 = clf.predict(X)
    
    if X.shape[1] > 2:
        U, s, V = np.linalg.svd(cd.X, full_matrices=False)
        X = U
        X = StandardScaler().fit_transform(X)
        
    plt.subplot(len(datasets), 2, plot_num)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

    plt.subplot(len(datasets), 2, plot_num)
    plt.scatter(X[:, 0], X[:, 1], color=colors[y1].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1
 
plt.show()
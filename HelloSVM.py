import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from SimulateModelData import simulateModelData

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
n_features = 10
n_classes = 4
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

meanVectors = np.random.uniform(0.0,10.0,size=[n_classes,n_features])
sigmaVectors = 4.0*np.ones([n_classes, n_features])
cd = simulateModelData(n_features,n_samples)
cd.simulateClass(n_classes, mu=meanVectors, sigma=sigmaVectors)
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
    ss = StandardScaler()
    ss.fit(X)
    XNew = ss.transform(X)
    clf = svm.SVC()
    clf.fit(XNew, y)  
    y1 = clf.predict(XNew)
    p1 = XNew[:, 0]
    p2 = XNew[:, 1]
    
    if XNew.shape[1] > 2:
        U, s, V = np.linalg.svd(XNew, full_matrices=False)
        p = StandardScaler().fit_transform(U)
        p1 = p[:, 0]
        p2 = p[:, 1]
    

    plt.subplot(len(datasets), 2, plot_num)
    plt.scatter(p1, p2, color=colors[y].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1

    if XNew.shape[1] > 2:
        cd1 = simulateModelData(n_features,10)
        cd1.simulateClass(n_classes, mu=meanVectors, sigma=sigmaVectors)
 
        y1 = clf.predict(ss.transform(cd1.X))
        p = StandardScaler().fit_transform(np.dot(cd1.X,V.transpose()))
        p1 =  p[:, 0]
        p2 =  p[:, 1]

    plt.subplot(len(datasets), 2, plot_num)
    plt.scatter(p1, p2, color=colors[y1].tolist(), s=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    plot_num += 1
 
plt.show()
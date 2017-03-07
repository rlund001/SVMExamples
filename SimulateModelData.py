'''
Created on Mar 1, 2017

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

class simulateModelData(object):
    '''
    classdocs
    '''
    def __init__(self, numFeatures, numSamples):
        self.numFeatures = numFeatures
        self.numSamples = numSamples
        self.meanVectors = None
        self.sigmaVectors = None
        self.X = None
        self.y = None
        self.numClasses = None
        self.theta = None
        return
    
    def simulateClassification(self, numClasses, mu=None, sigma=None):
        self.numClasses = numClasses
        if mu is None:
            self.meanVectors = np.zeros([self.numClasses,self.numFeatures])
        else:
            self.meanVectors = mu    
        if sigma is None:
            self.sigmaVectors = np.ones([self.numClasses,self.numFeatures])
        else:
            self.sigmaVectors = sigma        
        self.X = np.random.normal(size=[self.numSamples,self.numFeatures])
        self.y = np.random.randint(0,self.numClasses,size=self.numSamples)
        for i in range(0,self.numSamples):
            self.X[i,:] = self.X[i,:]/self.sigmaVectors[self.y[i],:] + self.meanVectors[self.y[i],:]    
        return 

    def simulateLinear(self, theta=None, sigma=1.0):
        if theta is None:
            self.theta = np.ones([self.numFeatures])
        else:
            self.theta = theta
        self.X = np.random.normal(size=[self.numSamples,self.numFeatures])
        
        self.y = np.dot(self.X, theta) + np.random.normal(0.0,sigma,size=[self.numSamples])
        return
    
    def simulateNonLinear(self, sigma=1.0):
        self.X = np.random.normal(size=[self.numSamples,self.numFeatures])
        r2 = np.sum(np.multiply(self.X,self.X),axis=1)
        self.y = np.where(r2 < 1.0, 2.0-r2, 1.0/r2) + np.random.normal(0.0,sigma,size=[self.numSamples])
        return
    
    def chooseBootstrapSamples(self, trainingPerc):
        tot = self.numSamples
        nTrain = int(np.round(trainingPerc*tot,0))
        nValidate = tot - nTrain
        perm = np.random.permutation(tot)
        iTrain = perm[0:nTrain]
        iValidate = perm[nTrain:tot]
        train = simulateModelData(self.numFeatures, nTrain)
        train.X = self.X[iTrain,:]
        train.y = self.y[iTrain]
        validate = simulateModelData(self.numFeatures, nValidate)
        validate.X = self.X[iValidate,:]
        validate.y = self.y[iValidate]
        return (train, validate)

        
if __name__ == '__main__':
    if (False):
        n_samples = 1500
        n_features = 10
        ld = simulateModelData(n_features, n_samples)
        theta = np.random.uniform(-10.0,10.0,size=[n_features])
        ld.simulateLinear(theta, 1.0) 
        thetaHat = np.dot(np.linalg.inv(np.dot(ld.X.transpose(),ld.X)),np.dot(ld.X.transpose(),ld.y)) 
        print theta
        print thetaHat   
    
    else:
        n_samples = 200
        n_features = 2
        nld = simulateModelData(n_features, n_samples)
        nld.simulateNonLinear()
         
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(nld.X[:,0], nld.X[:,1], nld.y)
        plt.show()
        
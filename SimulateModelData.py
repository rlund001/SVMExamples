'''
Created on Mar 1, 2017

@author: a3438
'''
import numpy as np

class simulateModelData(object):
    '''
    classdocs
    '''
    def __init__(self, numFeatures, numClasses, numSamples):
        self.numFeatures = numFeatures
        self.numClasses = numClasses
        self.numSamples = numSamples
        self.meanVectors = None
        self.sigmaVectors = None
        
        return
    
    def simulate(self, mu=None, sigma=None):
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

    
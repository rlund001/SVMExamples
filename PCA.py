'''
Created on Nov 4, 2016

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt

def reduceDimensionToN(X,n):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    sDiag = np.diag(s)
    Xn = np.dot(np.dot(U[:,0:n],sDiag[0:n,:]),V)
    return Xn

if __name__ == '__main__':
    theta = np.matrix([2.0, -1.0, 0.0, 0.0, 1.0]).transpose()
    
    samples = 200
    X = np.matrix([np.random.normal(10.0, 3.0, samples), 
                   np.random.normal(0.0, 1.0, samples), 
                   np.random.normal(0.0, 2.0, samples), 
                   np.random.normal(0.0, 1.0, samples), 
                   np.ones(samples)]).transpose()
       
    print 'Theta:', theta.transpose()
    print 'Shape of X is ', X.shape
    
    Y = np.dot(X,theta) + np.random.normal(0.0, 1.0, samples).reshape(samples,1)
    print 'Shape of Y is ', Y.shape
    
    U, s, V = np.linalg.svd(X, full_matrices=False)
    print 'Shape of U,s,V is ', U.shape, s.shape, V.shape
    print "s:", s

    thetaHat = np.dot(np.linalg.inv(np.dot(X.transpose(),X)),np.dot(X.transpose(),Y))
    YHat = np.dot(X,thetaHat)
    X2D = reduceDimensionToN(X,2)
    YHat2D = np.dot(X2D,theta)
    print 'ThetaHat:', thetaHat.transpose()
    
    plt.title('PCA Data')
    plt.xlabel('Y')
    plt.ylabel('Y2D')
    plt.grid(True)
    plt.scatter(YHat, YHat2D) 
    
    # plt.axis([0, 101, 0, 101])
    plt.show()

    
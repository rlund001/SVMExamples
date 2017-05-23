'''
Created on Dec 20, 2016

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt

def SetWave():
    s = 51
    w = np.zeros(s)
    t = np.zeros(s)
    for i in range(0,s):
        t[i] = (2.0*(i+1)/s - 1.0)*4.0*np.pi
        w[i] = np.sin(t[i])/t[i]
    return (t,w) 

if __name__ == '__main__':

    transmitterAngles = [0.0, 180.0,   60.0,  197.0,  120.0,  300.0]
    transmitterOffset = [0.0,   0.0, -41.72, -41.72, -83.43, -83.43]
    receiverAngles  = [317.0, 137.0,   60.0,  240.0,   77.0,  257.0]
    receiverOffset  = [  0.0,   0.0, -41.72, -41.72, -83.43, -83.43]
    interactions = [[True, True, False,False,False,False],
                    [True, True, False,False,False,False],
                    [False,False,True, True, False,False],
                    [False,False,True, True, False,False],
                    [False,False,False,False,True, True],
                    [False,False,False,False,True, True]]
    transmitterBias = [1.0,  1.1, 0.9, 1.0,  1.2,  .8]
    receiverBias    = [1.0, 0.95, 0.9, 1.0, 1.05, 1.1] 
    circumferentialSamples = 1000
    velocity = 100 
    
    (t,wave) = SetWave()
    
    plt.subplot(111)
    plt.title("sinch", size=18)
    plt.plot(t,wave)
    plt.show()
'''
Created on Oct 21, 2016

@author: a3438
'''
import numpy as np
import matplotlib.pyplot as plt


def MakeSumOfCosines(frequencies, x, length):
    curve = np.zeros(length)
    for f in frequencies:
        curve = curve + np.cos(f*x*2*np.pi)
    return curve

if __name__ == '__main__':
    length = 1024
    period = 4*np.pi
    t = np.arange(0, period, period/length)
    spect = length/(2*period)
    w = np.arange(0, spect+2*spect/length, 2*spect/length)
    
    frequencies = [1, 6.0]
    ft = MakeSumOfCosines(frequencies, t, length)
    fw = np.fft.rfft(ft)
    fw2 = np.zeros(len(fw), dtype=np.complex_)
    mid = (frequencies[0] + frequencies[1])/2
    for i in range(0,len(fw)):
        if w[i] < mid:
            fw2[i] = fw[i]
        else:
            fw2[i] = complex(0.0, 0.0)
    ft2 = np.fft.irfft(fw2)  
          
    print 't:',t.size,' w:',w.size,' ft:',ft.size,' fw:',fw.size
    
   
    plt.subplots_adjust(left=.1, right=.98, bottom=.1, top=.9, wspace=.1, hspace=.9)
    
    plt.subplot(611)
    plt.title("Time", size=25)
    plt.plot(t,ft)

    plt.subplot(612)
    plt.title("Freq -real", size=18)
    plt.plot(w,fw.real)
    
    plt.subplot(613)
    plt.title("Freq - imag", size=18)
    plt.plot(w,fw.imag)
    
    plt.subplot(614)
    plt.title("Freq - band pass real", size=18)
    plt.plot(w,fw2.real)

    plt.subplot(615)
    plt.title("Freq - band pass imag", size=18)
    plt.plot(w,fw2.imag)

    plt.subplot(616)
    plt.title("Time - inverted", size=18)
    plt.plot(t,ft2)

    plt.show()
        
    #
    #plt.show()
        
    #
    #plt.show()
    

import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np



plt.rcParams['figure.figsize'] = (10, 6)

def PlotHistNorm(data, bins=10, log=False):
    # distribution fitting
    param = stats.norm.fit(data) 
    mean = param[0]
    sd = param[1]

    #Plot histogram
    histdata = plt.hist(data,bins=bins,alpha=.3,log=log)

    #Generate X points
    x = np.linspace(min(data),max(data),100)
    #Get Y points via Normal PDF with fitted parameters
    pdf_fitted = stats.norm.pdf(x,loc=mean,scale=sd)
    

    #Get histogram data, in this case bin edges
    xh = [0.5 * (histdata[1][r] + histdata[1][r+1]) for r in xrange(len(histdata[1])-1)]
    #Get bin width from this
    binwidth = (max(xh) - min(xh)) / len(histdata[1])           
    #Scale the fitted PDF by area of the histogram
    pdf_fitted = pdf_fitted * (len(data) * binwidth)

    #Plot PDF
    plt.plot(x,pdf_fitted,'r-')
    plt.grid()
    
    return

if __name__ == '__main__':

    data = np.random.normal(.5, 2.0, 100)
  
    fig = plt.figure()

    fig.add_subplot(111)
    PlotHistNorm(data,bins=round(len(data)/20))
    plt.xlabel('Normal F')
    plt.show()
    
# fig.add_subplot(132)
# PlotHistNorm(fdf['Northing Error (m/km)'])
# plt.xlabel('Normalized Northing Error (m/km)')
# 
# fig.add_subplot(133)
# PlotHistNorm(fdf['Elevation Error (m/km)'])
# plt.xlabel('Normalized Elevation Error (m/km)')

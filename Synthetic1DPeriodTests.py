from TDA import *
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
from scipy import signal
from sklearn.decomposition import PCA
from CircularCoordinates import *

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(len(x))
    xidx = [] #Index the original samples into the sliding window array (assumes dT evenly divides 1)
    for i in range(NWindows):
        if dT*i == int(dT*i):
            xidx.append(i)
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    while len(xidx) < len(x):
        xidx.append(xidx[-1])
    return (X, xidx)

if __name__ == '__main__':
    #Step 1: Setup the signal
    T1 = 10 #The period of the first sine in number of samples
    T2 = T1*np.pi #The period of the second sine in number of samples
    NPeriods = 15 #How many periods to go through, relative to the second sinusoid
    N = T1*3*NPeriods #The total number of samples
    t = np.arange(N)*np.pi/3 #Time indices
    t1 = 2*np.pi*(1.0/T1)*t
    t2 = 2*np.pi*(1.0/T2)*t
    x = np.cos(t1) #The first sinusoid
    x += np.cos(t2) #The second sinusoid
    np.random.seed(2)
    x = x + 0.05*np.random.randn(len(x))

    dim = 30
    Tau = 1
    dT = 0.5
    (X, xidx) = getSlidingWindow(x, dim, Tau, dT)
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    sio.savemat("X.mat", {"X":X})
    extent = Tau*dim

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Perform PCA down to 2D for visualization
    pca = PCA(n_components = 10)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_

    #Do TDA
    p = 41
    (PDs, Cocycles) = doRipsFiltration(X, 1, coeff = p, cocycles = True)
    fig = plt.figure(figsize=(12, 10))
    
    ret = {"t1":t1, "t2":t2}
    for i in range(PDs[1].shape[0]):
        #if PDs[1][i, 1] - PDs[1][i, 0] < 0.1:
        #    continue
        ccl = Cocycles[1][i]
        thresh = PDs[1][i, 1] - 0.01
        (s, cclret) = getCircularCoordinates(X, ccl, p, thresh) 
        s = s - np.min(s)
        s = s/np.max(s)
        ret["ccl%i"%i] = s
        
        c = plt.get_cmap('spectral')
        C2 = c(np.array(np.round(255*s), dtype=np.int32))
        C2 = C2[:, 0:3]
    
        plt.clf()
        plt.subplot(221)
        drawLineColored(t, x, C2[xidx, :])
        ax = plt.gca()
        plotbgcolor = (0.15, 0.15, 0.15)
        ax.set_axis_bgcolor(plotbgcolor)


        plt.ylim([-3, 3])
        plt.title("Original Signal")
        plt.xlabel("t")

#        ax2 = fig.add_subplot(222, projection = '3d')
#        #plt.subplot(132)
#        ax2.set_title("PCA of Sliding Window Embedding")
#        ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 3], c=C2, edgecolors='none')
#        #plt.axis('equal')
#        #plt.axis('off')
#        #ax = plt.gca()
#        ax2.set_axis_bgcolor(plotbgcolor)
#        ax2.w_xaxis.set_pane_color(plotbgcolor)
#        ax2.w_yaxis.set_pane_color(plotbgcolor)
#        ax2.w_zaxis.set_pane_color(plotbgcolor)


        plt.subplot(223)
        H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
        plt.hold(True)
        plt.scatter([PDs[1][i, 0]], [PDs[1][i, 1]], 100, 'm')
        #H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
        #plt.legend(handles=[H1, H2])

        ax = plt.gca()
        ax.set_axis_bgcolor(plotbgcolor)
        plt.title('Persistence Diagram, Birth Time = %g'%PDs[1][i, 0])
        
        plt.subplot(222)
        plt.plot(s)
        
        plt.subplot(224)
        s2 = np.unwrap(s*2*np.pi)/(2*np.pi)
        plt.plot(s2)
        slope = (s2[-1] - s2[0])/len(s2)
        plt.title("Unwrapped, Slope = %g"%slope)
        
        plt.savefig("Cocycle%i.svg"%i)
    sio.savemat("cocycles.mat", ret)

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
from CSMSSMTools import *
from Laplacian import *

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
    plotbgcolor = (0.15, 0.15, 0.15)

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

    #x = sio.loadmat('Curto.mat')['x'].flatten()
    print "len(x) = ", len(x)

    dim = 30
    Tau = 1
    dT = 0.5
    (X, xidx) = getSlidingWindow(x, dim, Tau, dT)
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    sio.savemat("X.mat", {"X":X})

    fig = plt.figure(figsize=(12, 12))
    getTorusCoordinatesPC(X, weighted = True, doPlot = True)
    plt.subplot(3, 3, 1)
    plt.plot(x)
    plt.title("Original Signal")
    #drawLineColored(t, x, C[xidx, :])
    #ax = plt.gca()
    #ax.set_axis_bgcolor(plotbgcolor)
    plt.show()

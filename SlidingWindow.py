"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: Some basic code to do sliding window embeddings of 1D signals
"""
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def getSlidingWindow(x, dim, Tau, dT):
    """
    A function that computes the sliding window embedding of a
    discrete signal.  If the requested windows and samples do not
    coincide with sampels in the original signal, spline interpolation
    is used to fill in intermediate values
    :param x: The discrete signal
    :param dim: The dimension of the sliding window embedding
    :param Tau: The increment between samples in the sliding window 
    :param dT: The hop size between windows
    :returns: An Nxdim Euclidean vector of sliding windows
    """
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end >= len(x):
            X = X[0:i, :]
            break
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    return X

def getSlidingWindowNoInterp(x, dim):
    """
    A function that computes the sliding window embedding of a
    discrete signal.  It is assumed that Tau = 1 and dT = 1.
    This function is faster than getSlidingWindow() in this case
    :param x: The discrete signal
    :param dim: The dimension of the sliding window embedding
    :returns: An Nxdim Euclidean vector of sliding windows
    """
    N = len(x)
    NWindows = N - dim + 1
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        X[i, :] = x[i:i+dim]
    return X

def getSlidingWindowInteger(x, dim, Tau, dT):
    """
    Similar to the above function
    """
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT)) #The number of windows
    if NWindows <= 0:
        print("Error: Tau too large for signal extent")
        return np.zeros((3, dim))
    X = np.zeros((NWindows, dim)) #Create a 2D array which will store all windows
    idx = np.arange(N)
    for i in range(NWindows):
        #Figure out the indices of the samples in this window
        idxx = np.array(dT*i + Tau*np.arange(dim), dtype=np.int32)
        X[i, :] = x[idxx]
    return X
        

"""
Below is an example that shows how to do the quasiperiodic signal,
which lies on a 2-torus
"""
if __name__ == '__main__':
    from TDA import *
    plotbgcolor = (0.15, 0.15, 0.15)
    #Step 1: Setup the signal
    T1 = 10 #The period of the first sine in number of samples
    T2 = T1*np.pi #The period of the second sine in number of samples
    NPeriods = 9 #How many periods to go through, relative to the second sinusoid
    N = T1*3*NPeriods #The total number of samples
    t = np.arange(N)*np.pi/3 #Time indices
    x = np.cos(2*np.pi*(1.0/T1)*t) #The first sinusoid
    x += np.cos(2*np.pi*(1.0/T2)*t) #The second sinusoid

    dim = 30
    Tau = 1
    dT = 0.5
    X = getSlidingWindow(x, dim, Tau, dT)
    #Point center and sphere normalize
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
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
    PDs = doRipsFiltration(X, 2, thresh = -1, coeff = 2)

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.plot(t, x)
    ax = plt.gca()

    plt.ylim([-3, 3])
    plt.title("Original Signal")
    plt.xlabel("t")

    #ax2 = fig.add_subplot(132, projection = '3d')
    plt.subplot(132)
    plt.title("PCA of Sliding Window Embedding")
    plt.scatter(Y[:, 0], Y[:, 1], c=C, edgecolors='none')
    plt.axis('equal')
    #plt.axis('off')
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)


    plt.subplot(133)
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.legend(handles=[H1, H2])

    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    plt.title('Persistence Diagrams')

    plt.show()

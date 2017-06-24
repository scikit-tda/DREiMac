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

    x = sio.loadmat('Curto.mat')['x'].flatten()
    print "len(x) = ", len(x)

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
    print "Doing TDA..."
    PDs = doRipsFiltration(X, 1, coeff = p)
    fig = plt.figure(figsize=(12, 12))
    print "Finished TDA"

    I = PDs[1]
    diff = I[:, 1] - I[:, 0]
    idx = np.argsort(-diff)
    I = I[idx, :]
    thresh = np.mean(I[1, :])

    print "Doing Laplacian..."
    res = getTorusCoordinates(X, thresh, weighted = True)
    print "Finished Laplacian"
    v = res['v']
    w = res['w']
    A = res['A']
    #A = res['A'].toarray()

    plt.clf()
    plt.subplot(3, 3, 1)
    plt.plot(x)
    #drawLineColored(t, x, C[xidx, :])
    #ax = plt.gca()
    #ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim([-3, 3])
    plt.title("Original Signal")
    plt.xlabel("t")

    plt.subplot(3, 3, 4)
    plt.imshow(A, cmap = 'gray', interpolation = 'none')
    plt.title("SSM")

    plt.subplot(3, 3, 7)
    H1 = plotDGM(PDs[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    plt.title("Persistence Diagram, Thresh = %g"%thresh)


    plt.subplot(3, 3, 2)
    N = X.shape[0]
    plt.scatter(v[0:N, 1], v[0:N, 2], 20, c=C, edgecolor = 'none')
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    plt.xlabel("Eigvec 1")
    plt.ylabel("Eigvec 2")

    plt.subplot(3, 3, 3)
    plt.plot(res['theta'])
    plt.title('theta')

    plt.subplot(3, 3, 5)
    N = X.shape[0]
    plt.scatter(v[0:N, 3], v[0:N, 4], 20, c=C, edgecolor = 'none')
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    plt.xlabel("Eigvec 3")
    plt.ylabel("Eigvec 4")

    plt.subplot(3, 3, 6)
    plt.plot(res['phi'])
    plt.title('phi')

    plt.subplot(3, 3, 8)
    plt.stem(w)
    plt.title('Eigenvalues')

    plt.subplot(3, 3, 9)
    plt.imshow(v, cmap = 'spectral', aspect = 'auto', interpolation = 'none')
    plt.title('Eigenvectors')

    plt.show()

import subprocess
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from TDA import *
from Hodge import *

def integrateCocycle(ccl):
    print "TODO"

def getCircularCoordinates(X, ccl, p, thresh):
    """
    :param X: Nxd array of points
    :param ccl: Nx3 array holding cocycle.  First dimension is edge index 1,
    second dimension is edge index 2, and third dimension is value mod p
    :param p: Prime used in cocycle
    :param thresh: Threshold at which to find representative cocycle
    """
    N = X.shape[0]
    D = getSSM(X)

    #Lift to integer cocycle
    val = np.array(ccl[:, 2])
    val[val > (p-1)/2] -= p
    Y = np.zeros(D.shape)
    Y[ccl[:, 0], ccl[:, 1]] = val
    Y = Y + Y.T

    #Select edges that are under the threshold
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    Y = Y[np.triu_indices(N, 1)]
    idx = np.arange(len(I))
    idx = idx[D[I, J] <= thresh]
    I = I[idx]
    J = J[idx]
    Y = Y[idx]

    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I

    #Check to make sure Y is still a cocycle
    Delta1 = makeDelta1(R)
    res = np.sum(np.abs(Delta1.dot(Y)))
    isCocycle = (res == 0)
    if not isCocycle:
        print "\n\nERROR: Not a cocycle"
        print "Delta1.dot(Y) = ", Delta1.dot(Y)
        print "res = ", res
        print "\n"

    W = np.ones(len(Y))

    (s, I, H) = doHodge(R, W, Y, verbose = True)

    print "sum(abs(Delta1.dot(H))) = ", np.sum(np.abs(Delta1.dot(H)))

    #Cocycle resides in the harmonic component H
    cclret = np.zeros((R.shape[0], 3))
    cclret[:, 0:2] = R
    cclret[:, 2] = H
    return (s, cclret)

if __name__ == '__main__':
    p = 41
    np.random.seed(10)
    N = 50
    X = np.zeros((N*2, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    X[N::, 0] = 2*np.cos(t) + 4
    X[N::, 1] = 2*np.sin(t) + 4
    X = X[np.random.permutation(X.shape[0]), :]
    X = X + 0.2*np.random.randn(X.shape[0], 2)

    (PDs, Cocycles) = doRipsFiltration(X, 1, coeff = p, cocycles = True)
    
    plt.figure(figsize=(12, 5))
    for i in range(PDs[1].shape[0]):
        ccl = Cocycles[1][i]
        thresh = PDs[1][i, 1] - 0.01
        (s, cclret) = getCircularCoordinates(X, ccl, p, thresh)
        
        plt.clf()
        plt.subplot(121)
        plotDGM(PDs[1])
        plt.scatter([PDs[1][i, 0]], [PDs[1][i, 1]], 40, 'r')
        #plt.subplot(132)
        #plotCocycle(X, ccl, thresh)
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], 100, s, cmap = 'afmhot', edgecolor = 'none')
        plt.colorbar()
        plt.savefig("Cocycle%i.svg"%i, bbox_inches = 'tight')

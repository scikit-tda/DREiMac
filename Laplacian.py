import sys
import scipy.sparse as sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from CSMSSMTools import *

def getLaplacianEigs(A, NEigs):
    DEG = sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v, L)

def getLaplacianEigsDense(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG.toarray() - A
    w, v = linalg.eigh(L)
    return (w[0:NEigs], v[:, 0:NEigs], L)

def getKleinAdj(N):
    #Create an N x N grid
    idx = np.arange(N*N)
    idx = np.reshape(idx, (N, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(N))
    I = []
    J = []
    for i in range(N):
        for j in range(N):
            i1 = YPos[i, j]
            i2 = (i1+1)%N
            j1 = XPos[i, j]
            j2 = (j1+1)%N
            a = idx[i1, j1]
            b = idx[i2, j1]
            c = idx[i1, j2]
            if j == N-1:
                i2 = N - i2 - 1 #The Klein flip
                c = idx[i2, j2]
            I += [a, b, a, c]
            J += [b, a, c, a]
    I = np.array(I)
    J = np.array(J)
    V = np.ones(len(I))
    A = sparse.coo_matrix((V, (I, J)), shape=(N*N, N*N)).tocsr()
    return A

def getTorusAdj(M, N):
    #Create an M x N grid
    idx = np.arange(M*N)
    idx = np.reshape(idx, (M, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(M))
    I = []
    J = []
    for i in range(M):
        for j in range(N):
            i1 = YPos[i, j]
            i2 = (i1+1)%M
            j1 = XPos[i, j]
            j2 = (j1+1)%N
            a = idx[i1, j1]
            b = idx[i2, j1]
            c = idx[i1, j2]
            I += [a, b, a, c]
            J += [b, a, c, a]
    I = np.array(I)
    J = np.array(J)
    V = np.ones(len(I))
    A = sparse.coo_matrix((V, (I, J)), shape=(M*N, M*N)).tocsr()
    return A

def getThetas(pv, eig1, eig2):
    """
    Use arctangent of mean-centered eigenvectors as estimates of
    circular coordinates
    """
    v = np.array(pv[:, [eig1, eig2]])
    v = v - np.mean(v, 0, keepdims=True)
    theta = np.arctan2(v[:, 1], v[:, 0])
    thetau = np.unwrap(theta)
    #Without loss of generality, switch theta to overall increasing
    if thetau[-1] - thetau[0] < 0:
        thetau = -thetau
    return (theta, thetau - thetau[0])

def getSlopes(thetas, sWin = 10):
    """
    Estimate smoothed versions of slopes in radians per sample
    2*sWin is the size of the rectangular window used to smooth
    """
    N = len(thetas)
    slopes = np.zeros(N)
    deriv = np.zeros(sWin*2)
    deriv[0:sWin] = np.ones(sWin)
    deriv[sWin:] = -np.ones(sWin)
    slopes[sWin-1:-sWin] = np.convolve(thetas, deriv, 'valid')/float(sWin**2)
    slopes[0:sWin-1] = slopes[sWin-1]
    slopes[-(sWin+1):] = slopes[-(sWin+1)]
    return slopes

def getCircularCoordinates(X, sigma, weighted = False):
    D = getSSM(X)
    if weighted:
        A = np.exp(-D*D/(2*sigma**2))
        (w, v, L) = getLaplacianEigsDense(A, 10)
    else:
        A = getSSMAdj(D, sigma)
        (w, v, L) = getLaplacianEigs(A, 10)
    (theta, thetau) = getThetas(v, 1, 2)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'D':D}

def getTorusCoordinates(X, sigma, weighted = False):
    D = getSSM(X)
    if weighted:
        A = np.exp(-D*D/(2*sigma**2))
        (w, v, L) = getLaplacianEigsDense(A, 16)
    else:
        A = getSSMAdj(D, sigma)
        (w, v, L) = getLaplacianEigs(A, 16)
    (theta, thetau) = getThetas(v, 1, 2)
    (phi, phiu) = getThetas(v, 3, 4)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'phi':phi, 'phiu':phiu, 'A':A, 'D':D}

if __name__ == '__main__':
    M = 20
    N = 15
    A = getTorusAdj(M, N)
    print A
    NEigs = 20
    (w, v, L) = getLaplacianEigs(A, NEigs)
    plt.stem(w)
    plt.show()
    K = int(np.ceil(np.sqrt(NEigs)))
    for i in range(NEigs):
        plt.subplot(K, K, i+1)
        I = np.reshape(v[:, i], (M, N))
        plt.imshow(I, interpolation = 'none', cmap = 'afmhot')
        plt.title("%.3g"%w[i])
        plt.axis('off')
    plt.savefig("TorusBasis.png", bbox_inches='tight')
    #sio.savemat("KleinBasis.mat", {"w":w, "v":v, "A":A, "L":L})

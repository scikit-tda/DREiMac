"""
Programmer: Chris Tralie, 12/2016 (ctralie@alumni.princeton.edu)
Purpose: To provide tools for quickly computing all pairs self-similarity
and cross-similarity matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.sparse as sparse

#############################################################################
## Code for dealing with cross-similarity matrices
#############################################################################

def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getSSM(X):
    return getCSM(X, X)

def getCSMCosine(X, Y):
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y**2, 1))
    YNorm[YNorm == 0] = 1
    D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
    D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
    return D

def CSMToBinaryKappa(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
    """
    M = D.shape[0]
    N = D.shape[1]
    if Kappa == 0:
        return np.ones((M, N))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*N))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(M)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(M, N))
    return ret.toarray()

def CSMToBinaryThresh(D, sigma):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    with a 1 if the distance is under the threshold sigma and a zero
    otherwise
    """
    M = D.shape[0]
    N = D.shape[1]
    [J, I] = np.meshgrid(np.arange(N), np.arange(M))
    I = I[D < sigma]
    J = J[D < sigma]
    V = np.ones(len(I))
    ret = sparse.coo_matrix((V, (I, J)), shape=(M, N))
    return ret

def getSSMAdj(pD, sigma):
    """
    Get an adjacency matrix for an SSM
    """
    D = np.array(pD)
    M = D.shape[0]
    N = D.shape[1]

    np.fill_diagonal(D, np.inf) #Exclude diagonal
    [J, I] = np.meshgrid(np.arange(N), np.arange(M))
    I = I[D < sigma]
    J = J[D < sigma]
    V = np.ones(len(I))
    ret = sparse.coo_matrix((V, (I, J)), shape=(M, N))
    return ret

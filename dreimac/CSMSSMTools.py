"""
Programmer: Chris Tralie, 12/2016 (ctralie@alumni.princeton.edu)
Purpose: To provide tools for quickly computing all pairs self-similarity
and cross-similarity matrices, and for doing "greedy permutations"
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
    
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points
    Return
    ------
    D : ndarray (M, N)
        An MxN Euclidean cross-similarity matrix
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

def getGreedyPermEuclidean(X, M, verbose = False):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a Euclidean point cloud.  This saves computation
    over having compute the full distance matrix if the number of
    landmarks M << N
    
    Parameters
    ----------
    X : ndarray (N, d) 
        An Nxd Euclidean point cloud
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress

    Return
    ------
    result: Dictionary
        {'Y': An Mxd array of landmarks, 
         'perm': An array of indices into X of the greedy permutation
         'lambdas': Insertion radii of the landmarks
         'D': An MxN array of distances from landmarks to points in X}
    """
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    N = X.shape[0]
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = getCSM(X[0, :][None, :], X).flatten()
    D = np.zeros((M, N))
    D[0, :] = ds
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i-1] = ds[idx]
        thisds = getCSM(X[idx, :][None, :], X).flatten()
        D[i, :] = thisds
        ds = np.minimum(ds, thisds)
    lambdas[-1] = np.max(ds)
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas, 'D':D}

def getGreedyPermDM(D, M, verbose = False):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a N x N distance matrix
    
    Parameters
    ----------
    D : ndarray (N, N) 
        An N x N distance matrix
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress

    Return
    ------
    result: Dictionary
        {'perm': An array of indices into X of the greedy permutation
         'lambdas': Insertion radii of the landmarks
         'DLandmarks': An MxN array of distances from landmarks to points in the point cloud}
    """
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    N = D.shape[0]
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = D[0, :]
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    DLandmarks = D[perm, :] 
    return {'perm':perm, 'lambdas':lambdas, 'DLandmarks':DLandmarks}



from sklearn.metrics.pairwise import pairwise_distances


def dpoint2pointcloud(X, i, metric):
    """
    Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data 
    i: int
        The index of the point from which to return all distances
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
    """
    ds = pairwise_distances(X, X[i, :][None, :], metric=metric).flatten()
    ds[i] = 0
    #ds = getCSM(X[i, :][None, :], X).flatten()
    #ds[i] = 0
    return ds


def get_greedy_perm(X, n_perm=None, distance_matrix=False, metric="euclidean"):
    """
    Compute a furthest point sampling permutation of a set of points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix
    distance_matrix: bool
        Indicator that X is a distance matrix, if not we compute 
        distances in X using the chosen metric.
    n_perm: int
        Number of points to take in the permutation
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lambdas: ndarray(n_perm)
        Covering radii at different points
    dperm2all: ndarray(n_perm, n_samples)
        Distances from points in the greedy permutation to points
        in the original point set
    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    if distance_matrix:
        dpoint2all = lambda i: X[i, :]
    else:
        dpoint2all = lambda i: dpoint2pointcloud(X, i, metric)
    ds = dpoint2all(0)
    dperm2all = [ds]
    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    dperm2all = np.array(dperm2all)
    return (idx_perm, lambdas, dperm2all)


def testGreedyPermEuclidean():
    """
    t = np.linspace(0, 2*np.pi, 10000)
    np.random.seed(0)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    """
    import scipy.io as sio
    X = sio.loadmat("P.mat")["P"]
    #X += 0.01*np.random.randn(X.shape[0], X.shape[1])
    res = getGreedyPermEuclidean(X, 100, True)
    idx_perm = res['perm']
    print(idx_perm)
    Y, D = res['Y'], res['D']
    
    (idx_perm2, lambdas, _) = get_greedy_perm(X, 100)
    print(idx_perm2)
    
    plt.plot(res['lambdas'])
    plt.plot(lambdas)
    plt.show()
    
    """
    plt.figure()
    
    plt.subplot(121)
    plt.scatter(X[idx_perm, 0], X[idx_perm, 1], c = np.arange(len(idx_perm)))

    plt.subplot(122)
    plt.scatter(X[idx_perm2, 0], X[idx_perm2, 1], c = np.arange(len(idx_perm2)))
    plt.show()
    """
    
    """
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 10)
    plt.scatter(Y[:, 0], Y[:, 1], 40)
    plt.subplot(122)
    plt.imshow(D, aspect = 'auto')
    plt.show()
    """


if __name__ == '__main__':
    testGreedyPermEuclidean()

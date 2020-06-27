"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To provide tools for quickly computing all pairs self-similarity
and cross-similarity matrices, for doing "greedy permutations," and for
some topological operations like adding cocycles and creating partitions of unity
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.sparse as sparse

"""#########################################
   Self-Similarity And Cross-Similarity
#########################################"""

def get_csm(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points
    Returns
    ------
    D : ndarray (M, N)
        An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def get_csm_projarc(X, Y):
    """
    Return the projective arc length cross-similarity between two point
    clouds specified as points on the sphere
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points on RP^{d-1}
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points on RP^{d-1}
    Returns
    ------
    D : ndarray (M, N)
        An MxN  cross-similarity matrix
    """
    D = np.abs(X.dot(Y.T))
    D[D < -1] = -1
    D[D > 1] = 1
    D = np.arccos(np.abs(D))
    return D

def get_ssm(X):
    return get_csm(X, X)


"""#########################################
         Greedy Permutations
#########################################"""

def get_greedy_perm_pc(X, M, verbose = False, csm_fn = get_csm):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a point cloud specified in Euclidean space.  This saves 
    computation over having compute the full distance matrix if the number
    of landmarks M << N
    
    Parameters
    ----------
    X : ndarray (N, d) 
        An Nxd Euclidean point cloud
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress
    csm_fn: function X, Y -> D
        Cross-similarity function (Euclidean by default)

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
    ds = csm_fn(X[0, :][None, :], X).flatten()
    D = np.zeros((M, N))
    D[0, :] = ds
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        thisds = csm_fn(X[idx, :][None, :], X).flatten()
        D[i, :] = thisds
        ds = np.minimum(ds, thisds)
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas, 'D':D}

def get_greedy_perm_dm(D, M, verbose = False):
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

def test_greedy_perm_euclidean():
    t = np.linspace(0, 2*np.pi, 10000)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    res = get_greedy_perm_euclidean(X, 50, True)
    Y, D = res['Y'], res['D']
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 10)
    plt.scatter(Y[:, 0], Y[:, 1], 40)
    plt.subplot(122)
    plt.imshow(D, aspect = 'auto')
    plt.show()




"""#########################################
        Cohomology Utility Functions
#########################################"""

def add_cocycles(c1, c2, p = 2, real = False):
    S = {}
    c = np.concatenate((c1, c2), 0)
    for k in range(c.shape[0]):
        [i, j, v] = c[k, :]
        i, j = min(i, j), max(i, j)
        if not (i, j) in S:
            S[(i, j)] = v
        else:
            S[(i, j)] += v
    cret = np.zeros((len(S), 3))
    cret[:, 0:2] = np.array([s for s in S])
    cret[:, 2] = np.array([np.mod(S[s], p) for s in S])
    dtype = np.int64
    if real:
        dtype = np.float32
    cret = np.array(cret[cret[:, -1] > 0, :], dtype = dtype)
    return cret

def make_delta0(R):
    """
    Return the delta0 coboundary matrix
    :param R: NEdges x 2 matrix specifying edges, where orientation
    is taken from the first column to the second column
    R specifies the "natural orientation" of the edges, with the
    understanding that the ranking will be specified later
    It is assumed that there is at least one edge incident
    on every vertex
    """
    NVertices = int(np.max(R) + 1)
    NEdges = R.shape[0]
    
    #Two entries per edge
    I = np.zeros((NEdges, 2))
    I[:, 0] = np.arange(NEdges)
    I[:, 1] = np.arange(NEdges)
    I = I.flatten()
    
    J = R[:, 0:2].flatten()
    
    V = np.zeros((NEdges, 2))
    V[:, 0] = -1
    V[:, 1] = 1
    V = V.flatten()
    I = np.array(I, dtype=int)
    J = np.array(J, dtype=int)
    Delta = sparse.coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
    return Delta

def reindex_cocycles(cocycles, idx_land, N):
    """
    Convert the indices of a set of cocycles to be relative
    to a list of indices in a greedy permutation
    Parameters
    ----------
    cocycles: list of list of ndarray
        The cocycles
    idx_land: ndarray(M, dtype=int)
        Indices of the landmarks in the greedy permutation, with
        respect to all points
    N: int
        Number of total points
    """
    idx_map = -1*np.ones(N, dtype=int)
    idx_map[idx_land] = np.arange(idx_land.size)
    for ck in cocycles:
        for c in ck:
            c[:, 0:-1] = idx_map[c[:, 0:-1]]


"""#########################################
        Partition of Unity Functions
#########################################"""

def partunity_linear(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return r_cover - ds

def partunity_quadratic(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return (r_cover - ds)**2

def partunity_exp(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return np.exp(r_cover**2/(ds**2-r_cover**2))

if __name__ == '__main__':
    test_greedy_perm_euclidean()

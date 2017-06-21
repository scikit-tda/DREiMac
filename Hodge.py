import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
import time
import jellyfish
from CliqueAlgorithms import *


def getKendallTau(order1, order2):
    """
    Given two global rankings, return the Kendall Tau Score
    """
    N = len(order1)
    rank1 = np.zeros(N)
    rank1[order1] = np.arange(N)
    rank2 = np.zeros(N)
    rank2[order2] = np.arange(N)
    A = np.sign(rank1[None, :] - rank1[:, None])
    B = np.sign(rank2[None, :] - rank2[:, None])
    return np.sum(A*B)/float(N*(N-1))
    #tau, p_value = scipy.stats.kendalltau(rank1, rank2)


def getJWDistance(order1, order2):
    """
    Given two global rankings, return the Jaro Winkler Distance
    """
    s = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    s1 = u""
    s2 = u""
    for i in range(len(order1)):
        s1 += s[order1[i]]
    for i in range(len(order2)):
        s2 += s[order2[i]]
    return jellyfish.jaro_winkler(s1, s2)
    

def makeDelta0(R):
    """
    Return the delta0 coboundary matrix
    :param R: NEdges x 2 matrix specifying edges, where orientation
    is taken from the first column to the second column
    R specifies the "natural orientation" of the edges, with the
    understanding that the ranking will be specified later
    It is assumed that there is at least one edge incident
    on every vertex
    """
    NVertices = np.max(R) + 1
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
    
    Delta = sparse.coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
    return Delta
    

def makeDelta1(R):
    """Make the delta1 coboundary matrix
    :param R: Edge list NEdges x 2. It is assumed that 
    there is at least one edge incident on every vertex
    """
    NEdges = R.shape[0]
    NVertices = int(np.max(R))+1
    #Make a list of edges for fast lookup
    Edges = []
    for i in range(NVertices):
        Edges.append({})
    for i in range(R.shape[0]):
        [a, b] = [int(R[i, 0]), int(R[i, 1])]
        Edges[a][b] = i
        Edges[b][a] = i    
    
    tic = time.time()
    (I, J, V) = get3CliquesBrute(Edges)
    toc = time.time()
    print "Elapsed time 3 cliques brute: ", toc - tic
    [I, J, V] = [a.flatten() for a in [I, J, V]]
    TriNum = len(I)/3
    Delta1 = sparse.coo_matrix((V, (I, J)), shape = (TriNum, NEdges)).tocsr()
    
    return Delta1


def doHodge(R, W, Y, verbose = False):
    """
    Given 
    :param R: NEdges x 2 matrix specfiying comparisons that have been made
    :param W: A flat array of NEdges weights parallel to the rows of R
    :param Y: A flat array of NEdges specifying preferences
    :returns: (s, I, H): s is scalar function, I is local inconsistency vector,
        H is global inconsistency vector
    """
    #Step 1: Get s
    if verbose:
        print "Making Delta0..."
    tic = time.time()
    D0 = makeDelta0(R)
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc-tic, " seconds"
    wSqrt = np.sqrt(W).flatten()
    WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
    WSqrtRecip = scipy.sparse.spdiags(1/wSqrt, 0, len(W), len(W))
    A = WSqrt*D0
    b = WSqrt.dot(Y)
    s = lsqr(A, b)[0]
    
    #Step 2: Get local inconsistencies
    if verbose:
        print "Making Delta1..."
    tic = time.time()
    D1 = makeDelta1(R)
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc-tic, " seconds"
    B = WSqrtRecip*D1.T
    resid = Y - D0.dot(s)  #This has been verified to be orthogonal under <resid, D0*s>_W
    
    u = wSqrt*resid
    if verbose:
        print "Solving for Phi..."
    tic = time.time()
    Phi = lsqr(B, u)[0]
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc - tic, " seconds"
    I = WSqrtRecip.dot(B.dot(Phi)) #Delta1* dot Phi, since Delta1* = (1/W) Delta1^T
    
    #Step 3: Get harmonic cocycle
    H = resid - I
    return (s, I, H)

def getWNorm(X, W):
    return np.sqrt(np.sum(W*X*X))


#Do an experiment with a full 4-clique to make sure 
#that delta0 and delta1 look right
if __name__ == '__main__':
    np.random.seed(10)
    N = 600
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I    
    #R = R[np.random.permutation(R.shape[0])[0:R.shape[0]/2], :]
    makeDelta1(R)
    #print makeDelta0(R).toarray()
    #print makeDelta1(R).toarray()

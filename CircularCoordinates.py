import subprocess
import os
import numpy as np
import scipy
from scipy.sparse.linalg import lsqr
import time
import matplotlib.pyplot as plt
from CSMSSMTools import getSSM, getGreedyPermDM, getGreedyPermEuclidean
from TDAUtils import add_cocycles, makeDelta0
from ripser import Rips


def CircularCoords(P, n_landmarks, distance_matrix = False, perc = 0.99, \
                prime = 41, maxdim = 1, do_weighted = False, cocycle_idx = [0], verbose = False):
    """
    Perform circular coordinates via persistent cohomology of 
    sparse filtrations (Jose Perea 2018)
    Parameters
    ----------
    P : ndarray (n_data, d)
        n_data x d array of points
    n_landmarks : integer
        Number of landmarks to sample
    distance_matrix : boolean
        If true, then X is a distance matrix, not a Euclidean point cloud
    perc : float
        Percent coverage
    prime : int
        Field coefficient with which to compute rips on landmarks
    maxdim : int
        Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
        but it may be of interest to see other dimensions (e.g. for a torus)
    do_weighted : boolean
        Whether to make a weighted cocycle on the representatives
    cocycle_idx : list
        Add the cocycles together, sorted from most to least persistent
    verbose : boolean
        Whether to print detailed information during the computation
    """
    n_data = P.shape[0]
    rips = Rips(coeff=prime, maxdim=maxdim, do_cocycles=True)
    
    # Step 1: Compute greedy permutation
    tic = time.time()
    if distance_matrix:
        res = getGreedyPermDM(P, n_landmarks, verbose)
        perm, dist_land_data = res['perm'], res['DLandmarks']
        dist_land_land = P[perm, :]
        dist_land_land = dist_land_land[:, perm]
    else:    
        res = getGreedyPermEuclidean(P, n_landmarks, verbose)
        Y, dist_land_data = res['Y'], res['D']
        dist_land_land = getSSM(Y)
    if verbose:
        print("Elapsed time greedy permutation: %.3g seconds"%(time.time() - tic))
    np.fill_diagonal(dist_land_land, 0)


    # Step 2: Compute H1 with cocycles on the landmarks
    tic = time.time()
    dgms = rips.fit_transform(dist_land_land, distance_matrix=True)
    dgm1 = dgms[1]
    dgm1 = dgm1/2.0 #Need so that Cech is included in rips
    if verbose:
        print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
    idx_p1 = np.argsort(dgm1[:, 0] - dgm1[:, 1])
    cohomdeath = -np.inf
    cohombirth = np.inf
    cocycle = np.zeros((0, 3))
    for k in range(len(cocycle_idx)):
        cocycle = add_cocycles(cocycle, rips.cocycles_[1][idx_p1[cocycle_idx[k]]], p=prime)
        cohomdeath = max(cohomdeath, dgm1[idx_p1[cocycle_idx[k]], 0])
        cohombirth = min(cohombirth, dgm1[idx_p1[cocycle_idx[k]], 1])

    # Step 3: Determine radius for balls
    coverage = np.max(np.min(dist_land_data, 1))
    r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
    if verbose:
        print("r_cover = %.3g"%r_cover)
    

    # Step 4: Setup coboundary matrix, delta_0, for Cech_{r_cover }
    # and use it to find the harmonic cocycle
    #Lift to integer cocycle
    val = np.array(cocycle[:, 2])
    val[val > (prime-1)/2] -= prime
    Y = np.zeros((n_landmarks, n_landmarks))
    Y[cocycle[:, 0], cocycle[:, 1]] = val
    Y = Y + Y.T
    #Select edges that are under the threshold
    [I, J] = np.meshgrid(np.arange(n_landmarks), np.arange(n_landmarks))
    I = I[np.triu_indices(n_landmarks, 1)]
    J = J[np.triu_indices(n_landmarks, 1)]
    Y = Y[np.triu_indices(n_landmarks, 1)]
    idx = np.arange(len(I))
    idx = idx[dist_land_land[I, J] < 2*r_cover]
    I = I[idx]
    J = J[idx]
    Y = Y[idx]

    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I
    #Make a flat array of NEdges weights parallel to the rows of R
    if do_weighted:
        W = dist_land_land[I, J]
    else:
        W = np.ones(NEdges)
    delta_0 = makeDelta0(R)
    wSqrt = np.sqrt(W).flatten()
    WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
    A = WSqrt*delta_0
    b = WSqrt.dot(Y)
    tau = lsqr(A, b)[0]
    theta = np.zeros((NEdges, 3))
    theta[:, 0] = J
    theta[:, 1] = I
    theta[:, 2] = -delta_0.dot(tau)
    theta = add_cocycles(cocycle, theta, real=True)
    

    # Step 5: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

    # Let U_j be the set of data points whose distance to l_j is less than
    # r_cover
    U = dist_land_data < r_cover
    # Compute subordinated partition of unity varphi_1,...,varphi_{s+1}
    # Compute the bump phi_j(b) on each data point b in U_j. phi_j = 0 outside U_j.
    phi = np.zeros_like(dist_land_data)
    phi[U] = r_cover - dist_land_data[U]

    # Compute the partition of unity varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{s+1}(b))
    varphi = phi / np.sum(phi, 0)[None, :]

    # To each data point, associate the index of the first open set it belongs to
    ball_indx = np.argmax(U, 0)

    # Step 6: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map

    # compute all transition functions
    theta_matrix = np.zeros((n_landmarks, n_landmarks))
    I = np.array(theta[:, 0], dtype = np.int64)
    J = np.array(theta[:, 1], dtype = np.int64)
    theta = theta[:, 2]
    theta = np.mod(theta + 0.5, 1) - 0.5
    theta_matrix[I, J] = theta
    theta_matrix[J, I] = -theta
    class_map = -tau[ball_indx]
    for i in range(n_data):
        class_map[i] += theta_matrix[ball_indx[i], :].dot(varphi[:, i])    
    thetas = np.mod(2*np.pi*class_map, 2*np.pi)

    res["cocycle"] = cocycle
    res["dist_land_land"] = dist_land_land
    res["dist_land_data"] = dist_land_data
    res["dgm1"] = dgm1
    res["idx_p1"] = idx_p1
    res["rips"] = rips
    res["thetas"] = thetas
    res["r_cover"] = r_cover
    res["prime"] = prime
    res["tau"] = tau
    return res



def doTwoCircleTest():
    from ripser import Rips
    prime = 41
    np.random.seed(2)
    N = 500
    X = np.zeros((N*2, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    X[N::, 0] = 2*np.cos(t) + 4
    X[N::, 1] = 2*np.sin(t) + 4
    X = X[np.random.permutation(X.shape[0]), :]
    X = X + 0.2*np.random.randn(X.shape[0], 2)
    
    plt.figure(figsize=(12, 5))
    for i in range(2):
        res = CircularCoords(X, 100, prime = prime, cocycle_idx = [i])
        res["X"] = X
        import scipy.io as sio
        I = 2*res["dgm1"][res["idx_p1"][i], :]
        plt.clf()
        plt.subplot(121)
        res["rips"].plot(show=False)
        plt.scatter(I[0], I[1], 20, 'r')
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], 100, res['thetas'], cmap = 'afmhot', edgecolor = 'none')
        plt.axis('equal')
        plt.colorbar()
        plt.savefig("Cocycle%i.svg"%i, bbox_inches = 'tight')
        res.pop("rips")
        sio.savemat("JoseMatlabCode/Circ.mat", res)

if __name__ == '__main__':
    doTwoCircleTest()

import subprocess
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from Hodge import *
from CSMSSMTools import getSSM, getGreedyPermDM, getGreedyPermEuclidean
from TDAUtils import add_cocycles


def CircularCoords(P, n_landmarks, distance_matrix = False, perc = 0.99, \
                prime = 41, cocycle_idx = [0], verbose = False):
    """
    Perform multiscale projective coordinates via persistent cohomology of 
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
    cocycle_idx : list
        Add the cocycles together, sorted from most to least persistent
    prime : int
        Field coefficient with which to compute rips on landmarks
    verbose : boolean
        Whether to print detailed information during the computation
    """
    n_data = P.shape[0]
    rips = Rips(coeff=2, maxdim=1, do_cocycles=True)
    
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



    # Step 2: Compute H1 with cocycles on the landmarks
    tic = time.time()
    dgms = rips.fit_transform(dist_land_land, distance_matrix=True)
    dgm1 = dgms[1]
    dgm1 = dgm1/2.0 #Need so that Cech is included in rips
    if verbose:
        print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
    idx_p1 = np.argsort(dgm1[:, 0] - dgm1[:, 1])
    cocycle = rips.cocycles_[1][idx_p1[cocycle_idx[0]]]
    cohomdeath = -np.inf
    cohombirth = np.inf
    cocycle = np.zeros((0, 3))
    for k in range(len(cocycle_idx)):
        cocycle = add_cocycles(cocycle, rips.cocycles_[1][idx_p1[cocycle_idx[k]]])
        cohomdeath = max(cohomdeath, dgm1[idx_p1[cocycle_idx[k]], 0])
        cohombirth = min(cohombirth, dgm1[idx_p1[cocycle_idx[k]], 1])

    # Step 3: Determine radius for balls ( = interpolant btw data coverage and cohomological birth)
    coverage = np.max(np.min(dist_land_data, 1))
    r_birth = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
    if verbose:
        print("r_birth = %.3g"%r_birth)
    

    # Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

    # Let U_j be the set of data points whose distance to l_j is less than
    # r_birth
    U = dist_land_data < r_birth
    # Compute subordinated partition of unity varphi_1,...,varphi_{s+1}
    # Compute the bump phi_j(b) on each data point b in U_j. phi_j = 0 outside U_j.
    phi = np.zeros_like(dist_land_data)
    phi[U] = r_birth - dist_land_data[U]

    # Compute the partition of unity varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{s+1}(b))
    varphi = phi / np.sum(phi, 0)[None, :]

    # To each data point, associate the index of the first open set it belongs to
    indx = np.argmax(U, 0)

    # Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map

    # compute all transition functions
    cocycle_matrix = np.ones((n_landmarks, n_landmarks))
    cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
    cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
    class_map = np.sqrt(varphi.T)
    for i in range(n_data):
        class_map[i, :] *= cocycle_matrix[indx[i], :]
    
    res = PPCA(class_map, proj_dim, verbose)
    res["cocycle"] = cocycle[:, 0:2]
    res["dist_land_land"] = dist_land_land
    res["dist_land_data"] = dist_land_data
    res["dgm1"] = dgm1
    res["rips"] = rips
    return res



def doTwoCircleTest():
    from ripser import Rips
    p = 41
    r = Rips(coeff=p, do_cocycles=True, thresh = 3.0)
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
    D = getSSM(X)
    """
    fout = open("noisycircles.txt", "w")
    for i in range(D.shape[0]):
        for j in range(0, i):
            fout.write("%g"%D[i, j])
            if j < i-1:
                fout.write(",")
            else:
                fout.write("\n")
    fout.close()
    """
    
    tic = time.time()
    PDs = r.fit_transform(X)
    print("Elapsed Time Rips: %g"%(time.time() - tic))
    cocycles = r.cocycles_[1]
    
    plt.figure(figsize=(12, 5))
    for i in range(PDs[1].shape[0]):
        pers = PDs[1][i, 1]-PDs[1][i, 0]
        if pers < 0.5:
            continue
        ccl = cocycles[i]
        thresh = PDs[1][i, 1] - 0.01
        (s, cclret) = getCircularCoordinates(X, ccl, p, thresh)
        
        plt.clf()
        plt.subplot(121)
        r.plot(diagrams=PDs, show=False)
        plt.scatter([PDs[1][i, 0]], [PDs[1][i, 1]], 80, 'k')
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], 100, s, cmap = 'afmhot', edgecolor = 'none')
        plt.colorbar()
        plt.savefig("Cocycle%i.svg"%i, bbox_inches = 'tight')

if __name__ == '__main__':
    doTwoCircleTest()

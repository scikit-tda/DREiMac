import subprocess
import os
import numpy as np
import scipy
from scipy.sparse.linalg import lsqr
import time
import matplotlib.pyplot as plt
from TDAUtils import add_cocycles, makeDelta0
from ripser import ripser


class CircularCoords(object):
    def __init__(self, X, n_landmarks, prime=41, maxdim=1, verbose=False):
        """
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        assert(maxdim >= 1)
        self.verbose = verbose
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        res = ripser(X, coeff=prime, maxdim=maxdim, n_perm=n_landmarks, do_cocycles=True)
        if verbose:
            print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
        self.X_ = X
        self.prime_ = prime
        self.dgms_ = res['dgms']
        self.dist_land_data_ = res['dperm2all']
        self.idx_perm_ = res['idx_perm']
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_perm_]
        self.cocycles_ = res['cocycles']
        self.n_landmarks_ = n_landmarks

    def fit_transform(self, perc = 0.99, do_weighted = False, cocycle_idx = [0]):
        """
        Perform circular coordinates via persistent cohomology of 
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        do_weighted : boolean
            Whether to make a weighted cocycle on the representatives
        cocycle_idx : list
            Add the cocycles together, sorted from most to least persistent
        """
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        idx_p1 = np.argsort(dgm1[:, 0] - dgm1[:, 1])
        cohomdeath = -np.inf
        cohombirth = np.inf
        cocycle = np.zeros((0, 3))
        prime = self.prime_
        for k in range(len(cocycle_idx)):
            cocycle = add_cocycles(cocycle, self.cocycles_[1][idx_p1[cocycle_idx[k]]], p=prime)
            cohomdeath = max(cohomdeath, dgm1[idx_p1[cocycle_idx[k]], 0])
            cohombirth = min(cohombirth, dgm1[idx_p1[cocycle_idx[k]], 1])

        # Step 3: Determine radius for balls
        dist_land_data = self.dist_land_data_
        dist_land_land = self.dist_land_land_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        if self.verbose:
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

        res = {}
        res['dgms'] = self.dgms_
        res["cocycle"] = cocycle
        res["dist_land_land"] = dist_land_land
        res["dist_land_data"] = dist_land_data
        res["dgm1"] = dgm1
        res["idx_p1"] = idx_p1
        res["thetas"] = thetas
        res["r_cover"] = r_cover
        res["prime"] = prime
        res["tau"] = tau
        res["perm"] = self.idx_perm_
        return res



def doTwoCircleTest():
    from persim import plot_diagrams as plot_dgms
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
    
    c = CircularCoords(X, 100, prime = prime)

    plt.figure(figsize=(12, 5))
    for i in range(2):
        res = c.fit_transform(cocycle_idx = [i])
        print(res["perm"])
        import scipy.io as sio
        I = 2*res["dgm1"][res["idx_p1"][i], :]
        plt.clf()
        plt.subplot(121)
        plot_dgms(res["dgms"], show=False)
        plt.scatter(I[0], I[1], 20, 'r')
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], 100, res['thetas'], cmap = 'afmhot', edgecolor = 'none')
        plt.axis('equal')
        plt.colorbar()
        plt.savefig("Cocycle%i.svg"%i, bbox_inches = 'tight')

if __name__ == '__main__':
    doTwoCircleTest()

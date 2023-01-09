import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import scipy
from scipy.sparse.linalg import lsqr
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, RadioButtons
from .geomtools import *
from .emcoords import *
from ripser import ripser
import warnings


"""#########################################
        Main Circular Coordinates Class
#########################################"""
SCATTER_SIZE = 50

class CircularCoords(EMCoords):
    def __init__(self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "circ"

    def get_coordinates(self, perc = 0.99, do_weighted = False, cocycle_idx = [0], partunity_fn = PartUnity.linear):
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
            Add the cocycles together in this list
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        """
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        cohomdeath = -np.inf
        cohombirth = np.inf
        cocycle = np.zeros((0, 3))
        prime = self.prime_
        for k in range(len(cocycle_idx)):
            cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
            cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
            cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])

        ## Step 2: Determine radius for balls
        dist_land_data = self.dist_land_data_
        dist_land_land = self.dist_land_land_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        self.r_cover_ = r_cover # Store covering radius for reference
        if self.verbose:
            print("r_cover = %.3g"%r_cover)
        

        ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
        ## and use it to find a projection of the cocycle
        ## onto the image of delta0

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
        delta0 = make_delta0(R)
        wSqrt = np.sqrt(W).flatten()
        WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
        A = WSqrt*delta0
        b = WSqrt.dot(Y)
        tau = lsqr(A, b)[0]
        theta = np.zeros((NEdges, 3))
        theta[:, 0] = J
        theta[:, 1] = I
        theta[:, 2] = -delta0.dot(tau)
        theta = add_cocycles(cocycle, theta, real=True)
        

        ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity 
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are %i point not covered by a landmark"%nzero)
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]

        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)

        ## Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        
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

        return thetas
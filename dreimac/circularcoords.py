import numpy as np
import scipy
from scipy.sparse.linalg import lsqr
from .utils import *
from .emcoords import *


"""#########################################
        Main Circular Coordinates Class
#########################################"""

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
        EMCoords.__init__(self, X=X, n_landmarks=n_landmarks, distance_matrix=distance_matrix, prime=prime, maxdim=maxdim, verbose=verbose)
        self.type_ = "circ"

    def get_cocycle_projection(self, cocycle, r_cover, do_weighted):
        """
        Setup coboundary matrix, delta_0, for Cech_{r_cover }
        and use it to find a projection of the cocycle
        onto the image of delta0
        
        Parameters
        ----------
        cocycle: ndarray(K, 3, dtype=int)
            Representative cocycle.  First two columns are vertex indices,
            and third column is value in field of prime self.prime_
        r_cover: float
            Covering radius
        do_weighted : boolean
            Whether to make a weighted cocycle on the representatives
        
        Returns
        -------
        tau: ndarray(n_landmarks)
            Scalar function on the landmarks whose image under delta0 best matches cocycle
        theta: ndarray(n_edges)
            The image of tau under delta0
        """
        prime = self.prime_
        n_landmarks = self.n_landmarks_
        dist_land_land = self.dist_land_land_
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
        return tau, theta

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
            Add the cocycles together at the indices in this list
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        
        Returns
        -------
        thetas: ndarray(N)
            Circular coordinates
        """
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        cohomdeath, cohombirth, cocycle = EMCoords.get_rep_cocycle(self, cocycle_idx)
        

        ## Step 2: Determine radius for balls
        r_cover = EMCoords.get_cover_radius(self, perc, cohomdeath, cohombirth)
        

        ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
        ## and use it to find a projection of the cocycle
        ## onto the image of delta0
        tau, theta = self.get_cocycle_projection(cocycle, r_cover, do_weighted)
        

        ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

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
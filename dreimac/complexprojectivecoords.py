import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp
from .utils import PartUnity, CircleMapUtils
from .emcoords import *
import warnings


class ComplexProjectiveCoords(EMCoords):
    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=2, verbose=False
    ):
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
            Maximum dimension of homology.  Only dimension 2 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "complexprojective"

    def get_coordinates(
        self,
        perc=0.99,
        cohomology_class=0,
        partunity_fn=PartUnity.linear,
        check_and_fix_cocycle_condition=True,
    ):
        """

        TODO
        
        Parameters
        ----------
        perc : float
            Percent coverage
        inner_product : string
            Either 'uniform' or 'exponential'
        cohomology_class : integer
            TODO: explain
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function

        Returns
        -------
        thetas: ndarray(n, N)
            TODO
        """


        # TODO: generalize
        data = self.X_
        landmarks = np.array(
            [
                [0, 0, 1],
                [2 * np.sqrt(2) / 3, 0, -1 / 3],
                [-np.sqrt(2) / 3, np.sqrt(2 / 3), -1 / 3],
                [-np.sqrt(2) / 3, -np.sqrt(2 / 3), -1 / 3],
            ]
        )

        # get representative cocycle
        # TODO: generalize
        cohomdeath_rips, cohombirth_rips, cocycle = (
            1.92,
            4,
            np.array(
                [[1, 2, 3, 1],
                 [1, 2, 4, 0],
                 [1, 3, 4, 0],
                 [2, 3, 4, 0]]
            )
        )

        ##homological_dimension = 2
        ##cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(cohomology_class,homological_dimension)



        # determine radius for balls
        USE_CECH = True
        if USE_CECH:
            r_cover = EMCoords.get_cover_radius(
                self, perc, cohomdeath_rips, cohombirth_rips
            )
            threshold = 2 * r_cover
        else:
            r_cover = EMCoords.get_cover_radius(
                self, perc, cohomdeath_rips, cohombirth_rips * 2
            )
            threshold = r_cover

        self.threshold_ = threshold

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)



        # compute boundary matrix
        # TODO: generalize

        # NOTE: taking minimum between numbers and 1 because
        # when number is slighlty larger than 1 get nan with arccos
        dist_land_land = np.arccos(np.minimum(landmarks @ landmarks.T, 1))

        ##dist_land_land = self.dist_land_land_

        dist_land_data = np.arccos(np.minimum(landmarks @ data.T, 1))

        ##dist_land_data = self.dist_land_data_

        delta0, edge_pair_to_row_index = CohomologyUtils.make_delta0(dist_land_land, threshold)
        delta1, simplex_to_vector_index = CohomologyUtils.make_delta1(dist_land_land, edge_pair_to_row_index, threshold)

        # lift to integer cocycles
        integer_cocycle = CohomologyUtils.lift_to_integer_cocycle(cocycle, prime=self.prime_)
        integer_cocycle_as_vector = CohomologyUtils.sparse_cocycle_to_vector(integer_cocycle, simplex_to_vector_index,int)

        # integrate cocycle
        nu = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ nu
       
       
        N, d = data.shape
        s = landmarks.shape[0]
        #s = self.n_landmarks_


        #### harcoded cover and partition of unity
        r = np.sort(dist_land_land, axis=1)[:, 1]
        import numpy.matlib
        U = dist_land_data < np.matlib.repmat(r, N, 1).T
        ##print("U", U)
        
        varphi = np.zeros((s, N))
        for j in range(0, s):
            varphi[j, U[j, :]] = (r[j] - dist_land_data[j, U[j, :]]) ** 2
        sum_phi = np.sum(varphi, axis=0)
        varphi = varphi / sum_phi[np.newaxis, :]
        
        indx = np.zeros(N, dtype=int)
        
        for j in range(N):
            indx[j] = np.argwhere(U[:, j])[0][0]
        
        # NOTE: the cover is not great
        #print("idx counts", sum(indx[indx==0].shape),sum(indx[indx==1].shape),sum(indx[indx==2].shape))
        ##print("indx", indx)
        ####
        
        
        h = np.zeros((s,s,N), dtype=complex)
        
        for j in range(s):
            for k in range(s):
                unordered_simplex = np.array([j,k],dtype=int)
                ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
                if ordered_simplex in edge_pair_to_row_index:
                    nu_val = sign * nu[edge_pair_to_row_index[ordered_simplex]]
                else:
                    nu_val = 0

                theta_average = 0
                for l in range(s):
                    unordered_simplex = np.array([j,k,l],dtype=int)
                    ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
                    if ordered_simplex in simplex_to_vector_index:
                        theta_average += sign * harmonic_representative[simplex_to_vector_index[ordered_simplex]] * varphi[l]

                h[j,k] = np.exp( 2 * np.pi * 1j * (theta_average + nu_val))
                
        class_map = np.array(np.sqrt(varphi),dtype=complex)
        
        for j in range(N):
            h_k_ind_j = h[:,indx[j]]
            class_map[:,j] = class_map[:,j] * np.conjugate(h_k_ind_j[:,j])
            
        ##print("class map ", class_map)
        
        X = class_map
        #variance = np.zeros(X.shape[0])
        # dimension of projective space to project onto
        proj_dim = 1
        
        #for i in range(class_map.shape[0]-proj_dim-1):
        for i in [1,2]:
            UU, S, _ = np.linalg.svd(X)
            ##print("singular vals", S)
            #print("norm UU", np.linalg.norm(UU))
            ##print("norm X", np.linalg.norm(X,axis=1))
            #variance[-i] = np.mean(
            #    (np.pi/2 - np.arccos(np.abs(UU[:,-1].T @ X)))**2
            #)
            Y = np.conjugate(UU.T) @ X
            y = Y[-1,:]
            ##print(np.linalg.norm(y))
            Y = Y[:-1,:]
            #print(Y.shape)
            ##print(np.sqrt( 1 - np.abs(y)**2 ).shape)
            X = np.divide(Y, np.sqrt( 1 - np.abs(y)**2 ))
            ##print("X", X)
            
        Z = np.zeros((2 * X.shape[0], X.shape[1]))
        Z[::2,:] = np.real(X)
        Z[1::2,:] = np.imag(X)
        projData = Z
            
        #XX = X
        #print(np.linalg.norm(XX))
        
        #for j in [1]:
        #    UU, _, _ = np.linalg.svd(XX)
        #    print("norm UU", np.linalg.norm(UU))
        #    #variance[-i] = np.mean(
        #    #    (np.pi/2 - np.arccos(np.abs(UU[:,-1].T @ X)))**2
        #    #)
        #    Y = UU.T @ XX
        #    #print(np.linalg.norm(Y,axis=1))
        #    y = Y[-1,:]
        #    #print(np.linalg.norm(y))
        #    Y = Y[:-1,:]
        #    XX = Y / np.sqrt( 1 - np.abs(y)**2 )

        return projData


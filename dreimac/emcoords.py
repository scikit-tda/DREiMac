"""
A superclass for shared code across all different types of coordinates
"""
import numpy as np
from scipy.sparse.linalg import lsqr
import time
from .utils import CohomologyUtils
from ripser import ripser
import warnings


class EMCoords(object):
    def __init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose):
        """
        Perform persistent homology on the landmarks, store distance
        from the landmarks to themselves and to the rest of the points,
        and sort persistence diagrams and cocycles in decreasing order of persistence

        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
            TODO: describe None case
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
            TODO: describe non-square distance matrix case
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology. 

        """
        assert maxdim >= 1
        self.verbose = verbose
        self.X_ = X
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        if distance_matrix is False:
            ripser_metric_input = X 
        elif X.shape[0] == X.shape[1]:
            ripser_metric_input = X 
        else:
            ripser_metric_input = X[:,:X.shape[0]]
        res = ripser(
            ripser_metric_input,
            distance_matrix=distance_matrix,
            coeff=prime,
            maxdim=maxdim,
            n_perm=n_landmarks,
            do_cocycles=True,
        )
        if verbose:
            print("Elapsed time persistence: %.3g seconds" % (time.time() - tic))
        self.prime_ = prime
        self.dgms_ = res["dgms"]
        self.idx_land_ = res["idx_perm"]
        self.n_landmarks_ = len(self.idx_land_)
        #self.dist_land_data_ = res["dperm2all"]
        if distance_matrix is False:
            self.dist_land_data_ = res["dperm2all"]
        elif X.shape[0] == X.shape[1]:
            self.dist_land_data_ = res["dperm2all"]
        else:
            self.dist_land_data_ = X[self.idx_land_,:]
        self.coverage_ = np.max(np.min(self.dist_land_data_, 1))
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
        self.cocycles_ = res["cocycles"]
        # Sort persistence diagrams in descending order of persistence
        for i in range(1, maxdim+1):
            idxs = np.argsort(self.dgms_[i][:, 0] - self.dgms_[i][:, 1])
            self.dgms_[i] = self.dgms_[i][idxs, :]
            dgm_lifetime = np.array(self.dgms_[i])
            dgm_lifetime[:, 1] -= dgm_lifetime[:, 0]
            self.cocycles_[i] = [self.cocycles_[i][idx] for idx in idxs]
        CohomologyUtils.reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])

        self.type_ = "emcoords"

    def get_representative_cocycle(self, cohomology_class, homological_dimension):
        """
        Compute the representative cocycle, given a list of cohomology classes

        Parameters
        ----------
        cohomology_class : integer
            Integer representing the index of the persistent cohomology class.
            Persistent cohomology classes are ordered by persistence, from largest to smallest.

        Returns
        -------
        cohomdeath: float
            Cohomological death of the linear combination or single cocycle
        cohombirth: float
            Cohomological birth of the linear combination or single cocycle
        cocycle: ndarray(K, homological_dimension+2, dtype=int)
            Representative cocycle. First homological_dimension+1 columns are vertex indices,
            and last column takes values in finite field corresponding to self.prime_
        """

        assert isinstance(cohomology_class, int)

        dgm = self.dgms_[homological_dimension]
        cocycles = self.cocycles_[homological_dimension]
        return (
            dgm[cohomology_class, 0],
            dgm[cohomology_class, 1],
            cocycles[cohomology_class],
        )

    def get_cover_radius(self, perc, cohomdeath_rips, cohombirth_rips, standard_range):
        """
        Determine radius for covering balls

        Parameters
        ----------
        perc : float
            Percent coverage
        cohomdeath: float
            Cohomological death
        cohombirth: float
            Cohomological birth
        standard_range: float
            Whether or not to use the range that guarantees that the cohomology class selected
            is a non-trivial cohomology class in the Cech complex. If False, the class is only
            guaranteed to be non-trivial in the Rips complex.

        Returns
        -------
        r_cover : float
        rips_threshold : float

        """
        start = 2*cohomdeath_rips if standard_range else cohomdeath_rips
        end = cohombirth_rips
        if start > end:
            raise Exception(
                "The cohomology class selected is too short, try setting standard_range to False."
            )
        self.rips_threshold_ = (1 - perc) * start + perc * end
        self.r_cover_ = self.rips_threshold_ / 2

        return self.r_cover_, self.rips_threshold_

    def get_covering_partition(self, r_cover, partunity_fn):
        """
        Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

        Parameters
        ----------
        r_cover: float
            Covering radius
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function

        Returns
        -------
        varphi: ndarray(n_data, dtype=float)
            varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b)),
        ball_indx: ndarray(n_data, dtype=int)
            The index of the first open set each data point belongs to
        """
        dist_land_data = self.dist_land_data_
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are {} point not covered by a landmark".format(nzero))
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]
        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)
        return varphi, ball_indx

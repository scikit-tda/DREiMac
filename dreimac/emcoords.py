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
            A point cloud with N points in d dimensions, or a matrix of distances from N points to d points.
            See distance_matrix, below, for a description of the second scenario.
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud.
            If X is square, then the i-th row should represent the same point as the i-th column, meaning
            that the matrix should be symmetric.
            If X is not square, then it should have more columns than rows (i.e., N > d).
            Moreover, if i < N, the i-th row should represent the same point as the i-th column.
            When X is not square, the rows of X are interpreted as a subsample and the columns as all available points; thus
            X represents the distance from the points in the subsample to all available points.
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology. 

        """
        assert maxdim >= 1
        self.verbose = verbose
        self._X = X
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
        self._prime = prime
        self._dgms = res["dgms"]
        # TODO: the following is kept for backwards compatibility, remove in next interface-breaking version
        self.dgms_ = self._dgms
        self._idx_land = res["idx_perm"]
        self._n_landmarks = len(self._idx_land)
        #self.dist_land_data_ = res["dperm2all"]
        if distance_matrix is False:
            self._dist_land_data = res["dperm2all"]
        elif X.shape[0] == X.shape[1]:
            self._dist_land_data = res["dperm2all"]
        else:
            self._dist_land_data = X[self._idx_land,:]
        self._coverage = np.max(np.min(self._dist_land_data, 1))
        self._dist_land_land = self._dist_land_data[:, self._idx_land]
        self._cocycles = res["cocycles"]
        # Sort persistence diagrams in descending order of persistence
        for i in range(1, maxdim+1):
            idxs = np.argsort(self._dgms[i][:, 0] - self._dgms[i][:, 1])
            self._dgms[i] = self._dgms[i][idxs, :]
            dgm_lifetime = np.array(self._dgms[i])
            dgm_lifetime[:, 1] -= dgm_lifetime[:, 0]
            self._cocycles[i] = [self._cocycles[i][idx] for idx in idxs]
        CohomologyUtils.reindex_cocycles(self._cocycles, self._idx_land, X.shape[0])


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
            and last column takes values in finite field corresponding to self._prime.
            The number of rows K is the number of simplices on which the cocycle is non-zero.
        """

        assert isinstance(cohomology_class, int)

        dgm = self._dgms[homological_dimension]
        cocycles = self._cocycles[homological_dimension]
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
        self._rips_threshold = (1 - perc) * start + perc * end
        self._r_cover = self._rips_threshold / 2

        return self._r_cover, self._rips_threshold

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
        dist_land_data = self._dist_land_data
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

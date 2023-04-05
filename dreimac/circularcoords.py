import numpy as np
import scipy
from scipy.sparse.linalg import lsqr
from .utils import *
from .emcoords import *
from .toroidalcoords import ToroidalCoords


"""#########################################
        Main Circular Coordinates Class
#########################################"""


class CircularCoords(ToroidalCoords):
    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False
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
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        ToroidalCoords.__init__(
            self,
            X=X,
            n_landmarks=n_landmarks,
            distance_matrix=distance_matrix,
            prime=prime,
            maxdim=maxdim,
            verbose=verbose,
        )
        self.type_ = "circ"

    def get_coordinates(
        self,
        perc=0.99,
        cocycle_idx=0,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        check_and_fix_cocycle_condition=True,
    ):
        """
        Perform circular coordinates via persistent cohomology of
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        cocycle_idx : integer
            TODO: explain
        partunity_fn : (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            TODO: explain
        check_and_fix_cocycle_condition : bool
            TODO: explain

        Returns
        -------
        thetas : ndarray(N)
            Circular coordinates
        """

        return ToroidalCoords.get_coordinates(
            self,
            perc,
            [cocycle_idx],
            partunity_fn,
            standard_range,
            check_and_fix_cocycle_condition,
        )[0]

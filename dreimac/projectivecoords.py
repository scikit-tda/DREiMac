import numpy as np
import time
from .utils import PartUnity, EquivariantPCA
from .lenscoords import LensCoords


class ProjectiveCoords(LensCoords):
    """
    Object that performs multiscale real projective coordinates via
    persistent cohomology of sparse filtrations (Jose Perea 2018).

    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    n_landmarks: int
        Number of landmarks to use
    distance_matrix: boolean
        If true, treat X as a distance matrix instead of a point cloud
    maxdim : int
        Maximum dimension of homology. Only dimension 1 is needed for real projective coordinates,
        but it may be of interest to see other dimensions (e.g. for a torus)
    partunity_fn: ndarray(n_landmarks, N) -> ndarray(n_landmarks, N)
        A partition of unity function

    """

    def __init__(self, X, n_landmarks, distance_matrix=False, maxdim=1, verbose=False):
        prime = 2
        LensCoords.__init__(
            self,
            X=X,
            n_landmarks=n_landmarks,
            distance_matrix=distance_matrix,
            prime=prime,
            maxdim=maxdim,
            verbose=verbose,
        )
        self.type_ = "proj"

    def get_coordinates(
        self,
        perc=0.9,
        cocycle_idx=0,
        proj_dim=2,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        projective_dim_red_mode="one-by-one"
    ):
        """
        Get real projective coordinates.

        Parameters
        ----------
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : list
            Add the cocycles together, sorted from most to least persistent
        proj_dim : integer
            Dimension down to which to project the data
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a class in the Cech complex.

        Returns
        -------
        {'variance': ndarray(N-1)
            The variance captured by each dimension
        'X': ndarray(N, proj_dim+1)
            The projective coordinates
        }

        """

        return LensCoords.get_coordinates(
            self,
            perc=perc,
            cocycle_idx=cocycle_idx,
            lens_dim=proj_dim+1,
            partunity_fn=partunity_fn,
            standard_range=standard_range,
            projective_dim_red_mode=projective_dim_red_mode,
        )

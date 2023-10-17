import numpy as np
import time
from .utils import PartUnity, EquivariantPCA
from .emcoords import EMCoords

class ProjectiveCoords(EMCoords):
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
        EMCoords.__init__(
            self,
            X=X,
            n_landmarks=n_landmarks,
            distance_matrix=distance_matrix,
            prime=prime,
            maxdim=maxdim,
            verbose=verbose,
        )

    def get_coordinates(
        self,
        perc=0.9,
        cocycle_idx=0,
        proj_dim=2,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        projective_dim_red_mode="exponential",
    ):
        """
        Get real projective coordinates.

        Parameters
        ----------
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : list
            Integer representing the index of the persistent cohomology class
            used to construct the Eilenberg-MacLane coordinate. Persistent cohomology
            classes are ordered by persistence, from largest to smallest.
        proj_dim : integer
            Dimension down to which to project the data.
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a class in the Cech complex.
        projective_dim_red_mode : string
            Either "one-by-one", "exponential", or "direct". How to perform equivariant
            dimensionality reduction. "exponential" usually works best, being fast
            without compromising quality.

        Returns
        -------
        ndarray(N, proj_dim+1)
            The projective coordinates

        """

        n_landmarks = self._n_landmarks

        homological_dimension = 1
        cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
            cocycle_idx, homological_dimension
        )

        r_cover, _ = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        root_of_unity = -1

        cocycle_matrix = np.ones((n_landmarks, n_landmarks), dtype=float)
        cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = root_of_unity ** cocycle[:, 2]
        cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = (1 / root_of_unity) ** cocycle[:,2]

        class_map = np.sqrt(varphi.T) * cocycle_matrix[ball_indx[:], :]

        epca = EquivariantPCA.ppca(
            class_map, proj_dim, projective_dim_red_mode, self.verbose
        )
        self._variance = epca["variance"]

        return epca["X"]

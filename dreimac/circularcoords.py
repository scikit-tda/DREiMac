from .utils import PartUnity
from .toroidalcoords import ToroidalCoords


class CircularCoords(ToroidalCoords):
    """
    Object that performs circular coordinates via persistent cohomology of
    sparse filtrations (Jose Perea 2020).

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
    verbose : bool
        Print debug information.
    """

    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False
    ):
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
        perc=0.5,
        cocycle_idx=0,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        check_cocycle_condition=True,
    ):
        """
        Get circular coordinates.


        Parameters
        ----------
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : integer
            Integer representing the index of the persistent cohomology class
            used to construct the Eilenberg-MacLane coordinate. Persistent cohomology
            classes are ordered by persistence, from largest to smallest.
        partunity_fn : (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a non-trivial class in the Cech complex.
        check_cocycle_condition : bool
            Whether to check, and fix if necessary, that the integer cocycle constructed
            using finite field coefficients satisfies the cocycle condition.

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
            check_cocycle_condition,
        )[0]

import numpy as np
import time
from .utils import PartUnity
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
        Maximum dimension of homology. Only dimension 1 is needed for circular coordinates,
        but it may be of interest to see other dimensions (e.g. for a torus)
    partunity_fn: ndarray(n_landmarks, N) -> ndarray(n_landmarks, N)
        A partition of unity function

    """

    def __init__(self, X, n_landmarks, distance_matrix=False, maxdim=1, verbose=False):
        EMCoords.__init__(
            self,
            X=X,
            n_landmarks=n_landmarks,
            distance_matrix=distance_matrix,
            prime=2,
            maxdim=maxdim,
            verbose=verbose,
        )
        self.type_ = "proj"
        # GUI variables
        self.selected = set([])
        self.u = np.array([0, 0, 1])

    def get_coordinates(
        self,
        perc=0.9,
        cocycle_idx=0,
        proj_dim=2,
        partunity_fn=PartUnity.linear,
        standard_range=True,
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
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        homological_dimension = 1
        cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
            cocycle_idx, homological_dimension
        )

        ## Step 2: Determine radius for balls
        r_cover, _ = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        ## Step 3: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        ## Step 4: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        # compute all transition functions
        cocycle_matrix = np.ones((n_landmarks, n_landmarks))
        cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
        cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
        class_map = np.sqrt(varphi.T)
        for i in range(n_data):
            class_map[i, :] *= cocycle_matrix[ball_indx[i], :]
        res = _ppca(class_map, proj_dim, self.verbose)
        return res


def _ppca(class_map, proj_dim, verbose=False):
    """
    Principal Projective Component Analysis (Jose Perea 2017)

    Parameters
    ----------
    class_map : ndarray (N, d)
        For all N points of the dataset, membership weights to
        d different classes are the coordinates
    proj_dim : integer
        The dimension of the projective space onto which to project
    verbose : boolean
        Whether to print information during iterations

    Returns
    -------
    {'variance': ndarray(N-1)
        The variance captured by each dimension
    'X': ndarray(N, proj_dim+1)
        The projective coordinates
    }

    """
    if verbose:
        print(
            "Doing ppca on %i points in %i dimensions down to %i dimensions"
            % (class_map.shape[0], class_map.shape[1], proj_dim)
        )
    X = class_map.T
    variance = np.zeros(X.shape[0] - 1)

    n_dim = class_map.shape[1]
    tic = time.time()
    # Projective dimensionality reduction : Main Loop
    XRet = None
    for i in range(n_dim - 1):
        # Project onto an "equator"
        try:
            _, U = np.linalg.eigh(X.dot(X.T))
            U = np.fliplr(U)
        except:
            U = np.eye(X.shape[0])
        variance[-i - 1] = np.mean(
            (np.pi / 2 - np.real(np.arccos(np.abs(U[:, -1][None, :].dot(X))))) ** 2
        )
        Y = (U.T).dot(X)
        y = np.array(Y[-1, :])
        Y = Y[0:-1, :]
        X = Y / np.sqrt(1 - np.abs(y) ** 2)[None, :]
        if i == n_dim - proj_dim - 2:
            XRet = np.array(X)
    if verbose:
        print("Elapsed time ppca: %.3g" % (time.time() - tic))
    # Return the variance and the projective coordinates
    return {"variance": variance, "X": XRet.T}

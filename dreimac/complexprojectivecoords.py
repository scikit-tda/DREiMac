import numpy as np
from numba import jit
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp
from .utils import PartUnity, CohomologyUtils, EquivariantPCA, PPCA
from .emcoords import EMCoords
from .combinatorial import (
    combinatorial_number_system_table,
    combinatorial_number_system_d1_forward,
    combinatorial_number_system_d2_forward,
)


class ComplexProjectiveCoords(EMCoords):
    """
    Object that performs multiscale complex projective coordinates via
    persistent cohomology of sparse filtrations (Perea 2018).


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
        Maximum dimension of homology.  Only dimension 2 is needed for complex projective coordinates,
        but it may be of interest to see other dimensions
    """

    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=2, verbose=False
    ):
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        simplicial_complex_dimension = 3
        self._cns_lookup_table = combinatorial_number_system_table(
            n_landmarks, simplicial_complex_dimension
        )
        self.ppca = None

    def get_coordinates(
        self,
        X_query=None,
        distance_matrix_query=False,
        perc=0.5,
        cocycle_idx=0,
        proj_dim=1,
        partunity_fn=PartUnity.linear,
        standard_range=True,
        check_cocycle_condition=True,
        projective_dim_red_mode="exponential",
        save_projections=False
    ):
        """
        Get complex projective coordinates.


        Parameters
        ----------
        X_query: ndarray(M, d) or None
            A point cloud to compute the toroidal coordinates on. If None, uses self.X.
        distance_matrix_query: boolean
            If true, treat X_query as the distances of landmarks to the query point cloud
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : integer
            Integer representing the index of the persistent cohomology class
            used to construct the Eilenberg-MacLane coordinate. Persistent cohomology
            classes are ordered by persistence, from largest to smallest.
        proj_dim : integer
            Complex dimension down to which to project the data.
        partunity_fn : (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a non-trivial class in the Cech complex.
        check_cocycle_condition : bool
            Whether to check, and fix if necessary, that the integer cocycle constructed
            using finite field coefficients satisfies the cocycle condition.
        projective_dim_red_mode : string
            Either "one-by-one", "exponential", or "direct". How to perform equivariant
            dimensionality reduction. "exponential" usually works best, being fast
            without compromising quality.

        Returns
        -------
        ndarray(N, proj_dim+1)
            Complex projective coordinates
        """
        
        homological_dimension = 2
        cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
            cocycle_idx, homological_dimension
        )

        # determine radius for balls
        r_cover, rips_threshold = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn, X_query, distance_matrix_query)

        # compute boundary matrix
        delta1 = CohomologyUtils.make_delta1(
            self._dist_land_land, rips_threshold, self._cns_lookup_table
        )

        # lift to integer cocycles
        integer_cocycle = CohomologyUtils.lift_to_integer_cocycle(
            cocycle, prime=self._prime
        )

        # go from sparse to dense representation of cocycles
        integer_cocycle_as_vector = CohomologyUtils.sparse_cocycle_to_vector(
            integer_cocycle, self._cns_lookup_table, self._n_landmarks, int
        )

        if check_cocycle_condition:
            is_cocycle = _is_two_cocycle(
                integer_cocycle_as_vector,
                self._dist_land_land,
                rips_threshold,
                self._cns_lookup_table,
            )
            if not is_cocycle:
                delta2 = CohomologyUtils.make_delta2_compact(
                    self._dist_land_land, rips_threshold, self._cns_lookup_table
                )
                d2cocycle = delta2 @ integer_cocycle_as_vector.T

                y = d2cocycle // self._prime

                constraints = LinearConstraint(delta2, y, y, keep_feasible=True)
                n_edges = delta2.shape[1]
                objective = np.zeros((n_edges), dtype=int)
                integrality = np.ones((n_edges), dtype=int)
                bounds = scipy.optimize.Bounds()  # empty bounds

                optimizer_solution = milp(
                    objective,
                    integrality=integrality,
                    constraints=constraints,
                    bounds=bounds,
                )

                if not optimizer_solution["success"]:
                    raise Exception(
                        "The cohomology class at index "
                        + str(cocycle_idx)
                        + " does not have an integral lift."
                    )
                else:
                    solution = optimizer_solution["x"]
                    new_cocycle_as_vector = (
                        integer_cocycle_as_vector
                        - self._prime * np.array(np.rint(solution), dtype=int)
                    )
                    integer_cocycle_as_vector = new_cocycle_as_vector

        # compute harmonic representatives of cocycles and their projective-valued integrals
        integral = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ integral

        # compute complex projective coordinates on data points
        class_map = _sparse_integrate(
            harmonic_representative,
            integral,
            varphi,
            ball_indx,
            self._dist_land_land,
            rips_threshold,
            self._cns_lookup_table,
        )

        # reduce dimensionality of complex projective space
        self.ppca = PPCA(n_components=proj_dim, projective_dim_red_mode= projective_dim_red_mode)

        X = self.ppca.fit_transform(
            class_map, self.verbose, save=save_projections
        )
        self._variance = self.ppca.variance

        return X


# turn cocycle into tensor
def _two_cocycle_to_tensor(
    cocycle: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    n_points = dist_mat.shape[0]

    res = np.zeros((n_points, n_points, n_points))

    @jit(fastmath=True, nopython=True)
    def _get_res(
        cocycle: np.ndarray,
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        res: np.ndarray,
    ):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    for k in range(j + 1, n_points):
                        if dist_mat[i, k] < threshold and dist_mat[j, k] < threshold:
                            flat_index = combinatorial_number_system_d2_forward(
                                i, j, k, lookup_table
                            )
                            val = cocycle[flat_index]
                            # 012
                            res[i, j, k] = val
                            # 021
                            res[i, k, j] = -val
                            # 102
                            res[j, i, k] = -val
                            # 210
                            res[k, j, i] = -val
                            # 201
                            res[k, i, j] = val
                            # 120
                            res[j, k, i] = val

    _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)

    return res


def _one_cocycle_to_tensor(
    cocycle: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    n_points = dist_mat.shape[0]

    res = np.zeros((n_points, n_points))

    @jit(fastmath=True, nopython=True)
    def _get_res(
        cocycle: np.ndarray,
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        res: np.ndarray,
    ):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    flat_index = combinatorial_number_system_d1_forward(
                        i, j, lookup_table
                    )
                    val = cocycle[flat_index]
                    res[i, j] = val
                    res[j, i] = -val

    _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)

    return res


def _sparse_integrate(
    harm_rep,
    integral,
    part_unity,
    membership_function,
    dist_mat,
    threshold,
    lookup_table,
):
    nu = _one_cocycle_to_tensor(integral, dist_mat, threshold, lookup_table)

    eta = _two_cocycle_to_tensor(
        harm_rep,
        dist_mat,
        threshold,
        lookup_table,
    )

    class_map0 = np.zeros_like(part_unity.T)

    @jit(nopython=True)
    def _assemble(
        class_map: np.ndarray,
        nu: np.ndarray,
        eta: np.ndarray,
        varphi: np.ndarray,
        n_landmarks: int,
        n_data: int,
        ball_indx: np.ndarray,
    ):
        for b in range(n_data):
            for i in range(n_landmarks):
                class_map[b, i] += nu[i, ball_indx[b]]
                for t in range(n_landmarks):
                    class_map[b, i] += varphi[t, b] * eta[i, ball_indx[b], t]
        return np.exp(2 * np.pi * 1j * class_map) * np.sqrt(varphi.T)

    return _assemble(
        class_map0,
        nu,
        eta,
        part_unity,
        dist_mat.shape[0],
        part_unity.shape[1],
        membership_function,
    )


@jit(fastmath=True, nopython=True)
def _is_two_cocycle(
    cochain: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    is_cocycle = True
    n_points = dist_mat.shape[0]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_mat[i, j] < threshold:
                for k in range(j + 1, n_points):
                    if dist_mat[i, k] < threshold and dist_mat[j, k] < threshold:
                        for l in range(k + 1, n_points):
                            if (
                                dist_mat[i, l] < threshold
                                and dist_mat[j, l] < threshold
                                and dist_mat[k, l] < threshold
                            ):
                                index_ijk = combinatorial_number_system_d2_forward(
                                    i, j, k, lookup_table
                                )
                                index_ijl = combinatorial_number_system_d2_forward(
                                    i, j, l, lookup_table
                                )
                                index_ikl = combinatorial_number_system_d2_forward(
                                    i, k, l, lookup_table
                                )
                                index_jkl = combinatorial_number_system_d2_forward(
                                    j, k, l, lookup_table
                                )

                                if (
                                    cochain[index_ijk]
                                    - cochain[index_ijl]
                                    + cochain[index_ikl]
                                    - cochain[index_jkl]
                                    != 0
                                ):
                                    is_cocycle = False
    return is_cocycle

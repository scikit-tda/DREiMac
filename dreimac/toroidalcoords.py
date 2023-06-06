import numpy as np
import scipy
from numba import jit
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp
from .utils import PartUnity, CircleMapUtils, CohomologyUtils
from .emcoords import EMCoords
from .combinatorial import (
    combinatorial_number_system_table,
    combinatorial_number_system_d1_forward,
)


class ToroidalCoords(EMCoords):
    """
    Object that performs sparse toroidal coordinates via persistent cohomology as in
    (L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou, J. Perea 2022)

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

    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False
    ):
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "toroidal"
        simplicial_complex_dimension = 2
        self.cns_lookup_table_ = combinatorial_number_system_table(
            n_landmarks, simplicial_complex_dimension
        )

    def get_coordinates(
        self,
        perc=0.5,
        cocycle_idxs=[0],
        partunity_fn=PartUnity.linear,
        standard_range=True,
        check_cocycle_condition=True,
    ):
        """
        Get toroidal coordinates.

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
        thetas : ndarray(n, N)
            List of circular coordinates, with n the length of cocycle_idxs

        """

        # get representative cocycles and the intersection of their supports
        homological_dimension = 1
        cohomdeaths, cohombirths, cocycles = zip(
            *[
                self.get_representative_cocycle(cohomology_class, homological_dimension)
                for cohomology_class in cocycle_idxs
            ]
        )

        cohomdeath_rips = max(cohomdeaths)
        cohombirth_rips = min(cohombirths)

        if cohomdeath_rips >= cohombirth_rips:
            raise Exception(
                "\
                The supports of the chosen persistent cohomology classes do not intersect"
            )

        # lift to integer cocycles
        integer_cocycles = [
            CohomologyUtils.lift_to_integer_cocycle(cocycle, prime=self.prime_)
            for cocycle in cocycles
        ]

        # determine radius for balls
        r_cover, rips_threshold = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        # compute boundary matrix
        dist_land_land = self.dist_land_land_
        delta0 = CohomologyUtils.make_delta0(
            dist_land_land, rips_threshold, self.cns_lookup_table_
        )

        # go from sparse to dense representation of cocycles
        integer_cocycles_as_vectors = [
            CohomologyUtils.sparse_cocycle_to_vector(
                sparse_cocycle, self.cns_lookup_table_, self.n_landmarks_, int
            )
            for sparse_cocycle in integer_cocycles
        ]

        if check_cocycle_condition:
            delta1 = None
            fixed_cocycles = []

            for class_idx, cocycle_as_vector in enumerate(integer_cocycles_as_vectors):
                is_cocycle, _ = _is_one_cocycle(
                    cocycle_as_vector,
                    dist_land_land,
                    rips_threshold,
                    self.cns_lookup_table_,
                )
                if is_cocycle:
                    fixed_cocycles.append(cocycle_as_vector)
                else:
                    delta1 = (
                        delta1
                        if delta1
                        else CohomologyUtils.make_delta1_compact(
                            dist_land_land, rips_threshold, self.cns_lookup_table_
                        )
                    )
                    d1cocycle = delta1 @ cocycle_as_vector.T

                    y = d1cocycle // self.prime_

                    constraints = LinearConstraint(delta1, y, y, keep_feasible=True)
                    n_edges = delta1.shape[1]
                    objective = np.zeros((n_edges), dtype=int)
                    integrality = np.ones((n_edges), dtype=int)
                    optimizer_solution = milp(
                        objective,
                        integrality=integrality,
                        constraints=constraints,
                    )

                    if not optimizer_solution["success"]:
                        raise Exception(
                            "The cohomology class at index "
                            + str(class_idx)
                            + " does not have an integral lift."
                        )
                    else:
                        solution = optimizer_solution["x"]
                        new_cocycle_as_vector = (
                            cocycle_as_vector
                            - self.prime_ * np.array(np.rint(solution), dtype=int)
                        )

                        fixed_cocycles.append(new_cocycle_as_vector)
            integer_cocycles_as_vectors = fixed_cocycles

        # compute harmonic representatives of cocycles and their circle-valued integrals
        inner_product_matrix, sqrt_inner_product_matrix = _make_inner_product(
            dist_land_land, rips_threshold, self.cns_lookup_table_
        )

        harm_reps_and_integrals = [
            _integrate_harmonic_representative(
                cocycle, delta0, sqrt_inner_product_matrix
            )
            for cocycle in integer_cocycles_as_vectors
        ]
        harm_reps, _ = zip(*harm_reps_and_integrals)

        # compute circular coordinates on data points
        circ_coords = [
            _sparse_integrate(
                harm_rep,
                integral,
                varphi,
                ball_indx,
                dist_land_land,
                rips_threshold,
                self.cns_lookup_table_,
            )
            for harm_rep, integral in harm_reps_and_integrals
        ]

        # if more than one cohomology class was selected
        if len(integer_cocycles) > 1:
            # compute gram matrix with inner products between harmonic representative cocycles
            gram_mat = _gram_matrix(harm_reps, inner_product_matrix)

            # perform lattice reduction on the compute circular coordinates
            # using the gram matrix of the harmonic representative cocycles
            (
                circ_coords,
                change_basis,
                decorrelated_vectors,
            ) = _reduce_circular_coordinates(circ_coords, gram_mat)

            self.original_gram_matrix_ = gram_mat
            self.gram_matrix_ = decorrelated_vectors @ decorrelated_vectors.T
            self.change_basis_ = change_basis

        return circ_coords


def _integrate_harmonic_representative(cocycle, boundary_matrix, sqrt_inner_product):
    integral = lsqr(
        sqrt_inner_product @ boundary_matrix, sqrt_inner_product @ cocycle.T
    )[0]
    harm_rep = cocycle - boundary_matrix.dot(integral)
    return harm_rep, integral


def _make_inner_product(dist_mat, threshold, lookup_table):
    n_points = dist_mat.shape[0]
    n_edges = (n_points * (n_points - 1)) // 2

    max_n_entries = n_edges
    rows = np.empty((max_n_entries,), dtype=int)
    columns = np.empty((max_n_entries,), dtype=int)
    values = np.empty((max_n_entries,), dtype=float)

    @jit(fastmath=True)
    def _make_inner_product_get_row_columns_values(
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        rows: np.ndarray,
        columns: np.ndarray,
        values: np.ndarray,
    ):
        n_entries = 0
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    row_index = combinatorial_number_system_d1_forward(
                        i, j, lookup_table
                    )
                    rows[n_entries] = row_index
                    columns[n_entries] = row_index
                    values[n_entries] = 1
                    n_entries += 1
        return n_entries

    n_entries = _make_inner_product_get_row_columns_values(
        dist_mat, threshold, lookup_table, n_points, rows, columns, values
    )

    W = sparse.csr_array(
        (values[:n_entries], (rows[:n_entries], columns[:n_entries])),
        shape=(n_edges, n_edges),
    )

    WSqrt = W.copy()

    return W, WSqrt


def _sparse_integrate(
    harm_rep,
    integral,
    part_unity,
    membership_function,
    dist_mat,
    threshold,
    lookup_table,
):
    n_points = integral.shape[0]
    theta_matrix = np.zeros((n_points, n_points))

    @jit(fastmath=True)
    def _cocycle_to_matrix(
        dist_mat: np.ndarray,
        threshold: float,
        lookup_table: np.ndarray,
        n_points: int,
        theta_matrix: np.ndarray,
    ):
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if dist_mat[i, j] < threshold:
                    index = combinatorial_number_system_d1_forward(i, j, lookup_table)
                    theta_matrix[i, j] = harm_rep[index]

    _cocycle_to_matrix(dist_mat, threshold, lookup_table, n_points, theta_matrix)

    class_map = integral[membership_function].copy()
    for i in range(class_map.shape[0]):
        class_map[i] += theta_matrix[membership_function[i], :].dot(part_unity[:, i])
    return np.mod(2 * np.pi * class_map, 2 * np.pi)


@jit(fastmath=True)
def _is_one_cocycle(
    cochain: np.ndarray,
    dist_mat: np.ndarray,
    threshold: float,
    lookup_table: np.ndarray,
):
    is_cocycle = True
    first_failure = np.inf
    n_points = dist_mat.shape[0]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_mat[i, j] < threshold:
                for k in range(j + 1, n_points):
                    if (
                        dist_mat[i, k] < threshold
                        and dist_mat[j, k] < threshold
                    ):
                        index_ij = combinatorial_number_system_d1_forward(
                            i, j, lookup_table
                        )
                        index_jk = combinatorial_number_system_d1_forward(
                            j, k, lookup_table
                        )
                        index_ik = combinatorial_number_system_d1_forward(
                            i, k, lookup_table
                        )

                        if (
                            cochain[index_ij]
                            + cochain[index_jk]
                            - cochain[index_ik]
                            != 0
                        ):
                            is_cocycle = False
                            first_failure = min(
                                first_failure,
                                dist_mat[i, j],
                                dist_mat[j, k],
                                dist_mat[i, k],
                            )
    return is_cocycle, first_failure


# improve circular coordinates with lattice reduction
def _reduce_circular_coordinates(circ_coords, gram_matrix):
    lattice_red_input = np.linalg.cholesky(gram_matrix)
    decorrelated_vectors, change_basis = _lll(lattice_red_input.T)
    change_basis = change_basis.T
    new_circ_coords = CircleMapUtils.linear_combination(
        np.array(circ_coords), change_basis
    )
    return new_circ_coords, change_basis, decorrelated_vectors


def _gram_matrix(vectors, inner_product):
    n = len(vectors)
    res = np.zeros((n, n))
    for i, v in enumerate(vectors):
        for j, w in enumerate(vectors):
            res[i, j] = v.T @ inner_product @ w
    return res


## Lattice Reduction


# Gram-Schmidt (without normalization)
def _gram_schmidt(B):
    def gs_cofficient(v1, v2):
        return np.dot(v2, v1) / np.dot(v1, v1)

    # projects v2 onto v1
    def proj(v1, v2):
        return gs_cofficient(v1, v2) * v1

    n = len(B)
    A = np.zeros((n, n))
    A[:, 0] = B[:, 0]
    for i in range(1, n):
        Ai = B[:, i]
        for j in range(0, i):
            Aj = B[:, j]
            # t = np.dot(B[i],B[j])
            Ai = Ai - proj(Aj, Ai)
        A[:, i] = Ai
    return A


# LLL algorithm
def _lll(B, delta=3 / 4):
    B = B.copy()
    Q = _gram_schmidt(B)
    change = np.eye(B.shape[0])

    def mu(i, j):
        v = B[:, i]
        u = Q[:, j]
        return np.dot(v, u) / np.dot(u, u)

    n, k = len(B), 1
    while k < n:
        # length reduction step
        for j in reversed(range(k)):
            if abs(mu(k, j)) > 0.5:
                mu_kj = mu(k, j)
                B[:, k] = B[:, k] - round(mu_kj) * B[:, j]
                change[:, k] = change[:, k] - round(mu_kj) * change[:, j]
                Q = _gram_schmidt(B)

        # swap step
        if np.dot(Q[:, k], Q[:, k]) > (delta - mu(k, k - 1) ** 2) * (
            np.dot(Q[:, k - 1], Q[:, k - 1])
        ):
            k = k + 1
        else:
            B_k = B[:, k].copy()
            B_k1 = B[:, k - 1].copy()
            B[:, k], B[:, k - 1] = B_k1, B_k
            change_k = change[:, k].copy()
            change_k1 = change[:, k - 1].copy()
            change[:, k], change[:, k - 1] = change_k1, change_k

            Q = _gram_schmidt(B)
            k = max(k - 1, 1)

    return B, change

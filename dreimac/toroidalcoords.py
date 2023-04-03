import numpy as np
import scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp
from .utils import PartUnity, CircleMapUtils, CohomologyUtils
from .emcoords import *


class ToroidalCoords(EMCoords):
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
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        self.type_ = "toroidal"

    def get_coordinates(
        self,
        perc=0.99,
        cohomology_classes=[0],
        partunity_fn=PartUnity.linear,
        inner_product="uniform",
        check_and_fix_cocycle_condition=True,
    ):
        """
        Perform sparse toroidal coordinates via persistent cohomology as in
        (L. Scoccola, H. Gakhar, J. Bush, N. Schonsheck, T. Rask, L. Zhou, J. Perea 2022)

        Parameters
        ----------
        perc : float
            Percent coverage
        inner_product : string
            Either 'uniform' or 'exponential'
        cohomology_classes : list of integers
            TODO: explain
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        inner_product: TODO

        Returns
        -------
        thetas: ndarray(n, N)
            List of circular coordinates, with n the length of cocycle_idxs
        """

        # get representative cocycles and the intersection of their supports
        cohomdeaths, cohombirths, cocycles = zip(
            *[
                self.get_representative_one_cocycle(cohomology_class)
                for cohomology_class in cohomology_classes
            ]
        )

        cohomdeath_rips = max(cohomdeaths)
        cohombirth_rips = min(cohombirths)

        if cohomdeath_rips >= cohombirth_rips:
            raise Exception(
                "\
                The supports of the chosen persistent cohomology classes do not intersect"
            )

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
        dist_land_land = self.dist_land_land_
        delta0, edge_pair_to_row_index = _make_delta0(dist_land_land, threshold)

        # lift to integer cocycles
        integer_cocycles = [
            CohomologyUtils.lift_to_integer_cocycle(cocycle, prime=self.prime_) for cocycle in cocycles
        ]


        # go from sparse to dense representation of cocycles
        integer_cocycles_as_vectors = [
            _sparse_cocycle_to_vector(sparse_cocycle, edge_pair_to_row_index, int)
            for sparse_cocycle in integer_cocycles
        ]

        if check_and_fix_cocycle_condition:
            delta1 = _make_delta1(dist_land_land, edge_pair_to_row_index, threshold)

            fixed_cocycles = []

            cohomology_classes_ = []
            for i in range(len(integer_cocycles)):
                cohomology_classes_.append(
                    cohomology_classes[i % len(cohomology_classes)]
                )
            cohomology_classes = cohomology_classes_

            n_edges = len(edge_pair_to_row_index)
            for cohomology_idx, cocycle_as_vector in zip(
                cohomology_classes, integer_cocycles_as_vectors
            ):
                # compute delta1 of the cocycle to see if it is indeed an integer-valued cocycle
                d1cocycle = delta1 @ cocycle_as_vector.T
                # we perform exact comparison since delta1 and cocycle_as_vector are vectors of ints
                if np.linalg.norm(d1cocycle) == 0:
                    fixed_cocycles.append(cocycle_as_vector)
                else:
                    print("failure:", np.linalg.norm(delta1 @ cocycle_as_vector))
                    y = d1cocycle // self.prime_
                    constraints = LinearConstraint(delta1, y, y, keep_feasible=True)
                    objective = np.zeros((n_edges), dtype=int)
                    integrality = np.ones((n_edges), dtype=int)
                    optimizer_solution = milp(
                        objective, integrality=integrality, constraints=constraints
                    )
                    if not optimizer_solution["success"]:
                        raise Exception(
                            "The cohomology class at index "
                            + str(cohomology_idx)
                            + " does not have an integral lift."
                        )
                    else:
                        solution = optimizer_solution["x"]
                        new_cocycle_as_vector = (
                            cocycle_as_vector
                            - self.prime_ * np.array(np.rint(solution), dtype=int)
                        )
                        print(
                            "fixed failure:",
                            np.linalg.norm(delta1 @ new_cocycle_as_vector),
                        )

                        fixed_cocycles.append(new_cocycle_as_vector)
            integer_cocycles_as_vectors = fixed_cocycles

        # compute inner product matrix for cocycles
        inner_product_matrix, sqrt_inner_product_matrix = _make_inner_product(
            dist_land_land, threshold, edge_pair_to_row_index, inner_product
        )

        # compute harmonic representatives of cocycles and their circle-valued integrals
        harm_reps_and_integrals = [
            _integrate_harmonic_representative(
                cocycle, delta0, sqrt_inner_product_matrix
            )
            for cocycle in integer_cocycles_as_vectors
        ]
        harm_reps, _ = zip(*harm_reps_and_integrals)
        self._harm_reps = harm_reps

        # compute circular coordinates on data points
        circ_coords = [
            _sparse_integrate(
                harm_rep, integral, varphi, ball_indx, edge_pair_to_row_index
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


def _sparse_cocycle_to_vector(sparse_cocycle, edge_pair_to_row_index, dtype):
    n_edges = len(edge_pair_to_row_index)
    cocycle_as_vector = np.zeros((n_edges,), dtype=dtype)
    for i, j, val in sparse_cocycle:
        # if the cocycle takes a non-zero value on an edge (i,j) with i < j
        # then record that value in cocycle_as_vector at the index corresponding to (i,j)
        if (i, j) in edge_pair_to_row_index:
            cocycle_as_vector[edge_pair_to_row_index[(i, j)]] = val
        # if the cocycle takes a non-zero value on an edge (i,j) with j < i
        # then record *minus* that value in cocycle_as_vector at the index
        # corresponding to (j,i)
        if (j, i) in edge_pair_to_row_index:
            cocycle_as_vector[edge_pair_to_row_index[(j, i)]] = -val
    return cocycle_as_vector



def _integrate_harmonic_representative(cocycle, boundary_matrix, sqrt_inner_product):
    integral = lsqr(
        sqrt_inner_product @ boundary_matrix, sqrt_inner_product @ cocycle.T
    )[0]
    harm_rep = cocycle - boundary_matrix.dot(integral)
    return harm_rep, integral


def _make_inner_product(dist_matrix, threshold, edge_pair_to_row_index, kind):
    # n_edges = dist_matrix.shape[0] ** 2
    n_edges = len(edge_pair_to_row_index)
    if kind == "uniform":
        row_index = []
        col_index = []
        value = []
        for l in edge_pair_to_row_index.values():
            row_index.append(l)
            col_index.append(l)
            value.append(1)
        WSqrt = scipy.sparse.coo_matrix(
            (value, (row_index, col_index)), shape=(n_edges, n_edges)
        ).tocsr()
        W = scipy.sparse.coo_matrix(
            (value, (row_index, col_index)), shape=(n_edges, n_edges)
        ).tocsr()
    elif kind == "exponential":
        row_index = []
        col_index = []
        value = []
        sqrt_value = []
        # sigma = threshold * dist_matrix.shape[0]
        sigma = threshold
        for pl in edge_pair_to_row_index.items():
            p, l = pl
            i, j = p
            val = np.exp(-((dist_matrix[i, j] / sigma) ** 2))
            row_index.append(l)
            col_index.append(l)
            value.append(val)
            sqrt_value.append(np.sqrt(val))
        W = scipy.sparse.coo_matrix(
            (value, (row_index, col_index)), shape=(n_edges, n_edges)
        ).tocsr()
        WSqrt = scipy.sparse.coo_matrix(
            (sqrt_value, (row_index, col_index)), shape=(n_edges, n_edges)
        ).tocsr()
    else:
        raise Exception("Inner product kind must be uniform or exponential.")
    return W, WSqrt


def _make_delta0(dist_mat, threshold):
    n_points = dist_mat.shape[0]
    edge_pair_to_row_index = {}
    row_index = []
    col_index = []
    value = []
    n_edges = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_mat[i, j] < threshold:
                edge_pair_to_row_index[(i, j)] = n_edges

                row_index.append(n_edges)
                col_index.append(i)
                value.append(-1)

                row_index.append(n_edges)
                col_index.append(j)
                value.append(1)

                n_edges += 1

    delta0 = sparse.coo_matrix(
        (value, (row_index, col_index)), shape=(n_edges, n_points)
    ).tocsr()
    return delta0, edge_pair_to_row_index


def _make_delta1(dist_land_land, edge_pair_to_row_index, threshold):
    n_points = dist_land_land.shape[0]
    n_edges = len(edge_pair_to_row_index)
    face_triple_to_row_index = {}
    row_index = []
    col_index = []
    value = []
    n_faces = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_land_land[i, j] < threshold:
                for k in range(j + 1, n_points):
                    if (
                        dist_land_land[i, k] < threshold
                        and dist_land_land[j, k] < threshold
                    ):
                        face_triple_to_row_index[(i, j, k)] = n_faces

                        row_index.append(n_faces)
                        col_index.append(edge_pair_to_row_index[(i, j)])
                        value.append(1)

                        row_index.append(n_faces)
                        col_index.append(edge_pair_to_row_index[(j, k)])
                        value.append(1)

                        row_index.append(n_faces)
                        col_index.append(edge_pair_to_row_index[(i, k)])
                        value.append(-1)

                        n_faces += 1

    delta1 = sparse.coo_matrix(
        (value, (row_index, col_index)), shape=(n_faces, n_edges), dtype=int
    )
    print("number of faces: ", n_faces)
    return delta1


def _sparse_integrate(
    harm_rep, integral, part_unity, membership_function, edge_pair_to_row_index
):
    n = integral.shape[0]
    # from U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
    # compute all transition functions
    theta_matrix = np.zeros((n, n))

    for pl in edge_pair_to_row_index.items():
        p, l = pl
        i, j = p
        theta_matrix[i, j] = harm_rep[l]
    class_map = -integral[membership_function].copy()
    for i in range(class_map.shape[0]):
        class_map[i] += theta_matrix[membership_function[i], :].dot(part_unity[:, i])
    return np.mod(2 * np.pi * class_map, 2 * np.pi)


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

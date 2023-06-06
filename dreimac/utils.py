"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu) and Luis Scoccola
Purpose: To provide a number of utility functions, including
- Quickly computing all pairs self-similarity and cross-similarity matrices
- Doing "greedy permutations" 
- Some relevant geometric examples for tests
"""
import time
import numpy as np
import scipy.sparse as sparse
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from numba import jit
from .combinatorial import (
    combinatorial_number_system_forward,
    combinatorial_number_system_d1_forward,
    combinatorial_number_system_d2_forward,
    number_of_simplices_of_dimension,
)


class GeometryUtils:
    """
    Utilities for subsampling from point clouds and distance matrices.

    """

    @staticmethod
    def get_csm(X, Y):
        """
        Return the Euclidean cross-similarity matrix between the M points
        in the Mxd matrix X and the N points in the Nxd matrix Y.

        Parameters
        ----------
        X : ndarray (M, d)
            A matrix holding the coordinates of M points
        Y : ndarray (N, d)
            A matrix holding the coordinates of N points

        Returns
        -------
        D : ndarray (M, N)
            An MxN Euclidean cross-similarity matrix

        """
        C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2 * X.dot(Y.T)
        C[C < 0] = 0
        return np.sqrt(C)

    @staticmethod
    def get_csm_projarc(X, Y):
        """
        Return the projective arc length cross-similarity between two point
        clouds specified as points on the sphere

        Parameters
        ----------
        X : ndarray (M, d)
            A matrix holding the coordinates of M points on RP^{d-1}
        Y : ndarray (N, d)
            A matrix holding the coordinates of N points on RP^{d-1}

        Returns
        -------
        D : ndarray (M, N)
            An MxN  cross-similarity matrix

        """
        D = np.abs(X.dot(Y.T))
        D[D < -1] = -1
        D[D > 1] = 1
        D = np.arccos(np.abs(D))
        return D

    @staticmethod
    def get_ssm(X):
        return GeometryUtils.get_csm(X, X)

    # Greedy Permutations
    @staticmethod
    def get_greedy_perm_pc(X, M, csm_fn=get_csm.__func__):
        """
        A Naive O(NM) algorithm to do furthest points sampling, assuming
        the input is a point cloud specified in Euclidean space.  This saves
        computation over having compute the full distance matrix if the number
        of landmarks M << N

        Parameters
        ----------
        X : ndarray (N, d)
            An Nxd Euclidean point cloud
        M : integer
            Number of landmarks to compute
        csm_fn: function X, Y -> D
            Cross-similarity function (Euclidean by default)

        Return
        ------
        result: Dictionary
            {'Y': An Mxd array of landmarks,
            'perm': An array of indices into X of the greedy permutation
            'lambdas': Insertion radii of the landmarks
            'D': An MxN array of distances from landmarks to points in X}

        """
        # By default, takes the first point in the permutation to be the
        # first point in the point cloud, but could be random
        N = X.shape[0]
        perm = np.zeros(M, dtype=np.int64)
        lambdas = np.zeros(M)
        ds = csm_fn(X[0, :][None, :], X).flatten()
        D = np.zeros((M, N))
        D[0, :] = ds
        for i in range(1, M):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            thisds = csm_fn(X[idx, :][None, :], X).flatten()
            D[i, :] = thisds
            ds = np.minimum(ds, thisds)
        Y = X[perm, :]
        return {"Y": Y, "perm": perm, "lambdas": lambdas, "D": D}

    @staticmethod
    def get_greedy_perm_dm(D, M):
        """
        A Naive O(NM) algorithm to do furthest points sampling, assuming
        the input is a N x N distance matrix

        Parameters
        ----------
        D : ndarray (N, N)
            An N x N distance matrix
        M : integer
            Number of landmarks to compute

        Return
        ------
        result: Dictionary
            {'perm': An array of indices into X of the greedy permutation
            'lambdas': Insertion radii of the landmarks
            'DLandmarks': An MxN array of distances from landmarks to points in the point cloud}

        """
        # By default, takes the first point in the permutation to be the
        # first point in the point cloud, but could be random
        perm = np.zeros(M, dtype=np.int64)
        lambdas = np.zeros(M)
        ds = D[0, :]
        for i in range(1, M):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            ds = np.minimum(ds, D[idx, :])
        DLandmarks = D[perm, :]
        return {"perm": perm, "lambdas": lambdas, "DLandmarks": DLandmarks}


    @staticmethod
    def landmark_geodesic_distance(X, n_landmarks, n_neighbors):
        spatial_tree = KDTree(X)
        distances_nn, indices_nn = spatial_tree.query(X,k=n_neighbors)
        # https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/neighbors/_base.py#L997
        n_queries = X.shape[0]
        n_nonzero = n_queries * n_neighbors
        indptr = np.arange(0, n_nonzero + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances_nn.ravel(), indices_nn.ravel(), indptr), shape=(n_queries, n_queries)
        )

        # furthest point sampling
        n_points = X.shape[0]
        perm = np.zeros(n_landmarks, dtype=np.int64)
        lambdas = np.zeros(n_landmarks)
        ds = shortest_path(kneighbors_graph, indices = 0, directed=False)
        D = np.zeros((n_landmarks, n_points))
        D[0, :] = ds
        for i in range(1, n_landmarks):
            idx = np.argmax(ds)
            perm[i] = idx
            lambdas[i] = ds[idx]
            thisds  = shortest_path(kneighbors_graph, indices = idx, directed=False)
            D[i, :] = thisds
            ds = np.minimum(ds, thisds)

        perm_rest_points = np.setdiff1d(np.arange(0,n_points, dtype=int), perm, assume_unique=True)
        perm_all_points = np.concatenate((perm,perm_rest_points))

        return D[:,perm_all_points], perm_all_points


class CohomologyUtils:
    @staticmethod
    def lift_to_integer_cocycle(cocycle, prime):
        """
        Lift the given cocycle with values in a prime field to a cocycle with integer coefficients.

        Parameters
        ----------
        cocycle : ndarray(K, n, dtype=int)
            Cocycle to be lifted to integer coefficients.

        Note
        ----
        This routine modifies the input cocycle.

        Returns
        -------
        cocycle : ndarray(K, n, dtype=int)
            Cocycle with same support as input cocycle and integer coefficients.

        """
        cocycle[cocycle[:, -1] > (prime - 1) / 2, -1] -= prime
        return cocycle

    @staticmethod
    def reindex_cocycles(cocycles, idx_land, N):
        """
        Convert the indices of a set of cocycles to be relative
        to a list of indices in a greedy permutation

        Parameters
        ----------
        cocycles: list of list of ndarray
            The cocycles
        idx_land: ndarray(M, dtype=int)
            Indices of the landmarks in the greedy permutation, with
            respect to all points
        N: int
            Number of total points

        """
        idx_map = -1 * np.ones(N, dtype=int)
        idx_map[idx_land] = np.arange(idx_land.size)
        for ck in cocycles:
            for c in ck:
                c[:, 0:-1] = idx_map[c[:, 0:-1]]

    @staticmethod
    def parity(permutation):
        """Compute the parity of a permutation"""
        permutation_length = len(permutation)
        elements_seen = [False for _ in range(permutation_length)]
        cycles = 0
        for index, already_seen in enumerate(elements_seen):
            if already_seen:
                continue
            cycles += 1
            current = index
            while not elements_seen[current]:
                elements_seen[current] = True
                current = permutation[current]
        return 1 if (permutation_length - cycles) % 2 == 0 else -1

    @staticmethod
    def order_simplex(unordered_simplex):
        ordering_permutation = np.argsort(unordered_simplex)
        sign = CohomologyUtils.parity(ordering_permutation)
        simplex = tuple(unordered_simplex[ordering_permutation])
        return simplex, sign

    @staticmethod
    def sparse_cocycle_to_vector(sparse_cocycle, lookup_table, n_vertices, dtype):
        dimension = sparse_cocycle.shape[1] - 2
        n_simplices = number_of_simplices_of_dimension(
            dimension, n_vertices, lookup_table
        )
        cocycle_as_vector = np.zeros((n_simplices,), dtype=dtype)
        for entry in sparse_cocycle:
            value = entry[-1]
            unordered_simplex = np.array(entry[:-1], dtype=int)
            ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
            simplex_index = combinatorial_number_system_forward(
                ordered_simplex, lookup_table
            )
            cocycle_as_vector[simplex_index] = value * sign
        return cocycle_as_vector

    @staticmethod
    def make_delta0(dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray):
        n_points = dist_mat.shape[0]
        n_edges = (n_points * (n_points - 1)) // 2

        max_n_entries = 2 * n_edges
        rows = np.empty((max_n_entries,), dtype=int)
        columns = np.empty((max_n_entries,), dtype=int)
        values = np.empty((max_n_entries,), dtype=float)

        @jit(fastmath=True)
        def _delta0_get_row_columns_values(
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
                        columns[n_entries] = i
                        values[n_entries] = 1
                        n_entries += 1

                        rows[n_entries] = row_index
                        columns[n_entries] = j
                        values[n_entries] = -1
                        n_entries += 1
            return n_entries

        n_entries = _delta0_get_row_columns_values(
            dist_mat, threshold, lookup_table, n_points, rows, columns, values
        )

        return sparse.csr_array(
            (values[:n_entries], (rows[:n_entries], columns[:n_entries])),
            shape=(n_edges, n_points),
        )

    @staticmethod
    def make_delta1(dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray):
        n_points = dist_mat.shape[0]
        n_edges = (n_points * (n_points - 1)) // 2
        n_faces = number_of_simplices_of_dimension(2, n_points, lookup_table)

        max_n_entries = 3 * n_faces
        rows = np.empty((max_n_entries,), dtype=int)
        columns = np.empty((max_n_entries,), dtype=int)
        values = np.empty((max_n_entries,), dtype=float)

        @jit(fastmath=True)
        def _delta1_get_row_columns_values(
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
                        for k in range(j + 1, n_points):
                            if (
                                dist_mat[i, k] < threshold
                                and dist_mat[j, k] < threshold
                            ):
                                row_index = combinatorial_number_system_d2_forward(
                                    i, j, k, lookup_table
                                )
                                column_index_ij = (
                                    combinatorial_number_system_d1_forward(
                                        i, j, lookup_table
                                    )
                                )
                                column_index_jk = (
                                    combinatorial_number_system_d1_forward(
                                        j, k, lookup_table
                                    )
                                )
                                column_index_ik = (
                                    combinatorial_number_system_d1_forward(
                                        i, k, lookup_table
                                    )
                                )
                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_ij
                                values[n_entries] = 1
                                n_entries += 1

                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_jk
                                values[n_entries] = 1
                                n_entries += 1

                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_ik
                                values[n_entries] = -1
                                n_entries += 1

            return n_entries

        n_entries = _delta1_get_row_columns_values(
            dist_mat, threshold, lookup_table, n_points, rows, columns, values
        )

        # print("delta 1")
        # print(n_entries)
        # print(max_n_entries)

        return sparse.csr_array(
            (values[:n_entries], (rows[:n_entries], columns[:n_entries])),
            shape=(n_faces, n_edges),
        )

    @staticmethod
    def make_delta1_compact(
        dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray
    ):
        """
        Like [make_delta1] but it does not use the combinatorial number system for the rows
        (i.e., for the 2-simplices, and only has a row per 2-simplex *that exists in the filtration at that point*.
        This is used to solve the integer linear problem since, in that case, we do not need to index the 2-simplices
        so only having the ones that exist in the filtration at that point results in a matrix with potentially much
        fewer rows.
        """
        n_points = dist_mat.shape[0]
        n_edges = (n_points * (n_points - 1)) // 2
        n_faces = number_of_simplices_of_dimension(2, n_points, lookup_table)

        max_n_entries = 3 * n_faces
        rows = np.empty((max_n_entries,), dtype=int)
        columns = np.empty((max_n_entries,), dtype=int)
        values = np.empty((max_n_entries,), dtype=float)

        @jit(fastmath=True)
        def _delta1_get_row_columns_values(
            dist_mat: np.ndarray,
            threshold: float,
            lookup_table: np.ndarray,
            n_points: int,
            rows: np.ndarray,
            columns: np.ndarray,
            values: np.ndarray,
        ):
            n_entries = 0
            n_actual_faces = 0
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if dist_mat[i, j] < threshold:
                        for k in range(j + 1, n_points):
                            if (
                                dist_mat[i, k] < threshold
                                and dist_mat[j, k] < threshold
                            ):
                                # row_index = combinatorial_number_system_d2_forward(
                                #    i, j, k, lookup_table
                                # )
                                row_index = n_actual_faces
                                column_index_ij = (
                                    combinatorial_number_system_d1_forward(
                                        i, j, lookup_table
                                    )
                                )
                                column_index_jk = (
                                    combinatorial_number_system_d1_forward(
                                        j, k, lookup_table
                                    )
                                )
                                column_index_ik = (
                                    combinatorial_number_system_d1_forward(
                                        i, k, lookup_table
                                    )
                                )
                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_ij
                                values[n_entries] = 1
                                n_entries += 1

                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_jk
                                values[n_entries] = 1
                                n_entries += 1

                                rows[n_entries] = row_index
                                columns[n_entries] = column_index_ik
                                values[n_entries] = -1
                                n_entries += 1

                                n_actual_faces += 1
            return n_entries, n_actual_faces

        n_entries, n_actual_faces = _delta1_get_row_columns_values(
            dist_mat, threshold, lookup_table, n_points, rows, columns, values
        )

        # print("delta 1")
        # print(n_entries)
        # print(max_n_entries)

        return sparse.csr_array(
            (values[:n_entries], (rows[:n_entries], columns[:n_entries])),
            shape=(n_actual_faces, n_edges),
        )

    @staticmethod
    def make_delta2_compact(
        dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray
    ):
        """
        Like [make_delta1_compact] but for delta2.
        """
        n_points = dist_mat.shape[0]
        n_faces = number_of_simplices_of_dimension(2, n_points, lookup_table)
        n_three_simplices = number_of_simplices_of_dimension(3, n_points, lookup_table)

        max_n_entries = 4 * n_three_simplices
        rows = np.empty((max_n_entries,), dtype=int)
        columns = np.empty((max_n_entries,), dtype=int)
        values = np.empty((max_n_entries,), dtype=float)

        @jit(fastmath=True)
        def _delta2_get_row_columns_values(
            dist_mat: np.ndarray,
            threshold: float,
            lookup_table: np.ndarray,
            n_points: int,
            rows: np.ndarray,
            columns: np.ndarray,
            values: np.ndarray,
        ):
            n_entries = 0
            n_actual_three_simplices = 0
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if dist_mat[i, j] < threshold:
                        for k in range(j + 1, n_points):
                            if (
                                dist_mat[i, k] < threshold
                                and dist_mat[j, k] < threshold
                            ):
                                for l in range(k + 1, n_points):
                                    if (
                                        dist_mat[i, l] < threshold
                                        and dist_mat[j, l] < threshold
                                        and dist_mat[k, l] < threshold
                                    ):
                                        row_index = n_actual_three_simplices

                                        column_index_ijk = (
                                            combinatorial_number_system_d2_forward(
                                                i, j, k, lookup_table
                                            )
                                        )
                                        column_index_ijl = (
                                            combinatorial_number_system_d2_forward(
                                                i, j, l, lookup_table
                                            )
                                        )
                                        column_index_ikl = (
                                            combinatorial_number_system_d2_forward(
                                                i, k, l, lookup_table
                                            )
                                        )
                                        column_index_jkl = (
                                            combinatorial_number_system_d2_forward(
                                                j, k, l, lookup_table
                                            )
                                        )

                                        rows[n_entries] = row_index
                                        columns[n_entries] = column_index_ijk
                                        values[n_entries] = 1
                                        n_entries += 1

                                        rows[n_entries] = row_index
                                        columns[n_entries] = column_index_ijl
                                        values[n_entries] = -1
                                        n_entries += 1

                                        rows[n_entries] = row_index
                                        columns[n_entries] = column_index_ikl
                                        values[n_entries] = 1
                                        n_entries += 1

                                        rows[n_entries] = row_index
                                        columns[n_entries] = column_index_jkl
                                        values[n_entries] = -1
                                        n_entries += 1

                                        n_actual_three_simplices += 1
            return n_entries, n_actual_three_simplices

        n_entries, n_actual_three_simplices = _delta2_get_row_columns_values(
            dist_mat, threshold, lookup_table, n_points, rows, columns, values
        )

        # print("delta 1")
        # print(n_entries)
        # print(max_n_entries)

        return sparse.csr_array(
            (values[:n_entries], (rows[:n_entries], columns[:n_entries])),
            shape=(n_actual_three_simplices, n_faces),
        )


class PartUnity:
    """
    Partitions of unity subordinate to open ball covers using
    standard bump functions.

    """

    @staticmethod
    def linear(ds, r_cover):
        """
        Linear partition of unity.

        Parameters
        ----------
        ds: ndarray(n)
            Some subset of distances between landmarks and
            data points
        r_cover: float
            Covering radius

        Returns
        -------
        varphi: ndarray(n)
            The bump function

        """
        return r_cover - ds

    @staticmethod
    def quadratic(ds, r_cover):
        """
        Quadratic partition of unity.

        Parameters
        ----------
        ds: ndarray(n)
            Some subset of distances between landmarks and
            data points
        r_cover: float
            Covering radius

        Returns
        -------
        varphi: ndarray(n)
            The bump function

        """
        return (r_cover - ds) ** 2

    @staticmethod
    def exp(ds, r_cover):
        """
        Exponential partition of unity.

        Parameters
        ----------
        ds: ndarray(n)
            Some subset of distances between landmarks and
            data points
        r_cover: float
            Covering radius

        Returns
        -------
        varphi: ndarray(n)
            The bump function

        """
        return np.exp(r_cover**2 / (ds**2 - r_cover**2))


class EquivariantPCA:
    @staticmethod
    def ppca(class_map, proj_dim, projective_dim_red_mode="one-by-one", verbose=False):
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

        def _one_step_linear_reduction(X, dims_to_keep):
            try:
                _, U = np.linalg.eigh(X.dot(np.conjugate(X).T))
                U = np.fliplr(U)
            except:
                U = np.eye(X.shape[0])
            Y = (np.conjugate(U).T).dot(X)
            Y = Y[:dims_to_keep, :]
            X = Y / np.linalg.norm(Y, axis=0)[None, :]
            return X

        total_dims_to_keep = proj_dim + 1

        modes = ["direct", "exponential", "one-by-one"]
        mode = projective_dim_red_mode
        if mode == "direct":
            XRet = _one_step_linear_reduction(X, total_dims_to_keep)
        elif mode == "exponential":
            to_keep_this_iter = (n_dim - total_dims_to_keep) // 2
            while to_keep_this_iter > 0:
                X = _one_step_linear_reduction(
                    X, total_dims_to_keep + to_keep_this_iter
                )
                to_keep_this_iter = to_keep_this_iter // 2
            if X.shape[0] > total_dims_to_keep:
                X = _one_step_linear_reduction(X, total_dims_to_keep)
            XRet = X

        elif mode == "one-by-one":
            tic = time.time()
            # Projective dimensionality reduction : Main Loop
            XRet = None
            for i in range(n_dim - 1):
                if i == n_dim - proj_dim - 1:
                    XRet = X
                try:
                    _, U = np.linalg.eigh(X.dot(np.conjugate(X).T))
                    U = np.fliplr(U)
                    # U, _, _ = np.linalg.svd(X)
                except:
                    U = np.eye(X.shape[0])
                variance[-i - 1] = np.mean(
                    (np.pi / 2 - np.real(np.arccos(np.abs(U[:, -1][None, :].dot(X)))))
                    ** 2
                )
                Y = (np.conjugate(U).T).dot(X)
                # y = np.array(Y[-1, :])
                Y = Y[0:-1, :]
                # X = Y / np.sqrt(1 - np.abs(y) ** 2)[None, :]
                X = Y / np.linalg.norm(Y, axis=0)[None, :]
            if verbose:
                print("Elapsed time ppca: %.3g" % (time.time() - tic))

        # Return the variance and the projective coordinates
        return {"variance": variance, "X": XRet.T}


class GeometryExamples:
    """
    Finite samples from topologically nontrivial spaces.

    """

    # TODO: These probably belong in tdasets, but I'll keep them here for now

    @staticmethod
    def line_patches(dim, n_angles, n_offsets, sigma):
        """
        Sample a set of line segments, as witnessed by square patches

        Parameters
        ----------
        dim: int
            Patches will be dim x dim
        n_angles: int
            Number of angles to sweep between 0 and pi
        n_offsets: int
            Number of offsets to sweep from the origin to the edge of the patch
        sigma: float
            The blur parameter.  Higher sigma is more blur

        Returns
        -------
        ndarray(n_angles*n_offsets, dim*dim)
            An array of all of the patches raveled into dim*dim dimensional Euclidean space

        """
        N = n_angles * n_offsets
        P = np.zeros((N, dim * dim))
        thetas = np.linspace(0, np.pi, n_angles + 1)[0:n_angles]
        # ps = np.linspace(-0.5*np.sqrt(2), 0.5*np.sqrt(2), n_offsets)
        ps = np.linspace(-1, 1, n_offsets)
        idx = 0
        [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
        for i in range(n_angles):
            c = np.cos(thetas[i])
            s = np.sin(thetas[i])
            for j in range(n_offsets):
                patch = X * c + Y * s + ps[j]
                patch = np.exp(-(patch**2) / sigma**2)
                P[idx, :] = patch.flatten()
                idx += 1
        return P

    @staticmethod
    def moving_dot(sqrt_num_images, sigma=3):
        """
        TODO
        """

        def _gkern(l=5, mu=0, sig=1.0):
            ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
            gauss_x = np.exp(-0.5 * np.square(ax - mu[0]) / np.square(sig))
            gauss_y = np.exp(-0.5 * np.square(ax - mu[1]) / np.square(sig))
            kernel = np.outer(gauss_x, gauss_y)
            return kernel

        img_len = 10
        P = np.zeros((sqrt_num_images**2, img_len * img_len))
        bound = 15
        xs = bound * np.power(np.linspace(-1, 1, sqrt_num_images), 3)
        # xs = bound * np.linspace(-1,1,sqrt_num_images)
        ys = -xs
        i = 0
        for x in xs:
            for y in ys:
                P[i] = _gkern(l=img_len, mu=np.array([x, y]), sig=sigma).flatten()
                i += 1
        return P

    @staticmethod
    def rp2_metric(n_samples, seed=None):
        """
        Return a distance matrix of points on the projective plane
        obtained by identifying antipodal Gaussian random samples
        of a sphere

        Parameters
        ----------
        n_samples : int
            Number of random samples on the projective plane
        seed: int
            Seed to use.  If omitted, use the number of samples as a seed

        Returns
        -------
        ndarray(n_samples, 3)
            Original points on the sphere
        ndarray(n_samples, n_samples)
            Distance matrix of rp2

        """
        if seed is None:
            seed = n_samples
        np.random.seed(seed)
        X = np.random.randn(n_samples, 3)
        X = X / np.sqrt(np.sum(X**2, 1))[:, None]
        return X, GeometryUtils.get_csm_projarc(X, X)

    @staticmethod
    def torus_3d(n_samples, R, r, seed=None):
        """
        Return points sampled on a 3D torus

        Parameters
        ----------
        n_samples : int
            Number of random samples on the torus
        R: float
            Outer radius
        r: float
            Inner radius
        seed: int
            Seed to use.  If omitted, use the number of samples as a seed

        Returns
        -------
        X: ndarray(n_samples, 4)
            3D torus samples

        """
        if seed is None:
            seed = n_samples
        np.random.seed(seed)
        X = np.zeros((n_samples, 3))
        s = np.random.rand(n_samples) * 2 * np.pi
        t = np.random.rand(n_samples) * 2 * np.pi
        X[:, 0] = (R + r * np.cos(s)) * np.cos(t)
        X[:, 1] = (R + r * np.cos(s)) * np.sin(t)
        X[:, 2] = r * np.sin(s)
        return X

    @staticmethod
    def klein_bottle_4d(n_samples, R, r, seed=None):
        """
        Return samples on a klein bottle in 4D

        Parameters
        ----------
        n_samples : int
            Number of random samples on the projective plane
        R: float
            Outer radius
        r: float
            Inner radius
        seed: int
            Seed to use.  If omitted, use the number of samples as a seed

        Returns
        -------
        X: ndarray(n_samples, 4)
            4D klein bottle samples

        """
        if seed is None:
            seed = n_samples
        np.random.seed(seed)
        theta = np.random.rand(n_samples) * 2 * np.pi
        phi = np.random.rand(n_samples) * 2 * np.pi
        X = np.zeros((n_samples, 4))
        X[:, 0] = (R + r * np.cos(theta)) * np.cos(phi)
        X[:, 1] = (R + r * np.cos(theta)) * np.sin(phi)
        X[:, 2] = r * np.sin(theta) * np.cos(phi / 2)
        X[:, 3] = r * np.sin(theta) * np.sin(phi / 2)
        return X

    @staticmethod
    def genus_two_surface():
        """
        Return samples on a genus two surface in 3D

        Returns
        -------
        X: ndarray(n_samples, 3)
            3D surface samples

        """

        R2 = 9
        R = 5
        r = 2
        Ns = 80
        Nt = 120
        N = Ns * Nt
        Y = np.zeros((N, 3))
        s = np.linspace(0, 2 * np.pi, Ns, endpoint=False)
        t = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        st = np.array([[x, y] for x in s for y in t])
        s = st[:, 0]
        t = st[:, 1]
        Y[:, 0] = (R + r * np.cos(s)) * np.cos(t)
        Y[:, 1] = (R + r * np.cos(s)) * np.sin(t)
        Y[:, 2] = r * np.sin(s)

        Z = np.zeros((N, 3))
        Z[:, 0] = R2 + (R + r * np.cos(s)) * np.cos(t)
        Z[:, 1] = (R + r * np.cos(s)) * np.sin(t)
        Z[:, 2] = r * np.sin(s)

        Y = Y[(Y[:, 0] <= 4.5)]
        Z = Z[(Z[:, 0] >= 4.5)]

        return np.concatenate((Y, Z), axis=0)

    @staticmethod
    def trefoil(n_samples=2500, horizontal_width=6, noisy=True):
        """
        Samples on a trefoil in 3D

        Parameters
        ----------
        n_samples : int
            Number of random samples.

        Returns
        -------
        X: ndarray(n_samples, 3)
            3D trefoil samples

        """

        if noisy:
            np.random.seed(0)
            u = 4 * np.pi * np.random.rand(n_samples)
            v = 2 * np.pi * np.random.rand(n_samples)
            X = np.zeros((n_samples, 3))
            X[:, 0] = np.cos(u) * np.cos(v) + horizontal_width * np.cos(u) * (
                1.5 + np.sin(1.5 * u) / 2
            )
            X[:, 1] = np.sin(u) * np.cos(v) + horizontal_width * np.sin(u) * (
                1.5 + np.sin(1.5 * u) / 2
            )
            X[:, 2] = np.sin(v) + 4 * np.cos(1.5 * u)
        else:
            np.random.seed(0)
            u = 4 * np.pi * np.linspace(0, 1, n_samples, endpoint=False)
            X = np.zeros((n_samples, 3))
            X[:, 0] = np.cos(u) + horizontal_width * np.cos(u) * (
                1.5 + np.sin(1.5 * u) / 2
            )
            X[:, 1] = np.sin(u) + horizontal_width * np.sin(u) * (
                1.5 + np.sin(1.5 * u) / 2
            )
            X[:, 2] = 4 * np.cos(1.5 * u)

        return X

    @staticmethod
    def bullseye():
        """
        Samples on three concentric noisy circles in 2D.

        Returns
        -------
        X: ndarray(n_samples, 2)
            2D circles samples

        """

        N = 200
        sample_interval = np.linspace(0, 2 * np.pi, N, endpoint=False)
        c1 = np.array([np.sin(sample_interval), np.cos(sample_interval)]).T
        c2 = np.array([2 * np.sin(sample_interval), 2 * np.cos(sample_interval)]).T
        c3 = np.array([3 * np.sin(sample_interval), 3 * np.cos(sample_interval)]).T
        X = np.vstack((c1, c2, c3))

        np.random.seed(0)
        eps = 0.1
        X += (np.random.random(X.shape) - 0.5) * eps

        return X

    @staticmethod
    def three_circles():
        """
        Samples on two circles of different radii in 2D.

        Returns
        -------
        X: ndarray(n_samples, 2)
            2D circles samples

        """

        sample_interval_small_circle = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        small_circle = np.array(
            [np.sin(sample_interval_small_circle), np.cos(sample_interval_small_circle)]
        ).T
        sample_interval_big_circle = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        big_circle1 = np.array(
            [
                2 * np.sin(sample_interval_big_circle),
                2 * np.cos(sample_interval_big_circle),
            ]
        ).T + np.array([4, 0])
        big_circle2 = np.array(
            [
                2.5 * np.sin(sample_interval_big_circle),
                2.5 * np.cos(sample_interval_big_circle),
            ]
        ).T + np.array([-4, 0])
        X = np.vstack((small_circle, big_circle1, big_circle2))

        np.random.seed(0)
        eps = 0.3
        X += (np.random.random(X.shape) - 0.5) * eps

        return X

    @staticmethod
    def sphere(n_samples):
        """
        Samples on a 2-sphere in 3D.

        Returns
        -------
        X: ndarray(n_samples, 3)
            3D sphere samples

        """
        np.random.seed(0)
        data = 2 * np.random.random_sample((n_samples, 3)) - 1
        return data / np.linalg.norm(data, axis=1)[:, np.newaxis]

    @staticmethod
    def noisy_circle(n_samples, seed=0):
        """
        Samples on a circle in 2D.

        Returns
        -------
        X: ndarray(n_samples, 2)
            2D circle samples

        """
        np.random.seed(seed)
        X = np.random.random((n_samples, 2)) - 0.5
        return (
            X / np.linalg.norm(X, axis=1).reshape((n_samples, 1))
            + (np.random.random((n_samples, 2)) - 0.5) * 0.2
        )

    @staticmethod
    def moore_space_distance_matrix(rough_n_points=2000, prime=3):
        np.random.seed(0)
        X = (np.random.random((rough_n_points,2)) - 0.5) * 2
        X = X[np.linalg.norm(X,axis=1)<= 1]
        q = prime

        def _rot_mat(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array(((c, -s), (s, c)))

        R = _rot_mat((2 * np.pi) / q)

        n_points = X.shape[0]
        dist_mat = np.zeros((n_points, n_points))

        @jit
        def _fill_dist_mat(X, dist_mat, rot_mat, prime):
            for i, x in enumerate(X):
                for j, y in enumerate(X):
                    proj_x_to_boundary = x / np.linalg.norm(x)

                    dist_mat[i, j] = min(
                        # stay inside disk
                        np.linalg.norm(x - y),
                        # go to boundary and then to y
                        np.linalg.norm(x - proj_x_to_boundary)
                        + min(
                            [
                                np.linalg.norm(
                                    np.linalg.matrix_power(rot_mat, i) @ proj_x_to_boundary - y
                                )
                                for i in range(prime)
                            ]
                        ),
                    )
        _fill_dist_mat(X, dist_mat, R, prime)
        return dist_mat, X

class CircleMapUtils:
    """

    Utilities for adding, rotating, and plotting circle-valued maps.

    """

    @staticmethod
    def to_sinebow(circle_map):
        """
        Given a circle map construct an array of the same shape that can be fed as color in a matplotlib
        scatterplot to simulate the sinebow colormap.

        Parameters
        ----------
        circle_map: ndarray
            A numpy array of numbers between 0 and 2pi representing
            points on the circle.

        Returns
        -------
        ndarray
            A numpy array of floats to be used as color in a matplotlib scatterplot.

        """
        h = np.mod(circle_map / (2 * np.pi) + 0.5, 1)
        f = lambda x: np.sin(np.pi * x) ** 2
        return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)

    @staticmethod
    def center(circle_map):
        """
        Rotationally offset a circle-valued map so that most
        of the points map to the center of the circle (i.e., pi).

        Parameters
        ----------
        circle_map: ndarray
            A numpy array of numbers between 0 and 2pi representing
            points on the circle.

        Returns
        -------
        ndarray
            A numpy array of numbers between 0 and 2pi representing
            the rotation of the given points by the given offset.
        """
        bins = 50
        vals, ticks = np.histogram(circle_map, bins=bins)
        centered = ((circle_map - ticks[np.argmax(vals)]) + np.pi) % (2 * np.pi)
        return centered

    @staticmethod
    def offset(circle_map, offset):
        """
        Rotationally offset a circle-valued map.

        ----------
        circle_map: ndarray
            A numpy array of numbers between 0 and 2pi representing
            points on the circle.

        offset: float
            A number between 0 and 1 representing a rotational offset.

        Returns
        -------
        ndarray
            A numpy array of numbers between 0 and 2pi representing
            the rotation of the given points by the given offset.

        """

        return (circle_map + offset * (2 * np.pi)) % (2 * np.pi)

    @staticmethod
    def linear_combination(circle_maps, linear_combination_matrix):
        """
        Given k circle-valued maps on a dataset with n points and an l x k
        matrix with integer coefficients, return the l linear combinations of the
        given circle-valued maps induced by the given matrix.

        Parameters
        ----------
        circle_maps: list or ndarray(k, n, dtype=float)
            A numpy array with rows containing n points in the circle represented as
            floats between 0 and 2pi.

        linear_combination_matrix: list or ndarray(l, k, dtype=int)
            A numpy array encoding l integer linear combinations of the given k
            circle-valued maps.

        Returns
        -------
        ndarray(l, n, dtype=float)
            A numpy array with rows containing n points in the circle representing
            the l linear combinations of the given k circle-valued maps.
        """

        if not isinstance(circle_maps, np.ndarray):
            circle_maps = np.array(circle_maps)
        assert len(circle_maps.shape) == 2
        if not isinstance(linear_combination_matrix, np.ndarray):
            linear_combination_matrix = np.array(linear_combination_matrix)
        assert (
            len(linear_combination_matrix.shape) == 2
            or len(linear_combination_matrix.shape) == 1
        )

        if len(linear_combination_matrix.shape) == 2:
            assert circle_maps.shape[0] == linear_combination_matrix.shape[1]
            return (linear_combination_matrix @ circle_maps) % (2 * np.pi)
        else:
            assert circle_maps.shape[0] == linear_combination_matrix.shape[0]
            return (
                (np.array([linear_combination_matrix]) @ circle_maps) % (2 * np.pi)
            )[0]

    @staticmethod
    def levelset_coloring(circle_map, n_levelsets=4, smoothing=0.25):
        """
        Given points on the circle and a number of levelsets subdivide the
        circle into the given number of levelsets and return a smoothened
        membership function to the levelsets. This is useful for coloring a
        dataset X according to a circle-valued map X -> S^1.

        Parameters
        ----------
        circle_map: ndarray
            A numpy array of numbers between 0 and 2pi representing
            points on the circle.

        n_levelset: int, optional, default is 4
            Number of levelsets to evenly cover the circle.

        smoothing: float, optional, default is 0.25
            How much to smoothen the membership function

        Returns
        -------
        ndarray
            The smoothened membership function of each of the given
            points in the circle.
        """
        assert isinstance(n_levelsets, int)
        assert n_levelsets > 0
        n_levelsets *= 2
        colors = circle_map / (2 * np.pi)
        # transition should be between 0 (very fast) and 1 (slow)
        if smoothing == 0:
            return np.array([np.floor(c * n_levelsets) % 2 for c in colors])
        k = smoothing

        def sigmoid(x):
            x = (x - 0.5) * 2
            s = 1 / (1 + np.exp(-x / k))
            return s

        def triangle(y):
            return y if y < 1 else 2 - y

        return np.array([sigmoid(triangle((c * n_levelsets) % 2)) for c in colors])


class ProjectiveMapUtils:
    """
    Utilities for manipulating projective space-valued maps.

    """

    @staticmethod
    def get_stereo_proj_codim1(pX, u=np.array([])):
        """
        Do a projective stereographic projection

        Parameters
        ----------
        pX: ndarray(N, d)
            A collection of N points in d dimensions
        u: ndarray(d)
            A unit vector representing the north pole
            for the stereographic projection

        Returns
        -------
        S: ndarray(N, d-1)
            The stereographically projected coordinates

        """

        X = pX.T
        # Put points all on the same hemisphere
        if u.size == 0:
            _, U = np.linalg.eigh(X.dot(X.T))
            u = U[:, 0]
        XX = ProjectiveMapUtils.rotmat(u).dot(X)
        ind = XX[-1, :] < 0
        XX[:, ind] *= -1
        # Do stereographic projection
        S = XX[0:-1, :] / (1 + XX[-1, :])[None, :]
        return S.T

    @staticmethod
    def circle_to_3dnorthpole(x):
        """
        Convert a point selected on the circle to a 3D
        unit vector on the upper hemisphere

        Parameters
        ----------
        x: ndarray(2)
            Selected point in the circle

        Returns
        -------
        x: ndarray(2)
            Selected point in the circle (possibly clipped to the circle)
        u: ndarray(3)
            Unit vector on the upper hemisphere

        """
        magSqr = np.sum(x**2)
        if magSqr > 1:
            x /= np.sqrt(magSqr)
            magSqr = 1
        u = np.zeros(3)
        u[0:2] = x
        u[2] = np.sqrt(1 - magSqr)
        return x, u

    @staticmethod
    def hopf_map(X):
        """
        TODO
        """
        Y = np.zeros((2 * X.shape[1], X.shape[0]))
        Y[::2, :] = np.real(X).T
        Y[1::2, :] = np.imag(X).T
        return np.array(
            [
                2 * (np.prod(Y[[0, 2], :], axis=0) + np.prod(Y[[1, 3], :], axis=0)),
                2 * (np.prod(Y[[1, 2], :], axis=0) - np.prod(Y[[0, 3], :], axis=0)),
                np.sum(Y[[0, 1], :] ** 2, axis=0) - np.sum(Y[[2, 3], :] ** 2, axis=0),
            ]
        ).T

    @staticmethod
    def stereographic_projection_hemispheres(X, center_vector=None):
        """
        TODO
        """

        def _stereo(v):
            return v[:, :-1] / (1 - v[:, -1])[:, None]

        n = X.shape[1]
        if center_vector is None:
            center_vector = np.zeros((n))
            center_vector[-1] = 1
        centering_rotation = ProjectiveMapUtils.rotmat(center_vector)
        X_ = X @ centering_rotation.T
        e1 = np.zeros((n - 1))
        e1[0] = 1
        res = np.zeros((X_.shape[0], n - 1))
        res[X_[:, -1] < 0, :] = _stereo(X_[X_[:, -1] < 0, :])
        Y = X_[X_[:, -1] >= 0, :]
        Y[:, -1] *= -1
        res[X_[:, -1] >= 0, :] = _stereo(Y) + 2.5 * e1
        return res

    @staticmethod
    def rotmat(a, b=np.array([])):
        """
        Construct a d x d rotation matrix that rotates
        a vector a so that it coincides with a vector b

        Parameters
        ----------
        a : ndarray (d)
            A d-dimensional vector that should be rotated to b
        b : ndarray(d)
            A d-dimensional vector that should end up residing at
            the north pole (0,0,...,0,1)

        """
        if (len(a.shape) > 1 and np.min(a.shape) > 1) or (
            len(b.shape) > 1 and np.min(b.shape) > 1
        ):
            print("Error: a and b need to be 1D vectors")
            return None
        a = a.flatten()
        a = a / np.sqrt(np.sum(a**2))
        d = a.size

        if b.size == 0:
            b = np.zeros(d)
            b[-1] = 1
        b = b / np.sqrt(np.sum(b**2))

        c = a - np.sum(b * a) * b
        # If a numerically coincides with b, don't rotate at all
        if np.sqrt(np.sum(c**2)) < 1e-15:
            return np.eye(d)

        # Otherwise, compute rotation matrix
        c = c / np.sqrt(np.sum(c**2))
        lam = np.sum(b * a)
        beta = np.sqrt(1 - np.abs(lam) ** 2)
        rot = (
            np.eye(d)
            - (1 - lam) * (c[:, None].dot(c[None, :]))
            - (1 - lam) * (b[:, None].dot(b[None, :]))
            + beta * (b[:, None].dot(c[None, :]) - c[:, None].dot(b[None, :]))
        )
        return rot


class LensMapUtils:
    # TODO: docstring

    @staticmethod
    def lens_3D_to_disk_3D(X,q):
        # TODO: docstring
        def _point_lens_to_sphere(p,q):
            p1 = p[0]
            p2 = p[1]
            arg_z = np.mod(np.angle(p1), 2 * np.pi)
            theta = np.mod(arg_z, 2 * np.pi / q)

            k = np.floor((arg_z - theta) / (2 * np.pi / q))

            phi = np.mod(np.angle(p2), 2 * np.pi) - 2 * k * np.pi / q

            r = np.abs(p2)
            x, y, z = (
                r * np.cos(phi),
                r * np.sin(phi),
                (q / np.pi) * (theta - np.pi / q) * np.sqrt(1 - r**2),
            )
            return [x,y,z]

        return np.array([_point_lens_to_sphere(p,q) for p in X])

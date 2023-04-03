"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To provide a number of utility functions, including
- Quickly computing all pairs self-similarity and cross-similarity matrices
- Doing "greedy permutations" 
- Dome topological operations like adding cocycles and creating partitions of unity
- Some relevant geometric examples for tests
- Some utilities for plotting
"""
import numpy as np
import scipy.sparse as sparse


"""#########################################
            Geometry Utilities
#########################################"""


class GeometryUtils:

    """#########################################
        Self-Similarity And Cross-Similarity
    #########################################"""

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
        ------
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
        ------
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

    """#########################################
            Greedy Permutations
    #########################################"""

    @staticmethod
    def get_greedy_perm_pc(X, M, verbose=False, csm_fn=get_csm.__func__):
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
        verbose: boolean
            Whether to print progress
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
    def get_greedy_perm_dm(D, M, verbose=False):
        """
        A Naive O(NM) algorithm to do furthest points sampling, assuming
        the input is a N x N distance matrix

        Parameters
        ----------
        D : ndarray (N, N)
            An N x N distance matrix
        M : integer
            Number of landmarks to compute
        verbose: boolean
            Whether to print progress

        Return
        ------
        result: Dictionary
            {'perm': An array of indices into X of the greedy permutation
            'lambdas': Insertion radii of the landmarks
            'DLandmarks': An MxN array of distances from landmarks to points in the point cloud}
        """
        # By default, takes the first point in the permutation to be the
        # first point in the point cloud, but could be random
        N = D.shape[0]
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


"""#########################################
        Cohomology Utility Functions
#########################################"""


def linear_combination_one_cocycles(
    cocycles, coefficients, characteristic=0, real=False
):
    """
    Compute a linear combination of cocycles

    Parameters
    ----------
    cocycles : list of triples [vertex index, vertex index, value]
        List representing a list of cocycles

    coefficients: list of numbers
        Numbers representing the coefficient of each cocycle in the linear combination

    characteristic : integer, optional
        Integer representing the characteristic to mod out after performing linear combination.
        If zero, then no mod operation is performed.

    real : boolean, optional
        Whether to treat the values in the cocycles as floats or as ints.
    """
    assert len(cocycles) == len(coefficients)
    assert len(cocycles) > 0
    assert isinstance(characteristic, int)
    assert (not real) or (characteristic == 0)

    res_as_dict = {}
    for cocycle, coefficient in zip(cocycles, coefficients):
        for i, j, v in cocycle:
            i, j = min(i, j), max(i, j)
            if not (i, j) in res_as_dict:
                res_as_dict[(i, j)] = v * coefficient
            else:
                res_as_dict[(i, j)] += v * coefficient

    dtype = np.float32 if real else np.int
    res_as_list = list(res_as_dict.items())
    res = np.zeros((len(res_as_dict), 3), dtype=dtype)
    res[:, 0:2] = np.array([ij for ij, _ in res_as_list])
    res[:, 2] = np.array([v for _, v in res_as_list])
    if characteristic > 0:
        res[:, 2] = np.mod(res[:, 2], characteristic)
    return res


def add_cocycles(c1, c2, p=0, real=False):
    """
    Add two cocycles together under a field

    Parameters
    ----------
    c1: ndarray(N)
        First cocycle
    c2: ndarray(N)
        Second cocycle
    p: int
        Field
    real: bool
        Whether this is meant to be a real cocycle
    """
    return linear_combination_one_cocycles(
        [c1, c2], [1, 1], characteristic=p, real=real
    )


def _make_delta0(R):
    """
    Return the delta0 coboundary matrix

    Parameters
    ----------
    R: ndarray(n_edges, 2, dtype=int)
        A matrix specifying edges, where orientation
        is taken from the first column to the second column
        R specifies the "natural orientation" of the edges, with the
        understanding that the ranking will be specified later
        It is assumed that there is at least one edge incident
        on every vertex

    Returns
    -------
    scipy.sparse.csr_matrix((n_edges, n_vertices))
        The coboundary 0 matrix
    """
    n_vertices = int(np.max(R) + 1)
    n_edges = R.shape[0]
    # Two entries per edge
    I = np.zeros((n_edges, 2))
    I[:, 0] = np.arange(n_edges)
    I[:, 1] = np.arange(n_edges)
    I = I.flatten()
    J = R[:, 0:2].flatten()
    V = np.zeros((n_edges, 2))
    V[:, 0] = -1
    V[:, 1] = 1
    V = V.flatten()
    I = np.array(I, dtype=int)
    J = np.array(J, dtype=int)
    Delta = sparse.coo_matrix((V, (I, J)), shape=(n_edges, n_vertices)).tocsr()
    return Delta


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


"""#########################################
        Partition of Unity Functions
#########################################"""


class PartUnity:
    @staticmethod
    def linear(ds, r_cover):
        """
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


"""#########################################
            Geometry Examples
#########################################"""

## TODO: These probably belong in tdasets, but I'll keep them here for now


class GeometryExamples:
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
        eps = 0.3
        X += (np.random.random(X.shape) - 0.5) * eps

        return X


"""#########################################
        Matplotlib Plotting Utilities
#########################################"""


class PlotUtils:
    @staticmethod
    def imscatter(X, P, dim, zoom=1):
        """
        Plot patches in specified locations in R2

        Parameters
        ----------
        X : ndarray (N, 2)
            The positions of each patch in R2
        P : ndarray (N, dim*dim)
            An array of all of the patches
        dim : int
            The dimension of each patch

        """
        # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.pyplot as plt

        ax = plt.gca()
        for i in range(P.shape[0]):
            patch = np.reshape(P[i, :], (dim, dim))
            x, y = X[i, :]
            im = OffsetImage(patch, zoom=zoom, cmap="gray")
            ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(X)
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def plot_patches(P, zoom=1):
        """
        Plot patches in a best fitting rectangular grid
        """
        N = P.shape[0]
        d = int(np.sqrt(P.shape[1]))
        dgrid = int(np.ceil(np.sqrt(N)))
        ex = np.arange(dgrid)
        x, y = np.meshgrid(ex, ex)
        X = np.zeros((N, 2))
        X[:, 0] = x.flatten()[0:N]
        X[:, 1] = y.flatten()[0:N]
        PlotUtils.imscatter(X, P, d, zoom)

    @staticmethod
    def plot_proj_boundary():
        """
        Depict the boundary of RP2 on the unit circle
        """
        import matplotlib.pyplot as plt

        t = np.linspace(0, 2 * np.pi, 200)
        plt.plot(np.cos(t), np.sin(t), "c")
        plt.axis("equal")
        ax = plt.gca()
        ax.arrow(
            -0.1, 1, 0.001, 0, head_width=0.15, head_length=0.2, fc="c", ec="c", width=0
        )
        ax.arrow(
            0.1,
            -1,
            -0.001,
            0,
            head_width=0.15,
            head_length=0.2,
            fc="c",
            ec="c",
            width=0,
        )
        ax.set_facecolor((0.35, 0.35, 0.35))

    @staticmethod
    def set_axes_equal(ax):
        # taken from https://stackoverflow.com/a/31364297/2171328
        """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    @staticmethod
    def plot_2d_scatter_with_different_colorings(
        X, colorings=[], cmap="viridis", width=10, point_size=2
    ):
        """
        Plot a 2D point cloud as many times as the number of colorings given.

        Parameters
        ----------
        X : ndarray (N, 2)
            A point cloud in 2D
        colorings : ndarray (n, N) or list of lists
            A list of n colorings, each one consisting of a list or array of N floats representing
            the color of each data point
        cmap : string
            Matplotlib colormap to use
        width : int
            The width of the final plot
        """
        import matplotlib.pyplot as plt

        if len(colorings) == 0:
            plt.figure(figsize=(4, 4))
            plt.scatter(X[:, 0], X[:, 1], s=point_size)
            plt.axis("equal")
            plt.axis("off")
        elif len(colorings) == 1:
            plt.figure(figsize=(4, 4))
            plt.scatter(X[:, 0], X[:, 1], s=point_size, c=colorings[0], cmap=cmap)
            plt.axis("equal")
            plt.axis("off")
        else:
            num_colorings = len(colorings)
            fig, axs = plt.subplots(1, num_colorings)
            fig.set_figwidth(width)

            for i, c in enumerate(colorings):
                axs[i].scatter(X[:, 0], X[:, 1], s=point_size, c=c, cmap=cmap)
                axs[i].set_aspect("equal")
                axs[i].axis("off")


"""#########################################
        Matplotlib Plotting Utilities
#########################################"""


class CircleMapUtils:
    @staticmethod
    def offset(circle_map, offset):
        """
        Rotationally offset a circle-valued map.

        ----------
        circle_map: ndarray
            A numpy array of numbers between 0 and 2pi representing
            points on the circle.

        offset: float
            A number between 0 and 2pi representing a rotational offset.

        Returns
        -------
        ndarray
            A numpy array of numbers between 0 and 2pi representing
            the rotation of the given points by the given offset.
        """

        return (circle_map + offset) % (2 * np.pi)

    @staticmethod
    def linear_combination(circle_maps, linear_combination_matrix):
        """
        Given k circle-valued maps on a dataset with n points and an l x k
        matrix with integer coefficients, return the l linear combinations of the
        given circle-valued maps induced by the given matrix.

        Parameters
        ----------
        circle_maps: ndarray(k, n, dtype=float)
            A numpy array with rows containing n points in the circle represented as
            floats between 0 and 2pi.

        linear_combination_matrix: ndarray(l, k, dtype=int)
            A numpy array encoding l integer linear combinations of the given k
            circle-valued maps.

        Returns
        -------
        ndarray(l, n, dtype=float)
            A numpy array with rows containing n points in the circle representing
            the l linear combinations of the given k circle-valued maps.
        """

        assert isinstance(circle_maps, np.ndarray)
        assert len(circle_maps.shape) == 2
        assert isinstance(linear_combination_matrix, np.ndarray)
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

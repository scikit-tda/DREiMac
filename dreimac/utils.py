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
        C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
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
        return {'Y':Y, 'perm':perm, 'lambdas':lambdas, 'D':D}

    @staticmethod
    def get_greedy_perm_dm(D, M, verbose = False):
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
        return {'perm':perm, 'lambdas':lambdas, 'DLandmarks':DLandmarks}



"""#########################################
        Cohomology Utility Functions
#########################################"""

def add_cocycles(c1, c2, p = 2, real = False):
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
    S = {}
    c = np.concatenate((c1, c2), 0)
    for k in range(c.shape[0]):
        [i, j, v] = c[k, :]
        i, j = min(i, j), max(i, j)
        if not (i, j) in S:
            S[(i, j)] = v
        else:
            S[(i, j)] += v
    cret = np.zeros((len(S), 3))
    cret[:, 0:2] = np.array([s for s in S])
    cret[:, 2] = np.array([np.mod(S[s], p) for s in S])
    dtype = np.int64
    if real:
        dtype = np.float32
    cret = np.array(cret[cret[:, -1] > 0, :], dtype = dtype)
    return cret

def make_delta0(R):
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
    #Two entries per edge
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
    idx_map = -1*np.ones(N, dtype=int)
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
        return (r_cover - ds)**2

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
        return np.exp(r_cover**2/(ds**2-r_cover**2))


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
        N = n_angles*n_offsets
        P = np.zeros((N, dim*dim))
        thetas = np.linspace(0, np.pi, n_angles+1)[0:n_angles]
        #ps = np.linspace(-0.5*np.sqrt(2), 0.5*np.sqrt(2), n_offsets)
        ps = np.linspace(-1, 1, n_offsets)
        idx = 0
        [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
        for i in range(n_angles):
            c = np.cos(thetas[i])
            s = np.sin(thetas[i])
            for j in range(n_offsets):
                patch = X*c + Y*s + ps[j]
                patch = np.exp(-patch**2/sigma**2)
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
        X = X/np.sqrt(np.sum(X**2, 1))[:, None]
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
        s = np.random.rand(n_samples)*2*np.pi
        t = np.random.rand(n_samples)*2*np.pi
        X[:, 0] = (R + r*np.cos(s))*np.cos(t)
        X[:, 1] = (R + r*np.cos(s))*np.sin(t)
        X[:, 2] = r*np.sin(s)
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
        theta = np.random.rand(n_samples)*2*np.pi
        phi = np.random.rand(n_samples)*2*np.pi
        X = np.zeros((n_samples, 4))
        X[:, 0] = (R + r*np.cos(theta))*np.cos(phi)
        X[:, 1] = (R + r*np.cos(theta))*np.sin(phi)
        X[:, 2] = r*np.sin(theta)*np.cos(phi/2)
        X[:, 3] = r*np.sin(theta)*np.sin(phi/2)
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
        #https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.pyplot as plt
        ax = plt.gca()
        for i in range(P.shape[0]):
            patch = np.reshape(P[i, :], (dim, dim))
            x, y = X[i, :]
            im = OffsetImage(patch, zoom=zoom, cmap = 'gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(X)
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def plot_patches(P, zoom = 1):
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
        t = np.linspace(0, 2*np.pi, 200)
        plt.plot(np.cos(t), np.sin(t), 'c')
        plt.axis('equal')
        ax = plt.gca()
        ax.arrow(-0.1, 1, 0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
        ax.arrow(0.1, -1, -0.001, 0, head_width = 0.15, head_length = 0.2, fc = 'c', ec = 'c', width = 0)
        ax.set_facecolor((0.35, 0.35, 0.35))
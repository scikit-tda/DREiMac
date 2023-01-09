"""
A superclass for shared code across all different types of coordinates
"""
import numpy as np
from scipy.sparse.linalg import lsqr
import time
from .utils import *
from ripser import ripser
import warnings

"""#########################################
    Some Window Management Utilities
#########################################"""

DREIMAC_FIG_RES = 5 # The resolution of a square cell in inches

def in_notebook(): # pragma: no cover
    """
    Return true if we're in a notebook session, and false otherwise
    with help from https://stackoverflow.com/a/22424821
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except:
        return False
    return True

def compute_dpi(width_cells, height_cells, width_frac=0.5, height_frac=0.65, verbose=False):
    """
    Automatically compute the dpi so that the figure takes
    up some proportion of the available screen width/height
    Parameters
    ----------
    width_cells: float
        The target width of the figure, in units of DREIMAC_FIG_RES
    height_inches: float
        The target height of the figure, in units of DREIMAC_FIG_RES
    width_frac: float
        The fraction of the available width to take up
    height_frac: float
        The fraction of the available height to take up
    verbose: boolean
        Whether to print information about the dpi calculation
    """
    width_inches = DREIMAC_FIG_RES*width_cells
    height_inches = DREIMAC_FIG_RES*height_cells
    # Try to use the screeninfo library to figure out the size of the screen
    width = 1200
    height = 900
    try:
        import screeninfo
        monitor = screeninfo.get_monitors()[0]
        width = monitor.width
        height = monitor.height
    except:
        warnings.warn("Could not accurately determine screen size")
    dpi_width = int(width_frac*width/width_inches)
    dpi_height = int(height_frac*height/height_inches)
    dpi = min(dpi_width, dpi_height)
    if verbose:
        print("width = ", width)
        print("height = ", height)
        print("dpi_width = ", dpi_width)
        print("dpi_height = ", dpi_height)
        print("dpi = ", dpi)
    return dpi

"""#########################################
        Main Circular Coordinates Class
#########################################"""

class EMCoords(object):
    def __init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose):
        """
        Perform persistent homology on the landmarks, store distance
        from the landmarks to themselves and to the rest of the points,
        and sort persistence diagrams and cocycles in decreasing order of persistence

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
        assert(maxdim >= 1)
        self.verbose = verbose
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        res = ripser(X, distance_matrix=distance_matrix, coeff=prime, maxdim=maxdim, n_perm=n_landmarks, do_cocycles=True)
        if verbose:
            print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
        self.X_ = X
        self.prime_ = prime
        self.dgms_ = res['dgms']
        self.dist_land_data_ = res['dperm2all']
        self.idx_land_ = res['idx_perm']
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
        self.cocycles_ = res['cocycles']
        # Sort persistence diagrams in descending order of persistence
        idxs = np.argsort(self.dgms_[1][:, 0]-self.dgms_[1][:, 1])
        self.dgms_[1] = self.dgms_[1][idxs, :]
        self.dgm1_lifetime = np.array(self.dgms_[1])
        self.dgm1_lifetime[:, 1] -= self.dgm1_lifetime[:, 0]
        self.cocycles_[1] = [self.cocycles_[1][idx] for idx in idxs]
        reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])
        self.n_landmarks_ = n_landmarks
        self.type_ = "emcoords"
    
    def get_rep_cocycle(self, cocycle_idx):
        """
        Compute the representative cocycle, given a list of cocycle indices

        Parameters
        ----------
        cocycle_idx : list
            Add the cocycles together at the indices in this list
        
        Returns
        -------
        cohomdeath: float
            Cohomological death
        cohombirth: float
            Cohomological birth
        cocycle: ndarray(K, 3, dtype=int)
            Representative cocycle.  First two columns are vertex indices,
            and third column is value in field of prime self.prime_
        """
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        cohomdeath = -np.inf
        cohombirth = np.inf
        cocycle = np.zeros((0, 3))
        prime = self.prime_
        for k in range(len(cocycle_idx)):
            cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
            cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
            cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])
        return cohomdeath, cohombirth, cocycle
    
    def get_cover_radius(self, perc, cohomdeath, cohombirth):
        """
        Determine radius for covering balls

        Parameters
        ----------
        perc : float
            Percent coverage
        cohomdeath: float
            Cohomological death
        cohombirth: float
            Cohomological birth
        
        Returns
        -------
        float: Covering radius
        """
        dist_land_data = self.dist_land_data_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        self.r_cover_ = r_cover # Store covering radius for reference
        if self.verbose:
            print("r_cover = %.3g"%r_cover)
        self.r_cover_ = r_cover
        return r_cover
    
    def get_covering_partition(self, r_cover, partunity_fn):
        """
        Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

        Parameters
        ----------
        r_cover: float
            Covering radius
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        
        Returns
        -------
        varphi: ndarray(n_data, dtype=float)
            varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b)),
        ball_indx: ndarray(n_data, dtype=int)
            The index of the first open set each data point belongs to
        """
        dist_land_data = self.dist_land_data_
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity 
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are {} point not covered by a landmark".format(nzero))
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]
        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)
        return varphi, ball_indx
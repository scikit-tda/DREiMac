"""
A superclass for shared code across all different types of coordinates
"""
import numpy as np
import scipy
from scipy.sparse.linalg import lsqr
import time
from .geomtools import *
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
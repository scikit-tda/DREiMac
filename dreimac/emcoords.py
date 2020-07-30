"""
A superclass for shared code across all different types of coordinates
"""
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import scipy
from scipy.sparse.linalg import lsqr
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, RadioButtons
from .geomtools import *
from ripser import ripser
import warnings


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
        self.cocycles_[1] = [self.cocycles_[1][idx] for idx in idxs]
        reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])
        self.n_landmarks_ = n_landmarks
        self.type_ = "emcoords"

    def setup_ax_persistence(self):
        """
        Setup the persistence plot in an interactive window
        """
        dgm = self.dgms_[1]
        ax_min, ax_max = np.min(dgm), np.max(dgm)
        x_r = ax_max - ax_min
        buffer = x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        y_down, y_up = x_down, x_up
        self.ax_persistence.plot([x_down, x_up], [x_down, x_up], "--", c=np.array([0.0, 0.0, 0.0]))
        self.dgmplot, = self.ax_persistence.plot(dgm[:, 0], dgm[:, 1], 'o', picker=5, c='C0')
        self.selected_plot = self.ax_persistence.scatter([], [], 100, c='C1')
        self.ax_persistence.set_xlim([x_down, x_up])
        self.ax_persistence.set_ylim([y_down, y_up])
        self.ax_persistence.set_aspect('equal', 'box')
        self.ax_persistence.set_title("Persistent H1")
        self.ax_persistence.set_xlabel("Birth")
        self.ax_persistence.set_ylabel("Death")

    def get_selected_info(self):
        """
        Return information about what the user selected in
        the interactive plot
        Returns
        -------
        {
            'partunity_fn': (dist_land_data, r_cover) -> phi
                The selected function handle for the partition of unity
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'perc': float
                The selected percent coverage,
            'u':ndarray(3, float)
                The stereographic north pole
        }
        """
        return {
                'partunity_fn':PARTUNITY_FNS[self.partunity_selector.value_selected], 
                'cocycle_idxs':np.array(list(self.selected)), 
                'perc':self.perc_slider.val,
                'u':self.u
                }
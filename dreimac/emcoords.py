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
from matplotlib.widgets import Slider, RadioButtons, Button
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

    def setup_ax_persistence(self, y_compress=1):
        """
        Setup the persistence plot in an interactive window
        Parameters
        ----------
        y_compress: float
            The factor by which to compress the y-axis to make room
            for a plot underneath
        """
        dgm = self.dgm1_lifetime
        # Switch to lifetime
        ax_min, ax_max = np.min(dgm), np.max(dgm)
        x_r = ax_max - ax_min
        buffer = x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        y_down, y_up = x_down, x_up
        y_up += (y_up-y_down)*(y_compress-1)
        self.ax_persistence.plot([0, ax_max], [0, 0], "--", c=np.array([0.0, 0.0, 0.0]))
        self.dgmplot, = self.ax_persistence.plot(dgm[:, 0], dgm[:, 1], 'o', c='C0', picker=5)
        self.selected_plot = self.ax_persistence.scatter([], [], 100, c='C1')
        self.ax_persistence.set_xlim([x_down, x_up])
        self.ax_persistence.set_ylim([y_down, y_up])
        self.ax_persistence.set_aspect('equal', 'box')
        self.ax_persistence.set_title("H1 Cocycle Selection")
        self.ax_persistence.set_xlabel("Birth")
        self.ax_persistence.set_ylabel("Lifetime")
        self.persistence_text_labels = [self.ax_persistence.text(dgm[i, 0], dgm[i, 1], '') for i in range(dgm.shape[0])]

    def recompute_coords(self, clicked=[], clear_persistence_text = False):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram
        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        clear_persistence_text: boolean
            Whether to clear all previously labeled dots
        """
        self.selected = self.selected.symmetric_difference(set(clicked))
        idxs = np.array(list(self.selected))
        fmt = "c%i +"*len(idxs)
        fmt = fmt[0:-1]
        self.selected_cocycle_text.set_text(fmt%tuple(idxs))
        if clear_persistence_text:
            for label in self.persistence_text_labels:
                label.set_text("")
        for idx in idxs:
            self.persistence_text_labels[idx].set_text("%i"%idx)
        if idxs.size > 0:
            ## Step 1: Highlight point on persistence diagram
            self.selected_plot.set_offsets(self.dgm1_lifetime[idxs, :])
            ## Step 2: Update coordinates
            perc = self.perc_slider.val
            partunity_fn = PARTUNITY_FNS[self.partunity_selector.value_selected]
            self.coords = self.get_coordinates(cocycle_idx = idxs, perc=perc, partunity_fn = partunity_fn)
        else:
            self.coords = {'X':np.zeros((0, 2))}
            self.selected_plot.set_offsets(np.zeros((0, 2)))

    def setup_param_chooser_gui(self, fig, xstart, ystart, width, height, init_params, button_idx = -1):
        """
        Setup a GUI area 
        Parameters
        ----------
        fig: matplotlib figure handle
            Handle to the interactive figure
        xstart: float
            Where this GUI element is starting along x
        ystart: float
            Where this GUI element is starting along y
        width: float
            Width of GUI element
        height: float
            Height of GUI element
        init_params: dict
            Initial parameters
        button_idx: int
            Index of the circular coordinate to create a press button, or None
            if no button is created
        Returns
        -------
        percslider: matplotlib.widgets.Slider
            Handle to to the slider for choosing coverage
        partunity_selector: matplotlib.widgets.RadioButtons
            Radio buttons for choosing partition of unity type
        """
        # Percent coverage slider
        ax_perc_slider = fig.add_axes([xstart, ystart+height*0.15, 0.5*width, 0.02])
        perc = init_params['perc']
        perc_slider = Slider(ax_perc_slider, "Coverage", valmin=0, valmax=1, valstep=0.01, valinit=perc)
        
        # Partition of unity radio button
        ax_part_unity_label = fig.add_axes([xstart-width*0.175, ystart, 0.3*width, 0.045])
        ax_part_unity_label.text(0.1, 0.3, "Partition\nof Unity")
        ax_part_unity_label.set_axis_off()
        ax_part_unity = fig.add_axes([xstart, ystart, 0.2*width, 0.045])
        active_idx = 0
        partunity_fn = init_params['partunity_fn']
        partunity_keys = tuple(PARTUNITY_FNS.keys())
        for i, key in enumerate(partunity_keys):
            if partunity_fn == PARTUNITY_FNS[key]:
                active_idx = i
        partunity_selector = RadioButtons(ax_part_unity, partunity_keys, active=active_idx)

        # Selected cocycle display
        ax_selected_cocycles_label = fig.add_axes([xstart-width*0.175, ystart-height*0.15, 0.3*width, 0.045])
        ax_selected_cocycles_label.text(0.1, 0.3, "Selected\nCocycle")
        ax_selected_cocycles_label.set_axis_off()
        ax_selected_cocycles = fig.add_axes([xstart, ystart-height*0.15, 0.2*width, 0.045])
        selected_cocycle_text = ax_selected_cocycles.text(0.02, 0.5, "")
        ax_selected_cocycles.set_axis_off()

        # Button to select this particular coordinate
        select_button = None
        if button_idx > -1:
            ax_button_label = fig.add_axes([xstart+width*0.25, ystart, 0.2*width, 0.045])
            select_button = Button(ax_button_label, "Coords {}".format(button_idx))
        return perc_slider, partunity_selector, selected_cocycle_text, select_button

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
        }
        """
        return {
                'partunity_fn':PARTUNITY_FNS[self.partunity_selector.value_selected], 
                'cocycle_idxs':np.array(list(self.selected)), 
                'perc':self.perc_slider.val,
                }

"""#########################################
        Miscellaneous Utilities
#########################################"""

def callback_factory(callback, k):
    """
    Setup a callback that's linked to a particular
    circular coordinate index.  Having this function takes
    care of scoping issues
    Parameters
    ----------
    callback: function
        The callback to use
    k: int
        The index of the circular coordinate
    """
    return lambda evt: callback(evt, k)

def set_pi_axis_labels(ax, labels):
    """
    Set the axis labels of plots to be the pi symbols
    Parameters
    ----------
    ax: matplotlib handle
        The axes of which to change the symbol labels
    labels: list of string
        The names of the axes
    """
    ax.set_xlabel(labels[0])
    ax.set_xlim([-0.2, 2*np.pi+0.2])
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax.set_ylabel(labels[1])
    ax.set_ylim([-0.2, 2*np.pi+0.2])
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
    if len(labels) > 2:
        ax.set_zlabel(labels[1])
        ax.set_zlim([-0.2, 2*np.pi+0.2])
        ax.set_zticks([0, np.pi, 2*np.pi])
        ax.set_zticklabels(["0", "$\\pi$", "$2\\pi$"])

def set_3dplot_equalaspect(ax, X, pad=0.1):
    """
    An equal aspect ratio hack for 3D
    Parameters
    ----------
    ax: matplotlib axis
        Handle to the axis to change
    X: ndarray(N, 3)
        Point cloud that's being plotted
    pad: float
        The factor of padding to use around the point cloud
    """
    maxes = np.max(X)
    mins = np.min(X)
    r = maxes - mins
    ax.set_xlim([mins-r*pad, maxes+r*pad])
    ax.set_ylim([mins-r*pad, maxes+r*pad])
    ax.set_zlim([mins-r*pad, maxes+r*pad])
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
from .emcoords import *
from ripser import ripser
import warnings


"""#########################################
        Main Circular Coordinates Class
#########################################"""

class CircularCoords(EMCoords):
    def __init__(self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=1, verbose=False):
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
        self.type_ = "circ"

    def get_coordinates(self, perc = 0.99, do_weighted = False, cocycle_idx = [0], partunity_fn = partunity_linear):
        """
        Perform circular coordinates via persistent cohomology of 
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        do_weighted : boolean
            Whether to make a weighted cocycle on the representatives
        cocycle_idx : list
            Add the cocycles together in this list
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        """
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        cohomdeath = -np.inf
        cohombirth = np.inf
        cocycle = np.zeros((0, 3))
        prime = self.prime_
        for k in range(len(cocycle_idx)):
            cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=prime)
            cohomdeath = max(cohomdeath, dgm1[cocycle_idx[k], 0])
            cohombirth = min(cohombirth, dgm1[cocycle_idx[k], 1])

        ## Step 2: Determine radius for balls
        dist_land_data = self.dist_land_data_
        dist_land_land = self.dist_land_land_
        coverage = np.max(np.min(dist_land_data, 1))
        r_cover = (1-perc)*max(cohomdeath, coverage) + perc*cohombirth
        self.r_cover_ = r_cover # Store covering radius for reference
        if self.verbose:
            print("r_cover = %.3g"%r_cover)
        

        ## Step 3: Setup coboundary matrix, delta_0, for Cech_{r_cover }
        ## and use it to find a projection of the cocycle
        ## onto the image of delta0

        #Lift to integer cocycle
        val = np.array(cocycle[:, 2])
        val[val > (prime-1)/2] -= prime
        Y = np.zeros((n_landmarks, n_landmarks))
        Y[cocycle[:, 0], cocycle[:, 1]] = val
        Y = Y + Y.T
        #Select edges that are under the threshold
        [I, J] = np.meshgrid(np.arange(n_landmarks), np.arange(n_landmarks))
        I = I[np.triu_indices(n_landmarks, 1)]
        J = J[np.triu_indices(n_landmarks, 1)]
        Y = Y[np.triu_indices(n_landmarks, 1)]
        idx = np.arange(len(I))
        idx = idx[dist_land_land[I, J] < 2*r_cover]
        I = I[idx]
        J = J[idx]
        Y = Y[idx]

        NEdges = len(I)
        R = np.zeros((NEdges, 2))
        R[:, 0] = J
        R[:, 1] = I
        #Make a flat array of NEdges weights parallel to the rows of R
        if do_weighted:
            W = dist_land_land[I, J]
        else:
            W = np.ones(NEdges)
        delta0 = make_delta0(R)
        wSqrt = np.sqrt(W).flatten()
        WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
        A = WSqrt*delta0
        b = WSqrt.dot(Y)
        tau = lsqr(A, b)[0]
        theta = np.zeros((NEdges, 3))
        theta[:, 0] = J
        theta[:, 1] = I
        theta[:, 2] = -delta0.dot(tau)
        theta = add_cocycles(cocycle, theta, real=True)
        

        ## Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        # Compute the partition of unity 
        # varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{n_landmarks}(b))
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are %i point not covered by a landmark"%nzero)
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]

        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)

        ## Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        
        # compute all transition functions
        theta_matrix = np.zeros((n_landmarks, n_landmarks))
        I = np.array(theta[:, 0], dtype = np.int64)
        J = np.array(theta[:, 1], dtype = np.int64)
        theta = theta[:, 2]
        theta = np.mod(theta + 0.5, 1) - 0.5
        theta_matrix[I, J] = theta
        theta_matrix[J, I] = -theta
        class_map = -tau[ball_indx]
        for i in range(n_data):
            class_map[i] += theta_matrix[ball_indx[i], :].dot(varphi[:, i])    
        thetas = np.mod(2*np.pi*class_map, 2*np.pi)

        return thetas

    def update_colors(self):
        if len(self.selected) > 0:
            idxs = np.array(list(self.selected))
            self.selected_plot.set_offsets(self.dgms_[1][idxs, :])
            ## Step 2: Update circular coordinates on point cloud
            thetas = self.coords
            c = plt.get_cmap('magma_r')
            thetas -= np.min(thetas)
            thetas /= np.max(thetas)
            thetas = np.array(np.round(thetas*255), dtype=int)
            C = c(thetas)
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color(C)
            else:
                self.coords_scatter._facecolor3d = C
                self.coords_scatter._edgecolor3d = C
        else:
            self.selected_plot.set_offsets(np.zeros((0, 2)))
            if self.Y.shape[1] == 2:
                self.coords_scatter.set_color('C0')
            else:
                self.coords_scatter._facecolor3d = 'C0'
                self.coords_scatter._edgecolor3d = 'C0'

    def recompute_coords_dimred(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        colors accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        self.update_colors()
        
    def onpick_dimred(self, evt):
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_dimred(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.ax_coords.figure.canvas.draw()
        return True

    def on_perc_slider_move_dimred(self, evt):
        self.recompute_coords_dimred()

    def on_partunity_selector_change_dimred(self, evt):
        self.recompute_coords_dimred()

    def plot_dimreduced(self, Y, init_params = {'cocycle_idxs':[], 'perc':0.99, 'partunity_fn':partunity_linear, 'azim':-60, 'elev':30}, figres=5, dpi=80):
        """
        Do an interactive plot of circular coordinates, coloring a dimension
        reduced version of the point cloud by the circular coordinates

        Parameters
        ----------
        Y: ndarray(N, d)
            A 2D point cloud with the same number of points as X
        init_params: dict
            The intial parameters.  Optional fields of the dictionary are as follows:
            {
                cocycle_idxs: list of int
                    A list of cocycles to start with
                u: ndarray(3, float)
                    The initial stereographic north pole
                perc: float
                    The percent coverage to start with
                partunity_fn: (dist_land_data, r_cover) -> phi
                    The partition of unity function to start with
                azim: float
                    Initial azimuth for 3d plots
                elev: float
                    Initial elevation for 3d plots
            }
        figres: float
            Resolution of each square subplot, in inches
        dpi: int
            Dot pixels per inch
        """
        if Y.shape[1] < 2 or Y.shape[1] > 3:
            raise Exception("Dimension reduced version must be in 2D or 3D")
        self.Y = Y
        fig = plt.figure(figsize=(figres*2, figres), dpi=dpi)
        ## Step 1: Plot H1
        self.ax_persistence = fig.add_subplot(121)
        self.setup_ax_persistence()
        fig.canvas.mpl_connect('pick_event', self.onpick_dimred)
        self.selected = set([])

        ## Step 2: Setup window for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle
        self.perc_slider, self.partunity_selector, self.selected_cocycle_text = EMCoords.setup_param_chooser_gui(self, fig, 0.3, 0.25, 0.35, 0.5, init_params)
        self.perc_slider.on_changed(self.on_perc_slider_move_dimred)
        self.partunity_selector.on_clicked(self.on_partunity_selector_change_dimred)

        ## Step 3: Setup axis for coordinates
        if Y.shape[1] == 3:
            self.ax_coords = fig.add_subplot(122, projection='3d')
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], Y[:, 2], cmap='magma_r')
            # Equal aspect ratio hack for 3D
            pad = 0.1
            maxes = np.max(Y)
            mins = np.min(Y)
            r = maxes - mins
            self.ax_coords.set_xlim([mins-r*pad, maxes+r*pad])
            self.ax_coords.set_ylim([mins-r*pad, maxes+r*pad])
            self.ax_coords.set_zlim([mins-r*pad, maxes+r*pad])
            if 'azim' in init_params:
                self.ax_coords.azim = init_params['azim']
            if 'elev' in init_params:
                self.ax_coords.elev = init_params['elev']
        else:
            self.ax_coords = fig.add_subplot(122)
            self.coords_scatter = self.ax_coords.scatter(Y[:, 0], Y[:, 1], cmap='magma_r')
            self.ax_coords.set_aspect('equal')
        self.ax_coords.set_title("Dimension Reduced Point Cloud")
        if len(init_params['cocycle_idxs']) > 0:
            # If some initial cocycle indices were chosen, update
            # the plot
            self.recompute_coords_dimred(init_params['cocycle_idxs'])
        plt.show()
    
    def get_selected_dimreduced_info(self):
        """
        Return information about what the user selected and their viewpoint in
        the interactive dimension reduced plot

        Returns
        -------
        {
            'partunity_fn': (dist_land_data, r_cover) -> phi
                The selected function handle for the partition of unity
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'perc': float
                The selected percent coverage,
            'azim':float
                Azumith if viewing in 3D
            'elev':float
                Elevation if viewing in 3D
        }
        """
        ret = EMCoords.get_selected_info(self)
        if self.Y.shape[1] == 3:
            ret['azim'] = self.ax_coords.azim
            ret['elev'] = self.ax_coords.elev
        return ret

    def update_plot_torii(self, circ_idx):
        """
        Update a joint plot of circular coordinates, switching between
        2D and 3D modes if necessary

        Parameters
        ----------
        circ_idx: int
            Index of the circular coordinates that have
            been updated
        """
        N = self.plots_in_one
        n_plots = len(self.plots)
        ## Step 1: Figure out the index of the involved plot
        plot_idx = int(np.floor(circ_idx/N))
        plot = self.plots[plot_idx]

        ## Step 2: Extract the circular coordinates from all
        ## plots that have at least one cochain representative selected
        labels = []
        coords = []
        for i in range(N):
            idx = plot_idx*N + i
            c_info = self.coords_info[idx]
            if len(c_info['selected']) > 0:
                # Only include circular coordinates that have at least
                # on persistence dot selected
                coords.append(c_info['coords'])
                labels.append("Coords {}".format(idx))
        
        ## Step 3: Adjust the plot accordingly
        if len(labels) > 0:
            X = np.array([])
            if len(labels) == 1:
                # Just a single coordinate; put it on a circle
                X = np.array([np.cos(self.coords), np.sin(self.coords)]).T
            else:
                X = np.array(coords).T
            updating_axes = False
            if X.shape[1] == 3 and plot['axis_2d']:
                # Need to switch from 2D to 3D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = False
                updating_axes = True
            elif X.shape[1] == 2 and not plot['axis_2d']:
                # Need to switch from 3D to 2D coordinates
                self.fig.delaxes(plot['ax'])
                plot['axis_2d'] = True
                updating_axes = True
            if X.shape[1] == 3:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx, projection='3d')
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], X[:, 2], c=self.coords_colors)
                else:
                    plot['coords_scatter'].set_offsets(X[:, 0], X[:, 1], X[:, 2])
                plot['ax'].set_xlabel(labels[0])
                plot['ax'].set_ylabel(labels[1])
                plot['ax'].set_zlabel(labels[2])
            else:
                if updating_axes:
                    plot['ax'] = self.fig.add_subplot(2, n_plots+1, n_plots+3+plot_idx)
                    plot['coords_scatter'] = plot['ax'].scatter(X[:, 0], X[:, 1], c=self.coords_colors)
                else:
                    plot['coords_scatter'].set_offsets(X[:, 0], X[:, 1])
                plot['ax'].set_xlabel(labels[0])
                plot['ax'].set_ylabel(labels[1])
    
    def recompute_coords_torii(self, clicked = []):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram, and update the circular coordinates
        joint torii plots accordingly

        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        EMCoords.recompute_coords(self, clicked)
        # Save away circular coordinates
        self.coords_info[self.selected_coord_idx]['coords'] = self.coords
        self.update_plot_torii(self.selected_coord_idx)

    def onpick_torii(self, evt):
        """
        Handle a pick even for the torii plot
        """
        if evt.artist == self.dgmplot:
            ## Step 1: Highlight point on persistence diagram
            clicked = set(evt.ind.tolist())
            self.recompute_coords_torii(clicked)
        self.ax_persistence.figure.canvas.draw()
        self.fig.canvas.draw()
        return True

    def select_torii_coord(self, idx):
        """
        Select a particular circular coordinate plot and un-select others
        
        Parameters
        ----------
        idx: int
            Index of the plot to select
        """
        self.selected_coord_idx = idx
        for i, coordsi in enumerate(self.coords_info):
            if i == idx:
                # Swap in the appropriate GUI objects for selection
                self.selected = coordsi['selected']
                self.selected_cocycle_text = coordsi['selected_cocycle_text']
                self.perc_slider = coordsi['perc_slider']
                self.partunity_selector = coordsi['partunity_selector']
                self.persistence_text_labels = coordsi['persistence_text_labels']
                self.coords = coordsi['coords']
                for j in np.array(list(self.selected)):
                    self.persistence_text_labels[j].set_text("%i"%j)
            else:
                # For other circular coordinate selections, clear the selected 
                # dots from the persistence plot
                for label in self.persistence_text_labels:
                    label.set_text("")

    def on_perc_slider_move_torii(self, evt):
        self.recompute_coords_torii()

    def on_partunity_selector_change_torii(self, evt):
        self.recompute_coords_torii()

    def plot_torii(self, f, zoom=1, max_disp=1000, figres=5, dpi=80, coords_info=2, plots_in_one = 2):
        """
        Do an interactive plot of circular coordinates, where points are drawn on S1, 
        on S1 x S1, or S1 x S1 x S1

        Parameters
        ----------
        f: Display information for the points
            On of three options:
            1) A scalar function with which to color the points, represented
               as a 1D array
            2) A list of colors with which to color the points, specified as
               an Nx3 array
            3) A list of images to place at each location
        zoom: int
            If using patches, the factor by which to zoom in on them
        max_disp: int
            The maximum number of points to display
        figres: float
            Dimension of each subplot square, in inches
        dpi: int
            Dot pixels per inch
        coords_info: Information about how to perform circular coordinates.  There will
            be as many plots as the ceil of the number of circular coordinates, and
            they will be plotted pairwise.
            This parameter is one of two options
            1) An int specifying the number of different circular coordinate
               functions to compute
            2) A list of dictionaries with pre-specified initial parameters for
               each circular coordinate.  Each dictionary has the following keys:
               {
                    'cocycle_reps': dictionary
                        A dictionary of cocycle representatives, with the key
                        as the cocycle index, and the value as the coefficient
                    TODO: Finish update to support this instead of a set
                    'perc': float
                        The percent coverage to start with,
                    'partunity_fn': (dist_land_data, r_cover) -> phi
                        The partition of unity function to start with
               }
        plots_in_one: int
            The max number of circular coordinates to put in one plot
        """
        if plots_in_one < 2 or plots_in_one > 3:
            raise Exception("Cannot be fewer than 2 or more than 3 circular coordinates in one plot")
        self.plots_in_one = plots_in_one
        self.f = f
        self.max_disp = max_disp
        ## Step 1: Figure out how many plots are needed to accommodate all
        ## circular coordinates
        n_plots = 1
        if type(coords_info) is int:
            n_plots = int(np.ceil(coords_info/plots_in_one))
            coords_info = []
        else:
            n_plots = int(np.ceil(len(coords_info)/plots_in_one))
        while len(coords_info) < n_plots*plots_in_one:
            coords_info.append({'selected':set([]), 'perc':0.99, 'partunity_fn':partunity_linear})
        self.selecting_idx = 0 # Index of circular coordinate which is currently being selected
        fig = plt.figure(figsize=(figres*(n_plots+1), figres*2), dpi=dpi)
        self.fig = fig

        ## Step 2: Setup H1 plot, along with initially empty text labels
        ## for each persistence point
        self.ax_persistence = fig.add_subplot(2, n_plots+1, 1)
        self.setup_ax_persistence()
        fig.canvas.mpl_connect('pick_event', self.onpick_torii)


        ## Step 2: Setup windows for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle for each circular coordinate.
        ## Also store variables for selecting cocycle representatives
        width = 1/(n_plots+1)
        height = 1/plots_in_one
        idx = 0
        partunity_keys = tuple(PARTUNITY_FNS.keys())
        for i in range(n_plots):
            xstart = width*(i+1.4)
            for j in range(plots_in_one):
                # Setup plots and state for a particular circular coordinate
                ystart = 0.8 - 0.4*height*j
                coords_info[idx]['perc_slider'], coords_info[idx]['partunity_selector'], coords_info[idx]['selected_cocycle_text'] = self.setup_param_chooser_gui(fig, xstart, ystart, width, height, coords_info[idx])
                coords_info[idx]['perc_slider'].on_changed(self.on_perc_slider_move_torii)
                coords_info[idx]['partunity_selector'].on_clicked = self.on_partunity_selector_change_torii
                dgm = self.dgms_[1]
                coords_info[idx]['persistence_text_labels'] = [self.ax_persistence.text(dgm[i, 0], dgm[i, 1], '') for i in range(dgm.shape[0])]
                coords_info[idx]['idx'] = idx
                coords_info[idx]['coords'] = np.zeros(self.X_.shape[0])
                idx += 1
        self.coords_info = coords_info

        ## Step 3: Figure out colors of coordinates
        self.coords_colors = None
        if not (type(f) is list):
            # Figure out colormap if images aren't passed along
            self.coords_colors = f
            if f.size == self.X_.shape[0]:
                # Scalar function, so need to apply colormap
                c = plt.get_cmap('magma_r')
                fscaled = f - np.min(f)
                fscaled = fscaled/np.max(fscaled)
                C = c(np.array(np.round(fscaled*255), dtype=np.int32))
                self.coords_colors = C[:, 0:3]
        
        ## Step 4: Setup plots
        plots = []
        self.n_plots = n_plots
        for i in range(n_plots):
            # 2D by default, but can change to 3D later
            ax = fig.add_subplot(2, n_plots+1, n_plots+3+i)
            idx_disp = np.arange(self.X_.shape[0])
            if self.X_.shape[0] > max_disp:
                idx_disp = np.random.permutation(self.X_.shape[0])[0:max_disp]
            # Setup some dummy points outside of the axis just to get the colors right
            pix = -2*np.ones(idx_disp.size)
            plot = {}
            plot['ax'] = ax
            plot['coords_scatter'] = ax.scatter(pix, pix) # Scatterplot for circular coordinates
            plot['axis_2d'] = True
            plot['patch_boxes'] = [] # Array of image patch display objects
            plot['idx_disp'] = idx_disp # Indices of subset of points to display
            plots.append(plot)
        self.plots = plots

        ## Step 5: Initialize plots with information passed along
        for i in range(len(coords_info)):
            self.select_torii_coord(i)
            self.recompute_coords_torii([])
        
        plt.show()


def do_two_circle_test():
    """
    Test interactive plotting with two noisy circles of different sizes
    """
    prime = 41
    np.random.seed(2)
    N = 500
    X = np.zeros((N*2, 2))
    t = np.linspace(0, 1, N+1)[0:N]**1.2
    t = 2*np.pi*t
    X[0:N, 0] = np.cos(t)
    X[0:N, 1] = np.sin(t)
    X[N::, 0] = 2*np.cos(t) + 4
    X[N::, 1] = 2*np.sin(t) + 4
    X = X[np.random.permutation(X.shape[0]), :]
    X = X + 0.2*np.random.randn(X.shape[0], 2)
    
    cc = CircularCoords(X, 100, prime = prime)
    #cc.plot_dimreduced(X)
    cc.plot_torii(np.concatenate((t, t + 2*np.max(t))), coords_info=2, plots_in_one=2)

def do_torus_test():
    """
    Test interactive plotting with a torus
    """
    prime = 41
    np.random.seed(2)
    N = 10000
    R = 5
    r = 2
    X = np.zeros((N, 3))
    s = np.random.rand(N)*2*np.pi
    t = np.random.rand(N)*2*np.pi
    t = 2*np.pi*t
    X[:, 0] = (R + r*np.cos(s))*np.cos(t)
    X[:, 1] = (R + r*np.cos(s))*np.sin(t)
    X[:, 2] = r*np.sin(s)
    
    cc = CircularCoords(X, 100, prime = prime)
    cc.plot_dimreduced(X)
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
        np.savetxt("R.txt", R, delimiter=',')
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

    def plot_dimreduced(self, Y, init_params = {'cocycle_idxs':[], 'perc':0.99, 'partunity_fn':partunity_linear, 'azim':-60, 'elev':30}):
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
        """
        if Y.shape[1] < 2 or Y.shape[1] > 3:
            raise Exception("Dimension reduced version must be in 2D or 3D")
        self.Y = Y
        fig = plt.figure(figsize=(12, 6))
        ## Step 1: Plot H1
        self.ax_persistence = fig.add_subplot(121)
        self.setup_ax_persistence()
        fig.canvas.mpl_connect('pick_event', self.onpick_dimred)
        self.selected = set([])

        ## Step 2: Setup window for choosing coverage / partition of unity type
        ## and for displaying the chosen cocycle
        self.perc_slider, self.partunity_selector = EMCoords.setup_param_chooser_gui(self, fig, 0.3, 0.25, 0.35, 0.5, init_params)
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
    cc.plot_dimreduced(X)
    #cc.plot_circles(np.concatenate((t, t + 2*np.max(t))), coords=7, plots_in_one=3)

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
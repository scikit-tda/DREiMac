import numpy as np 
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from ripser import ripser
import time
import warnings
from geomtools import *

"""#########################################
    Projective Coordinates Utilities
#########################################"""

def ppca(class_map, proj_dim, verbose = False):
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
        print("Doing ppca on %i points in %i dimensions down to %i dimensions"%\
                (class_map.shape[0], class_map.shape[1], proj_dim))
    X = class_map.T
    variance = np.zeros(X.shape[0]-1)

    n_dim = class_map.shape[1]
    tic = time.time()
    # Projective dimensionality reduction : Main Loop
    XRet = None
    for i in range(n_dim-1):
        # Project onto an "equator"
        try:
            _, U = linalg.eigh(X.dot(X.T))
            U = np.fliplr(U)
        except:
            U = np.eye(X.shape[0])
        variance[-i-1] = np.mean((np.pi/2-np.real(np.arccos(np.abs(U[:, -1][None, :].dot(X)))))**2)
        Y = (U.T).dot(X)
        y = np.array(Y[-1, :])
        Y = Y[0:-1, :]
        X = Y/np.sqrt(1-np.abs(y)**2)[None, :]
        if i == n_dim-proj_dim-2:
            XRet = np.array(X)
    if verbose:
        print("Elapsed time ppca: %.3g"%(time.time()-tic))
    #Return the variance and the projective coordinates
    return {'variance':variance, 'X':XRet.T}


def rotmat(a, b = np.array([])):
    """
    Construct a d x d rotation matrix that rotates
    a vector a so that it coincides with a vector b

    Parameters
    ----------
    a : ndarray (d)
        A d-dimensional vector that should be rotated to b
    b : ndarray(d)
        A d-dimensional vector that shoudl end up residing at 
        the north pole (0,0,...,0,1)
    """
    if (len(a.shape) > 1 and np.min(a.shape) > 1)\
         or (len(b.shape) > 1 and np.min(b.shape) > 1):
        print("Error: a and b need to be 1D vectors")
        return None
    a = a.flatten()
    a = a/np.sqrt(np.sum(a**2))
    d = a.size

    if b.size == 0:
        b = np.zeros(d)
        b[-1] = 1
    b = b/np.sqrt(np.sum(b**2))
    
    c = a - np.sum(b*a)*b
    # If a numerically coincides with b, don't rotate at all
    if np.sqrt(np.sum(c**2)) < 1e-15:
        return np.eye(d)

    # Otherwise, compute rotation matrix
    c = c/np.sqrt(np.sum(c**2))
    lam = np.sum(b*a)
    beta = np.sqrt(1 - np.abs(lam)**2)
    rot = np.eye(d) - (1-lam)*(c[:, None].dot(c[None, :])) \
                    - (1-lam)*(b[:, None].dot(b[None, :])) \
                    + beta*(b[:, None].dot(c[None, :]) - c[:, None].dot(b[None, :]))
    return rot


def get_stereo_proj_codim1(pX, u = np.array([])):
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
    from sklearn.decomposition import PCA
    X = pX.T
    # Put points all on the same hemisphere
    if u.size == 0:
        _, U = linalg.eigh(X.dot(X.T))
        u = U[:, 0]
    XX = rotmat(u).dot(X)
    ind = XX[-1, :] < 0
    XX[:, ind] *= -1
    # Do stereographic projection
    S = XX[0:-1, :]/(1+XX[-1, :])[None, :]
    return S.T

def plot_rp2_circle(ax, arrowcolor = 'c', facecolor = (0.15, 0.15, 0.15), do_arrows = True):
    """
    Plot a circle with arrows showing the identifications for RP2.
    Set an equal aspect ratio and get rid of the axis ticks, since
    they are clear from the circle
    Parameters
    ----------
    ax: matplotlib axis
        Axis onto which to plot the circle+arrows
    arrowcolor: string or ndarray(3) or ndarray(4)
        Color for the circle and arrows
    facecolor: string or ndarray(3) or ndarray(4)
        Color for background of the plot
    do_arrows: boolean
        Whether to draw the arrows
    """
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), c=arrowcolor)
    ax.axis('equal')
    ax = plt.gca()
    if do_arrows:
        ax.arrow(-0.1, 1, 0.001, 0, head_width = 0.15, head_length = 0.2, fc = arrowcolor, ec = arrowcolor, width = 0)
        ax.arrow(0.1, -1, -0.001, 0, head_width = 0.15, head_length = 0.2, fc = arrowcolor, ec = arrowcolor, width = 0)
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    pad = 1.1
    ax.set_xlim([-pad, pad])
    ax.set_ylim([-pad, pad])

def plot_rp2_stereo(S, f, arrowcolor = 'c', facecolor = (0.15, 0.15, 0.15)):
    """
    Plot a 2D Stereographic Projection
    Parameters
    ----------
    S : ndarray (N, 2)
        An Nx2 array of N points to plot on RP2
    f : ndarray (N) or ndarray (N, 3)
        A function with which to color the points, or a list of colors
    """
    if not (S.shape[1] == 2):
        warnings.warn("Plotting stereographic RP2 projection, but points are not 2 dimensional")
    plot_rp2_circle(plt.gca(), arrowcolor, facecolor)
    if f.size > S.shape[0]:
        plt.scatter(S[:, 0], S[:, 1], 20, c=f, cmap='afmhot')
    else:
        plt.scatter(S[:, 0], S[:, 1], 20, f, cmap='afmhot')


def plot_rp3_stereo(ax, S, f, draw_sphere = False):
    """
    Plot a 3D Stereographic Projection

    Parameters
    ----------
    ax : matplotlib axis
        3D subplotting axis
    S : ndarray (N, 3)
        An Nx3 array of N points to plot on RP3
    f : ndarray (N) or ndarray (N, 3)
        A function with which to color the points, or a list of colors
    draw_sphere : boolean
        Whether to draw the 2-sphere
    """
    if not (S.shape[1] == 3):
        warnings.warn("Plotting stereographic RP3 projection, but points are not 4 dimensional")
    if f.size > S.shape[0]:
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=f, cmap='afmhot')
    else:
        c = plt.get_cmap('afmhot')
        C = f - np.min(f)
        C = C/np.max(C)
        C = c(np.array(np.round(C*255), dtype=np.int32))
        C = C[:, 0:3]
        ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=C, cmap='afmhot')
    if draw_sphere:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        ax.set_aspect('equal')    
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="k")

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
    u[2] = np.sqrt(1-magSqr)
    return x, u



"""#########################################
        Main Projective Coordinates Class
#########################################"""

class ProjectiveCoords(object):
    def __init__(self, X, n_landmarks, distance_matrix=False, maxdim=1, verbose=False):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        maxdim : int
            Maximum dimension of homology.  Only dimension 1 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        partunity_fn: ndarray(n_landmarks, N) -> ndarray(n_landmarks, N)
            A partition of unity function
        """
        assert(maxdim >= 1)
        self.verbose = verbose
        if verbose:
            tic = time.time()
            print("Doing TDA...")
        res = ripser(X, distance_matrix=distance_matrix, coeff=2, maxdim=maxdim, n_perm=n_landmarks, do_cocycles=True)
        if verbose:
            print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
        self.X_ = X
        self.dist_land_data_ = res['dperm2all']
        self.idx_land_ = res['idx_perm']
        self.dist_land_land_ = self.dist_land_data_[:, self.idx_land_]
        self.dgms_ = res['dgms']
        self.cocycles_ = res['cocycles']
        # Sort persistence diagrams in descending order of persistence
        idxs = np.argsort(self.dgms_[1][:, 0]-self.dgms_[1][:, 1])
        self.dgms_[1] = self.dgms_[1][idxs, :]
        self.cocycles_[1] = [self.cocycles_[1][idx] for idx in idxs]
        reindex_cocycles(self.cocycles_, self.idx_land_, X.shape[0])
        self.n_landmarks_ = n_landmarks
        self.type_ = "proj"
        # GUI variables
        self.selected = set([])
        self.u = np.array([0, 0, 1])

    def get_coordinates(self, perc = 0.99, cocycle_idx = [0], proj_dim = 3, partunity_fn = partunity_linear):
        """
        Perform multiscale projective coordinates via persistent cohomology of 
        sparse filtrations (Jose Perea 2018)
        Parameters
        ----------
        perc : float
            Percent coverage
        cocycle_idx : list
            Add the cocycles together, sorted from most to least persistent
        proj_dim : integer
            Dimension down to which to project the data
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        
        Returns
        -------
        {'variance': ndarray(N-1)
            The variance captured by each dimension
        'X': ndarray(N, proj_dim+1)
            The projective coordinates
        }
        """
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        dgm1 = self.dgms_[1]/2.0 #Need so that Cech is included in rips
        cohomdeath = -np.inf
        cohombirth = np.inf
        cocycle = np.zeros((0, 3))
        for k in range(len(cocycle_idx)):
            cocycle = add_cocycles(cocycle, self.cocycles_[1][cocycle_idx[k]], p=2)
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

        ## Step 3: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        U = dist_land_data < r_cover
        phi = np.zeros_like(dist_land_data)
        phi[U] = partunity_fn(dist_land_data[U], r_cover)
        denom = np.sum(phi, 0)
        nzero = np.sum(denom == 0)
        if nzero > 0:
            warnings.warn("There are %i point not covered by a landmark"%nzero)
            denom[denom == 0] = 1
        varphi = phi / denom[None, :]
        # To each data point, associate the index of the first open set it belongs to
        ball_indx = np.argmax(U, 0)

        ## Step 4: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        # compute all transition functions
        cocycle_matrix = np.ones((n_landmarks, n_landmarks))
        cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
        cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
        class_map = np.sqrt(varphi.T)
        for i in range(n_data):
            class_map[i, :] *= cocycle_matrix[ball_indx[i], :]
        res = ppca(class_map, proj_dim, self.verbose)
        return res

    def update_display_coords(self):
        S = -2*np.ones((self.idx_disp.size, 2))
        if len(self.selected) > 0 and self.coords.size > 0:
            # Update projective coordinates if at least
            # one point in the persistence diagram is selected
            S = get_stereo_proj_codim1(self.coords[self.idx_disp, :], self.u)
        if self.coords_scatter:
            # Updated scatterplot
            self.coords_scatter.set_offsets(S)
            C = self.coords_colors[self.idx_disp, :]
            self.coords_scatter.set_color(C)
        else:
            for i, (im, ab) in enumerate(self.patch_boxes):
                ab.xybox = S[i, :]
                idx = self.idx_disp[i]
                im.set_data(self.f[idx])
        self.ax_coords.figure.canvas.draw()

    def toggle_persistence(self, clicked):
        """
        Toggle including a cocycle from a set of points in the 
        persistence diagram
        Parameters
        ----------
        clicked: list of int
            Indices to toggle
        """
        self.selected = self.selected.symmetric_difference(set(clicked))
        idxs = np.array(list(self.selected))
        if idxs.size > 0:
            ## Step 1: Highlight point on persistence diagram
            self.selected_plot.set_offsets(self.dgms_[1][idxs, :])
            ## Step 2: Update projective coordinates
            res = self.get_coordinates(proj_dim=2, cocycle_idx = idxs)
            self.coords = res['X']
            # If the number of points exceeds the maximum to plot, select
            # a subset with a greedy permutation, using the arclength
            # metric on RP2
            if self.X_.shape[0] > self.max_disp:
                self.idx_disp = get_greedy_perm_pc(self.coords, self.max_disp, csm_fn = get_csm_projarc)['perm']
        else:
            self.selected_plot.set_offsets(np.zeros((0, 2)))
        self.update_display_coords()
        self.ax_coords.figure.canvas.draw()

    def onpick(self, evt):
        """
        Toggle a point in the persistence diagram
        """
        if evt.artist == self.dgmplot:
            self.toggle_persistence(evt.ind.tolist())
        return True
    
    def onstereo_click(self, evt):
        self.stereo_pressed = True
        self.onstereo_move(evt)
    
    def onstereo_release(self, evt):
        self.stereo_pressed = False

    def onstereo_move(self, evt):
        """
        Change the north pole on the projective plane for
        stereographic projection
        """
        if evt.inaxes == self.ax_pickstereo and self.stereo_pressed:
            x = np.array([evt.xdata, evt.ydata])
            x, self.u = circle_to_3dnorthpole(x)
            self.selected_northpole_plot.set_offsets(x)
            self.ax_pickstereo.figure.canvas.draw()
            self.update_display_coords()

    def plot(self, f, zoom=1, max_disp = 1000, cocycle_idxs = [], u = np.array([0, 0, 1])):
        """
        Do an interactive plot, with H1 on the left and a 
        2D dimension reduced version of the point cloud on the right.
        The right plot will be colored by the specified scalar function, or
        a set of specified images will be placed on top of them.  
        Left click on points in the persistence  diagram to toggle their
        inclusion in the circular coordinates
        Parameters
        ----------
        f : Display information for the points
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
        cocycle_idxs: list of int
            A list of cocycles to start with
        u: ndarray(3, float)
            The initial stereographic north pole
        """
        self.f = f
        self.max_disp = max_disp
        fig = plt.figure(figsize=(15, 5))
        ## Step 1: Plot H1
        dgm_size = 20
        self.ax_persistence = fig.add_subplot(131)
        dgm = self.dgms_[1]
        ax_min, ax_max = np.min(dgm), np.max(dgm)
        x_r = ax_max - ax_min
        buffer = x_r / 5
        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer
        y_down, y_up = x_down, x_up
        yr = y_up - y_down
        self.ax_persistence.plot([x_down, x_up], [x_down, x_up], "--", c=np.array([0.0, 0.0, 0.0]))
        self.dgmplot, = self.ax_persistence.plot(dgm[:, 0], dgm[:, 1], 'o', picker=5, c='C0')
        self.selected_plot = self.ax_persistence.scatter([], [], 100, c='C1')
        self.ax_persistence.set_xlim([x_down, x_up])
        self.ax_persistence.set_ylim([y_down, y_up])
        self.ax_persistence.set_aspect('equal', 'box')
        self.ax_persistence.set_title("Persistent H1")
        self.ax_persistence.set_xlabel("Birth")
        self.ax_persistence.set_ylabel("Death")
        fig.canvas.mpl_connect('pick_event', self.onpick)
        self.selected = set([])

        ## Step 2: Setup axis for picking stereographic north pole
        self.ax_pickstereo = fig.add_subplot(132)
        self.selected_northpole_plot = self.ax_pickstereo.scatter([0], [0], 100, c='C1')
        plot_rp2_circle(self.ax_pickstereo, do_arrows=False)
        self.u = u # Initial u
        self.ax_pickstereo.set_title("Stereographic North Pole")
        self.stereo_pressed = False
        fig.canvas.mpl_connect('motion_notify_event', self.onstereo_move)
        fig.canvas.mpl_connect('button_press_event', self.onstereo_click)
        fig.canvas.mpl_connect('button_release_event', self.onstereo_release)

        ## Step 3: Setup axis for coordinates.  Start with axis 
        ## which is the ordinary north pole
        self.ax_coords = fig.add_subplot(133)
        # Figure out which points to display.  Start off with a random subset
        self.idx_disp = np.arange(self.X_.shape[0])
        if self.X_.shape[0] > max_disp:
            self.idx_disp = np.random.permutation(self.X_.shape[0])[0:max_disp]
        # Setup some dummy points outside of the axis just to get the colors right
        pix = -2*np.ones(self.idx_disp.size)
        self.coords_scatter = None
        self.patch_boxes = []
        if type(f) is list:
            # Figure out range of patches
            minpatch = np.inf
            maxpatch = -np.inf
            for patch in f:
                minpatch = min(minpatch, np.min(patch))
                maxpatch = max(maxpatch, np.max(patch))
            for i in range(self.idx_disp.size):
                # Setup a bunch of empty images
                im = OffsetImage(np.array([[minpatch, maxpatch]]), zoom=zoom, cmap = 'gray')
                ab = AnnotationBbox(im, (0, 0), xycoords='data', frameon=False)
                self.ax_coords.add_artist(ab)
                self.patch_boxes.append((im, ab))
        else:
            self.coords_colors = f
            if f.size == self.X_.shape[0]:
                # Scalar function, so need to apply colormap
                c = plt.get_cmap('magma_r')
                fscaled = f - np.min(f)
                fscaled = fscaled/np.max(fscaled)
                C = c(np.array(np.round(fscaled*255), dtype=np.int32))
                self.coords_colors = C[:, 0:3]
            self.coords_scatter = self.ax_coords.scatter(pix, pix)
            self.coords = np.array([[]])
        plot_rp2_circle(self.ax_coords)
        self.ax_coords.set_title("Projective Coordinates")
        if len(cocycle_idxs) > 0:
            # If some initial cocycle indices were chosen, update
            # the plot
            self.toggle_persistence(cocycle_idxs)
            self.selected_northpole_plot.set_offsets(self.u[0:2])
        return fig
    
    def get_selected_info(self):
        """
        Return information about what the user selected in
        the interactive plot
        Returns
        -------
        {
            'cocycle_idxs':ndarray(dtype = int)
                Indices of the selected cocycles,
            'u':ndarray(3, float)
                The stereographic north pole
        }
        """
        return {'cocycle_idxs':np.array(list(self.selected)), 'u':self.u}


def testProjCoordsRP2(NSamples, NLandmarks):
    """
    Test projective coordinates on the projective plane
    Parameters
    ----------
    NSamples : int
        Number of random samples on the projective plane
    NLandmarks : int
        Number of landmarks to take in the projective coordinates computation
    """
    from persim import plot_diagrams
    np.random.seed(NSamples)
    X = np.random.randn(NSamples, 3)
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    D = X.dot(X.T)
    D = np.abs(D)
    D[D > 1.0] = 1.0
    D = np.arccos(D)

    # Coming up with ground truth theta and phi for RP2 for colors
    SOrig = get_stereo_proj_codim1(X)
    phi = np.sqrt(np.sum(SOrig**2, 1))
    theta = np.arccos(np.abs(SOrig[:, 0]))
    
    pc = ProjectiveCoords(D, NLandmarks, distance_matrix=True, verbose=True)
    pc.plot(phi, max_disp=X.shape[0])


def testProjCoordsKleinBottle(NSamples, NLandmarks):
    """
    Test projective coordinates on the Klein bottle
    Parameters
    ----------
    NSamples : int
        Number of random samples on the projective plane
    NLandmarks : int
        Number of landmarks to take in the projective coordinates computation
    """
    theta = np.random.rand(NSamples)*2*np.pi
    phi = np.random.rand(NSamples)*2*np.pi
    R = 2
    r = 1
    X = np.zeros((NSamples, 4))
    X[:, 0] = (R + r*np.cos(theta))*np.cos(phi)
    X[:, 1] = (R + r*np.cos(theta))*np.sin(phi)
    X[:, 2] = r*np.sin(theta)*np.cos(phi/2)
    X[:, 3] = r*np.sin(theta)*np.sin(phi/2)
    pc = ProjectiveCoords(X, NLandmarks, verbose=True)
    pc.plot(theta)


def getLinePatches(dim, NAngles, NOffsets, sigma):
    """
    Sample a set of line segments, as witnessed by square patches
    Parameters
    ----------
    dim: int
        Patches will be dim x dim
    NAngles: int
        Number of angles to sweep between 0 and pi
    NOffsets: int
        Number of offsets to sweep from the origin to the edge of the patch
    sigma: float
        The blur parameter.  Higher sigma is more blur
    """
    N = NAngles*NOffsets
    P = np.zeros((N, dim*dim))
    thetas = np.linspace(0, np.pi, NAngles+1)[0:NAngles]
    #ps = np.linspace(-0.5*np.sqrt(2), 0.5*np.sqrt(2), NOffsets)
    ps = np.linspace(-1, 1, NOffsets)
    idx = 0
    [Y, X] = np.meshgrid(np.linspace(-0.5, 0.5, dim), np.linspace(-0.5, 0.5, dim))
    for i in range(NAngles):
        c = np.cos(thetas[i])
        s = np.sin(thetas[i])
        for j in range(NOffsets):
            patch = X*c + Y*s + ps[j]
            patch = np.exp(-patch**2/sigma**2)
            P[idx, :] = patch.flatten()
            idx += 1
    return P

def testProjCoordsLinePatches():
    """
    Test projective coordinates on a set of line segment patches
    """
    dim = 10
    P = getLinePatches(dim=dim, NAngles = 200, NOffsets = 200, sigma=0.25)
    patches = [np.reshape(P[i, :], (dim, dim)) for i in range(P.shape[0])]
    pc = ProjectiveCoords(P, n_landmarks=100)
    fig = pc.plot(patches, max_disp=200, cocycle_idxs=[0])
    plt.show()


if __name__ == '__main__':
    #testProjCoordsRP2(10000, 60)
    #testProjCoordsKleinBottle(10000, 300)
    testProjCoordsLinePatches()
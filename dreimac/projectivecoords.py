import numpy as np 
import numpy.linalg as linalg
import matplotlib.pyplot as plt 
import time
import warnings
from .utils import *
from .emcoords import *

class ProjectiveCoords(EMCoords):
    """#########################################
        Projective Coordinates Utilities
    #########################################"""
    @staticmethod
    def ppca(class_map, proj_dim, verbose=False):
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
            _, U = linalg.eigh(X.dot(X.T))
            u = U[:, 0]
        XX = ProjectiveCoords.rotmat(u).dot(X)
        ind = XX[-1, :] < 0
        XX[:, ind] *= -1
        # Do stereographic projection
        S = XX[0:-1, :]/(1+XX[-1, :])[None, :]
        return S.T
    
    @staticmethod
    def plot_rp2_circle(ax, arrowcolor='c', facecolor=(0.15, 0.15, 0.15), do_arrows=True, pad=1.1):
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
        pad: float
            The dimensions of the window around the unit square
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
        ax.set_xlim([-pad, pad])
        ax.set_ylim([-pad, pad])
    
    @staticmethod
    def plot_rp2_stereo(S, f, arrowcolor='c', facecolor=(0.15, 0.15, 0.15)):
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

    @staticmethod
    def plot_rp3_stereo(ax, S, f, draw_sphere=False):
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
        u[2] = np.sqrt(1-magSqr)
        return x, u


    """#########################################
     Main Projective Coordinates Functionality
    #########################################"""
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
        EMCoords.__init__(self, X=X, n_landmarks=n_landmarks, distance_matrix=distance_matrix, prime=2, maxdim=maxdim, verbose=verbose)
        self.type_ = "proj"
        # GUI variables
        self.selected = set([])
        self.u = np.array([0, 0, 1])

    def get_coordinates(self, perc=0.99, cocycle_idx=[0], proj_dim=2, partunity_fn=PartUnity.linear):
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
        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]
        ## Step 1: Come up with the representative cocycle as a formal sum
        ## of the chosen cocycles
        cohomdeath, cohombirth, cocycle = EMCoords.get_rep_cocycle(self, cocycle_idx)

        ## Step 2: Determine radius for balls
        r_cover = EMCoords.get_cover_radius(self, perc, cohomdeath, cohombirth)

        ## Step 3: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        ## Step 4: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
        # compute all transition functions
        cocycle_matrix = np.ones((n_landmarks, n_landmarks))
        cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
        cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
        class_map = np.sqrt(varphi.T)
        for i in range(n_data):
            class_map[i, :] *= cocycle_matrix[ball_indx[i], :]
        res = ProjectiveCoords.ppca(class_map, proj_dim, self.verbose)
        return res


import numpy as np 
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from ripser import Rips
import time
import warnings

def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    
    Parameters
    ----------
    X : ndarray (M, d)
        A matrix holding the coordinates of M points
    Y : ndarray (N, d) 
        A matrix holding the coordinates of N points
    Return
    ------
    D : ndarray (M, N)
        An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getGreedyPermEuclidean(X, M, verbose = False):
    """
    A Naive O(NM) algorithm to do furthest points sampling, assuming
    the input is a Euclidean point cloud.  This saves computation
    over having compute the full distance matrix if the number of
    landmarks M << N
    
    Parameters
    ----------
    X : ndarray (N, d) 
        An Nxd Euclidean point cloud
    M : integer
        Number of landmarks to compute
    verbose: boolean
        Whether to print progress

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
    ds = getCSM(X[0, :][None, :], X).flatten()
    D = np.zeros((M, N))
    D[0, :] = ds
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        thisds = getCSM(X[idx, :][None, :], X).flatten()
        D[i, :] = thisds
        ds = np.minimum(ds, thisds)
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas, 'D':D}

def getGreedyPermDM(D, M, verbose = False):
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

def PPCA(class_map, proj_dim, verbose = False):
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
    """
    if verbose:
        print("Doing PPCA on %i points in %i dimensions down to %i dimensions"%\
                (class_map.shape[0], class_map.shape[1], proj_dim))
    X = class_map.T
    variance = np.zeros(X.shape[0]-1)

    n_dim = class_map.shape[1]
    tic = time.time()
    # Projective dimensionality reduction : Main Loop
    XRet = None
    for i in range(n_dim-1):
        # Project onto an "equator"
        _, U = linalg.eigh(X.dot(X.T))
        U = np.fliplr(U)
        variance[-i-1] = np.mean((np.pi/2-np.real(np.arccos(np.abs(U[:, -1][None, :].dot(X)))))**2)
        Y = (U.T).dot(X)
        y = np.array(Y[-1, :])
        Y = Y[0:-1, :]
        X = Y/np.sqrt(1-np.abs(y)**2)[None, :]
        if i == n_dim-proj_dim-2:
            XRet = np.array(X)
    if verbose:
        print("Elapsed time PPCA: %.3g"%(time.time()-tic))
    #Return the variance and the projective coordinates
    return {'variance':variance, 'X':XRet.T}

def ProjCoords(P, n_landmarks, distance_matrix = False, perc = 0.99, \
                proj_dim = 3, verbose = False):
    """
    Perform multiscale projective coordinates via persistent cohomology of 
    sparse filtrations (Jose Perea 2018)
    Parameters
    ----------
    P : ndarray (n_data, d)
        n_data x d array of points
    n_landmarks : integer
        Number of landmarks to sample
    distance_matrix : boolean
        If true, then X is a distance matrix, not a Euclidean point cloud
    perc : float
        Percent coverage
    proj_dim : integer
        Dimension down to which to project the data
    verbose : boolean
        Whether to print detailed information during the computation
    """
    n_data = P.shape[0]
    rips = Rips(coeff=2, maxdim=1, do_cocycles=True)
    
    # Step 1: Compute greedy permutation
    tic = time.time()
    if distance_matrix:
        res = getGreedyPermDM(P, n_landmarks, verbose)
        perm, dist_land_data = res['perm'], res['DLandmarks']
        dist_land_land = P[perm, :]
        dist_land_land = dist_land_land[:, perm]
    else:    
        res = getGreedyPermEuclidean(P, n_landmarks, verbose)
        Y, dist_land_data = res['Y'], res['D']
        dist_land_land = getCSM(Y, Y)
    if verbose:
        print("Elapsed time greedy permutation: %.3g seconds"%(time.time() - tic))



    # Step 2: Compute H1 with cocycles on the landmarks
    tic = time.time()
    dgms = rips.fit_transform(dist_land_land, distance_matrix=True)
    dgm1 = dgms[1]
    dgm1 = dgm1/2.0 #Need so that Cech is included in rips
    if verbose:
        print("Elapsed time persistence: %.3g seconds"%(time.time() - tic))
        rips.plot()
    idx_mp1 = np.argmax(dgm1[:, 1] - dgm1[:, 0])
    cocycle = rips.cocycles_[1][idx_mp1]

    # Step 3: Determine radius for balls ( = interpolant btw data coverage and cohomological birth)
    coverage = np.max(np.min(dist_land_data, 1))
    r_birth = (1-perc)*max(dgm1[idx_mp1, 0], coverage) + perc*dgm1[idx_mp1, 1]
    print("r_birth = %.3g"%r_birth)
    

    # Step 4: Create the open covering U = {U_1,..., U_{s+1}} and partition of unity

    # Let U_j be the set of data points whose distance to l_j is less than
    # r_birth
    U = dist_land_data < r_birth
    # Compute subordinated partition of unity varphi_1,...,varphi_{s+1}
    # Compute the bump phi_j(b) on each data point b in U_j. phi_j = 0 outside U_j.
    phi = np.zeros_like(dist_land_data)
    phi[U] = r_birth - dist_land_data[U]

    # Compute the partition of unity varphi_j(b) = phi_j(b)/(phi_1(b) + ... + phi_{s+1}(b))
    varphi = phi / np.sum(phi, 0)[None, :]

    # To each data point, associate the index of the first open set it belongs to
    indx = np.argmax(U, 0)


    # Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map

    # compute all transition functions
    cocycle_matrix = np.ones((n_landmarks, n_landmarks))
    cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
    cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
    class_map = np.sqrt(varphi.T)
    for i in range(n_data):
        class_map[i, :] *= cocycle_matrix[indx[i], :]
    
    res = PPCA(class_map, proj_dim, verbose)
    res["cocycle"] = cocycle[:, 0:2]+1
    res["dist_land_land"] = dist_land_land
    res["dist_land_data"] = dist_land_data
    res["dgm1"] = dgm1
    return res

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


def getStereoRP2(pX):
    if not (pX.shape[1] == 3):
        warnings.warn("Doing stereographic RP2 projection, but points are not 3 dimensional")
    X = pX.T
    # Put points all on the same hemisphere
    _, U = linalg.eigh(X.dot(X.T))
    XX = rotmat(U[:, 0]).dot(X)
    ind = XX[-1, :] < 0
    XX[:, ind] *= -1
    # Do stereographic projection
    S = XX[0:-1, :]/(1+XX[-1, :])[None, :]
    return S.T

def plotRP2Stereo(S, f):
    plt.scatter(S[:, 0], S[:, 1], 20, f, cmap='afmhot')
    t = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(t), np.sin(t))
    plt.axis('equal')


def testGreedyPermEuclidean():
    t = np.linspace(0, 2*np.pi, 10000)
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    res = getGreedyPermEuclidean(X, 50, True)
    Y, D = res['Y'], res['D']
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 10)
    plt.scatter(Y[:, 0], Y[:, 1], 40)
    plt.subplot(122)
    plt.imshow(D, aspect = 'auto')
    plt.show()

def testProjCoordsRP2(NSamples, NLandmarks):
    np.random.seed(NSamples)
    X = np.random.randn(NSamples, 3)
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]

    SOrig = getStereoRP2(X)
    phi = np.sqrt(np.sum(SOrig**2, 1))
    
    D = X.dot(X.T)
    D = np.abs(D)
    D[D > 1.0] = 1.0
    D = np.arccos(D)
    
    res = ProjCoords(D, NLandmarks, proj_dim=2, distance_matrix=True, verbose=True)
    import scipy.io as sio
    res["rp2_phi"] = phi
    sio.savemat("JoseMatlabCode/RP2.mat", res)
    variance, X = res['variance'], res['X']

    SFinal = getStereoRP2(X)
    
    vcumu = np.cumsum(variance)
    vcumu = vcumu/vcumu[-1]
    plt.subplot(121)
    plotRP2Stereo(SOrig, phi)
    plt.colorbar()
    plt.title("Ground Truth RP2")
    plt.subplot(122)
    plotRP2Stereo(SFinal, phi)
    plt.colorbar()
    plt.title("My Projective Coordinates")
    plt.show()

if __name__ == '__main__':
    #testGreedyPermEuclidean()
    testProjCoordsRP2(10000, 60)

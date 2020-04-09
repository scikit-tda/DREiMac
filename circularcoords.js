const DEFAULT_PRIME_FIELD = 41; // Field for homology

class CircularCoords {

    // TODO: Have this class add on menu options to the tda instance

    /**
     * A constructor for a circular coordinate instance
     * 
     * @param {TDA} tda A handle to the TDA object
     * @param {DOM Element} canvas2D A canvas to which to draw the results
     * @param {String} dgmsCanvasName The string of the canvas on which to plot
     *                                the interactive persistence diagrams
     * @param {int} nlandmarks The number of landmarks to use
     * @param {int} prime The homology coefficient field
     * @param {int} maxdim The maximum dimension of homology
     */
    constructor(tda, canvas2D, dgmsCanvasName, nlandmarks, prime, maxdim) {
        if (nlandmarks === undefined) {
            nlandmarks = 100;
        }
        if (prime === undefined) {
            prime = DEFAULT_PRIME_FIELD;
        }
        if (maxdim === undefined) {
            maxdim = 1;
        }

        this.tda = tda;
        this.rips = new Ripser(tda, prime, maxdim, true);
        this.rips.nlandmarks = nlandmarks;
        this.dgmsCanvasName = dgmsCanvasName;
        this.canvas2D = canvas2D;

        // Circular coordinate options
        this.doWeighted = false;
        this.cocyle_idx = [];
        this.perc = 0.99;

        this.setupMenu();
    }

    /**
     * Setup the menu for circular coordinate options
     */
    setupMenu() {
        const gui = this.tda.gui;
        let ccOpts = gui.addFolder('Circular Coordinates');

        // Rips options and computation are separate from other options
        let ripsOpts = ccOpts.addFolder('Rips Options');
        ripsOpts.add(this.rips, 'field').min(2).step(1);
        ripsOpts.add(this.rips, 'homdim').min(1).step(1);
        // Update landmarks if there aren't enough with .listen()
        ripsOpts.add(this.rips, 'nlandmarks').min(1).step(1).listen(); 
        ripsOpts.add(this, 'recomputeRips');

        ccOpts.add(this, 'perc').min(0).max(1).step(0.01);
        ccOpts.add(this, 'doWeighted');
        ccOpts.add(this, 'updateCoordinates');

    }

    /**
     * Load in a point cloud and compute rips with the 
     * current parameters
     * 
     * @param {list} X A list of lists of coordinates in a Euclidean point cloud
     */
    addEuclideanPointCloud() {
        this.X = this.tda.points;
        this.rips.computeRipsPC(X, this.rips.nlandmarks);
    }

    /**
     * If the user has chosen different parameters for
     * rips, then recompute
     */
    recomputeRips() {
        if (this.X === undefined) {
            alert("Point cloud not loaded in yet");
        }
        else {
            this.rips.computeRipsPC(this.X, this.rips.nlandmarks);
        }
    }

    /**
      * Perform circular coordinates via persistent cohomology of 
      * sparse filtrations (Jose Perea 2018)
     */
    updateCoordinates() {
        /**## Step 1: Come up with the representative cocycle as a formal sum
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
        phi[U] = partunity_fn(phi[U], r_cover)
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

        return thetas**/
    }



}

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
        this.ripsPromise = null;
        this.dgmsCanvasName = dgmsCanvasName;
        this.canvas2D = canvas2D;

        // Circular coordinate options
        this.doWeighted = true;
        this.cocyle_idx = [];
        this.perc = 0.99;
        this.partUnityFn = 'partUnityLinear';

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
        ripsOpts.landmarksListener = ripsOpts.add(this.rips, 'nlandmarks').min(1).step(1); 
        ripsOpts.add(this, 'recomputeRips');
        this.ripsOpts = ripsOpts;

        ccOpts.add(this, 'perc').min(0).max(1).step(0.01);
        ccOpts.add(this, 'doWeighted');
        ccOpts.add(this, 'partUnityFn', ['partUnityLinear', 'partUnityQuadratic', 'partUnityExp']);
        ccOpts.add(this, 'updateCoords');

        this.ccOpts = ccOpts;

    }

    /**
     * Load in a point cloud and compute rips with the 
     * current parameters
     * 
     * @param {list} X A list of lists of coordinates in a Euclidean point cloud
     */
    addEuclideanPointCloud() {
        this.X = this.tda.points;
        this.thetas = new Float32Array(this.X.length);
        this.ripsPromise = this.rips.computeRipsPC(this.X, this.rips.nlandmarks);
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
            this.ripsPromise = this.rips.computeRipsPC(this.X, this.rips.nlandmarks);
        }
    }

    /**
      * Perform circular coordinates via persistent cohomology of 
      * sparse filtrations (Jose Perea 2018)
     */
    updateCoords() {
        /*if (this.cocyle_idx.length == 0) {
            alert("Must choose at least one representative cocycle");
            return;
        }*/
        if (this.X === undefined) {
            this.addEuclideanPointCloud();
        }
        if (this.ripsPromise === null) {
            alert("Rips computation has not been initiated yet");
            return;
        }

        let that = this;
        this.ripsPromise.then(function() {
            let partUnityFn = eval(that.partUnityFn);
            let nlandmarks = that.rips.nlandmarks;
            let perc = that.perc;
            that.ripsOpts.landmarksListener.updateDisplay();

            let dgm1 = that.rips.dgms[1];

            //////////////////////////////////////
            // Step 0: Make the cocycle index the index of
            // the largest persistence point (TODO: Update this to GUI element)
            let cocycle_idx = [];
            if (dgm1.births.length > 0) {
                cocycle_idx = [0];
                let max = dgm1.deaths[0] - dgm1.births[0];
                for (let i = 1; i < dgm1.births.length; i++) {
                    let pers = dgm1.deaths[i] - dgm1.births[i];
                    if (pers > max) {
                        max = pers;
                        cocycle_idx[0] = i;
                    }
                }
            }
            //////////////////////////////////////


            let distLandLand = that.rips.distLandLand;
            let distLandData = that.rips.distLandData;
            let doWeighted = that.doWeighted;

            // Step 1: Come up with the representative cocycle as a formal sum
            // of the chosen cocycles
            let cohomdeath = null;
            let cohombirth = null;
            let cocycle = [];
            let prime = that.rips.field;
            for (let idx of cocycle_idx) {
                cocycle = addCochains(cocycle, that.rips.cocycles1[idx], prime);
                if (cohomdeath === null) {
                    cohomdeath = dgm1.births[idx];
                    cohombirth = dgm1.deaths[idx];
                }
                else {
                    cohomdeath = Math.max(cohomdeath, dgm1.births[idx]);
                    cohombirth = Math.min(cohombirth, dgm1.deaths[idx]);
                }
            }
            
            // Step 2: Determine radius for balls
            // coverage = np.max(np.min(distLandData, 1))
            let coverage = 0.0;
            for (let i = 0; i < distLandData.length; i++) {
                let row = distLandData[i];
                if (row.length > 0) {
                    let min = row[0];
                    for (let i = 1; i < row.length; i++) {
                        if (row[i] < min) {
                            min = row[i];
                        }
                    }
                    if (min > coverage) {
                        coverage = min;
                    }
                }
            }
            let rCover = (1-perc)*Math.max(cohomdeath, coverage) + perc*cohombirth;
            that.rCover = rCover // Store covering radius for reference
            
            
            // Step 3: Setup coboundary matrix, delta_0, for Cech_{rCover }
            // and use it to find a projection of the cocycle
            // onto the image of delta0

            // Lift to integer cocycle
            for (let i = 0; i < cocycle.length; i++) {
                let vidx = cocycle[i].length-1;
                if (cocycle[i][vidx] > (prime-1)/2) {
                    cocycle[i][vidx] -= prime;
                }
            }
            // Turn cocycle into a dictionary for easier lookup
            let cocycleDict = getCochainDict(cocycle);
            // Select edges that are under the threshold
            let R = [];
            let Y = [];
            let wSqrt = []; // Square root of distances
            let j = 0;
            let i = 1;
            // Loop through lower triangle
            for (let k = 0; i < distLandLand.length; k++) {
                if (distLandLand[k] < 2*rCover) {
                    R.push([j, i]);
                    let idx = j + "_" + i;
                    if (idx in cocycleDict) {
                        Y.push(cocycleDict[idx]);
                    }
                    else {
                        Y.push(0);
                    }
                    if (doWeighted) {
                        wSqrt.push(Math.sqrt(distLandLand[k]));
                    }
                }
                i++;
                if (i == nlandmarks) {
                    j++;
                    i = j+1;
                }
            }
            // Setup and solve linear system
            let delta0 = makeDelta0(R, nlandmarks); // Coboundary matrix
            let A = delta0;
            let b = Y;
            if (doWeighted) {
                A = numeric.clone(delta0);
                // Point-multiply y by the square root of the weights
                for (let i = 0; i < b.length; i++) {
                    b[i] *= wSqrt[i];
                }
                // Multiply the rows of a by the square root of the weights
                for (let i = 0; i < A[1].length; i++) {
                    A[2][i] *= wSqrt[A[1][i]];
                }
            }
            // Solve the sparse system of linear equations
            A = numeric.ccsFull(A);
            let AT = numeric.transpose(A);
            let ATA = numeric.dot(AT, A);
            let ATb = numeric.dot(AT, b);
            let tau = numeric.solve(ATA, ATb);
            // TODO: Figure out why sparse isn't working here
            delta0 = numeric.ccsFull(delta0); 
            let tauGrad = numeric.dot(delta0, tau);

            let theta = [];
            for (let i = 0; i < R.length; i++) {
                theta.push([R[i][0], R[i][1], -tauGrad[i]]);
            }
            theta = addCochains(cocycle, theta);


            // Step 4: Create the open covering U = {U_1,..., U_{s+1}} and 
            // partition of unity
            let npoints = distLandData[0].length;
            let varphi = [];
            let ballIndx = [];
            // Allocate space for varphi
            for (let i = 0; i < distLandData.length; i++) {
                varphi[i] = new Float32Array(distLandData[i].length);
            }
            let notCovered = 0;
            for (let j = 0; j < npoints; j++) {
                let idxs = [];
                let phis = [];
                let total = 0.0;
                for (let i = 0; i < nlandmarks; i++) {
                    if (distLandData[i][j] < rCover) {
                        let phi = partUnityFn(distLandData[i][j], rCover);
                        idxs.push(i);
                        phis.push(phi);
                        total += phi;
                    }
                }
                if (idxs.length == 0) {
                    notCovered++;
                    ballIndx[j] = 0;
                }
                else {
                    // To each data point, associate the index of the first 
                    // open set it belongs to
                    ballIndx[j] = idxs[0];
                    for (let k = 0; k < idxs.length; k++) {
                        let i = idxs[k];
                        varphi[i][j] = phis[k]/total;
                    }
                }

            }
            if (notCovered > 0) {
                console.log("WARNING: There are " + notCovered + "points not covered by a landmark");
            }


            // Step 5: From U_1 to U_{s+1} - (U_1 \cup ... \cup U_s), apply classifying map
            // compute all transition functions
            varphi = numeric.transpose(varphi);
            let thetaMatrix = [];
            for (let i = 0; i < nlandmarks; i++) {
                thetaMatrix[i] = new Float32Array(nlandmarks);
            }
            for (let k = 0; k < theta.length; k++) {
                let i = theta[k][0];
                let j = theta[k][1];
                let thetak = (theta[k][2] + 0.5) % 1 - 0.5;
                thetaMatrix[i][j] = thetak;
                thetaMatrix[j][i] = -thetak;
            }
            let thetas = new Float32Array(npoints);
            for (let j = 0; j < npoints; j++) {
                thetas[j] = numeric.dot(thetaMatrix[ballIndx[j]], varphi[j]);
                thetas[j] -= tau[ballIndx[j]];
                while (thetas[j] < 0) {
                    thetas[j] += 1;
                }
                thetas[j] = thetas[j] % 1;
            }
            that.thetas = thetas;
            that.repaint2DCanvas();
        });

    }

    /**
     * Draw points colored by their circular coordinates
     */
    repaint2DCanvas() {
        let dW = 5;
        let W = this.tda.canvas2D.width;
        let H = this.tda.canvas2D.height;
        this.tda.canvas2D.ctx2D.clearRect(0, 0, W, H);
        // Draw all of the points in black
        for (let i = 0; i < this.tda.points.length; i++) {
            let r = Math.round(255*this.thetas[i]);
            let g = r;
            let b = r;
            this.tda.canvas2D.ctx2D.fillStyle = "rgb("+r+","+g+","+b+")";
            let x = this.tda.points[i][0];
            let y = this.tda.points[i][1];
            this.canvas2D.ctx2D.fillRect(x, y, dW, dW);
        }
    }



}

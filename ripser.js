class Ripser {
    /**
     * Initialize a Ripser object and setup place holders for all 
     * Emscripten variables to hold distances, diagrams, and cocycles
     * 
     * @param {TDA} tda A handle to the TDA object
     * @param {int} field Prime number for field coefficients for homology (default 2)
     * @param {int} homdim Maximum dimension of homology to compute (default 1)
     * @param {boolean} do_cocycles Whether to compute representative cocycles (default false)
     * @param {string} canvasName A string ID of the DOM element where the plots will be 
     *                            drawn.  If undefined, results won't be drawn automatically
     */
    constructor(tda, field, homdim, do_cocycles, canvasName) {
        this.tda = tda;
        this.field = field;
        this.canvasName = canvasName;
        if (field === undefined) {
            this.field = 2;
        }
        this.homdim = homdim;
        if (homdim === undefined) {
            this.homdim = 1;
        }
        this.do_cocycles = do_cocycles;
        if (do_cocycles === undefined) {
            this.do_cocycles = false;
        }

        this.X = null; // C++ copy of euclidean points
        this.idxPerm = [];  // Indices chosen in greedy permutation
    
        this.distLandLand = null; // VectorFloat holding inter-landmark distances
        this.distLandData = null; // VectorVectorFloat holding landmark to point cloud distances
            
        this.dgms = null; // VectorVectorFloat holding persistence diagrams
        this.cocycles = null; // VectorVectorVectorInt holding representative cocycles
        this.nperm = 1;
        return this;
    }

    /**
     * Setup a menu for doing a rips computation.
     * This is optional; this class can be used without a menu
     */
    setupMenu() {
        const gui = this.tda.gui;
        let ripsOpts = gui.addFolder('Rips Options');
        ripsOpts.add(this, 'field').min(2).step(1);
        ripsOpts.add(this, 'homdim').min(0).step(1);
        ripsOpts.add(this, 'nperm').min(1).step(1);
        ripsOpts.add(this, 'computeRips');
    }


    computeRips() {
        this.computeRipsPC(this.tda.points, this.nperm);
        this.drawLandmarks2DCanvas();
        if (!(this.canvasName === undefined)) {
            plotDGMS(this.dgms, this.canvasName);
        }
    }

    /**
     * Initialize all of the C++ objects needed for rips 
     * computation
     */
    init() {
        if (this.distLandData === null) {
            this.distLandLand = new Module.VectorFloat();
            this.distLandData = new Module.VectorVectorFloat();
            this.X = new Module.VectorVectorFloat();
            this.dgms = new Module.VectorVectorFloat();
            this.cocycles = new Module.VectorVectorVectorInt();
        }
    }

     /**
      * Compute rips on a Euclidean point cloud
      * @param {array} points An array of arrays of coordinates
      * @param {int} nperm The number of points to take in the greedy permutation
      * @param {double} thresh The threshold at which to stop rips
      */
    computeRipsPC(points, nperm, thresh) {
        if (this.tda.isCompiled) {
            this.init();
            // Step 1: Clear the vectors that will hold the output
            Module.clearVector(this.distLandLand);
            Module.clearVectorVector(this.distLandData);
            Module.clearVectorVector(this.X);

            // Step 2: Perform the greedy permutation on a Euclidean
            // point cloud
            for (let i = 0; i < points.length; i++) {
                let x = new Module.VectorFloat();
                for (let j = 0; j < points[i].length; j++) {
                    x.push_back(points[i][j]);
                }
                this.X.push_back(x);
            }
            if (nperm === undefined) {
                // If the number of points in the permutation was not
                // specified, simply make it the number of points in the point cloud
                nperm = points.length;
            }
            nperm = Math.min(nperm, points.length);
            this.nperm = nperm;
            let perm = new Module.getGreedyPerm(this.X, nperm, this.distLandLand, this.distLandData);
            this.idxPerm.length = 0;
            for (let i = 0; i < nperm; i++) {
                this.idxPerm.push(perm.get(i));
            }

            // Step 3: Run ripser
            Module.clearVectorVector(this.dgms);
            Module.clearVectorVectorVectorInt(this.cocycles);
            // Automatically determine the threshold to be greater
            // than the max inter-landmark distance if the threshold
            // was not specified
            if (thresh === undefined) {
                // Make thresh twice the max distance
                thresh = 0.0;
                for (let i = 0; i < this.distLandLand.size(); i++) {
                    let dist = this.distLandLand.get(i);
                    if (dist > thresh) {
                        thresh = dist;
                    }
                }
                thresh = 1.1*thresh;
            }
            Module.jsRipsDM(this.distLandLand, this.field, this.homdim, thresh, this.do_cocycles, 
            this.dgms, this.cocycles);
        }
        else {
            alert("Not Compiled Yet");
        }
    }

    /**
     * Draw the landmarks on the 2D canvas
     */
    drawLandmarks2DCanvas() {
        this.tda.repaint2DCanvas();
        // Draw the landmark slightly bigger and in red
        this.tda.ctx2D.fillStyle = "#ff0000";
        let dW = 10;
        for (let i = 0; i < this.idxPerm.length; i++) {
            let x = this.tda.points[this.idxPerm[i]][0];
            let y = this.tda.points[this.idxPerm[i]][1];
            this.tda.ctx2D.fillRect(x, y, dW, dW);
        }
    }

}

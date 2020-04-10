class Ripser {
    /**
     * Initialize a Ripser object and setup place holders for all 
     * Emscripten variables to hold distances, diagrams, and cocycles
     * 
     * @param {TDA} tda A handle to the TDA object
     * @param {int} field Prime number for field coefficients for homology (default 2)
     * @param {int} homdim Maximum dimension of homology to compute (default 1)
     * @param {boolean} do_cocycles Whether to compute representative cocycles (default false)
     * @param {string} dgmsCanvasName A string ID of the DOM element where the persistence 
     *                                diagrams will be drawn.  If undefined, results won't be 
     *                                drawn automatically
     */
    constructor(tda, field, homdim, do_cocycles, dgmsCanvasName) {
        this.tda = tda;
        this.field = field;
        this.dgmsCanvasName = dgmsCanvasName;
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
        this.cocycles = null; // VectorVectorVector holding representative cocycles
        this.nlandmarks = 100;
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
        // Update landmarks if there aren't enough with .listen()
        ripsOpts.add(this, 'nlandmarks').min(1).step(1).listen(); 
        ripsOpts.add(this, 'computeRips');
    }


    computeRips() {
        this.computeRipsPC(this.tda.points, this.nlandmarks);
        this.drawLandmarks2DCanvas();
        if (!(this.dgmsCanvasName === undefined)) {
            plotDGMS(this.dgms, this.dgmsCanvasName);
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
            this.cocycles = new Module.VectorVectorVectorFloat();
        }
    }

    /**
     * Return the cocycle at a particular index
     * @param {int} dim Dimension of homology
     * @param {int} index Index of persistence point
     * 
     * @returns {array 2d} The nonzero elements of the cocycle
     */
    getCocycle(dim, index) {
        let ret = [];
        let cocycle = this.cocycles.get(dim).get(index);
        for (let i = 0; i < cocycle.size(); i += dim+2) {
            let elem = [];
            for (let k = 0; k < dim+2; k++) {
                elem.push(cocycle.get(i+k));
            }
            ret.push(elem);
        }
        return ret;
    }

     /**
      * Compute rips on a Euclidean point cloud
      * @param {array} points An array of arrays of coordinates
      * @param {int} nlandmarks The number of points to take in the greedy permutation
      * @param {double} thresh The threshold at which to stop rips
      */
    computeRipsPC(points, nlandmarks, thresh) {
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
            if (nlandmarks === undefined) {
                // If the number of points in the permutation was not
                // specified, simply make it the number of points in the point cloud
                nlandmarks = points.length;
            }
            nlandmarks = Math.min(nlandmarks, points.length);
            this.nlandmarks = nlandmarks;
            let perm = new Module.getGreedyPerm(this.X, nlandmarks, this.distLandLand, this.distLandData);
            this.idxPerm.length = 0;
            for (let i = 0; i < nlandmarks; i++) {
                this.idxPerm.push(perm.get(i));
            }

            // Step 3: Run ripser
            Module.clearVectorVector(this.dgms);
            Module.clearVectorVectorVector(this.cocycles);
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
        if (this.tda.canvas2D.style.display != "none") {
            this.tda.repaint2DCanvas();
            // Draw the landmark slightly bigger and in red
            this.tda.canvas2D.ctx2D.fillStyle = "#ff0000";
            let dW = 10;
            for (let i = 0; i < this.idxPerm.length; i++) {
                let x = this.tda.points[this.idxPerm[i]][0];
                let y = this.tda.points[this.idxPerm[i]][1];
                this.tda.canvas2D.ctx2D.fillRect(x, y, dW, dW);
            }
        }
    }

}

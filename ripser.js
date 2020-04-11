class Ripser {
    /**
     * Initialize a Ripser object and setup place holders for 
     * permutation indices, persistence diagrams, cocycles, and distances
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
        this.idxPerm = [];  // Indices chosen in greedy permutation
        this.dgms = {'births':[], 'deaths':[]}; // Persistence diagrams
        this.cocycles1 = []; // Representative cocycles for 1D cohomology
        this.nlandmarks = 100; // Number of landmarks
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
        let that = this;
        this.computeRipsPC(this.tda.points, this.nlandmarks).then(
            function() {
                that.drawLandmarks2DCanvas();
                if (!(that.dgmsCanvasName === undefined)) {
                    plotDGMS(that.dgms, that.dgmsCanvasName);
                }
            }
        );

    }

     /**
      * Compute rips on a Euclidean point cloud
      * @param {array} points An array of arrays of coordinates
      * @param {int} nlandmarks The number of points to take in the greedy permutation
      * @param {double} thresh The threshold at which to stop rips
      */
    computeRipsPC(points, nlandmarks, thresh) {
        let that = this;
        return new Promise(function(resolve, reject) {
            that.tda.progressBar.startLoading("Computing");
            let worker = new Worker("ripserworker.js");
            worker.postMessage({
                points:points, nlandmarks:nlandmarks, thresh:thresh,
                homdim:that.homdim, field:that.field,
                do_cocycles:that.do_cocycles
            });
            worker.onmessage = function(event) {
                if (event.data.message == "finished") {
                    that.tda.progressBar.changeToReady();
                    that.tda.feedbackCanvas.innerHTML = "";
                    // Return back nlandmarks, idxPerm
                    that.idxPerm = event.data.idxPerm;
                    that.nlandmarks = event.data.nlandmarks;
                    that.dgms = event.data.dgms;
                    that.cocycles1 = event.data.cocycles1;
                    console.log(that.cocycles1);
                    resolve();
                }
                else {
                    that.tda.feedbackCanvas.innerHTML = "<h3>" + event.data.message + "</h3>";
                }
            }
        });
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

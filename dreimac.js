/**
 * A class that holds the state of a DREiMac instance, including
 * emscripten compiled modules, current point clouds that are being
 * processed, etc
 * 
 * @param {DOM Element} pccanvas Handle to the HTML where the 2D point cloud
 *                               will be drawn
 */
function dreimacContext(pccanvas) {
    // Member variables
    this.isCompiled = false;
    this.points = []; // Javascript copy of Euclidean points
    this.X = null; // C++ copy of euclidean points
    this.idxPerm = [];  // Indices chosen in greedy permutation

    this.distLandLand = null; // VectorFloat holding inter-landmark distances
    this.distLandData = null; // VectorVectorFloat holding landmark to point cloud distances
        
    this.dgms = null; // VectorVectorFloat holding persistence diagrams
    this.cocycles = null; // VectorVectorVectorInt holding representative cocycles
    
    this.numPermsInput = document.getElementById("numPerms");
    
    this.pccanvas = pccanvas;
    this.ctx2D = pccanvas.getContext("2d");

    Module.onRuntimeInitialized = function () {
        isCompiled = true;
        distLandLand = new Module.VectorFloat();
        distLandData = new Module.VectorVectorFloat();
        X = new Module.VectorVectorFloat();
        dgms = new Module.VectorVectorFloat();
        cocycles = new Module.VectorVectorVectorInt();
        console.log("Finished compiling");
    }

    this.repaint = function() {
        let dW = 5;
        let W = this.pccanvas.width;
        let H = this.pccanvas.height;
        this.ctx2D.clearRect(0, 0, W, H);
        // Draw all of the points in black
        this.ctx2D.fillStyle = "#000000";
        for (let i = 0; i < this.points.length; i++) {
            let x = this.points[i][0];
            let y = this.points[i][1];
            this.ctx2D.fillRect(x, y, dW, dW);
        }

        // Draw the landmark slightly bigger and in red
        this.ctx2D.fillStyle = "#ff0000";
        dW = 10;
        for (let i = 0; i < this.idxPerm.length; i++) {
            let x = this.points[this.idxPerm[i]][0];
            let y = this.points[this.idxPerm[i]][1];
            this.ctx2D.fillRect(x, y, dW, dW);
        }
    }

    this.clickPoint = function() {
        let rect = this.pccanvas.getBoundingClientRect();
        let x = event.clientX - rect.left;
        let y = event.clientY - rect.top;
        this.points.push([x, y]);
        this.repaint();
    }

    this.pccanvas.addEventListener("mousedown", this.clickPoint.bind(this));

    this.computeRips = function() {
        if (this.isCompiled) {
            // Step 1: Clear the vectors that will hold the output
            Module.clearVector(this.distLandLand);
            Module.clearVectorVector(this.distLandData);
            Module.clearVectorVector(this.X);

            // Step 2: Perform the greedy permutation on a Euclidean
            // point cloud
            for (let i = 0; i < this.points.length; i++) {
                let x = new Module.VectorFloat();
                for (let j = 0; j < this.points[i].length; j++) {
                    x.push_back(this.points[i][j]);
                }
                X.push_back(x);
            }
            let k = parseInt(this.numPermsInput.value);
            k = Math.min(k, this.points.length);
            let perm = new Module.getGreedyPerm(this.X, k, this.distLandLand, this.distLandData);
            this.idxPerm.length = 0;
            for (let i = 0; i < k; i++) {
                this.idxPerm.push(perm.get(i));
            }

            // Step 3: Run ripser
            Module.clearVectorVector(this.dgms);
            Module.clearVectorVectorVectorInt(this.cocycles);
            // TODO: Send over infinity as threshold
            let field = 2;
            let homdim = 2;
            let thresh = 1e12;
            Module.jsRipsDM(this.distLandLand, field, homdim, thresh, 1, 
            this.dgms, this.cocycles);
            plotDGMS(this.dgms, 'dgmsCanvas');

            this.repaint();
        }
        else {
            alert("Not Compiled Yet");
        }
    }

}
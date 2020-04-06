/**
 * A class that holds the state of a TDA instance, including info about
 * Emscripten compiled modules, menu handles, and canvas handles
 * for drawing
 */

class TDA {
    // TODO: Add dat.gui menu handles

    /** 
     * @param {DOM Element} pccanvas Handle to the HTML where the 2D point cloud
     *                               will be drawn
     */
    constructor(pccanvas) {
        this.isCompiled = false;
        Module.onRuntimeInitialized = this.init.bind(this);

        this.points = []; // Javascript copy of Euclidean points

        // Variables for a 2D canvas
        this.pccanvas = pccanvas;
        this.ctx2D = pccanvas.getContext("2d");
        this.pccanvas.addEventListener("mousedown", this.clickPoint2DCanvas.bind(this));

        this.numPermsInput = document.getElementById("numPerms");
    }

    /**
     * This is called when Emscripten is finished compiling
     */
    init() {
        this.isCompiled = true;
        console.log("Finished compiling");
    }

    /**
     * Add a point to the 2D canvas
     */
    clickPoint2DCanvas() {
        let rect = this.pccanvas.getBoundingClientRect();
        let x = event.clientX - rect.left;
        let y = event.clientY - rect.top;
        this.points.push([x, y]);
        this.repaint2DCanvas();
    }
    
    repaint2DCanvas() {
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
    }
}
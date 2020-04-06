/**
 * A class that holds the state of a TDA instance, including info about
 * Emscripten compiled modules, menu handles, and canvas handles
 * for drawing
 */

class TDA {
    /** 
     * @param {DOM Element} canvas2D Handle to the HTML where the 2D point cloud
     *                               will be drawn
     * @param {DOM Element} feedbackCanvas Handle to the HTML where the 2D point cloud
     *                               feedback text will be drawn
     */
    constructor(canvas2D, feedbackCanvas) {
        this.isCompiled = false;
        Module.onRuntimeInitialized = this.init.bind(this);

        this.feedbackCanvas = feedbackCanvas;

        this.points = []; // Javascript copy of Euclidean points

        // Variables for a 2D canvas
        this.canvas2D = canvas2D;
        this.ctx2D = canvas2D.getContext("2d");
        this.canvas2D.addEventListener("mousedown", this.clickPoint2DCanvas.bind(this));
        
        this.setupMenu();
    }

    /**
     * Setup a menu for choosing different point clouds
     */
    setupMenu() {
        this.gui = new dat.GUI();
        const gui = this.gui;
        this.dataType = {'canvas2D':true,
                         'synthetic':false,
                         'external':false};
        let dataTypeDisp = {'canvas2D':'2D Canvas', 'synthetic':'Synthetic Point Cloud', 'external':'External File'};
        this.dataTypeDisp = dataTypeDisp;
        let dataMenu = gui.addFolder("Dataset");
        this.dataMenu = dataMenu;
        this.dataTypeMenu = {};
        function setChecked(handle, name){
            for (let otherName in handle.dataType){
                handle.dataType[otherName] = false;
            }
            handle.dataType[name] = true;
            if (name == 'canvas2D') {
                handle.canvas2D.style.display = "block";
                handle.feedbackCanvas.innerHTML = "Please draw a 2D point cloud by left clicking";
            }
            else {
                handle.canvas2D.style.display = "none";
                handle.feedbackCanvas.innerHTML = "Please Select " + handle.dataTypeDisp[name];
            }
        }

        for (let name in this.dataType) {
            this.dataTypeMenu[name] = dataMenu.add(this.dataType, name)
                                        .name(dataTypeDisp[name])
                                        .listen().onChange( function() {
                                                                setChecked(this, name);
                                                            }.bind(this))
        }
        setChecked(this, 'canvas2D');
        this.syntheticMenu = dataMenu.addFolder("Synthetic Point Clouds");
        this.setupKleinMenu();
    }

    /**
     * Sample a klein bottle based on menu parameters, and set 
     * the point cloud to be equal to the samples
     */
    sampleKlein() {
        const ko = this.kleinOptions;
        this.points = sampleKleinBottle(ko.R, ko.P, ko.eps);
        this.feedbackCanvas.innerHTML = "Sampled Klein Bottle with " + this.points.length + " points";
    }

    /**
     * Make a menu for choosing klein bottle parameters
     */
    setupKleinMenu() {
        const syntheticMenu = this.syntheticMenu;
        let kleinMenu = syntheticMenu.addFolder("Klein Bottle");
        this.kleinOptions = {'R':1, 'P':1, 'eps':0.01};
        kleinMenu.add(this.kleinOptions, 'R').min(0);
        kleinMenu.add(this.kleinOptions, 'P').min(0);
        kleinMenu.add(this.kleinOptions, 'eps').min(0);
        kleinMenu.add(this, 'sampleKlein').name("Generate");
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
        let rect = this.canvas2D.getBoundingClientRect();
        let x = event.clientX - rect.left;
        let y = event.clientY - rect.top;
        this.points.push([x, y]);
        this.repaint2DCanvas();
    }
    
    /**
     * Draw all of the selected points on the 2D canvas
     */
    repaint2DCanvas() {
        let dW = 5;
        let W = this.canvas2D.width;
        let H = this.canvas2D.height;
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
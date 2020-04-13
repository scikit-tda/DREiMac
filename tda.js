/**
 * A class to show loading progress
 */
class ProgressBar {
    constructor() {
        //A function to show a progress bar
        this.loading = false;
        this.loadString = "Loading";
        this.loadColor = "#AAAA00";
        this.ndots = 0;
        this.waitingDisp = document.getElementById("pagestatus");
    }

    /**
     * Start the progress bar
     * @param {String} loadString What kind of work is being done
     *                            (e.g. "Loading", "Computing")
     */
    startLoading(loadString) {
        this.loadString = loadString;
        this.loading = true;
        this.changeLoad();
    }

    /**
     * A function to keep the indicated progress dots moving along
     */
    changeLoad() {
        if (!this.loading) {
            return;
        }
        var s = "<h3><font color = \"" + this.loadColor + "\">" + this.loadString;
        for (var i = 0; i < this.ndots; i++) {
            s += ".";
        }
        s += "</font></h3>";
        this.waitingDisp.innerHTML = s;
        if (this.loading) {
            this.ndots = (this.ndots + 1)%12;
            setTimeout(this.changeLoad.bind(this), 200);
        }
    };
    
    /**
     * Finish loading
     */
    changeToReady() {
        this.loading = false;
        this.waitingDisp.innerHTML = "<h3><font color = \"#00AA00\">Ready</font></h3>";
    };
    
    /**
     * Set loading to failed
     */
    setLoadingFailed() {
        this.loading = false;
        this.waitingDisp.innerHTML = "<h3><font color = \"red\">"  + this.loadString + " Failed :(</font></h3>";
    };
}


/**
 * A class that holds the state of a TDA instance, including info 
 * about menu handles and canvas handles for drawing
 */
class TDA {
    /** 
     * @param {DOM Element} canvas2D Handle to the HTML where the 2D point cloud
     *                               will be drawn
     * @param {DOM Element} feedbackCanvas Handle to the HTML where the 2D point cloud
     *                               feedback text will be drawn
     */
    constructor(canvas2D, feedbackCanvas) {
        this.points = []; // Javascript copy of Euclidean points
        this.isCompiled = false;
        

        // Variables for a 2D canvas
        if (!(canvas2D === undefined)) {
            this.canvas2D = canvas2D;
            this.canvas2D.ctx2D = canvas2D.getContext("2d");
            this.canvas2D.addEventListener("mousedown", this.clickPoint2DCanvas.bind(this));
        }

        this.feedbackCanvas = feedbackCanvas;
        this.progressBar = new ProgressBar();
        this.setupMenu();
    }

    /**
     * Toggle the type of selected data
     * @param {string} name Name of the data type
     */
    setDataTypeChecked(name){
        for (let otherName in this.dataType){
            this.dataType[otherName] = false;
        }
        this.dataType[name] = true;
        if (name == 'canvas2D') {
            this.canvas2D.style.display = "block";
            this.feedbackCanvas.innerHTML = "Please draw a 2D point cloud by left clicking. Then compute rips with your chosen parameters";
            this.points = [];
        }
        else {
            this.canvas2D.style.display = "none";
            this.feedbackCanvas.innerHTML = "Please Select " + this.dataTypeDisp[name];
        }
    }

    /**
     * Setup a menu for choosing different point clouds
     */
    setupMenu() {
        if (window.dat === undefined) {
            // Dat.gui hasn't been included, so skip this
            return;
        }
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


        for (let name in this.dataType) {
            this.dataTypeMenu[name] = dataMenu.add(this.dataType, name)
                                        .name(dataTypeDisp[name])
                                        .listen().onChange( function() {
                                                               this.setDataTypeChecked(name);
                                                            }.bind(this))
        }
        this.setDataTypeChecked('canvas2D');
        this.syntheticMenu = dataMenu.addFolder("Synthetic Point Clouds");
        this.setupCircleMenu();
        this.setupKleinMenu();
        this.setupFlatTorusMenu();
    }



    /**
     * Sample a flat torus based on menu parameters, and set 
     * the point cloud to be equal to the samples
     */
    sampleCircle() {
        const co = this.circleOptions;
        this.setDataTypeChecked("canvas2D");
        this.points = sampleCircle(co.R, co.cx, co.cy, co.noise, co.N);
        this.repaint2DCanvas();
        this.feedbackCanvas.innerHTML = "Sampled circle with " + this.points.length + " points.  Now compute rips";
    }

    /**
     * Make a menu for choosing klein bottle parameters
     */
    setupCircleMenu() {
        const syntheticMenu = this.syntheticMenu;
        let circleMenu = syntheticMenu.addFolder("Circle");
        this.circleOptions = {'R':100, 'cx':200, 'cy':200, 'noise':20, 'N':500};
        circleMenu.add(this.circleOptions, 'R').min(0);
        circleMenu.add(this.circleOptions, 'cx');
        circleMenu.add(this.circleOptions, 'cy');
        circleMenu.add(this.circleOptions, 'noise').min(0);
        circleMenu.add(this.circleOptions, 'N').min(1).name("Points");
        circleMenu.add(this, 'sampleCircle').name("Generate");
    }

    /**
     * Sample a klein bottle based on menu parameters, and set 
     * the point cloud to be equal to the samples
     */
    sampleKlein() {
        const ko = this.kleinOptions;
        this.points = sampleKleinBottle(ko.R, ko.P, ko.eps);
        this.setDataTypeChecked("synthetic");
        this.feedbackCanvas.innerHTML = "Sampled Klein Bottle with " + this.points.length + " points.  Now compute rips";
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
     * Sample a flat torus based on menu parameters, and set 
     * the point cloud to be equal to the samples
     */
    sampleFlatTorus() {
        const fto = this.flatTorusOptions;
        this.points = sampleFlatTorus(fto.R1, fto.R2, fto.N);
        this.setDataTypeChecked("synthetic");
        this.feedbackCanvas.innerHTML = "Sampled Flat Torus with " + this.points.length + " points.  Now compute rips";
    }

    /**
     * Make a menu for choosing klein bottle parameters
     */
    setupFlatTorusMenu() {
        const syntheticMenu = this.syntheticMenu;
        let flatTorusMenu = syntheticMenu.addFolder("Flat Torus");
        this.flatTorusOptions = {'R1':1, 'R2':1, 'N':200};
        flatTorusMenu.add(this.flatTorusOptions, 'R1').min(0);
        flatTorusMenu.add(this.flatTorusOptions, 'R2').min(0);
        flatTorusMenu.add(this.flatTorusOptions, 'N').min(1).name("Points");
        flatTorusMenu.add(this, 'sampleFlatTorus').name("Generate");
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
        this.canvas2D.ctx2D.clearRect(0, 0, W, H);
        // Draw all of the points in black
        this.canvas2D.ctx2D.fillStyle = "#000000";
        for (let i = 0; i < this.points.length; i++) {
            let x = this.points[i][0];
            let y = this.points[i][1];
            this.canvas2D.ctx2D.fillRect(x, y, dW, dW);
        }
    }
}
/**
 * Plot a set of persistence diagrams using plotly
 * 
 * @param {VectorVector} dgms All of the persistence diagrams. It
 *                            is assumed that they start at H0
 * @param {string} elemStr A string ID of the DOM element where
 *                         the plots will be placed
 * @param {list} toPlot    A list of diagrams to plot.  By default, plot
 *                         all available
 */
function plotDGMS(dgms, elemStr, toPlot) {
    if (toPlot === undefined) {
        toPlot = [];
        for (let k = 0; k < dgms.length; k++) {
            toPlot.push(k);
        }
    }
    let allPlots = [];
    let axMin = null;
    let axMax = null;
    for (let k of toPlot) {
        let births = dgms[k].births;
        let deaths = dgms[k].deaths;
        let dgmPoints = {x:births, y:deaths, mode:'markers', name:'H'+k};
        allPlots.push(dgmPoints);
        // TODO: Add another persistence diagram for each dimension
        let axMink = Math.min(Math.min.apply(null, births), Math.min.apply(null, deaths));
        let axMaxk = Math.max(Math.max.apply(null, births), Math.max.apply(null, deaths.filter(function(x){
            return x < Infinity;
        })));
        if (axMin === null) {
            axMin = axMink;
            axMax = axMaxk;
        }
        else {
            axMin = Math.min(axMin, axMink);
            axMax = Math.max(axMax, axMaxk);
        }
    }
    let axRange = axMax - axMin;
    let diagonal = {x:[axMin-axRange/5, axMax+axRange/5], y:[axMin-axRange/5, axMax+axRange/5], mode:'lines', name:'diagonal'};
    allPlots.push(diagonal);
    let layout = {title:'Persistence Diagrams',
                  autosize: false,
                  width: 600,
                  height: 600};
    Plotly.newPlot(elemStr, allPlots, layout);
}


/**
 * Plot a set of persistence diagrams using plotly
 * Allows the user the toggle different points to be returned to
 * the place where the plot was called
 * 
 * @param {VectorVector} dgms All of the persistence diagrams. It
 *                            is assumed that they start at H0
 * @param {string} elemStr A string ID of the DOM element where
 *                         the plots will be placed
 * @param {list} toPlot    A list of diagrams to plot.  By default, plot
 *                         all available
 */
function togglePlotDGMS(dgms, elemStr, toPlot){
    if (toPlot === undefined) {
        toPlot = [];
        for (let k = 0; k < dgms.length; k++) {
            toPlot.push(k);
        }
    }
    let allPlots = [];
    let axMin = null;
    let axMax = null;
    for (let k of toPlot) {
        let births = dgms[k].births;
        let deaths = dgms[k].deaths;
        let dgmPoints = {x:births, y:deaths, mode:'markers', name:'H'+k};
        allPlots.push(dgmPoints);
        // TODO: Add another persistence diagram for each dimension
        let axMink = Math.min(Math.min.apply(null, births), Math.min.apply(null, deaths));
        let axMaxk = Math.max(Math.max.apply(null, births), Math.max.apply(null, deaths.filter(function(x){
            return x < Infinity;
        })));
        if (axMin === null) {
            axMin = axMink;
            axMax = axMaxk;
        }
        else {
            axMin = Math.min(axMin, axMink);
            axMax = Math.max(axMax, axMaxk);
        }
    }
    let axRange = axMax - axMin;
    let diagonal = {x:[axMin-axRange/5, axMax+axRange/5], y:[axMin-axRange/5, axMax+axRange/5], mode:'lines', name:'diagonal'};
    allPlots.push(diagonal);
    let layout = {title:'Persistence Diagrams',
                  autosize: false,
                  width: 600,
                  height: 600};
    Plotly.newPlot(elemStr, allPlots, layout);
    let myPlot = document.getElementById(elemStr);
    let idxs = [];
    let xCoords = [];
    let yCoords = [];
    myPlot.on('plotly_click', function(data){
        for(var i=0; i < data.points.length; i++){
            ptIdx = data.points[i].pointNumber;
            let x = data.points[i].x;
            let y = data.points[i].y
            
            if(idxs.includes(ptIdx)){
                let pointLocation = idxs.indexOf(ptIdx);
                idxs.splice(pointLocation,1);
                xCoords.splice(pointLocation,1);
                yCoords.splice(pointLocation,1);
            }else{
                idxs.push(ptIdx);
                xCoords.push(x);
                yCoords.push(y);
            }
        }
        let toggledPts = {x:xCoords, y:yCoords, mode:'markers', name:'Toggled'};
        allPlots.push(toggledPts);
        Plotly.newPlot(elemStr, allPlots , layout);
    });
}

importScripts("ripserem.js");

onmessage = function(event) {
    let data = event.data;
    let points = event.data.points;
    let nlandmarks = event.data.nlandmarks;
    let thresh = event.data.thresh;

    // Step 1: Clear the vectors that will hold the output
    Module.onRuntimeInitialized = function() {
        postMessage({"message":"Initializing STL vectors"});
        let distLandLand = new Module.VectorFloat();
        let distLandData = new Module.VectorVectorFloat();
        let X = new Module.VectorVectorFloat();
        let dgms = new Module.VectorVectorFloat();
        let cocycles = new Module.VectorVectorVectorFloat();

    
        // Step 2: Perform the greedy permutation on a Euclidean
        // point cloud
        postMessage({"message":"Performing greedy permutation"});
        for (let i = 0; i < points.length; i++) {
            let x = new Module.VectorFloat();
            for (let j = 0; j < points[i].length; j++) {
                x.push_back(points[i][j]);
            }
            X.push_back(x);
        }
        if (nlandmarks === undefined) {
            // If the number of points in the permutation was not
            // specified, simply make it the number of points in the point cloud
            nlandmarks = points.length;
        }
        nlandmarks = Math.min(nlandmarks, points.length);
        let perm = new Module.getGreedyPerm(X, nlandmarks, distLandLand, distLandData);
        let idxPerm = [];
        for (let i = 0; i < nlandmarks; i++) {
            idxPerm.push(perm.get(i));
        }
    
        // Step 3: Run ripser
        postMessage({"message":"Running ripser"});
        // Automatically determine the threshold to be greater
        // than the max inter-landmark distance if the threshold
        // was not specified
        if (thresh === undefined) {
            // Make thresh twice the max distance
            thresh = 0.0;
            for (let i = 0; i < distLandLand.size(); i++) {
                let dist = distLandLand.get(i);
                if (dist > thresh) {
                    thresh = dist;
                }
            }
            thresh = 1.1*thresh;
        }
        postMessage({"message":"Running ripser, thresh = " + thresh});
        Module.jsRipsDM(distLandLand, data.field, data.homdim, thresh, data.do_cocycles, 
        dgms, cocycles);

        let ret = {"message":"finished", "nlandmarks":nlandmarks, "idxPerm":idxPerm};
        
        // Step 4: Copy over diagrams and cocycles
        postMessage({"message":"Copying over diagrams and cocycles"});
        ret.dgms = [];
        for (let i = 0; i < dgms.size(); i++) {
            let dgmin = dgms.get(i);
            let dgmout = {'births':[], 'deaths':[]};
            if (!(dgmin === undefined)) {
                let N = dgmin.size()/2;
                for (let k = 0; k < N; k++) {
                    dgmout.births.push(dgmin.get(k*2));
                    dgmout.deaths.push(dgmin.get(k*2+1));
                }
            }
            ret.dgms.push(dgmout);
        }
        ret.cocycles1 = [];
        if (cocycles.size() > 1) {
            let dim = 1;
            let cocycles1_in = cocycles.get(1);
            for (let index = 0; index < cocycles1_in.size(); index++) {
                let cocycle_in = cocycles1_in.get(index);
                let cocycle_out = [];
                for (let i = 0; i < cocycle_in.size(); i += dim+2) {
                    let elem = [];
                    for (let k = 0; k < dim+2; k++) {
                        elem.push(cocycle_in.get(i+k));
                    }
                    cocycle_out.push(elem);
                }
                ret.cocycles1.push(cocycle_out);
            }
        }



        // Step 5: Copy over distances
        postMessage({"message":"Copying over distances"});
        // TODO: Finish this

    
        postMessage(ret);
    }
    
}

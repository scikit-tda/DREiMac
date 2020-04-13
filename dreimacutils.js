
//#########################################
//        Cohomology Utility Functions
//#########################################

/**
 * A function for helping to hash cocycle indices
 * @param {array} indices A list of sorted indices
 * @returns {string} "idx1_idx2", where idx1 < idx2 
 */
function indices_to_string(indices) {
    let s = "";
    for (let i = 0; i < indices.length; i++) {
        s = s + indices[i];
        if (i < indices.length-1) {
            s = s + "_";
        }
    }
    return s;
}

/**
 * A function to convert back from a hash string to indices
 * @param {String} s "idx1_idx2", where idx1 < idx2
 * @return {array} The indices as ints
 */
function string_to_indices(s) {
    let indices = s.split("_");
    for (let i = 0; i < indices.length; i++) {
        indices[i] = parseInt(indices[i]);
    }
    return indices;
}

/**
 * 
 * @param {array} c A list of [idx1, idx2, value] for the cochain
 * @returns {dictionary} A dictionary of index strings to cochain values
 */
function getCochainDict(c) {
    let S = {};
    for (elem of c) {
        let indices = elem.slice(0, -1);
        let v = parseFloat(elem[elem.length-1]);
        if (indices[1] < indices[0]) {
            indices.sort();
            v = -v;
        }
        let key = indices_to_string(indices);
        if (!(key in S)) {
            S[key] = v;
        }
        else {
            S[key] += v;
        }
    }
    return S;
}

/**
 * Perform the formal sum of two cochains over some field
 * 
 * @param {array} c1 A list of [idx1, idx2, value] for the 1st cochain
 * @param {array} c2 A list of [idx1, idx2, value] for the 2nd cochain
 * @param {int} p Prime for the field
 * 
 * @returns {array} The formal sum of the two cochains, expressed in
 *                  the same format
 */
function addCochains(c1, c2, p) {
    if (p === undefined) {
        p = 2;
    }
    let c = [];
    c = c.concat(c1);
    c = c.concat(c2);
    let S = getCochainDict(c);
    let cret = [];
    for (let key of Object.keys(S)) {
        let elem = string_to_indices(key);
        let v = S[key];
        v = v % p;
        while (v < 0) {
            v += p;
        }
        if (v != 0) {
            elem.push(v);
            cret.push(elem);
        }
    }
    return cret;
}

/**
 * Return the delta0 coboundary matrix
 * 
 * @param {array} R NEdges x 2 matrix specifying edges, where
    orientation is taken from the first column to the second column
    R specifies the "natural orientation" of the edges, with the
    understanding that the ranking will be specified later
    It is assumed that there is at least one edge incident
    on every vertex
 * @param {int} NVertices The number of vertices
 * @returns {numeric sparse} An NEdges x NVertices sparse matrix
 *                          holding the coboundary matrix
 */
function makeDelta0(R, NVertices) {
    let NEdges = R.length;
    Delta0 = [];
    for (let i = 0; i < NEdges; i++) {
        Delta0[i] = Array(NVertices);
    }
    for (let i = 0; i < NEdges; i++) {
        Delta0[i][R[i][0]] = -1;
        Delta0[i][R[i][1]] = 1;
    }
    return numeric.ccsSparse(Delta0);
}




//#########################################
//        Partition of Unity Functions
//#########################################

/**
 * Linear partition of unity
 * 
 * @param {float} d A distance between a landmark and a data point
 * @param {float} rCover Covering radius
 * @returns {float} The bump function
 */
function partUnityLinear(d, rCover) {
    return rCover - d;
}

/**
 * Quadratic partition of unity
 * 
 * @param {float} d A distance between a landmark and a data point
 * @param {float} rCover Covering radius
 * @returns {float} The bump function
 */
function partUnityQuadratic(d, rCover) {
    return (rCover - d)*(rCover-d);
}

/**
 * Exponential partition of unity
 * 
 * @param {float} d A distance between a landmark and a data point
 * @param {float} rCover Covering radius
 * @returns {float} The bump function
 */
function partUnityExp(d, rCover) {
    return Math.exp(rCover*rCover/((rCover - d)*(rCover-d)));
}

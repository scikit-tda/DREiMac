
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
    S = {};
    c = [];
    c = c.concat(c1);
    c = c.concat(c2);
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
    return cret
}

/*
def make_delta0(R):
    """
    Return the delta0 coboundary matrix
    :param R: NEdges x 2 matrix specifying edges, where orientation
    is taken from the first column to the second column
    R specifies the "natural orientation" of the edges, with the
    understanding that the ranking will be specified later
    It is assumed that there is at least one edge incident
    on every vertex
    """
    NVertices = int(np.max(R) + 1)
    NEdges = R.shape[0]
    
    #Two entries per edge
    I = np.zeros((NEdges, 2))
    I[:, 0] = np.arange(NEdges)
    I[:, 1] = np.arange(NEdges)
    I = I.flatten()
    
    J = R[:, 0:2].flatten()
    
    V = np.zeros((NEdges, 2))
    V[:, 0] = -1
    V[:, 1] = 1
    V = V.flatten()
    I = np.array(I, dtype=int)
    J = np.array(J, dtype=int)
    Delta = sparse.coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
    return Delta


//#########################################
//        Partition of Unity Functions
//#########################################

def partunity_linear(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return r_cover - ds

def partunity_quadratic(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return (r_cover - ds)**2

def partunity_exp(ds, r_cover):
    """
    Parameters
    ----------
    ds: ndarray(n)
        Some subset of distances between landmarks and 
        data points
    r_cover: float
        Covering radius
    Returns
    -------
    varphi: ndarray(n)
        The bump function
    """
    return np.exp(r_cover**2/(ds**2-r_cover**2))*/
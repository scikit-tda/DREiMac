//Will be used to check that the brute force computed distaLandData are the same as the internally computed ones
function compareVectorVectorFloats(a, b) {
    let isEqual = true;
    if (a.size() == b.size() && a.get(0).size() == b.get(0).size()) {
        for (let i = 0; i < a.size(); i++) {
            for (let j = 0; j < a.get(0).size();j++) {
                if (a.get(i).get(j) != b.get(i).get(j)) {
                    isEqual = false;
                }
            }
        }
    } else {
        isEqual = false;
    }


    return isEqual;
}

//Will be used to check that the brute force computed distLandLand are the same as the internally computed ones
function compareVectorFloats(a, b) {
    let isEqual = true;
    if (a.size() == b.size()) {
        for (let i = 0; i < a.size(); i++) {
            if (a.get(i) != b.get(i)) {
                isEqual = false;
            }
        }
    } else {
        isEqual = false;
    }
    return isEqual;
}

//Changed distLandLandCheck from raw calc to just pulling from distLandDataCheck
function computeDistLandVec(X, idxPerm) {
    for (let i = 0; i < idxPerm.size(); i++) {
        let x = new Module.VectorFloat();
        let point1 = X.get(idxPerm.get(i));
        for (let j = 0; j < X.size(); j++) {
            let point2 = X.get(j);
            let distance = 0.0;
            for (let k = 0; k < point1.size(); k++) {
                distance += (point1.get(k) - point2.get(k)) * (point1.get(k) - point2.get(k));
            }
            distance = Math.sqrt(distance);
            x.push_back(distance);
        }
        distLandDataCheck.push_back(x);
    }
    for (let j = 0; j < idxPerm.size(); j++) {
        let point1 = X.get(idxPerm.get(j));
        for (let i = j+1; i < idxPerm.size(); i++) {
            let point2 = X.get(idxPerm.get(i));
            let distance = 0.0;
            for (let k = 0; k < point1.size(); k++) {
                distance += (point1.get(k) - point2.get(k)) * (point1.get(k) - point2.get(k));
            }
            distance = Math.sqrt(distance);
            distLandLandCheck.push_back(distance);
        }
    }
}

function cochainsEqual(c1, c2) {
    let S1 = {};
    let S2 = {};
    for (let i = 0; i < c1.length; i++) {
        idx1 = indices_to_string(c1[i].slice(0, -1));
        S1[idx1] = c1[i][c1[i].length-1];
    }
    for (let i = 0; i < c2.length; i++) {
        idx1 = indices_to_string(c2[i].slice(0, -1));
        S2[idx1] = c2[i][c2[i].length-1];
    }
    for (let key of Object.keys(S1)) {
        proclaim.isTrue(key in S2);
        if (key in S2) {
            proclaim.equal(S1[key], S2[key]);
        }
    }
    for (let key of Object.keys(S2)) {
        proclaim.isTrue(key in S1);
        if (key in S1) {
            proclaim.equal(S1[key], S2[key]);
        }
    }
}

tda = new TDA();

describe('ripser', function() {

});

describe('dreimacutils', function() {
  describe('#addCochains1()', function() {
    it('Adding 1-cochains under a field', function() {
        let c1 = [[0, 1, 3], [1, 2, 4], [0, 3, 3]];
        let c2 = [[1, 0, 1], [1, 2, 1]];
        let p = 5;
        let expValue = [[0, 1, 2], [0, 3, 3]];
        let result = addCochains(c1, c2, p);
        cochainsEqual(expValue, result);
    });
  });
});


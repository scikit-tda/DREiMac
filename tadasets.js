/**
 * Sample a parameterized Klein bottle in R^4
 * 
 * @param R constant to scale x,y
 * @param P constant to scale z,w
 * @param eps tiny constant used to avoid self intersection
 * 
 * @returns A list of lists of point coordinates
 */
function sampleKleinBottle(R, P, eps) {
    let points = [];
    for (let theta = 0.0; theta < 4 * Math.PI; theta += 0.5) {
        for (let v = 0.0; v < 2 * Math.PI; v += 0.5) {
            //Calcs x,y,z,w
            let x = R * (Math.cos(theta / 2.0) * Math.cos(v) - Math.sin(theta / 2.0) * Math.sin(2 * v));
            let y = R * (Math.sin(theta / 2.0) * Math.cos(v) + Math.cos(theta / 2.0) * Math.sin(2 * v));
            let z = P * Math.cos(theta) * (1.0 + eps * Math.sin(v));
            let w = P * Math.sin(theta) * (1.0 + eps * Math.sin(v));
            //Pushes x,y,z,w into point
            points.push([x, y, z, w]);
        }
    }
    return points;
}


/**
 * Sample a parameterized flat torus bottle in R^4
 * 
 * @param R1 Radius of the first loop
 * @param R2 Radius of the second loop
 * @param N Number of points to sample
 * 
 * @returns A list of lists of point coordinates
 */
function sampleFlatTorus(R1, R2, N) {
    let points = [];
    for (let i = 0; i < N; i++) {
        let theta = Math.random()*2*Math.PI;
        let phi = Math.random()*2*Math.PI;
        let x = R1 * Math.cos(theta);
        let y = R1 * Math.sin(theta);
        let z = R2 * Math.cos(phi);
        let w = R2 * Math.sin(phi);
        points.push([x, y, z, w]);
    }
    return points;
}
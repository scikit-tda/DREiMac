/**
 * Sample a parameterized Klein bottle in R^4
 * 
 * @param kleinPoints A VectorVector object to fill in with
 *                    the coordinates of the Klein bottle
 * @param R constant to scale x,y
 * @param P constant to scale z,w
 * @param eps tiny constant used to avoid self intersection
 */
function sampleKleinBottle(kleinPoints, R, P, eps) {
    Module.clearVectorVector(kleinPoints);
    let x = 0;
    let y = 0;
    let z = 0;
    let w = 0;
    
    for (let theta = 0.0; theta < 4 * Math.PI; theta += 0.5) {
        for (let v = 0.0; v < 2 * Math.PI; v += 0.5) {
            let point = new Module.VectorFloat();//Creates empty point
            //Calcs x,y,z,w
            x = R * (Math.cos(theta / 2.0) * Math.cos(v) - Math.sin(theta / 2.0) * Math.sin(2 * v));
            y = R * (Math.sin(theta / 2.0) * Math.cos(v) + Math.cos(theta / 2.0) * Math.sin(2 * v));
            z = P * Math.cos(theta) * (1.0 + eps * Math.sin(v));
            w = P * Math.sin(theta) * (1.0 + eps * Math.sin(v));
            //Pushes x,y,z,w into point
            point.push_back(x);
            point.push_back(y);
            point.push_back(z);
            point.push_back(w);
            //Pushes point back on list
            kleinPoints.push_back(point);
        }
    }
}

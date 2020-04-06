const PRIME_FIELD = 41; // Field for homology

class CircularCoords {

    // TODO: Have this class add on menu options to the tda instance

    /**
     * 
     * @param {TDA} tda A handle to the TDA object
     */
    constructor(tda) {
        this.tda = tda;
        this.rips = new Ripser(tda, PRIME_FIELD, 1, true);
    }

}

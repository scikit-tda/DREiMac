import numpy as np

from dreimac import CircularCoords, ToroidalCoords, GeometryExamples


class TestCircular:
    def test_input_warnings(self):
        pass

    def test_two_circle(self):
        """
        Test two noisy circles of different sizes
        """
        prime = 41
        np.random.seed(2)
        N = 500
        X = np.zeros((N*2, 2))
        t = np.linspace(0, 1, N+1)[0:N]**1.2
        t = 2*np.pi*t
        X[0:N, 0] = np.cos(t)
        X[0:N, 1] = np.sin(t)
        X[N::, 0] = 2*np.cos(t) + 4
        X[N::, 1] = 2*np.sin(t) + 4
        perm = np.random.permutation(X.shape[0])
        X = X[perm, :]
        X = X + 0.2*np.random.randn(X.shape[0], 2)

        f = np.concatenate((t, t + np.max(t)))
        f = f[perm]
        fscaled = f - np.min(f)
        fscaled = fscaled/np.max(fscaled)
        
        cc = CircularCoords(X, 100, prime = prime)
        coords = cc.get_coordinates(cocycle_idx=0)

        assert len(coords) == len(X)
        ## TODO: Check something about circular coordinates

    def test_torus(self):
        """
        Test a 3D torus
        """
        prime = 41
        n_samples = 10000
        n_landmarks = 100
        R = 5
        r = 2
        X = GeometryExamples.torus_3d(n_samples, R, r)
        cc = CircularCoords(X, n_landmarks, prime=prime)
        coords = cc.get_coordinates(cocycle_idx=0)

        assert len(coords) == len(X)

    def test_trefoil(self):
        X = GeometryExamples.trefoil(n_samples = 2500, horizontal_width=10)
        prime = 41
        perc = 0.1
        cocycle_idx_index = 0
        cc = CircularCoords(X, 500, prime=prime)
        coords = cc.get_coordinates(perc=perc, cocycle_idx=cocycle_idx_index, check_cocycle_condition =False)

        assert len(coords) == len(X)

        prime = 3
        large_perc = 0.8
        cc = CircularCoords(X, 300, prime=prime)
        coords_large_perc_fixed = cc.get_coordinates(perc=large_perc, cocycle_idx=cocycle_idx_index)
        assert len(coords_large_perc_fixed) == len(X)

        coords_large_perc_not_fixed = cc.get_coordinates(perc=large_perc, cocycle_idx=cocycle_idx_index, check_cocycle_condition=False)

        assert len(coords_large_perc_not_fixed) == len(X)
        
    def test_genus_two_surface(self):
        # TODO: use the following instead
        # try:
        #   ...
        # except ...:
        #   self.fail(...)
        X = GeometryExamples.genus_two_surface()
        tc = ToroidalCoords(X, n_landmarks=1000)
        cocycle_idxs = [0,1,2,3]
        toroidal_coords = tc.get_coordinates(cocycle_idxs = cocycle_idxs)

        assert toroidal_coords.shape == (4,X.shape[0])

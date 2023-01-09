import numpy as np

from dreimac import CircularCoords, GeometryExamples


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
        ## TODO: Check something about circular coordinates
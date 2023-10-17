import numpy as np
from scipy.spatial import KDTree

from dreimac import CircularCoords, ToroidalCoords, GeometryExamples


def _less_than_or_equal_with_tolerance(x, y):
    return np.allclose(x, y) or x <= y


class TestCircular:
    def test_toroidal_coordinates_independent(self):
        """
        Test that toroidal coordinates returns two circle-valued maps, each one constant in
        one of the two circles.
        """
        prime = 41
        np.random.seed(2)
        N = 500
        X = np.zeros((N * 2, 2))
        t = np.linspace(0, 1, N + 1)[0:N] ** 1.2
        t = 2 * np.pi * t
        X[0:N, 0] = np.cos(t)
        X[0:N, 1] = np.sin(t)
        X[N::, 0] = 2 * np.cos(t) + 4
        X[N::, 1] = 2 * np.sin(t) + 4
        X = X + 0.2 * np.random.randn(X.shape[0], 2)

        tc = ToroidalCoords(X, 200, prime=prime)
        coords1, coords2 = tc.get_coordinates(cocycle_idxs=[0, 1])

        assert len(coords1) == len(X)
        assert len(coords2) == len(X)
        print(
            _maximum_circle_distance(coords1[:N]),
            _maximum_circle_distance(coords2[N:]),
            _maximum_circle_distance(coords2[:N]),
            _maximum_circle_distance(coords1[N:]),
        )
        assert (
            np.isclose(_maximum_circle_distance(coords1[:N]), 0)
            and np.isclose(_maximum_circle_distance(coords2[N:]), 0)
        ) or (
            np.isclose(_maximum_circle_distance(coords2[:N]), 0)
            and np.isclose(_maximum_circle_distance(coords1[N:]), 0)
        )

    def test_toroidal_coordinates_less_energy(self):
        """
        Test that toroidal coordinates returns circle-valued maps with less Dirichlet
        energy in total.
        """
        # TODO: use the following instead
        # try:
        #   ...
        # except ...:
        #   self.fail(...)
        X = GeometryExamples.genus_two_surface()
        tc = ToroidalCoords(X, n_landmarks=1000)
        cocycle_idxs = [0, 1, 2, 3]
        toroidal_coords = tc.get_coordinates(cocycle_idxs=cocycle_idxs)

        assert toroidal_coords.shape == (4, X.shape[0])
        assert _less_than_or_equal_with_tolerance(
            np.linalg.norm(tc._gram_matrix), np.linalg.norm(tc._original_gram_matrix)
        )

        X = GeometryExamples.torus_3d(2000, 5, 1, seed=0)
        tc = ToroidalCoords(X, n_landmarks=500)
        cocycle_idxs = [0, 1]
        toroidal_coords = tc.get_coordinates(cocycle_idxs=cocycle_idxs)

        assert toroidal_coords.shape == (2, X.shape[0])
        assert _less_than_or_equal_with_tolerance(
            np.linalg.norm(tc._gram_matrix), np.linalg.norm(tc._original_gram_matrix)
        )


        X = GeometryExamples.bullseye()
        tc = ToroidalCoords(X, n_landmarks=300)
        cocycle_idxs = [0, 1, 2]
        toroidal_coords = tc.get_coordinates(cocycle_idxs=cocycle_idxs)

        assert toroidal_coords.shape == (3, X.shape[0])
        assert _less_than_or_equal_with_tolerance(
            np.linalg.norm(tc._gram_matrix), np.linalg.norm(tc._original_gram_matrix)
        )


        X = GeometryExamples.three_circles()
        tc = ToroidalCoords(X, n_landmarks=300)
        cocycle_idxs = [0, 1, 2]
        toroidal_coords = tc.get_coordinates(cocycle_idxs=cocycle_idxs)

        assert toroidal_coords.shape == (3, X.shape[0])
        assert _less_than_or_equal_with_tolerance(
            np.linalg.norm(tc._gram_matrix), np.linalg.norm(tc._original_gram_matrix)
        )


    def test_trefoil(self):
        """Check that circular coordinates returns a continuous map, even when the lifted
        cochain may fail to be a cocycle and fix the cocycle using the integer linear system
        """
        for noisy in [True, False]:
            X = GeometryExamples.trefoil(n_samples=2500, horizontal_width=10, noisy=noisy)

            prime = 3
            large_perc = 0.8
            cc = CircularCoords(X, 300, prime=prime)
            coords = cc.get_coordinates(
                perc=large_perc,
                cocycle_idx=0,
                check_cocycle_condition=True,
            )
            assert len(coords) == len(X)

            tree = KDTree(X)
            k = 5
            _, nns = tree.query(X, k=k)

            tolerance = 5 / 100 * (2 * np.pi)  # 5% of the full circle

            for i in range(X.shape[0]):
                assert _maximum_circle_distance(coords[nns[i]]) <= tolerance


def _circle_distance(x, y):
    return np.minimum(
        np.minimum(np.abs(x - y), np.abs((x - 2 * np.pi) - y)),
        np.abs(x - (y - 2 * np.pi)),
    )


def _maximum_circle_distance(xs):
    return max([_circle_distance(a, b) for a in xs for b in xs])

import numpy as np

from dreimac import ProjectiveCoords, ComplexProjectiveCoords, GeometryExamples

class TestRealProjective:

    def test_rp2(self):
        """
        Test projective coordinates on the projective plane
        Parameters
        ----------
        NSamples : int
            Number of random samples on the projective plane
        NLandmarks : int
            Number of landmarks to take in the projective coordinates computation
        """
        n_samples = 10000
        n_landmarks = 100
        X, D = GeometryExamples.rp2_metric(n_samples)

        ## Coming up with ground truth theta and phi for RP2 for colors
        #SOrig = ProjectiveCoords.get_stereo_proj_codim1(X)
        #phi = np.sqrt(np.sum(SOrig**2, 1))
        #theta = np.arccos(np.abs(SOrig[:, 0]))
        
        pc = ProjectiveCoords(D, n_landmarks, distance_matrix=True, verbose=True)
        coordinates = pc.get_coordinates()
        assert len(coordinates) == len(X)
        total_variance = np.linalg.norm(pc.variance_)
        target_coordinates = 2
        variance_threshold = 0.8
        assert np.linalg.norm(pc.variance_[:target_coordinates]) >= total_variance * variance_threshold



    def test_klein_bottle(self):
        """
        Test projective coordinates on the Klein bottle
        """
        n_samples = 10000
        n_landmarks = 200
        R = 2
        r = 1
        X = GeometryExamples.klein_bottle_4d(n_samples, R, r)
        pc = ProjectiveCoords(X, n_landmarks, verbose=True)
        coordinates = pc.get_coordinates()
        assert len(coordinates) == len(X)
        total_variance = np.linalg.norm(pc.variance_)
        target_coordinates = 2
        variance_threshold = 0.3
        assert np.linalg.norm(pc.variance_[:target_coordinates]) >= total_variance * variance_threshold


    def test_line_segment(self):
        """
        Test projective coordinates on a set of line segment patches
        """
        dim = 10
        P = GeometryExamples.line_patches(dim=dim, n_angles=200, n_offsets=200, sigma=0.25)
        pc = ProjectiveCoords(P, n_landmarks=100)
        coordinates = pc.get_coordinates()
        assert len(coordinates) == len(P)
        total_variance = np.linalg.norm(pc.variance_)
        target_coordinates = 2
        variance_threshold = 0.3
        assert np.linalg.norm(pc.variance_[:target_coordinates]) >= total_variance * variance_threshold


    def test_image_patches(self):
        X = GeometryExamples.line_patches(dim=10, n_angles=200, n_offsets=200, sigma=0.25)
        proj = ProjectiveCoords(X, n_landmarks=200)
        coordinates = proj.get_coordinates(proj_dim=2, perc=0.8, cocycle_idx=0)
        assert len(coordinates) == len(X)
        total_variance = np.linalg.norm(proj.variance_)
        target_coordinates = 2
        variance_threshold = 0.3
        assert np.linalg.norm(proj.variance_[:target_coordinates]) >= total_variance * variance_threshold


class TestComplexProjective:

    def test_sphere(self):
        data = GeometryExamples.sphere(2000)
        cpc = ComplexProjectiveCoords(data, n_landmarks=100)
        target_coordinates = 1
        coordinates = cpc.get_coordinates(proj_dim=target_coordinates)
        assert len(coordinates) == len(data)
        total_variance = np.linalg.norm(cpc.variance_)
        variance_threshold = 0.3
        assert np.linalg.norm(cpc.variance_[:target_coordinates]) >= total_variance * variance_threshold

    def test_moving_dot(self):
        P = GeometryExamples.moving_dot(40)
        cpc = ComplexProjectiveCoords(P, n_landmarks=300)
        target_coordinates = 1
        coordinates = cpc.get_coordinates(proj_dim=target_coordinates)
        assert len(coordinates) == len(P)
        total_variance = np.linalg.norm(cpc.variance_)
        variance_threshold = 0.3
        assert np.linalg.norm(cpc.variance_[:target_coordinates]) >= total_variance * variance_threshold

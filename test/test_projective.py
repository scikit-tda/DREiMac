import numpy as np

from dreimac import ProjectiveCoords, GeometryExamples

class TestProjective:
    def test_input_warnings(self):
        pass

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
        print("RP2 Test")
        n_samples = 10000
        n_landmarks = 100
        X, D = GeometryExamples.rp2_metric(n_samples)

        # Coming up with ground truth theta and phi for RP2 for colors
        SOrig = ProjectiveCoords.get_stereo_proj_codim1(X)
        phi = np.sqrt(np.sum(SOrig**2, 1))
        theta = np.arccos(np.abs(SOrig[:, 0]))
        
        pc = ProjectiveCoords(D, n_landmarks, distance_matrix=True, verbose=True)
        ## TODO: Check something about projective coordinates


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
        ## TODO: Check something about projective coordinates


    def test_line_segment(self):
        """
        Test projective coordinates on a set of line segment patches
        """
        dim = 10
        P = GeometryExamples.line_patches(dim=dim, n_angles=200, n_offsets=200, sigma=0.25)
        pc = ProjectiveCoords(P, n_landmarks=100)
        ## TODO: Test something about projective coordinates

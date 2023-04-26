class TestLibrary:
    # Does the library install in scope? Are the objects in scope?
    def test_import(self):
        from dreimac import CircularCoords, ToroidalCoords, ProjectiveCoords, PartUnity, GeometryExamples, CircleMapUtils, PlotUtils
        assert True


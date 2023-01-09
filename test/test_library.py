class TestLibrary:
    # Does the library install in scope? Are the objects in scope?
    def test_import(self):
        import dreimac
        from dreimac import CircularCoords, ProjectiveCoords, PartUnity, GeometryExamples
        assert 1


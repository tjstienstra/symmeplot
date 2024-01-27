import numpy as np
import pytest

try:
    from pyqtgraph.opengl import MeshData
    from symmeplot.pyqtgraph.artists import create_tube_mesh_data
except ImportError:
    pytest.skip("PyQtGraph not installed.", allow_module_level=True)

class TestTubeMeshData:
    def test_straight_tube(self):
        mesh_data = create_tube_mesh_data((0, 1), (2, 2), (0, 0, 0), (0, 0, 1), 4)
        verts, faces = mesh_data._vertexes, mesh_data._faces
        assert isinstance(mesh_data, MeshData)
        assert verts.shape == (8, 3)
        assert faces.shape == (8, 3)
        np.testing.assert_almost_equal(verts, np.array([
            [ 2.,  0.,  0.], [ 0.,  2.,  0.], [-2.,  0.,  0.], [-0., -2.,  0.],
            [ 2.,  0.,  1.], [ 0.,  2.,  1.], [-2.,  0.,  1.], [-0., -2.,  1.]]))
        np.testing.assert_equal(faces, np.array([
            [0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7],
            [4, 5, 1], [5, 6, 2], [6, 7, 3], [7, 4, 0]]))

    def test_cone0(self):
        mesh_data = create_tube_mesh_data((0, 2), (0, 1), (0, 0, 0), (0, 0, 1), 4)
        verts, faces = mesh_data._vertexes, mesh_data._faces
        assert verts.shape == (5, 3)
        assert faces.shape == (4, 3)
        np.testing.assert_almost_equal(verts, np.array([
            [ 0.,  0.,  0.],
            [ 1.,  0.,  2.], [ 0.,  1.,  2.], [-1.,  0.,  2.], [-0., -1.,  2.]]))
        np.testing.assert_equal(faces, np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]))

    def test_cone1(self):
        mesh_data = create_tube_mesh_data((0, 2), (1, 0), (0, 0, 0), (0, 0, 1), 4)
        verts, faces = mesh_data._vertexes, mesh_data._faces
        assert verts.shape == (5, 3)
        assert faces.shape == (4, 3)
        np.testing.assert_almost_equal(verts, np.array([
            [ 1.,  0.,  0.], [ 0.,  1.,  0.], [-1.,  0.,  0.], [-0., -1.,  0.],
            [ 0.,  0.,  2.]]))
        np.testing.assert_equal(faces, np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))

    def test_line(self):
        mesh_data = create_tube_mesh_data((0, 2), (0, 0), (0, 0, 0), (0, 0, 1), 4)
        verts, faces = mesh_data._vertexes, mesh_data._faces
        assert verts.shape == (2, 3)
        assert faces.shape == (0, 3)

    def test_direction(self):
        mesh_data = create_tube_mesh_data((0, 2), (0, 0), (0, 0, 0), (1, 1, 1))
        verts = mesh_data._vertexes
        np.testing.assert_almost_equal(verts[0], np.array([ 0.,  0.,  0.]))
        np.testing.assert_almost_equal(
            verts[1], np.array([ 1.,  1.,  1.]) / np.sqrt(3) * 2)

    def test_position(self):
        mesh_data = create_tube_mesh_data((0, 4), (0, 0), (1, 2, 3), (0, 0, 1))
        verts = mesh_data._vertexes
        np.testing.assert_almost_equal(verts, np.array([
            [ 1.,  2.,  3.], [ 1.,  2.,  7.]]))

    def test_complex_shape(self):
        # Vector like shape
        mesh_data = create_tube_mesh_data(
            (0, 0, 3, 3, 5), (0, 2, 2, 4, 0), (1, 2, 3), (1, 2, 1), 50)
        verts, faces = mesh_data._vertexes, mesh_data._faces
        assert verts.shape == (152, 3)  # 1 + 50 + 50 + 50 + 1
        assert faces.shape == (300, 3)  # 50 + 2 * 50 + 2 * 50 + 50
        assert len(np.unique(verts.round(4), axis=0)) == 152
        assert len(np.unique(faces, axis=0)) == 300
        np.testing.assert_almost_equal(verts[0], np.array([1, 2, 3]))
        last_vertex = np.array([1, 2, 3]) + 5 * np.array([1, 2, 1]) / np.sqrt(6)
        np.testing.assert_almost_equal(verts[-1], last_vertex.astype(np.float32))

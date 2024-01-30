from unittest.mock import patch

import numpy as np
import pytest

try:
    from pyqtgraph.opengl import GLLinePlotItem, GLMeshItem, MeshData
    from symmeplot.pyqtgraph.artists import (
        Line3D,
        Point3D,
        Vector3D,
        create_tube_mesh_data,
    )
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


class TestPoint3D:
    def test_init(self):
        point = Point3D(0, 1, 2, color=(1, 0, 0, 1))
        assert point.gl_items[0].color == (1, 0, 0, 1)
        np.testing.assert_almost_equal(point.gl_items[0].pos, np.array([[0, 1, 2]]))

    def test_update_data(self):
        point = Point3D(0, 1, 2)
        point.update_data(5, 2, 9)
        np.testing.assert_almost_equal(point.gl_items[0].pos, np.array([[5, 2, 9]]))


class TestLine3D:
    def test_init(self):
        line = Line3D([0, 1, 2], [0, 0, 0], [1, 1, 1], color=(1, 0, 0, 1))
        assert line.gl_items[0].color == (1, 0, 0, 1)
        np.testing.assert_almost_equal(line.gl_items[0].pos, np.array([
            [0, 0, 1], [1, 0, 1], [2, 0, 1]]))

    def test_update_data(self):
        line = Line3D([0, 1, 2], [0, 0, 0], [1, 1, 1])
        line.update_data([1, 2, 3], [1, 1, 1], [1, 2, 3])
        np.testing.assert_almost_equal(line.gl_items[0].pos, np.array([
            [1, 1, 1], [2, 1, 2], [3, 1, 3]]))


class TestVector3D:
    def test_init(self):
        vector = Vector3D([.4, .2, .1], [.5, .6, .7], color=(1, 0, 0, 1))
        assert isinstance(vector.gl_items[0], GLLinePlotItem)  # Default
        assert vector.gl_items[0].color == (1, 0, 0, 1)
        np.testing.assert_almost_equal(vector.gl_items[0].pos, np.array([
            [.4, .2, .1], [.9, .8, .8]]))

    def test_update_data(self):
        vector = Vector3D([.4, .2, .1], [.5, .6, .7])
        vector.update_data([.1, .2, .3], [.4, .5, .6])
        np.testing.assert_almost_equal(vector.gl_items[0].pos, np.array([
            [.1, .2, .3], [.5, .7, .9]]))

    @patch("symmeplot.pyqtgraph.artists.create_tube_mesh_data", return_value=MeshData())
    def test_as_mesh(self, mock):
        vector = Vector3D([.4, .2, .1], [.5, .6, .7], as_mesh=True)
        assert isinstance(vector.gl_items[0], GLMeshItem)
        mock.assert_called_once()

    def test_auto_reuse_vector_props(self):
        v1 = Vector3D([.4, .2, .1], [.5, .6, .7], as_mesh=True)
        vector_radius = v1.vector_radius
        head_length = v1.head_length
        head_width = v1.head_width
        v2 = Vector3D([6, 2, 9], [9, 2, 0], as_mesh=True)
        assert v2.vector_radius == vector_radius
        assert v2.head_length == head_length
        assert v2.head_width == head_width
        # Change the default for all vectors.
        assert vector_radius != 0.1  # Test value should be different.
        Vector3D.vector_radius = 0.1
        assert v1.vector_radius == 0.1
        assert v2.vector_radius == 0.1
        # Change the head_width of a single vector.
        assert v1.head_width != 0.2  # Test value should be different.
        v1.head_width = 0.2
        assert v1.head_width == 0.2
        assert v2.head_width != 0.2

    @pytest.mark.parametrize("name, value", [
        ("vector_radius", 0.1324),
        ("head_length", 0.21345),
        ("head_width", 0.314365),
        ("mesh_resolution", 10)]
    )
    def test_set_mesh_prop(self, name, value):
        v_ref = Vector3D([.4, .2, .1], [.5, .6, .7], as_mesh=True)
        assert getattr(v_ref, name) != value  # Test value should be different.
        v = Vector3D([.4, .2, .1], [.5, .6, .7], as_mesh=True, **{name: value})
        assert getattr(v, name) == value

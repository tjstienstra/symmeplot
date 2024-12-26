from __future__ import annotations

import numpy as np
import pytest

from symmeplot.utilities.testing import ON_CI

try:
    from symmeplot.matplotlib.artists import Circle3D, Line3D, Vector3D
except ImportError:
    if ON_CI:
        raise
    pytest.skip("Matplotlib not installed.", allow_module_level=True)


class TestLine3D:
    def test_init(self):
        line = Line3D([0, 1, 2], [0, 0, 0], [1, 1, 1], color="g")
        assert line.get_color() == "g"
        np.testing.assert_almost_equal(line.get_data_3d()[0], [0, 1, 2])
        np.testing.assert_almost_equal(line.get_data_3d()[1], [0, 0, 0])
        np.testing.assert_almost_equal(line.get_data_3d()[2], [1, 1, 1])

    def test_update_data(self):
        line = Line3D([0, 1, 2], [0, 0, 0], [1, 1, 1])
        line.update_data([1, 2, 3], [1, 1, 1], [1, 2, 3])
        np.testing.assert_almost_equal(line.get_data_3d()[0], [1, 2, 3])
        np.testing.assert_almost_equal(line.get_data_3d()[1], [1, 1, 1])
        np.testing.assert_almost_equal(line.get_data_3d()[2], [1, 2, 3])

    def test_min(self):
        line = Line3D([3, 0, 4], [-1, 6, 0], [1, 1, -4])
        np.testing.assert_almost_equal(line.min(), [0, -1, -4])

    def test_max(self):
        line = Line3D([0, 1, 4], [6, -1, 0], [1, 1, -4])
        np.testing.assert_almost_equal(line.max(), [4, 6, 1])


class TestVector3D:
    def test_init(self):
        vector = Vector3D([0.4, 0.2, 0.1], [0.5, 0.6, 0.7], color=(1, 0, 0, 1))
        assert vector.get_edgecolor() == (1, 0, 0, 1)
        assert vector.get_facecolor() == (1, 0, 0, 1)
        np.testing.assert_almost_equal(vector._origin, [0.4, 0.2, 0.1])  # noqa: SLF001
        np.testing.assert_almost_equal(vector._vector, [0.5, 0.6, 0.7])  # noqa: SLF001

    def test_update_data(self):
        vector = Vector3D([0.4, 0.2, 0.1], [0.5, 0.6, 0.7])
        vector.update_data([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        np.testing.assert_almost_equal(vector._origin, [0.1, 0.2, 0.3])  # noqa: SLF001
        np.testing.assert_almost_equal(vector._vector, [0.4, 0.5, 0.6])  # noqa: SLF001

    def test_min(self):
        vector = Vector3D([0.4, 0.2, 0.1], [0.5, -0.6, 0.7])
        np.testing.assert_almost_equal(vector.min(), [0.4, -0.4, 0.1])

    def test_max(self):
        vector = Vector3D([0.4, 0.2, 0.1], [0.5, -0.6, 0.7])
        np.testing.assert_almost_equal(vector.max(), [0.9, 0.2, 0.8])


class TestCircle3D:
    def test_init(self):
        circle = Circle3D([0.4, 0.2, 0.1], 0.5, [1, 0, 0], color=(1, 0, 0, 1))
        assert circle.get_edgecolor() == (1, 0, 0, 1)
        assert circle.get_facecolor() == (1, 0, 0, 1)
        np.testing.assert_almost_equal(circle.min(), [0.4, -0.3, -0.4])
        np.testing.assert_almost_equal(circle.max(), [0.4, 0.7, 0.6])

    def test_update_data(self):
        circle = Circle3D([0.4, 0.2, 0.1], 0.5, [1, 0, 0])
        circle.update_data([0.1, 0.2, 0.3], 0.3, [0, 1, 0])
        np.testing.assert_almost_equal(circle.min(), [-0.2, 0.2, 0])
        np.testing.assert_almost_equal(circle.max(), [0.4, 0.2, 0.6])

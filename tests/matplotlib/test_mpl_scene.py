from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sympy.physics.mechanics as me

from symmeplot.utilities.testing import ON_CI

try:
    import matplotlib.pyplot as plt

    from symmeplot.matplotlib import PlotFrame, Scene3D
except ImportError:
    if ON_CI:
        raise
    pytest.skip("Matplotlib not installed.", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def mock_visualization():
    with patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())):
        yield


class TestScene3D:
    @pytest.fixture(autouse=True)
    def _define_system(self):
        self.q = me.dynamicsymbols("q:3")
        self.rf = me.ReferenceFrame("rf")
        self.zp = me.Point("zp")
        self.f = me.ReferenceFrame("frame")
        self.f.orient_axis(self.rf, self.rf.z, self.q[0])
        self.p1 = self.zp.locatenew("p1", self.q[1] * self.rf.x + 0.5 * self.rf.z)
        self.p2 = self.p1.locatenew("p2", self.q[2] * self.f.y)
        self.p3 = self.p2.locatenew("p3", 0.5 * self.f.z)
        self.line = (self.p1, self.p2, self.p3)
        self.rb = me.RigidBody(
            "rb", self.p2, self.f, 1, (self.f.x.outer(self.f.x), self.p2)
        )
        self.pt = me.Particle("pt", self.p3, 1)

    def test_scene_init(self):
        scene = Scene3D(self.rf, self.zp)
        assert scene.inertial_frame == self.rf
        assert scene.zero_point == self.zp
        assert len(scene.plot_objects) == 1
        assert isinstance(scene.plot_objects[0], PlotFrame)
        assert hasattr(scene.axes, "get_zlim")

    def test_scene_init_with_ax(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        scene = Scene3D(self.rf, self.zp, ax=ax)
        assert scene.axes == ax

    def test_scene_in_orthogonal_proj_no_arguments_view_of_zy_plane_of_rf(self):
        axes_mock = MagicMock()
        scene = Scene3D(self.rf, self.zp, ax=axes_mock)

        scene.as_orthogonal_projection_plot()

        kwargs = {
            "elev": 0.0,
            "azim": 0.0,
            "roll": 0.0,
        }
        axes_mock.view_init.assert_called_once_with(**kwargs)

    def test_scene_in_orthogonal_proj_rf_as_arguments_view_of_zy_plane_of_rf(self):
        axes_mock = MagicMock()
        scene = Scene3D(self.rf, self.zp, ax=axes_mock)

        scene.as_orthogonal_projection_plot(self.rf)

        kwargs = {
            "elev": 0.0,
            "azim": 0.0,
            "roll": 0.0,
        }
        axes_mock.view_init.assert_called_once_with(**kwargs)

    def test_scene_in_orthogonal_projection_view_of_xy_plane_of_rf(self):
        axes_mock = MagicMock()
        scene = Scene3D(self.rf, self.zp, ax=axes_mock)

        f1 = me.ReferenceFrame("A")
        f1.orient_axis(self.rf, self.rf.z, np.pi / 2)
        f2 = me.ReferenceFrame("B")
        f2.orient_axis(f1, self.rf.y, -np.pi / 2)

        scene.as_orthogonal_projection_plot(f2)

        kwargs = {
            "elev": 90.0,
            "azim": -90.0,
            "roll": 0.0,
        }
        axes_mock.view_init.assert_called_once_with(**kwargs)

    def test_scene_in_orthogonal_projection_view_of_xz_plane_of_rf(self):
        axes_mock = MagicMock()
        scene = Scene3D(self.rf, self.zp, ax=axes_mock)

        frame = me.ReferenceFrame("A")
        frame.orient_axis(self.rf, self.rf.z, np.pi / 2)

        scene.as_orthogonal_projection_plot(frame)

        kwargs = {
            "elev": 0.0,
            "azim": -90.0,
            "roll": 0.0,
        }
        axes_mock.view_init.assert_called_once_with(**kwargs)

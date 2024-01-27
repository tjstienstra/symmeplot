from unittest.mock import MagicMock, patch

import pytest
import sympy.physics.mechanics as me

try:
    import matplotlib.pyplot as plt
    from symmeplot.matplotlib import PlotFrame, Scene3D
except ImportError:
    pytest.skip("Matplotlib not installed.")


@pytest.fixture(scope="module", autouse=True)
def mock_visualization():
    with (patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock()))):
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
            "rb", self.p2, self.f, 1, (self.f.x.outer(self.f.x), self.p2))
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

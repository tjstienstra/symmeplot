from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import symmeplot.matplotlib as matplotlib
import symmeplot.pyqtgraph as pyqtgraph
import symmeplot.utilities.dummy_backend as dummy
import sympy.physics.mechanics as me


@pytest.fixture(scope="module", autouse=True)
def mock_visualization():
    with (patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock())),
          patch("pyqtgraph.exec"),
          patch("pyqtgraph.opengl.GLViewWidget.GLViewWidget.show")):
        yield

@pytest.mark.parametrize("backend", [dummy, matplotlib, pyqtgraph])
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
        self.p3 = self.p2.locatenew("p3", 0.1 * self.f.x + 0.5 * self.f.z)
        self.line = (self.p1, self.p2, self.p3)
        self.rb = me.RigidBody(
            "rb", self.p2, self.f, 1, (self.f.x.outer(self.f.x), self.p2))
        self.pt = me.Particle("pt", self.p3, 1)
        self.p1_coords, self.p2_coords, self.p3_coords = None, None, None

    @pytest.fixture()
    def _filled_scene(self, backend):
        self.scene = backend.Scene3D(self.rf, self.zp)
        self.scene.add_point(self.p1)
        self.scene.add_line(self.line)
        self.scene.add_vector(self.p2.pos_from(self.p1), self.p1, name="my_vector")
        self.scene.add_frame(self.f, self.p1)
        self.scene.add_body(self.rb)
        self.scene.add_body(self.pt)

    def _evaluate1(self, scene):
        scene.lambdify_system(self.q)
        scene.evaluate_system(np.pi / 2, 0.7, 0.3)
        self.p1_coords = (0.7, 0, 0.5)
        self.p2_coords = (0.4, 0, 0.5)
        self.p3_coords = (0.4, 0.1, 1)
        scene.plot()

    def test_scene_init(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        assert scene.inertial_frame == self.rf
        assert scene.zero_point == self.zp
        assert len(scene.plot_objects) == 1
        assert isinstance(scene.plot_objects[0], backend.PlotFrame)
        assert len(scene.artists) == 3
        self._evaluate1(scene)

    def test_add_point(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_point(self.p1)
        assert len(scene.plot_objects) == 2
        plot_point = scene.plot_objects[-1]
        assert isinstance(plot_point, backend.PlotPoint)
        assert plot_point.point == self.p1
        assert len(scene.artists) == 4
        self._evaluate1(scene)
        np.testing.assert_almost_equal(plot_point.point_coords, self.p1_coords)

    def test_add_line(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_line(self.line)
        assert len(scene.plot_objects) == 2
        plot_line = scene.plot_objects[-1]
        assert isinstance(plot_line, backend.PlotLine)
        assert plot_line.line == self.line
        assert len(scene.artists) == 4
        self._evaluate1(scene)
        np.testing.assert_almost_equal(
            plot_line.line_coords,
            np.array([self.p1_coords, self.p2_coords, self.p3_coords]).T)

    def test_add_vector(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_vector(self.p2.pos_from(self.p1), self.p1)
        assert len(scene.plot_objects) == 2
        plot_vector = scene.plot_objects[-1]
        assert isinstance(plot_vector, backend.PlotVector)
        assert plot_vector.vector == self.q[2] * self.f.y
        assert len(scene.artists) == 4
        self._evaluate1(scene)
        np.testing.assert_almost_equal(plot_vector.origin_coords, self.p1_coords)
        np.testing.assert_almost_equal(plot_vector.vector_values, (-0.3, 0, 0))

    def test_add_frame(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_frame(self.f, self.p2)
        assert len(scene.plot_objects) == 2
        plot_frame = scene.plot_objects[-1]
        assert isinstance(plot_frame, backend.PlotFrame)
        assert plot_frame.frame is self.f
        assert len(scene.artists) == 6
        self._evaluate1(scene)
        for v in plot_frame.vectors:
            np.testing.assert_almost_equal(v.origin_coords, self.p2_coords)
        np.testing.assert_almost_equal(plot_frame.x.vector_values, (0, 0.1, 0))
        np.testing.assert_almost_equal(plot_frame.y.vector_values, (-0.1, 0, 0))
        np.testing.assert_almost_equal(plot_frame.z.vector_values, (0, 0, 0.1))

    def test_add_rigid_body(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_body(self.rb)
        assert len(scene.plot_objects) == 2
        plot_body = scene.plot_objects[-1]
        assert isinstance(plot_body, backend.PlotBody)
        assert plot_body.body is self.rb
        assert len(scene.artists) == 7
        self._evaluate1(scene)
        assert isinstance(plot_body.plot_masscenter, backend.PlotPoint)
        assert isinstance(plot_body.plot_frame, backend.PlotFrame)
        self._evaluate1(scene)
        np.testing.assert_almost_equal(
            plot_body.plot_masscenter.point_coords, self.p2_coords)

    def test_add_particle(self, backend):
        scene = backend.Scene3D(self.rf, self.zp)
        scene.add_body(self.pt)
        assert len(scene.plot_objects) == 2
        plot_body = scene.plot_objects[-1]
        assert isinstance(plot_body, backend.PlotBody)
        assert plot_body.body is self.pt
        assert len(scene.artists) == 4
        assert isinstance(plot_body.plot_masscenter, backend.PlotPoint)
        assert plot_body.plot_frame is None
        self._evaluate1(scene)
        np.testing.assert_almost_equal(
            plot_body.plot_masscenter.point_coords, self.p3_coords)

    def test_get_plot_object(self, backend, _filled_scene):
        # Get inertial frame by sympy object
        rf_obj = self.scene.get_plot_object(self.rf)
        assert isinstance(rf_obj, backend.PlotFrame)
        # Get added point by sympy object
        p1_obj = self.scene.get_plot_object(self.p1)
        assert isinstance(p1_obj, backend.PlotPoint)
        assert p1_obj.point is self.p1
        # Get added point by name
        assert self.scene.get_plot_object("p1") is p1_obj
        # Get added vector by name
        v_obj = self.scene.get_plot_object("my_vector")
        assert isinstance(v_obj, backend.PlotVector)
        assert v_obj.vector == self.q[2] * self.f.y
        # Get rigid body by sympy object
        rb_obj = self.scene.get_plot_object(self.rb)
        assert isinstance(rb_obj, backend.PlotBody)
        assert rb_obj.body is self.rb
        # Get particle by name
        pt_obj = self.scene.get_plot_object("pt")
        assert isinstance(pt_obj, backend.PlotBody)
        assert pt_obj.body is self.pt
        # Get nested point by sympy object
        p3_obj = self.scene.get_plot_object(self.p3)
        assert isinstance(p3_obj, backend.PlotPoint)
        assert p3_obj.point is self.p3

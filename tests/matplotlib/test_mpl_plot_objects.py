import numpy as np
import pytest
import sympy as sm
import sympy.physics.mechanics as me
from symmeplot.utilities.testing import ON_CI

try:
    from symmeplot.matplotlib import (
        PlotBody,
        PlotFrame,
        PlotLine,
        PlotPoint,
        PlotVector,
    )
except ImportError as e:
    if ON_CI:
        raise e
    pytest.skip("Matplotlib not installed.", allow_module_level=True)


class TestPlotPoint:
    @pytest.fixture(autouse=True)
    def _define_point(self):
        self.s = sm.symbols("s:3")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        self.p = self.zp.locatenew(
            "point", sum(si * v for si, v in zip(self.s, self.rf)))

    @pytest.fixture()
    def _basic_plot_point(self):
        self.plot_point = PlotPoint(self.rf, self.zp, self.p)
        self.evalf = sm.lambdify(self.s, self.plot_point.get_expressions_to_evaluate())
        self.plot_point.values = self.evalf(0.2, 0.6, 0.3)
        self.plot_point.update()

    def test_artist(self, _basic_plot_point):
        assert len(self.plot_point.artists) == 1
        line = self.plot_point.artists[0]
        assert line.get_marker() == "o"
        np.testing.assert_almost_equal(line.get_data_3d(), [[0.2], [0.6], [0.3]])

    def test_update(self):
        plot_point = PlotPoint(self.rf, self.zp, self.p)
        f = sm.lambdify(self.s, plot_point.get_expressions_to_evaluate())
        plot_point.values = f(0, 0, 0)
        np.testing.assert_almost_equal(plot_point.point_coords, np.zeros(3))
        plot_point.values = f(0.2, 0.6, 0.3)
        np.testing.assert_almost_equal(
            plot_point.point_coords, np.array([0.2, 0.6, 0.3]))

    def test_annot_coords(self, _basic_plot_point):
        np.testing.assert_almost_equal(self.plot_point.annot_coords,
                                       np.array([0.2, 0.6, 0.3]))


class TestPlotLine:
    @pytest.fixture(autouse=True)
    def _define_line(self):
        self.s = sm.symbols("s:3")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        self.p1 = self.zp.locatenew(
            "p1", sum(si * v for si, v in zip(self.s, self.rf)))
        self.p2 = self.p1.locatenew("p2", 0.6 * self.rf.x + 0.3 * self.rf.z)
        self.p3 = self.p2.locatenew("p3", 0.6 * self.rf.y)
        self.line = (self.p1, self.p2, self.p3)

    @pytest.fixture()
    def _basic_plot_line(self):
        self.plot_line = PlotLine(self.rf, self.zp, self.line)
        self.evalf = sm.lambdify(self.s, self.plot_line.get_expressions_to_evaluate())
        self.plot_line.values = self.evalf(0.1, 0.6, 0.2)
        self.plot_line.update()

    def test_artist(self, _basic_plot_line):
        assert len(self.plot_line.artists) == 1
        line = self.plot_line.artists[0]
        np.testing.assert_almost_equal(
            line.get_data_3d(), [[0.1, 0.7, 0.7], [0.6, 0.6, 1.2], [0.2, 0.5, 0.5]])

    def test_update(self):
        plot_line = PlotLine(self.rf, self.zp, self.line)
        f = sm.lambdify(self.s, plot_line.get_expressions_to_evaluate())
        plot_line.values = f(0, 0, 0)
        np.testing.assert_almost_equal(
            plot_line.line_coords, [[0, 0.6, 0.6], [0, 0, 0.6], [0, 0.3, 0.3]])
        plot_line.values = f(0.1, 0.6, 0.2)
        np.testing.assert_almost_equal(
            plot_line.line_coords, [[0.1, 0.7, 0.7], [0.6, 0.6, 1.2], [0.2, 0.5, 0.5]])

    def test_annot_coords(self, _basic_plot_line):
        np.testing.assert_almost_equal(self.plot_line.annot_coords, [0.5, 0.8, 0.4])


class TestPlotVector:
    @pytest.fixture(autouse=True)
    def _define_vector(self):
        self.s = sm.symbols("s:3")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        self.o = self.zp.locatenew("origin", 0.3 * self.rf.x + 0.2 * self.rf.y)
        self.v = sum(si * v for si, v in zip(self.s, self.rf))

    @pytest.fixture()
    def _basic_plot_vector(self):
        self.plot_vector = PlotVector(self.rf, self.zp, self.v, self.o)
        self.evalf = sm.lambdify(self.s, self.plot_vector.get_expressions_to_evaluate())
        self.plot_vector.values = self.evalf(0.2, -0.6, 0.3)
        self.plot_vector.update()

    def test_artist(self, _basic_plot_vector):
        assert len(self.plot_vector.artists) == 1
        line = self.plot_vector.artists[0]
        np.testing.assert_almost_equal(line.min(), [0.3, -0.4, 0])
        np.testing.assert_almost_equal(line.max(), [0.5, 0.2, 0.3])

    def test_update(self):
        plot_vector = PlotVector(self.rf, self.zp, self.v, self.o)
        f = sm.lambdify(self.s, plot_vector.get_expressions_to_evaluate())
        plot_vector.values = f(0, 0, 0)
        np.testing.assert_almost_equal(plot_vector.origin_coords, [0.3, 0.2, 0])
        np.testing.assert_almost_equal(plot_vector.vector_values, [0, 0, 0])
        plot_vector.values = f(0.2, -0.6, 0.3)
        np.testing.assert_almost_equal(plot_vector.origin_coords, [0.3, 0.2, 0])
        np.testing.assert_almost_equal(plot_vector.vector_values, [0.2, -0.6, 0.3])

    def test_annot_coords(self, _basic_plot_vector):
        np.testing.assert_almost_equal(self.plot_vector.annot_coords, [0.4, -0.1, 0.15])


class TestPlotFrame:
    @pytest.fixture(autouse=True)
    def _define_frame(self):
        self.q = sm.symbols("q")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        self.f = me.ReferenceFrame("frame")
        self.f.orient_axis(self.rf, self.q, self.rf.z)
        self.o = self.zp.locatenew("origin", 0.3 * self.rf.x + 0.2 * self.rf.y)

    @pytest.fixture()
    def _basic_plot_frame(self):
        self.plot_frame = PlotFrame(self.rf, self.zp, self.f, self.o, scale=1)
        self.evalf = sm.lambdify(self.q, self.plot_frame.get_expressions_to_evaluate())
        self.plot_frame.values = self.evalf(np.pi / 2)
        self.plot_frame.update()

    def test_children(self, _basic_plot_frame):
        assert len(self.plot_frame.children) == 3
        assert all(isinstance(child, PlotVector) for child in self.plot_frame.children)
        assert len(self.plot_frame.artists) == 3

    def test_update(self):
        plot_frame = PlotFrame(self.rf, self.zp, self.f, self.o, scale=1)
        f = sm.lambdify(self.q, plot_frame.get_expressions_to_evaluate())
        plot_frame.values = f(0)
        np.testing.assert_almost_equal(plot_frame.x.vector_values, [1, 0, 0])
        np.testing.assert_almost_equal(plot_frame.y.vector_values, [0, 1, 0])
        np.testing.assert_almost_equal(plot_frame.z.vector_values, [0, 0, 1])
        plot_frame.values = f(np.pi / 2)
        np.testing.assert_almost_equal(plot_frame.x.vector_values, [0, 1, 0])
        np.testing.assert_almost_equal(plot_frame.y.vector_values, [-1, 0, 0])
        np.testing.assert_almost_equal(plot_frame.z.vector_values, [0, 0, 1])

    def test_annot_coords(self, _basic_plot_frame):
        np.testing.assert_almost_equal(self.plot_frame.annot_coords, [0, 0.5, 0.3])


class TestPlotBody:
    @pytest.fixture(autouse=True)
    def _define_body(self):
        self.q, self.s = sm.symbols("q s")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        f, mc = me.ReferenceFrame("frame"), me.Point("center_of_mass")
        f.orient_axis(self.rf, self.q, self.rf.z)
        mc.set_pos(self.zp, 0.3 * self.rf.x + 0.2 * self.rf.y + self.s * self.rf.z)
        self.rb = me.RigidBody("body", mc, f, 1, (f.x.outer(f.x), mc))
        self.p = me.Particle("particle", mc, 1)

    @pytest.fixture()
    def _basic_plot_body(self):
        self.plot_body = PlotBody(self.rf, self.zp, self.rb)
        self.evalf = sm.lambdify(
            (self.q, self.s), self.plot_body.get_expressions_to_evaluate())
        self.plot_body.values = self.evalf(np.pi / 2, 0.5)
        self.plot_body.update()

    def test_children(self):
        plot_rb = PlotBody(self.rf, self.zp, self.rb)
        assert len(plot_rb.children) == 2
        assert isinstance(plot_rb.children[0], PlotPoint)
        assert isinstance(plot_rb.children[1], PlotFrame)
        assert len(plot_rb.artists) == 4
        plot_p = PlotBody(self.rf, self.zp, self.p)
        assert len(plot_p.children) == 1
        assert isinstance(plot_p.children[0], PlotPoint)
        assert len(plot_p.artists) == 1

    def test_update(self):
        plot_rb = PlotBody(self.rf, self.zp, self.rb)
        mc, frame = plot_rb.plot_masscenter, plot_rb.plot_frame
        f = sm.lambdify((self.q, self.s), plot_rb.get_expressions_to_evaluate())
        plot_rb.values = f(0, 0)
        np.testing.assert_almost_equal(mc.point_coords, [0.3, 0.2, 0])
        np.testing.assert_almost_equal(frame.x.vector_values, [.1, 0, 0])
        np.testing.assert_almost_equal(frame.y.vector_values, [0, .1, 0])
        np.testing.assert_almost_equal(frame.z.vector_values, [0, 0, .1])
        plot_rb.values = f(np.pi / 2, 0.5)
        np.testing.assert_almost_equal(mc.point_coords, [0.3, 0.2, 0.5])
        np.testing.assert_almost_equal(frame.x.vector_values, [0, .1, 0])
        np.testing.assert_almost_equal(frame.y.vector_values, [-.1, 0, 0])
        np.testing.assert_almost_equal(frame.z.vector_values, [0, 0, .1])

    def test_annot_coords(self, _basic_plot_body):
        np.testing.assert_almost_equal(self.plot_body.annot_coords, [0.3, 0.2, 0.5])


import numpy as np
import pytest
import symmeplot.utilities.dummy_backend as dummy
import sympy as sm
import sympy.physics.mechanics as me

try:
    import symmeplot.matplotlib as matplotlib
except ImportError:
    matplotlib = None
try:
    import symmeplot.pyqtgraph as pyqtgraph
except ImportError:
    pyqtgraph = None

backends = [dummy, matplotlib, pyqtgraph]


@pytest.mark.parametrize("backend", backends)
class TestPlotPointMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        if backend is None:
            pytest.skip("Backend not installed.")
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.P1 = self.O.locatenew(
            "P1", sum(li * self.Ni for li, self.Ni in zip(self.l, self.N)))

        self.plot_object = backend.PlotPoint(self.N, self.O, self.P1)
        self.eval = sm.lambdify(self.l, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.sympy_object == self.P1
        assert self.plot_object.name == "P1"
        assert self.plot_object.visible is True
        assert self.plot_object.point == self.P1

    def test_expressions(self):
        assert self.plot_object.get_sympy_object_exprs() == self.l
        self.plot_object.values = self.eval(4, 5, 6)
        np.testing.assert_equal(self.plot_object.point_coords, (4, 5, 6))


@pytest.mark.parametrize("backend", backends)
class TestPlotLineMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        if backend is None:
            pytest.skip("Backend not installed.")
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.line = (self.O.locatenew("P1", self.l[0] * self.N.x),
                       self.O.locatenew("P2", self.l[1] * self.N.y),
                       self.O.locatenew("P3", self.l[2] * self.N.z))

        self.plot_object = backend.PlotLine(self.N, self.O, self.line, name="line")
        self.eval = sm.lambdify(self.l, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.sympy_object == self.line
        assert self.plot_object.name == "line"
        assert self.plot_object.visible is True
        assert self.plot_object.line == self.line

    def test_expressions(self):
        assert self.plot_object.get_sympy_object_exprs() == (
            (self.l[0], 0, 0),
            (0, self.l[1], 0),
            (0, 0, self.l[2])
        )
        self.plot_object.values = self.eval(4, 5, 6)
        np.testing.assert_equal(self.plot_object.line_coords, np.diag((4, 5, 6)))


@pytest.mark.parametrize("backend", backends)
class TestPlotVectorMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        if backend is None:
            pytest.skip("Backend not installed.")
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.P = self.O.locatenew("P", self.N.x)
        self.v = sum(li * self.Ni for li, self.Ni in zip(self.l, self.N))

        self.plot_object = backend.PlotVector(
            self.N, self.O, self.v, self.P, name="vector")
        self.eval = sm.lambdify(self.l, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.origin == self.P
        assert self.plot_object.sympy_object == self.v
        assert self.plot_object.name == "vector"
        assert self.plot_object.visible is True
        assert self.plot_object.vector == self.v

    def test_expressions(self):
        assert self.plot_object.get_sympy_object_exprs() == ((1, 0, 0), tuple(self.l))
        self.plot_object.values = self.eval(4, 5, 6)
        np.testing.assert_equal(self.plot_object.origin_coords, (1, 0, 0))
        np.testing.assert_equal(self.plot_object.vector_values, (4, 5, 6))


@pytest.mark.parametrize("backend", backends)
class TestPlotFrameMixin:
    @pytest.fixture(autouse=True)
    def _define_system(self, backend):
        if backend is None:
            pytest.skip("Backend not installed.")
        self.q = sm.symbols("q")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.A = me.ReferenceFrame("A")
        self.A.orient_axis(self.N, self.q, self.N.z)
        self.P = self.O.locatenew("P", self.N.x)

    @pytest.fixture()
    def _instantiate_plot_object(self, backend):
        self.plot_object = backend.PlotFrame(self.N, self.O, self.A, origin=self.P,
                                             name="frame", scale=2.0)
        self.eval = sm.lambdify(self.q, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self, _instantiate_plot_object):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.frame == self.A
        assert self.plot_object.origin == self.P
        assert self.plot_object.sympy_object == self.A
        assert self.plot_object.name == "frame"
        assert self.plot_object.visible is True
        assert self.plot_object.vectors == (
            self.plot_object.x, self.plot_object.y, self.plot_object.z)

    def test_expressions(self, _instantiate_plot_object):
        self.plot_object.values = self.eval(np.pi / 2)
        np.testing.assert_almost_equal(self.plot_object.x.origin_coords, (1, 0, 0))
        np.testing.assert_almost_equal(self.plot_object.x.vector_values, (0, 2, 0))
        np.testing.assert_almost_equal(self.plot_object.y.origin_coords, (1, 0, 0))
        np.testing.assert_almost_equal(self.plot_object.y.vector_values, (-2, 0, 0))
        np.testing.assert_almost_equal(self.plot_object.z.origin_coords, (1, 0, 0))
        np.testing.assert_almost_equal(self.plot_object.z.vector_values, (0, 0, 2))

    def test_name_init(self, backend):
        assert backend.PlotFrame(self.N, self.O, self.A).name == "A"
        assert backend.PlotFrame(self.N, self.O, self.A, name="B").name == "B"

    def test_scale(self, backend):
        assert backend.PlotFrame(self.N, self.O, self.N).x.get_sympy_object_exprs(
            ) == ((0, 0, 0), (0.1, 0, 0))
        assert backend.PlotFrame(self.N, self.O, self.N, scale=2
                                 ).x.get_sympy_object_exprs() == ((0, 0, 0), (2, 0, 0))


@pytest.mark.parametrize("backend", backends)
class TestPlotBodyMixin:
    @pytest.fixture(autouse=True)
    def _define_system(self, backend):
        if backend is None:
            pytest.skip("Backend not installed.")
        self.q = sm.symbols("q")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.A = me.ReferenceFrame("A")
        self.A.orient_axis(self.N, self.q, self.N.z)
        self.mc = self.O.locatenew("mc", self.N.x)
        self.rb = me.RigidBody(
            "rb", self.mc, self.A, 1.0, (self.A.x.outer(self.A.x), self.mc))
        self.pt = me.Particle("pt", self.mc, 1.0)

    @pytest.fixture()
    def _instantiate_plot_object(self, backend):
        self.plot_object = backend.PlotBody(self.N, self.O, self.rb, name="body")
        self.eval = sm.lambdify(self.q, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self, _instantiate_plot_object):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.body == self.rb
        assert self.plot_object.sympy_object == self.rb
        assert self.plot_object.name == "body"
        assert self.plot_object.visible is True
        assert self.plot_object.children == (
            self.plot_object.plot_masscenter, self.plot_object.plot_frame)

    def test_expressions(self, _instantiate_plot_object):
        self.plot_object.values = self.eval(np.pi / 2)
        np.testing.assert_almost_equal(
            self.plot_object.plot_masscenter.point_coords, (1, 0, 0))
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.x.vector_values, (0, .1, 0))
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.y.vector_values, (-.1, 0, 0))
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.z.vector_values, (0, 0, .1))

    def test_particle(self, backend):
        plot_object = backend.PlotBody(self.N, self.O, self.pt)
        plot_object.values = sm.lambdify((), plot_object.get_expressions_to_evaluate()
                                         )()
        np.testing.assert_equal(plot_object.plot_masscenter.point_coords, (1, 0, 0))
        assert len(plot_object.children) == 1

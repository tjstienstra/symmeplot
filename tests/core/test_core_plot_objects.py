from __future__ import annotations

import numpy as np
import pytest
import sympy as sm
import sympy.physics.mechanics as me

import symmeplot.utilities.dummy_backend as dummy
from symmeplot.utilities.testing import ON_CI

try:
    from symmeplot import matplotlib
except ImportError:
    matplotlib = None
try:
    from symmeplot import pyqtgraph
except ImportError:
    pyqtgraph = None


def parametrize_backends(*, exclude_backends=None):
    if exclude_backends is None:
        exclude_backends = set()
    else:
        exclude_backends = set(exclude_backends)
        exclude_backends.discard(None)
    return pytest.mark.parametrize(
        "backend",
        [
            pytest.param(
                backend,
                marks=pytest.mark.skipif(
                    backend is None and not ON_CI, reason="Backend not installed."
                ),
            )
            for backend in (dummy, matplotlib, pyqtgraph)
            if backend not in (exclude_backends)
        ],
    )


@parametrize_backends()
class TestPlotPointMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.P1 = self.O.locatenew(
            "P1", sum(li * self.Ni for li, self.Ni in zip(self.l, self.N, strict=True))
        )

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


@parametrize_backends(exclude_backends={pyqtgraph})
class TestPlotTracedPointMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        if not hasattr(backend, "PlotTracedPoint"):
            pytest.skip("Backend does not have PlotTracedPoint.")
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.P1 = self.O.locatenew(
            "P1", sum(li * self.Ni for li, self.Ni in zip(self.l, self.N, strict=True))
        )

        self.plot_object = backend.PlotTracedPoint(self.N, self.O, self.P1, frequency=2)
        self.eval = sm.lambdify(self.l, self.plot_object.get_expressions_to_evaluate())

    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.sympy_object == self.P1
        assert self.plot_object.name == "P1"
        assert self.plot_object.visible is True
        assert self.plot_object.point == self.P1
        assert self.plot_object.frequency == 2

    def test_expressions(self):
        assert self.plot_object.get_sympy_object_exprs() == self.l
        self.plot_object.values = self.eval(4, 5, 6)
        np.testing.assert_equal(self.plot_object.point_coords, (4, 5, 6))

    def test_trace_history(self):
        # First update - evaluation_count becomes 1, not logged (1 % 2 != 0)
        self.plot_object.values = self.eval(1, 2, 3)
        self.plot_object.update()
        assert len(self.plot_object.trace_history) == 0

        # Second update - evaluation_count becomes 2, logged (2 % 2 == 0)
        self.plot_object.values = self.eval(4, 5, 6)
        self.plot_object.update()
        assert len(self.plot_object.trace_history) == 1
        np.testing.assert_equal(self.plot_object.trace_history[0], (4, 5, 6))

        # Third update - evaluation_count becomes 3, not logged (3 % 2 != 0)
        self.plot_object.values = self.eval(7, 8, 9)
        self.plot_object.update()
        assert len(self.plot_object.trace_history) == 1

        # Fourth update - evaluation_count becomes 4, logged (4 % 2 == 0)
        self.plot_object.values = self.eval(10, 11, 12)
        self.plot_object.update()
        assert len(self.plot_object.trace_history) == 2
        np.testing.assert_equal(self.plot_object.trace_history[1], (10, 11, 12))

    def test_alpha_decays_default(self):
        # Default alpha_decays should return all ones
        ages = np.array([0, 2, 4, 6])
        alphas = self.plot_object.alpha_decays(ages)
        np.testing.assert_equal(alphas, np.ones(4))

    def test_alpha_decays_custom(self, backend):
        custom_decay = lambda ages: np.maximum(0.1, 1.0 - ages / 10.0)  # noqa: E731
        plot_object = backend.PlotTracedPoint(
            self.N, self.O, self.P1, alpha_decays=custom_decay
        )
        ages = np.array([0, 5, 10, 15])
        alphas = plot_object.alpha_decays(ages)
        np.testing.assert_almost_equal(alphas, [1.0, 0.5, 0.1, 0.1])

    def test_current_alpha_values(self, backend):
        custom_decay = lambda ages: np.maximum(0.1, 1.0 - ages / 10.0)  # noqa: E731
        plot_object = backend.PlotTracedPoint(
            self.N, self.O, self.P1, alpha_decays=custom_decay, frequency=2
        )
        # Perform several updates to build up history
        for i in range(6):
            plot_object.values = self.eval(i, i + 1, i + 2)
            plot_object.update()
        np.testing.assert_equal(plot_object._current_alpha_values, [0.6, 0.8, 1.0])  # noqa: SLF001


@parametrize_backends()
class TestPlotLineMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.line = (
            self.O.locatenew("P1", self.l[0] * self.N.x),
            self.O.locatenew("P2", self.l[1] * self.N.y),
            self.O.locatenew("P3", self.l[2] * self.N.z),
        )

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
            (0, 0, self.l[2]),
        )
        self.plot_object.values = self.eval(4, 5, 6)
        np.testing.assert_equal(self.plot_object.line_coords, np.diag((4, 5, 6)))


@parametrize_backends()
class TestPlotVectorMixin:
    @pytest.fixture(autouse=True)
    def _instantiate_plot_object(self, backend):
        self.l = sm.symbols("l:3")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.P = self.O.locatenew("P", self.N.x)
        self.v = sum(li * self.Ni for li, self.Ni in zip(self.l, self.N, strict=True))

        self.plot_object = backend.PlotVector(
            self.N, self.O, self.v, self.P, name="vector"
        )
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


@parametrize_backends()
class TestPlotFrameMixin:
    @pytest.fixture(autouse=True)
    def _define_system(self):
        self.q = sm.symbols("q")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.A = me.ReferenceFrame("A")
        self.A.orient_axis(self.N, self.q, self.N.z)
        self.P = self.O.locatenew("P", self.N.x)

    @pytest.fixture
    def _instantiate_plot_object(self, backend):
        self.plot_object = backend.PlotFrame(
            self.N, self.O, self.A, origin=self.P, name="frame", scale=2.0
        )
        self.eval = sm.lambdify(self.q, self.plot_object.get_expressions_to_evaluate())

    @pytest.mark.usefixtures("_instantiate_plot_object")
    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.frame == self.A
        assert self.plot_object.origin == self.P
        assert self.plot_object.sympy_object == self.A
        assert self.plot_object.name == "frame"
        assert self.plot_object.visible is True
        assert self.plot_object.vectors == (
            self.plot_object.x,
            self.plot_object.y,
            self.plot_object.z,
        )

    @pytest.mark.usefixtures("_instantiate_plot_object")
    def test_expressions(self):
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
        assert backend.PlotFrame(self.N, self.O, self.N).x.get_sympy_object_exprs() == (
            (0, 0, 0),
            (0.1, 0, 0),
        )
        assert backend.PlotFrame(
            self.N, self.O, self.N, scale=2
        ).x.get_sympy_object_exprs() == ((0, 0, 0), (2, 0, 0))


@parametrize_backends()
class TestPlotBodyMixin:
    @pytest.fixture(autouse=True)
    def _define_system(self, backend):
        if backend is None and not ON_CI:
            pytest.skip("Backend not installed.")
        self.q = sm.symbols("q")
        self.N, self.O = me.ReferenceFrame("N"), me.Point("O")
        self.A = me.ReferenceFrame("A")
        self.A.orient_axis(self.N, self.q, self.N.z)
        self.mc = self.O.locatenew("mc", self.N.x)
        self.rb = me.RigidBody(
            "rb", self.mc, self.A, 1.0, (self.A.x.outer(self.A.x), self.mc)
        )
        self.pt = me.Particle("pt", self.mc, 1.0)

    @pytest.fixture
    def _instantiate_plot_object(self, backend):
        self.plot_object = backend.PlotBody(self.N, self.O, self.rb, name="body")
        self.eval = sm.lambdify(self.q, self.plot_object.get_expressions_to_evaluate())

    @pytest.mark.usefixtures("_instantiate_plot_object")
    def test_basic_properties(self):
        assert self.plot_object.inertial_frame == self.N
        assert self.plot_object.zero_point == self.O
        assert self.plot_object.body == self.rb
        assert self.plot_object.sympy_object == self.rb
        assert self.plot_object.name == "body"
        assert self.plot_object.visible is True
        assert self.plot_object.children == (
            self.plot_object.plot_masscenter,
            self.plot_object.plot_frame,
        )

    @pytest.mark.usefixtures("_instantiate_plot_object")
    def test_expressions(self):
        self.plot_object.values = self.eval(np.pi / 2)
        np.testing.assert_almost_equal(
            self.plot_object.plot_masscenter.point_coords, (1, 0, 0)
        )
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.x.vector_values, (0, 0.1, 0)
        )
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.y.vector_values, (-0.1, 0, 0)
        )
        np.testing.assert_almost_equal(
            self.plot_object.plot_frame.z.vector_values, (0, 0, 0.1)
        )

    def test_particle(self, backend):
        plot_object = backend.PlotBody(self.N, self.O, self.pt)
        plot_object.values = sm.lambdify(
            (), plot_object.get_expressions_to_evaluate()
        )()
        np.testing.assert_equal(plot_object.plot_masscenter.point_coords, (1, 0, 0))
        assert len(plot_object.children) == 1

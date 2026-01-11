from __future__ import annotations

import numpy as np
import pytest
import sympy as sm
import sympy.physics.mechanics as me

from symmeplot.utilities.testing import ON_CI

try:
    from symmeplot.matplotlib import PlotTracedPoint, Scene3D
except ImportError:
    if ON_CI:
        raise
    pytest.skip("Matplotlib not installed.", allow_module_level=True)


class TestPlotTracedPoint:
    @pytest.fixture(autouse=True)
    def _define_point(self):
        self.t = sm.symbols("t")
        self.rf, self.zp = me.ReferenceFrame("inertial_frame"), me.Point("zero_point")
        self.p = self.zp.locatenew(
            "point", sm.cos(self.t) * self.rf.x + sm.sin(self.t) * self.rf.y
        )

    @pytest.fixture
    def _basic_plot_traced_point(self):
        self.plot_traced = PlotTracedPoint(self.rf, self.zp, self.p, frequency=1)
        self.evalf = sm.lambdify(
            self.t, self.plot_traced.get_expressions_to_evaluate()
        )
        # Evaluate at multiple time steps
        for t_val in np.linspace(0, np.pi, 10):
            self.plot_traced.values = self.evalf(t_val)
            self.plot_traced.update()

    @pytest.mark.usefixtures("_basic_plot_traced_point")
    def test_trace_history(self):
        # Should have 10 points in history
        assert len(self.plot_traced.trace_history) == 10
        # First point should be at (1, 0, 0)
        np.testing.assert_almost_equal(
            self.plot_traced.trace_history[0], np.array([1.0, 0.0, 0.0])
        )
        # Last point should be at (-1, 0, 0)
        np.testing.assert_almost_equal(
            self.plot_traced.trace_history[-1], np.array([-1.0, 0.0, 0.0]), decimal=5
        )

    @pytest.mark.usefixtures("_basic_plot_traced_point")
    def test_artist(self):
        assert len(self.plot_traced.artists) == 1
        line_collection = self.plot_traced.artists[0]
        # Should have 9 segments (10 points = 9 segments)
        assert len(line_collection._segments_3d) == 9

    @pytest.mark.usefixtures("_basic_plot_traced_point")
    def test_annot_coords(self):
        # Annotation should be at the last traced point
        np.testing.assert_almost_equal(
            self.plot_traced.annot_coords, self.plot_traced.trace_history[-1]
        )

    def test_frequency(self):
        plot_traced = PlotTracedPoint(self.rf, self.zp, self.p, frequency=3)
        f = sm.lambdify(self.t, plot_traced.get_expressions_to_evaluate())
        # Evaluate 12 times, but only every 3rd should be stored
        for t_val in np.linspace(0, 2 * np.pi, 12):
            plot_traced.values = f(t_val)
            plot_traced.update()
        # Should have 4 points (0, 3, 6, 9 evaluations)
        assert len(plot_traced.trace_history) == 4

    def test_alpha_decay_default(self):
        # Default alpha decay should be lambda _: 1.0
        plot_traced = PlotTracedPoint(self.rf, self.zp, self.p)
        assert plot_traced.alpha_decay(0) == 1.0
        assert plot_traced.alpha_decay(10) == 1.0

    def test_alpha_decay_custom(self):
        # Custom alpha decay
        alpha_func = lambda i: max(0.1, 1.0 - i / 10)
        plot_traced = PlotTracedPoint(self.rf, self.zp, self.p, alpha_decay=alpha_func)
        assert plot_traced.alpha_decay(0) == 1.0
        assert plot_traced.alpha_decay(5) == 0.5
        assert plot_traced.alpha_decay(10) == 0.1
        assert plot_traced.alpha_decay(20) == 0.1  # Should be clamped to 0.1


class TestScene3DAddPointTrace:
    @pytest.fixture(autouse=True)
    def _setup_scene(self):
        self.t = sm.symbols("t")
        self.N = me.ReferenceFrame("N")
        self.O = me.Point("O")
        self.P = self.O.locatenew("P", sm.cos(self.t) * self.N.x + sm.sin(self.t) * self.N.y)
        self.scene = Scene3D(self.N, self.O)

    def test_add_point_trace(self):
        traced = self.scene.add_point_trace(self.P, frequency=1, color="red")
        assert traced in self.scene.plot_objects
        assert isinstance(traced, PlotTracedPoint)

    def test_add_point_trace_with_params(self):
        traced = self.scene.add_point_trace(
            self.P,
            frequency=2,
            alpha_decay=lambda i: 1.0 - i / 10,
            color="blue",
            name="TestTrace",
        )
        assert traced.frequency == 2
        assert traced.name == "TestTrace"
        # Evaluate a few times to build trace
        self.scene.lambdify_system((self.t,))
        for t_val in np.linspace(0, np.pi, 8):
            self.scene.evaluate_system(t_val)
            self.scene.update()
        # With frequency=2, should have 4 points (0, 2, 4, 6 evaluations)
        assert len(traced.trace_history) == 4

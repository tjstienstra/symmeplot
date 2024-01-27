from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from symmeplot.core import SceneBase
from symmeplot.pyqtgraph.plot_base import PgPlotBase
from symmeplot.pyqtgraph.plot_objects import (
    PlotBody,
    PlotFrame,
    PlotLine,
    PlotPoint,
    PlotVector,
)

if TYPE_CHECKING:
    from sympy.physics.vector import Point, ReferenceFrame

__all__ = ["Scene3D"]


class Scene3D(SceneBase):
    """Class for plotting sympy mechanics in pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which all objects will be oriented.
    origin : Point
        The absolute origin with respect to which all objects will be positioned.
    view : pyqtgraph.opengl.GLViewWidget, optional
        The view in which the scene should be plotted. If None, a new view is created.
    **inertial_frame_properties : dict, optional
        Keyword arguments are parsed to the
        :class:`symmeplot.pyqtgraph.plot_objects.PlotFrame` representing the inertial
        reference frame.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy.physics.mechanics as me
        from symmeplot.pyqtgraph import Scene3D

        N = me.ReferenceFrame("N")
        A = me.ReferenceFrame("A")
        A.orient_axis(N, N.z, 1)
        N0 = me.Point("N_0")
        v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
        A0 = N0.locatenew("A_0", v)
        scene = Scene3D(N, N0, scale=0.5)
        scene.add_vector(v, name="v")
        scene.add_frame(A, A0, ls="--")
        scene.lambdify_system(())
        scene.evaluate_system()
        scene.plot()

    """

    _PlotPoint: type[PgPlotBase] = PlotPoint
    _PlotLine: type[PgPlotBase] = PlotLine
    _PlotVector: type[PgPlotBase] = PlotVector
    _PlotFrame: type[PgPlotBase] = PlotFrame
    _PlotBody: type[PgPlotBase] = PlotBody

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 view: gl.GLViewWidget | None = None, **inertial_frame_properties):
        if pg.QAPP is None:
            pg.mkQApp()
        if view is None:
            view = gl.GLViewWidget()
        if not isinstance(view, gl.GLViewWidget):
            raise TypeError(
                f"Expected a pyqtgraph.opengl.GLViewWidget, got {type(view)}.")
        self._view = view
        view.show()
        super().__init__(inertial_frame, zero_point, **inertial_frame_properties)

    @property
    def view(self):
        """The view in which the scene is plotted."""
        return self._view

    def plot(self):
        """Plot all plot objects."""
        for plot_object in self._children:
            plot_object.update()
            plot_object.plot(self.view)
        pg.exec()

from __future__ import annotations

from typing import Iterable

from sympy.physics.mechanics import Particle, Point, ReferenceFrame, RigidBody, Vector

from symmeplot.core import (
    PlotBodyMixin,
    PlotFrameMixin,
    PlotLineMixin,
    PlotPointMixin,
    PlotVectorMixin,
)
from symmeplot.pyqtgraph.artists import Line3D, Point3D, Vector3D
from symmeplot.pyqtgraph.plot_base import PgPlotBase

__all__ = ["PlotPoint", "PlotLine", "PlotVector", "PlotFrame", "PlotBody"]


class PlotPoint(PlotPointMixin, PgPlotBase):
    """A class for plotting a Point in 3D using pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    point : Point or Vector
        The point or vector that should be plotted with respect to the ``zero_point``.
        If a vector is provided, the ``origin`` will be at the tip of the vector with
        respect to the ``zero_point``. If not specified, the default is the
        ``zero_point``.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`pyqtgraph.opengl.items.GLScatterPlotItem`, so
        ``color=(1, 0, 0, 1)`` will make the plotted point red.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotPoint

        l1, l2, l3 = sm.symbols("l:3")
        N, O = me.ReferenceFrame("N"), me.Point("O")
        P1 = O.locatenew("P1", (l1 * N.x + l2 * N.y + l3 * N.z))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plot_point = PlotPoint(N, O, P1, color="k")
        f = sm.lambdify((l1, l2, l3), plot_point.get_expressions_to_evaluate())
        plot_point.values = f(0, 0, 0)
        plot_point.plot()  # Plot the point
        plot_point.values = f(0.2, 0.6, 0.3)
        plot_point.update()  # The point will now be on its new position

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 point: Point, name: str | None = None, **kwargs):
        super().__init__(inertial_frame, zero_point, point, name)
        self.add_artist(
            Point3D(0, 0, 0, **kwargs),
            self.get_sympy_object_exprs(),
        )


class PlotLine(PlotLineMixin, PgPlotBase):
    """A class for plotting lines in 3D using pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    points : list of Point or Vector
        The points or vectors through which the line should be plotted with respect to
        the ``zero_point``. If a vector is provided, the ``origin`` will be at the tip
        of the vector with respect to the ``zero_point``.
    name : str, optional
        The name of the line. Default is ``None``.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`pyqtgraph.opengl.items.GLLinePlotItem`, so
        ``color=(1, 0, 0, 1)`` will make the plotted line red.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotLine

        l1, l2, l3 = sm.symbols("l:3")
        N, O = me.ReferenceFrame("N"), me.Point("O")
        P1 = O.locatenew("P1", (l1 * N.x + l2 * N.y + l3 * N.z))
        P2 = P1.locatenew("P2", -0.3 * N.x)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        line_plot = PlotLine(N, O, [O, P1, P2], color="k")
        f = sm.lambdify((l1, l2, l3), line_plot.get_expressions_to_evaluate())
        line_plot.values = f(0, 0, 0)
        line_plot.plot()  # Plot the point
        line_plot.values = f(0.2, 0.6, 0.3)
        line_plot.update()  # The point will now be on its new position

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 line: Iterable[Point], name: str | None = None, **kwargs):
        super().__init__(inertial_frame, zero_point, line, name)
        self.add_artist(
            Line3D([0], [0], [0], **kwargs),
            self.get_sympy_object_exprs(),
        )


class PlotVector(PlotVectorMixin, PgPlotBase):
    """A class for plotting a Vector in 3D using pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    vector : Vector
        The vector that should be plotted with respect to the ``zero_point``.
    origin : Point or Vector, optional
        The origin of the vector with respect to the ``zero_point``. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the ``origin`` is at the
        tip of the vector with respect to the ``zero_point``. Default is ``zero_point``.
    name : str
        Name of the plot object. Default is the vector as string.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`pyqtgraph.opengl.items.GLLinePlotItem`, so
        ``color=(1, 0, 0, 1)`` will make the plotted vector red.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotVector

        N = me.ReferenceFrame("N")
        O = me.Point("O")
        O_v = O.locatenew("O_v", 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        v = 0.4 * N.x + 0.4 * N.y - 0.6 * N.z
        v_plot = PlotVector(N, O, v, O_v, color="r", ls="--")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        v_plot.values = sm.lambdify((), v_plot.get_expressions_to_evaluate())()
        v_plot.plot(ax)

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 vector: Vector, origin: Point | Vector | None = None,
                 name: str | None = None, **kwargs):
        super().__init__(inertial_frame, zero_point, vector, origin, name)
        self.add_artist(
            Vector3D([0, 0, 0], [0, 0, 0], **kwargs),
            self.get_sympy_object_exprs(),
        )


class PlotFrame(PlotFrameMixin, PgPlotBase):
    """A class for plotting a ReferenceFrame in 3D using pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    frame : ReferenceFrame
        The reference frame that should be plotted.
    origin : Point or Vector, optional
        The origin of the frame with respect to the ``zero_point``. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the `origin` is at the
        tip of the vector with respect to the ``zero_point``. Default is ``zero_point``.
    style : str, optional
        Reference to what style should be used for plotting the frame. The default style
        is ``'default'``. Available styles:
        - None: No properties of the vectors will be set.
        - 'default': Nice default frame with as color 'rgb' for xyz.
    scale : float, optional
        Length of the vectors of the reference frame.
    **kwargs : dict, optional
        Kwargs that are parsed to
        :class:`symmeplot.pyqtgraph.plot_objects.PlotVector`s, which possibly parses
        them to :class:`pyqtgraph.opengl.items.GLLinePlotItem`, so
        ``color=(1, 0, 0, 1)`` will make all vectors of the reference frame red.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotFrame

        N = me.ReferenceFrame("N")
        A = me.ReferenceFrame("A")
        A.orient_axis(N, N.z, 1)
        N0 = me.Point("N_0")
        A0 = N0.locatenew("A_0", 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        N_plot = PlotFrame(N, N0, N, scale=0.5)
        A_plot = PlotFrame(N, N0, A, A0, scale=0.5, ls="--")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        N_plot.values = sm.lambdify((), N_plot.get_expressions_to_evaluate())()
        A_plot.values = sm.lambdify((), A_plot.get_expressions_to_evaluate())()
        N_plot.plot(ax)
        A_plot.plot(ax)

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 frame: ReferenceFrame, origin: Point | Vector | None = None,
                 name: str | None = None, scale: float = 0.1, style: str = "default",
                 **kwargs):
        super().__init__(inertial_frame, zero_point, frame, origin, name, scale)
        properties = self._get_style_properties(style)
        for prop in properties:
            prop.update(kwargs)
        for vector, prop in zip(frame, properties):
            self._children.append(
                PlotVector(inertial_frame, zero_point, scale * vector, origin, **prop))

    def _get_style_properties(self, style):
        """Get the properties of the vectors belonging to a certain style."""
        properties = [{}, {}, {}]
        if style is None:
            return properties
        elif style == "default":
            colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
            for color, prop in zip(colors, properties):
                prop.update({
                    "color": color
                })
            return properties
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")


class PlotBody(PlotBodyMixin, PgPlotBase):
    """A class for plotting a body in 3D using pyqtgraph.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    body : RigidBody or Particle
        The body that should be plotted.
    origin : Point or Vector, optional
        The origin of the frame with respect to the ``zero_point``. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the ``origin`` is at the
        tip of the vector with respect to the ``zero_point``. Default is ``zero_point``.
    style : str, optional
        Reference to what style should be used for plotting the body. The default style
        is ``'default'``. Available styles:
        - None: No properties of the vectors will be set.
        - 'default': Uses a frame with as color 'rgb' for xyz.
    plot_frame_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`symemplot.pyqtgraph.plot_objects.PlotFrame`.
    plot_point_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`symemplot.pyqtgraph.plot_objects.PlotPoint` representing the center of
        mass.
    **kwargs : dict, optional
        Kwargs that are parsed to both internally used plot objects.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotBody

        N = me.ReferenceFrame("N")
        A = me.ReferenceFrame("A")
        A.orient_axis(N, N.z, 1)
        N0 = me.Point("N_0")
        A0 = N0.locatenew("A_0", 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        ground = me.RigidBody("ground", N0, N, 1, (N.x.outer(N.x), N0))
        body = me.RigidBody("body", A0, A, 1, (A.x.outer(A.x), A0))
        ground_plot = PlotBody(N, N0, ground)
        body_plot = PlotBody(N, N0, body)
        body_plot.attach_circle(body.masscenter, 0.3, A.x + A.y + A.z,
                                facecolor="none", edgecolor="k")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ground_plot.values = sm.lambdify((), ground_plot.get_expressions_to_evaluate()
                                        )()
        body_plot.values = sm.lambdify((), body_plot.get_expressions_to_evaluate())()
        ground_plot.plot(ax)
        body_plot.plot(ax)

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 body: Particle | RigidBody, name: str | None = None,
                 style: str = "default", plot_point_properties: dict | None = None,
                 plot_frame_properties: dict | None = None, **kwargs):
        super().__init__(inertial_frame, zero_point, body, name)
        properties = self._get_style_properties(style)
        if plot_point_properties is not None:
            properties[0].update(plot_point_properties)
        if plot_frame_properties is not None:
            properties[1].update(plot_frame_properties)
        for prop in properties:
            prop.update(kwargs)
        # Particle.masscenter does not yet exist in SymPy 1.12
        masscenter = getattr(body, "masscenter", getattr(body, "point", None))
        self._children.append(
            PlotPoint(inertial_frame, zero_point, masscenter, **properties[0]))
        if hasattr(body, "frame"):
            self._children.append(
                PlotFrame(inertial_frame, zero_point, body.frame, masscenter,
                          **properties[1]))

    def _get_style_properties(self, style):
        """Get the properties of the vectors belonging to a certain style."""
        properties = [{}, {}]
        if style is None:
            return properties
        elif style == "default":
            properties[0] = {}
            properties[1] = {"style": "default"}
            return properties
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")

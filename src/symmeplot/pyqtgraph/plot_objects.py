from __future__ import annotations

from collections.abc import Iterable

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

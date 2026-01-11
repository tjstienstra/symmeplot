"""Plot objects of the matplotlib backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sympy import Expr, sympify
from sympy.physics.mechanics import Particle, Point, ReferenceFrame, RigidBody, Vector

from symmeplot.core import (
    PlotBodyMixin,
    PlotFrameMixin,
    PlotLineMixin,
    PlotPointMixin,
    PlotTracedPointMixin,
    PlotVectorMixin,
)
from symmeplot.matplotlib.artists import Circle3D, Line3D, LineCollection3D, Vector3D
from symmeplot.matplotlib.plot_base import MplPlotBase

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "PlotBody",
    "PlotFrame",
    "PlotLine",
    "PlotPoint",
    "PlotTracedPoint",
    "PlotVector",
]


class PlotPoint(PlotPointMixin, MplPlotBase):
    """A class for plotting a Point in 3D using matplotlib.

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
    style : str, optional
        Reference to what style should be used for plotting the point. The default style
        is ``'default'``. Available styles:
        - None: Default of the :class:`mpl_toolkits.mplot3d.art3d.Line3D`.
        - 'default': Normal point.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        ``color='r'`` will make the plotted point red.

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
        plot_point.update()  # Updates the artist(s) to the new values
        plot_point.plot()  # Plot the point
        plot_point.values = f(0.2, 0.6, 0.3)
        plot_point.update()  # The point will now be on its new position

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        point: Point,
        name: str | None = None,
        style: str = "default",
        **kwargs: object,
    ) -> None:
        super().__init__(inertial_frame, zero_point, point, name)
        self.add_artist(
            Line3D([0], [0], [0], **{**self._get_style_properties(style), **kwargs}),
            tuple((expr,) for expr in self.get_sympy_object_exprs()),
        )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        return self.point_coords

    def _get_style_properties(self, style: str | None) -> dict:
        """Get the properties of the vector belonging to a certain style."""
        if style is None:
            return {}
        if style == "default":
            return {"marker": "o"}
        msg = f"Style '{style}' is not implemented."
        raise NotImplementedError(msg)


class PlotTracedPoint(PlotTracedPointMixin, MplPlotBase):
    """A class for plotting a traced Point in 3D using matplotlib.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    point : Point
        The point that should be traced with respect to the ``zero_point``.
    name : str, optional
        Name of the plot object.
    frequency : int, optional
        Frequency to log the point with. Default is 1 (shows every point).
    alpha_decay : callable, optional
        Function that returns the transparency of a point based on the number
        of evaluations since it was logged. The default is `lambda _: 1.0`
        (all points remain fully visible).
    color : str, optional
        Color of the traced points. Default is 'blue'.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`matplotlib.collections.LineCollection`.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import sympy as sm
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import PlotTracedPoint

        t = sm.symbols("t")
        N, O = me.ReferenceFrame("N"), me.Point("O")
        P = O.locatenew("P", sm.cos(t) * N.x + sm.sin(t) * N.y)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plot_traced = PlotTracedPoint(
            N, O, P, frequency=1, alpha_decay=lambda i: max(0.1, 1.0 - i / 20)
        )
        f = sm.lambdify(t, plot_traced.get_expressions_to_evaluate())
        for t_val in np.linspace(0, 2 * np.pi, 30):
            plot_traced.values = f(t_val)
            plot_traced.update()
        plot_traced.plot(ax)

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        point: Point,
        name: str | None = None,
        frequency: int = 1,
        alpha_decay: callable[[int], float] | None = None,
        color: str = "blue",
        **kwargs: object,
    ) -> None:
        super().__init__(
            inertial_frame, zero_point, point, name, frequency, alpha_decay
        )
        # Create colormap that fades from transparent to the specified color
        color_rgba = mcolors.to_rgba(color)
        transparent_color = (*color_rgba[:3], 0.0)
        self._cmap = mcolors.LinearSegmentedColormap.from_list(
            f"trace_{color}", [transparent_color, color_rgba]
        )
        # Set default linewidths if not provided
        if "linewidths" not in kwargs:
            kwargs["linewidths"] = 2
        self.add_artist(
            LineCollection3D(cmap=self._cmap, **kwargs),
            tuple((expr,) for expr in self.get_sympy_object_exprs()),
        )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        if len(self._trace_history) > 0:
            return self._trace_history[-1]
        return self.point_coords

    def plot(self, ax: plt.Axes | None = None) -> None:
        """Plot the traced point.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            Axes on which the artist should be added. The default is the active axes.

        """
        if ax is None:
            ax = plt.gca()
        # For LineCollection3D, we need to use add_collection instead of add_artist
        for artist, _ in self._artists:
            ax.add_collection(artist)
        for child in self._children:
            child.plot(ax)

    def update(self) -> None:
        """Update the objects on the scene, based on the current values."""
        # Update trace history
        self._update_trace_history()

        # Create segments and alphas for the line collection
        segments = []
        alphas = []
        if len(self._trace_history) > 1:
            for i in range(len(self._trace_history) - 1):
                segments.append([self._trace_history[i], self._trace_history[i + 1]])
                # Calculate alpha based on how old this segment is
                age = len(self._trace_history) - i - 1
                alphas.append(self._alpha_decay(age))

        # Update the artist with segments and alphas
        if segments:
            artist = self._artists[0][0]
            # Normalize alphas to [0, 1] range for colormap
            alphas_array = np.array(alphas)
            if alphas_array.max() > alphas_array.min():
                # Normalize to 0-1 range
                normalized_alphas = (alphas_array - alphas_array.min()) / (
                    alphas_array.max() - alphas_array.min()
                )
            else:
                # All alphas are the same, use them directly
                normalized_alphas = alphas_array
            artist.update_data(segments, normalized_alphas)

        # Update children
        for child in self._children:
            child.update()


class PlotLine(PlotLineMixin, MplPlotBase):
    """A class for plotting lines in 3D using matplotlib.

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
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        ``color='r'`` will make the plotted line red.

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
        line_plot.update()  # Updates the artist(s) to the new values
        line_plot.plot()  # Plot the point
        line_plot.values = f(0.2, 0.6, 0.3)
        line_plot.update()  # The point will now be on its new position

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        line: Iterable[Point],
        name: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(inertial_frame, zero_point, line, name)
        self.add_artist(
            Line3D([0], [0], [0], **kwargs),
            self.get_sympy_object_exprs(),
        )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        return np.array(self.line_coords, dtype=np.float64).mean(axis=1)


class PlotVector(PlotVectorMixin, MplPlotBase):
    """A class for plotting a Vector in 3D using matplotlib.

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
    style : str, optional
        Reference to what style should be used for plotting the vector. The default
        style is ``'default'``. Available styles:
        - None: Default of the :class:`mpl_toolkits.mplot3d.art3d.Line3D`.
        - 'default': Normal black arrow.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        ``color='r'`` will make the plotted arrow red.

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
        v_plot.update()  # Updates the artist(s) to the new values
        v_plot.plot(ax)

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        vector: Vector,
        origin: Point | Vector | None = None,
        name: str | None = None,
        style: str = "default",
        **kwargs: object,
    ) -> None:
        super().__init__(inertial_frame, zero_point, vector, origin, name)
        self._properties = {}
        self.add_artist(
            Vector3D(
                [0, 0, 0], [0, 0, 0], **{**self._get_style_properties(style), **kwargs}
            ),
            self.get_sympy_object_exprs(),
        )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        return self.origin_coords + self.vector_values / 2

    def _get_style_properties(self, style: str | None) -> dict:
        """Get the properties of the vector belonging to a certain style."""
        if style is None:
            return {}
        if style == "default":
            return {
                "color": "k",
                "mutation_scale": 10,
                "arrowstyle": "-|>",
                "shrinkA": 0,
                "shrinkB": 0,
                "picker": 20,
            }
        msg = f"Style '{style}' is not implemented."
        raise NotImplementedError(msg)


class PlotFrame(PlotFrameMixin, MplPlotBase):
    """A class for plotting a ReferenceFrame in 3D using matplotlib.

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
        :class:`symmeplot.matplotlib.plot_objects.PlotVector`s, which possibly parses
        them to :class:`matplotlib.patches.FancyArrow`, so ``color='r'`` will make all
        vectors of the reference frame red.

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
        N_plot.update()  # Updates the artist(s) to the new values
        A_plot.update()  # Updates the artist(s) to the new values
        N_plot.plot(ax)
        A_plot.plot(ax)

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        frame: ReferenceFrame,
        origin: Point | Vector | None = None,
        name: str | None = None,
        scale: float = 0.1,
        style: str = "default",
        **kwargs: object,
    ) -> None:
        super().__init__(inertial_frame, zero_point, frame, origin, name, scale)
        properties = self._get_style_properties(style)
        for prop in properties:
            prop.update(kwargs)
        for vector, prop in zip(frame, properties, strict=True):
            self._children.append(
                PlotVector(inertial_frame, zero_point, scale * vector, origin, **prop)
            )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        return self.vectors[0].origin_coords + 0.3 * sum(
            [v.vector_values for v in self.vectors]
        )

    def _get_style_properties(self, style: str | None) -> list[dict]:
        """Get the properties of the vectors belonging to a certain style."""
        properties = [{}, {}, {}]
        if style is None:
            return properties
        if style == "default":
            colors = "rgb"
            for color, prop in zip(colors, properties, strict=True):
                prop.update({"style": "default", "color": color})
            return properties
        msg = f"Style '{style}' is not implemented."
        raise NotImplementedError(msg)


class PlotBody(PlotBodyMixin, MplPlotBase):
    """A class for plotting a body in 3D using matplotlib.

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
        - 'default': Uses a special point for the center of mass and a frame with as
        color 'rgb' for xyz.
    plot_frame_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`symemplot.matplotlib.plot_objects.PlotFrame`.
    plot_point_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`symemplot.matplotlib.plot_objects.PlotPoint` representing the center of
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
        body_plot.attach_circle(
            body.masscenter, 0.3, A.x + A.y + A.z, facecolor="none", edgecolor="k"
        )
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ground_plot.values = sm.lambdify((), ground_plot.get_expressions_to_evaluate())()
        body_plot.values = sm.lambdify((), body_plot.get_expressions_to_evaluate())()
        ground_plot.update()  # Updates the artist(s) to the new values
        body_plot.update()  # Updates the artist(s) to the new values
        ground_plot.plot(ax)
        body_plot.plot(ax)

    """  # noqa: E501

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        body: Particle | RigidBody,
        name: str | None = None,
        style: str = "default",
        plot_point_properties: dict | None = None,
        plot_frame_properties: dict | None = None,
        **kwargs: object,
    ) -> None:
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
            PlotPoint(inertial_frame, zero_point, masscenter, **properties[0])
        )
        if hasattr(body, "frame"):
            self._children.append(
                PlotFrame(
                    inertial_frame, zero_point, body.frame, masscenter, **properties[1]
                )
            )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        return self.plot_masscenter.annot_coords

    def attach_circle(
        self, center: Point | Vector, radius: Expr, normal: Vector, **kwargs: object
    ) -> Circle3D:
        """Attaches a circle to a point to represent the body.

        Parameters
        ----------
        center : Point or Vector
            Center of the circle.
        radius : Sympifyable
            Radius of the circle.
        normal : Vector
            Normal of the circle.
        kwargs : dict
            Key word arguments are parsed to
            :class:`symmeplot.matplotlib.plot_artists.Circle3D`.

        Returns
        -------
        :class:`symmeplot.matplotlib.plot_artists.Circle3D`
            Circle artist.

        """
        if isinstance(center, Point):
            center = center.pos_from(self.zero_point)
        if isinstance(center, Vector):
            center = tuple(center.to_matrix(self.inertial_frame)[:])
        else:
            msg = f"'center' should be a {type(Point)} or {type(Vector)}."
            raise TypeError(msg)
        if isinstance(normal, Vector):
            normal = tuple(normal.to_matrix(self.inertial_frame)[:])
        else:
            msg = f"'normal' should be a {type(Vector)}."
            raise TypeError(msg)
        self.add_artist(
            Circle3D((0, 0, 0), 0, (0, 0, 1), **kwargs),
            (center, sympify(radius), normal),
        )

    def _get_style_properties(self, style: str | None) -> list[dict]:
        """Get the properties of the vectors belonging to a certain style."""
        properties = [{}, {}]
        if style is None:
            return properties
        if style == "default":
            properties[0] = {
                "color": "k",
                "marker": r"$\bigoplus$",
                "markersize": 8,
                "markeredgewidth": 0.5,
                "zorder": 10,
            }
            properties[1] = {"style": "default"}
            return properties
        msg = f"Style '{style}' is not implemented."
        raise NotImplementedError(msg)

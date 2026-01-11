"""Mixin classes to be used when creating plot objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sympy import Expr, latex
from sympy.physics.mechanics import Particle, Point, ReferenceFrame, RigidBody, Vector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from symmeplot.core.plot_base import PlotBase

__all__ = [
    "OriginMixin",
    "PlotBodyMixin",
    "PlotFrameMixin",
    "PlotLineMixin",
    "PlotPointMixin",
    "PlotTracedPointMixin",
    "PlotVectorMixin",
]


class OriginMixin:
    """Mixin class adding an origin property to a plot object."""

    @property
    def origin(self) -> Point:
        """The origin of the object with respect to the `zero_point`."""
        return self._origin

    @origin.setter
    def origin(self, new_origin: Point | Vector | None) -> None:
        if hasattr(self, "_origin"):
            msg = "The origin of a plot object cannot be changed."
            raise AttributeError(msg)
        if new_origin is None:
            new_origin = self.zero_point
        elif isinstance(new_origin, Vector):
            new_origin = self.zero_point.locatenew("", new_origin)
        if isinstance(new_origin, Point):
            for child in self._children:
                child.origin = new_origin
            self._origin = new_origin
        else:
            msg = "'origin' should be a valid Point object."
            raise TypeError(msg)


class PlotPointMixin:
    """Mixin class for plotting a Point in 3D.

    Notes
    -----
    The subclass should create and add the artist in the constructor.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        point: Point,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = point.name
        if not isinstance(point, Point):
            msg = "'point' should be a sympy Point object."
            raise TypeError(msg)
        super().__init__(inertial_frame, zero_point, point, name)

    @property
    def point(self) -> Point:
        """The sympy Point, which is being plotted."""
        return self._sympy_object

    @property
    def point_coords(self) -> np.ndarray[np.float64]:
        """Coordinate values of the plotted point."""
        return np.array(self._values[0]).reshape(3)

    def get_sympy_object_exprs(self) -> tuple[Expr, Expr, Expr]:
        """Get coordinate of the point as expressions."""
        return tuple(
            self.point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:]
        )


class PlotTracedPointMixin:
    """Mixin class for plotting a traced Point in 3D.

    Notes
    -----
    The subclass should create and add the artist in the constructor.
    The traced point tracks the history of a point's position and displays
    all previous positions with optional transparency decay.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        point: Point,
        name: str | None = None,
        frequency: int = 1,
        alpha_decay: callable[[int], float] | None = None,
    ) -> None:
        """Initialize the traced point.

        Parameters
        ----------
        inertial_frame : ReferenceFrame
            The reference frame with respect to which the object is oriented.
        zero_point : Point
            The absolute origin with respect to which the object is positioned.
        point : Point
            The point to be traced.
        name : str, optional
            Name of the point.
        frequency : int, optional
            Frequency to log the point with. Default is 1 (shows every point).
        alpha_decay : callable, optional
            Function that returns the transparency of a point based on the number
            of evaluations since it was logged. The default is `lambda _: 1.0`
            (all points remain fully visible).

        """
        if name is None:
            name = point.name
        if not isinstance(point, Point):
            msg = "'point' should be a sympy Point object."
            raise TypeError(msg)
        super().__init__(inertial_frame, zero_point, point, name)
        self._frequency = frequency
        self._alpha_decay = alpha_decay if alpha_decay is not None else lambda _: 1.0
        self._trace_history: list[np.ndarray] = []
        self._evaluation_count = 0

    @property
    def point(self) -> Point:
        """The sympy Point, which is being traced."""
        return self._sympy_object

    @property
    def frequency(self) -> int:
        """Frequency to log the point with."""
        return self._frequency

    @property
    def alpha_decay(self) -> callable:
        """Function that returns the transparency based on evaluation count."""
        return self._alpha_decay

    @property
    def trace_history(self) -> list[np.ndarray]:
        """History of point positions."""
        return self._trace_history

    @property
    def point_coords(self) -> np.ndarray[np.float64]:
        """Coordinate values of the current plotted point."""
        return np.array(self._values[0]).reshape(3)

    def get_sympy_object_exprs(self) -> tuple[Expr, Expr, Expr]:
        """Get coordinate of the point as expressions."""
        return tuple(
            self.point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:]
        )

    def _update_trace_history(self) -> None:
        """Update the trace history with the current point coordinates."""
        self._evaluation_count += 1
        if self._evaluation_count % self._frequency == 0:
            self._trace_history.append(self.point_coords.copy())


class PlotLineMixin:
    """Mixin class for plotting lines in 3D.

    Notes
    -----
    The subclass should create and add the artist in the constructor.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        line: Iterable[Point],
        name: str | None = None,
    ) -> None:
        _points = []
        if isinstance(line, Point):
            line = (line,)
        for point in line:
            if not isinstance(point, Point):
                msg = "'line' should be a list of Point objects."
                raise TypeError(msg)
            _points.append(point)
        super().__init__(inertial_frame, zero_point, tuple(_points), name)

    @property
    def line(self) -> tuple[Point, ...]:
        """The points that spawn the line."""
        return self._sympy_object

    @property
    def line_coords(self) -> np.ndarray[np.float64]:
        """Coordinate values of the plotted line."""
        return np.array(self._values[0]).reshape(3, -1)

    def get_sympy_object_exprs(self) -> tuple[tuple[Expr, ...], ...]:
        """Arguments of the sympy object artist in expression form.

        Notes
        -----
        The form of the expression is ``((x0, x1, ...), (y0, y1, ...), (z0, z1, ...))``.

        """
        arr = np.array(
            [
                point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:]
                for point in self.line
            ],
            dtype=object,
        ).T
        return tuple(map(tuple, arr))


class PlotVectorMixin(OriginMixin):
    """Mixin class for plotting a Vector in 3D.

    Notes
    -----
    The subclass should create and add the artist in the constructor.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        vector: Vector,
        origin: Point | Vector | None = None,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = str(latex(vector))
        if not isinstance(vector, Vector):
            msg = "'vector' should be a sympy Vector object."
            raise TypeError(msg)
        super().__init__(inertial_frame, zero_point, vector, name)
        self.origin = origin

    @property
    def vector(self) -> Vector:
        """The sympy Vector, which is being plotted."""
        return self._sympy_object

    @property
    def origin_coords(self) -> np.ndarray[np.float64]:
        """Coordinate values of the origin of the plotted vector."""
        return np.array([self._values[0][0]]).reshape(3)

    @property
    def vector_values(self) -> np.ndarray[np.float64]:
        """Values of the plotted vector."""
        return np.array(self._values[0][1]).reshape(3)

    def get_sympy_object_exprs(self) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
        """Arguments of the sympy object artist in expression form."""
        return tuple(
            tuple(v.to_matrix(self.inertial_frame)[:])
            for v in (self.origin.pos_from(self.zero_point), self.vector)
        )


class PlotFrameMixin(OriginMixin):
    """Mixin class for plotting a ReferenceFrame in 3D.

    Notes
    -----
    The subclass should instantiate the PlotVector objects in the constructor.
    The children should be added in the following order: x, y, z.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        frame: ReferenceFrame,
        origin: Point | Vector | None = None,
        name: str | None = None,
        scale: float = 0.1,  # noqa: ARG002
    ) -> None:
        if name is None:
            name = frame.name
        if not isinstance(frame, ReferenceFrame):
            msg = "'frame' should be a sympy ReferenceFrame object."
            raise TypeError(msg)
        super().__init__(inertial_frame, zero_point, frame, name)
        self.origin = origin

    @property
    def frame(self) -> ReferenceFrame:
        """The sympy ReferenceFrame, which is being plotted."""
        return self._sympy_object

    @property
    def vectors(self) -> tuple[PlotBase, PlotBase, PlotBase]:
        """The PlotVectors used to plot the reference frame."""
        return tuple(self._children)

    @property
    def x(self) -> PlotBase:
        """PlotVector used for the unit vector in the x direction."""
        return self._children[0]

    @property
    def y(self) -> PlotBase:
        """PlotVector used for the unit vector in the y direction."""
        return self._children[1]

    @property
    def z(self) -> PlotBase:
        """PlotVector used for the unit vector in the z direction."""
        return self._children[2]


class PlotBodyMixin:
    """Mixin class for plotting a body in 3D.

    Notes
    -----
    The subclass should instantiate the PlotFrame and PlotPoint objects in the
    constructor. If the body has a frame, then the PlotFrame should be the second child.
    The PlotPoint representing the center of mass should always be the first child.

    """

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        body: Particle | RigidBody,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = str(body)  # Particle.name does not yet exist in SymPy 1.12
        if not isinstance(body, (Particle, RigidBody)):
            msg = "'body' should be a sympy body."
            raise TypeError(msg)
        super().__init__(inertial_frame, zero_point, body, name)
        self._expressions_self = ()

    @property
    def body(self) -> Particle | RigidBody:
        """The sympy body, which is being plotted."""
        return self._sympy_object

    @property
    def plot_frame(self) -> PlotBase | None:
        """PlotFrame used for plotting the reference frame of the body."""
        if len(self._children) == 2:
            return self._children[1]
        return None

    @property
    def plot_masscenter(self) -> PlotBase:
        """PlotPoint used for plotting the center of mass of the body."""
        return self._children[0]

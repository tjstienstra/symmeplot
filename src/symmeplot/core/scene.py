from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Iterable

from sympy import lambdify
from sympy.physics.mechanics import Point, ReferenceFrame, Vector

from symmeplot.core.artists import ArtistBase
from symmeplot.core.plot_base import PlotBase


def _create_undefined_function(error: Exception, message: str) -> Callable[..., None]:
    def undefined_function(*args, **kwargs):
        raise error(message)

    return undefined_function


class SceneBase(ABC):
    """Base class for a visualization scene for sympy mechanics.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which all objects will be oriented.
    origin : Point
        The absolute origin with respect to which all objects will be positioned.
    **inertial_frame_properties
        Keyword arguments are parsed to the PlotFrame representing the inertial
        reference frame.

    """

    _PlotPoint: type[PlotBase] = _create_undefined_function(
        NotImplementedError, "'add_point' has not been implemented in this backend.")
    _PlotLine: type[PlotBase] = _create_undefined_function(
        NotImplementedError, "'add_line' has not been implemented in this backend.")
    _PlotVector: type[PlotBase] = _create_undefined_function(
        NotImplementedError, "'add_vector' has not been implemented in this backend.")
    _PlotFrame: type[PlotBase] = _create_undefined_function(
        NotImplementedError, "'add_frame' has not been implemented in this backend.")
    _PlotBody: type[PlotBase] = _create_undefined_function(
        NotImplementedError, "'add_body' has not been implemented in this backend.")

    def __init__(
        self, inertial_frame: ReferenceFrame, zero_point: Point,
        **inertial_frame_properties
    ):
        self._zero_point = zero_point
        self._inertial_frame = inertial_frame
        self._children = []
        self._lambdified_system = _create_undefined_function(
            ValueError,
            "System has not been lambdified. "
            "Use 'lambdify_system' to lambdify the system.",
        )
        self.add_frame(inertial_frame, **inertial_frame_properties)

    @property
    def zero_point(self) -> Point:
        """The absolute origin with respect to which all objects will be positioned."""
        return self._zero_point

    @property
    def inertial_frame(self) -> ReferenceFrame:
        """The reference frame with respect to which all objects will be oriented."""
        return self._inertial_frame

    @property
    def plot_objects(self) -> tuple[PlotBase, ...]:
        """Plot objects."""
        return tuple(self._children)

    @property
    def children(self) -> tuple[PlotBase, ...]:
        """Plot objects."""
        return tuple(self._children)

    @property
    def artists(self) -> tuple[ArtistBase, ...]:
        """Artists used to plot the object."""
        return tuple(a for plot_obj in self._children for a in plot_obj.artists)

    @property
    def values(self) -> tuple:
        """List of evaluated values for the object's variables."""
        return tuple(plot_obj.values for plot_obj in self._children)

    @values.setter
    def values(self, values: tuple):
        for plot_obj, vals in zip(self._children, values):
            plot_obj.values = vals

    def get_expressions_to_evaluate(self) -> tuple:
        """Return a tuple of the necessary expressions for plotting."""
        return tuple(po.get_expressions_to_evaluate() for po in self._children)

    def add_plot_object(self, plot_object: PlotBase):
        """Add a plot object to the scene.

        Parameters
        ----------
        plot_object : PlotBase
            The plot object that should be added.

        """
        self._children.append(plot_object)

    def add_point(self, point: Point | Vector, **kwargs) -> type[PlotBase]:
        """Add a sympy Vector to the scene.

        Parameters
        ----------
        point : Point or Vector
            The point or vector to be plotted as point in space.
        **kwargs :
            Keyword arguments are parsed to the plot object.

        Returns
        -------
        PlotPoint
            The added plot object.

        """
        obj = self._PlotPoint(self.inertial_frame, self.zero_point, point, **kwargs)
        self.add_plot_object(obj)
        return obj

    def add_line(
        self, points: Iterable[Point | Vector], name: str | None = None, **kwargs
    ):
        """Add a sympy Vector to the scene.

        Parameters
        ----------
        points : list of Point or Vector
            The points or vectors through which the line should be plotted.
        name : str, optional
            The name of the line. Default is `None`.
        **kwargs :
            Kwargs that are parsed to plot object.

        Returns
        -------
        PlotLine
            The added plot object.

        """
        obj = self._PlotLine(
            self.inertial_frame, self.zero_point, points, name, **kwargs
        )
        self.add_plot_object(obj)
        return obj

    def add_vector(self, vector: Vector, origin: Point | Vector | None = None,
                   **kwargs):
        """Add a sympy Vector to the scene.

        Parameters
        ----------
        vector : Vector
            The vector that should be plotted with respect to the `zero_point`.
        origin : Point, Vector, optional
            The origin of the reference frame. Default is the `zero_point`.
        **kwargs : dict, optional
            Kwargs that are parsed to the plot object.

        Returns
        -------
        PlotVector
            The added plot object.

        """
        obj = self._PlotVector(self.inertial_frame, self.zero_point, vector, origin,
                              **kwargs)
        self.add_plot_object(obj)
        return obj

    def add_frame(self, frame: ReferenceFrame, origin: Point | Vector | None = None,
                  **kwargs):
        """Add a sympy ReferenceFrame to the scene.

        Parameters
        ----------
        frame : ReferenceFrame
            The reference frame that should be plotted.
        origin : Point, Vector, optional
            The origin of the reference frame. Default is the `zero_point`.
        **kwargs : dict, optional
            Kwargs that are parsed to plot object.

        Returns
        -------
        PlotFrame
            The added plot object.

        """
        obj = self._PlotFrame(self.inertial_frame, self.zero_point, frame, origin,
                             **kwargs)
        self.add_plot_object(obj)
        return obj

    def add_body(self, body, **kwargs):
        """Add a sympy body to the scene.

        Parameters
        ----------
        body : RigidBody or Particle
            The body that should be plotted.
        **kwargs : dict, optional
            Kwargs that are parsed to both internally used plot objects.

        Returns
        -------
        PlotBody
            The added plot object.

        """
        obj = self._PlotBody(self.inertial_frame, self.zero_point, body, **kwargs)
        self.add_plot_object(obj)
        return obj

    def get_plot_object(self, sympy_object: Any | str) -> PlotBase | None:
        """Return the `plot_object` based on a sympy object.

        Explanation
        -----------
        Return the ``plot_object`` based on a provided sympy object. For example
        ``ReferenceFrame('N')`` will give the ``PlotFrame`` of that reference frame. If
        the ``plot_object`` has not been added it will return ``None``.

        Parameters
        ----------
        sympy_object : Any or str
            SymPy object to search for. If it is a string it will search for the name.

        Returns
        -------
        PlotBase or None
            Retrieved plot object.

        """
        queue = [self]
        while queue:
            for po in queue.pop()._children:
                if po.sympy_object is sympy_object or po.name == sympy_object:
                    return po
                queue.append(po)

    def lambdify_system(
        self, args, modules=None, printer=None, use_imps=True, dummify=False, cse=True
    ) -> Callable:
        """Lambdify the system.

        Explanation
        -----------
        Lambdify the system for faster evaluation, when combined with
        :meth:`symmeplot.core.scene.SceneBase.evaluate_system`. See
        :func:`sympy.utilities.lambdify.lambdify` for more information.

        """
        self._lambdified_system = lambdify(
            args,
            self.get_expressions_to_evaluate(),
            modules=modules,
            printer=printer,
            use_imps=use_imps,
            dummify=dummify,
            cse=cse,
        )
        return self.evaluate_system

    def evaluate_system(self, *args) -> None:
        """Evaluate the system using the function created with ``lambdify_system``."""
        self.values = self._lambdified_system(*args)

    def set_visibility(
        self, sympy_object: Any | str, is_visible: bool, raise_error: bool = True
    ) -> None:
        """Hide or show a ``plot_object`` based on a ``sympy_object``.

        Parameters
        ----------
        sympy_object : Point or Vector or ReferenceFrame or Particle or RigidBody or str
            SymPy object to show or hide.
        is_visible : bool
            If True show ``plot_object``, otherwise hide plot_object.
        raise_error : bool, optional
            If plot_object not found raise an error. Default is True.

        """
        plot_object = self.get_plot_object(sympy_object)
        if plot_object is not None:
            plot_object.visible = is_visible
            return
        if raise_error:
            raise ValueError(f"PlotObject corresponding to '{sympy_object}' not found.")

    def plot(self) -> None:
        """Plot all plot objects."""
        self.update()
        for plot_object in self._children:
            plot_object.plot()

    def update(self) -> None:
        """Update the objects on the scene, based on the currect values."""
        for plot_object in self._children:
            plot_object.update()

    def animate(self, get_args: Callable[[Any], tuple], frames: Iterable[Any] | int,
                interval: int = 30, **kwargs) -> None:
        """Animate the scene.

        Parameters
        ----------
        get_args : Callable
            Function that returns the arguments for the ``evaluate_system`` method. The
            function should takes the current frame as input.
        frames : int or iterable
            Number of frames or iterable with frames.
        interval : int, optional
            Time interval between frames in milliseconds. Default is 30.
        **kwargs
            Keyword arguments are parsed to the internally used animation function.
        """
        raise NotImplementedError("'animate' has not been implemented in this backend.")

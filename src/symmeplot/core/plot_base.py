from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sympy.physics.vector import Point, ReferenceFrame
from sympy.utilities.iterables import iterable

import symmeplot.utilities.sympy_patches  # noqa: F401
from symmeplot.core.artists import ArtistBase

if TYPE_CHECKING:
    from sympy import Expr

__all__ = ["PlotBase"]


class PlotBase(ABC):
    """Class with the basic attributes and methods for the plot objects.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    name : str, optional
        Name of the plot object. Default is the name of the object being plotted.

    Notes
    -----
    The format of expressions and therefore values is as follows:

        tuple(self._values, tuples of expressions for children artists)

        where self._values is tuple(expressions for own artists).

    """

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 sympy_object: Any, name: str | None = None):
        self._children = []
        if not isinstance(inertial_frame, ReferenceFrame):
            raise TypeError("'inertial_frame' should be a valid ReferenceFrame object.")
        if not isinstance(zero_point, Point):
            raise TypeError("'zero_point' should be a valid Point object.")
        self._inertial_frame = inertial_frame
        self._zero_point = zero_point
        self._sympy_object = sympy_object
        self.name = name
        self._artists = []
        self._values = []
        self.visible = True

    def __repr__(self):
        """Representation showing some basic information of the instance."""
        return (
            f"{self.__class__.__name__}(inertia_frame={self.inertial_frame}, "
            f"zero_point={self.zero_point}, origin={self.origin}, name={self.name})")

    def __str__(self):
        return self.name

    @property
    def inertial_frame(self) -> ReferenceFrame:
        """The reference frame with respect to which the object is oriented."""
        return self._inertial_frame

    @property
    def zero_point(self) -> Point:
        """The absolute origin with respect to which the object is positioned."""
        return self._zero_point

    @property
    def sympy_object(self) -> Any:
        """The absolute origin with respect to which the object is positioned."""
        return self._sympy_object

    @property
    def name(self) -> str:
        """Name of the plot object. Default is the name of the object being plotted."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = str(name)

    @property
    @abstractmethod
    def visible(self) -> bool:
        """If the object is be visible in the plot.

        Notes
        -----
        Subclasses should also implement the setter of this property.
        """
        return self._visible

    @property
    def children(self) -> tuple[PlotBase, ...]:
        """Child objects in the plot hierarchy."""
        return tuple(self._children)

    @property
    def artists(self) -> tuple[ArtistBase, ...]:
        """Artists used to plot the object."""
        return tuple(a for a, _ in self._artists) + tuple(
            a for child in self._children for a in child.artists)

    @property
    def values(self) -> tuple:
        """List of evaluated values for the object's variables."""
        return (self._values, *(child.values for child in self._children))

    @values.setter
    def values(self, values: tuple):
        self._values = values[0]
        for child, vals in zip(self._children, values[1:]):
            child.values = vals

    def get_expressions_to_evaluate(self) -> tuple:
        """Return a tuple of the necessary expressions for plotting."""
        return (tuple(expr for _, expr in self._artists), *tuple(
            child.get_expressions_to_evaluate() for child in self._children))

    def add_artist(self, artist: ArtistBase, exprs: Expr | tuple[Expr, ...]):
        """Add an artist to the plot object.

        Parameters
        ----------
        artist : ArtistBase
            The artist to be added.
        exprs : expression or tuple of expressions
            Args used to update the artist in the form of expressions.
        """
        if not isinstance(artist, ArtistBase):
            raise TypeError("'artist' should be a valid Artist object.")
        if not iterable(exprs):
            exprs = (exprs,)
        self._artists.append((artist, tuple(exprs)))

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        """Plot the associated plot objects."""

    def get_sympy_object_exprs(self) -> tuple[Any, ...]:
        """Return the expressions used in plotting the sympy object."""
        return ()

    def update(self) -> None:
        """Update the objects on the scene, based on the currect values."""
        for args, (artist, _) in zip(self._values, self._artists):
            artist.update_data(*args)
        for child in self._children:
            child.update()

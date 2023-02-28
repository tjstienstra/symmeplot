from abc import ABC, abstractmethod
from sympy import MatrixBase
from sympy.physics.vector import Vector, ReferenceFrame, Point
from matplotlib.pyplot import gca

__all__ = ['PlotBase']


class PlotBase(ABC):
    """
    Class with the basic attributes and methods for the plot objects.

    Attributes
    ----------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`. Default is
        `zero_point`.
    children : list of PlotBase objects
        Child objects in the plot hierarchy.
    artists : list of matplotlib artists
        Artists corresponding to the object and its children.
    values : list
        list of evaluated values for the object's variables.
    annot_coords : numpy.array
        Coordinate where the annotation text is displayed.
    visible : bool
        If the object is be visible in the plot.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point or Vector, optional
        The origin of the object with respect to the `zero_point`. Default is
        `zero_point`.
    name : str, optional
        Name of the plot object. Default is the name of the object being plotted.

    """

    def __init__(self, inertial_frame, zero_point, origin=None, name=None):
        self._children = []
        self._artists_self = tuple()
        self.inertial_frame = inertial_frame
        self.zero_point = zero_point
        self.origin = origin
        self.visible = True
        self.name = name
        self._values = []

    def __repr__(self):
        """Representation showing some basic information of the instance."""
        return (
            f"{self.__class__.__name__}(inertia_frame={self.inertial_frame}, "
            f"zero_point={self.zero_point}, origin={self.origin}, name={self.name})")

    def __str__(self):
        return self.name

    @property
    def children(self):
        return tuple(self._children)

    @property
    def artists(self):
        return self._artists_self + tuple(
            artist for child in self._children for artist in child.artists)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def inertial_frame(self):
        return self._inertial_frame

    @inertial_frame.setter
    def inertial_frame(self, new_inertial_frame):
        if not isinstance(new_inertial_frame, ReferenceFrame):
            raise TypeError("'inertial_frame' should be a valid ReferenceFrame object.")
        elif hasattr(self, '_inertial_frame'):
            raise NotImplementedError("'inertial_frame' cannot be changed.")
        else:
            for child in self._children:
                child.inertial_frame = new_inertial_frame
            self._inertial_frame = new_inertial_frame

    @property
    def zero_point(self):
        return self._zero_point

    @zero_point.setter
    def zero_point(self, new_zero_point):
        if hasattr(self, '_zero_point') and new_zero_point != self._zero_point:
            raise NotImplementedError("'zero_point' cannot be changed")
        if not isinstance(new_zero_point, Point):
            raise TypeError("'zero_point' should be a valid Point object.")
        else:
            for child in self._children:
                child.zero_point = new_zero_point
            self._zero_point = new_zero_point

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        if new_origin is None:
            new_origin = self.zero_point
        elif isinstance(new_origin, Vector):
            new_origin = self.zero_point.locatenew('', new_origin)
        if isinstance(new_origin, Point):
            for child in self._children:
                child.origin = new_origin
            self._origin = new_origin
        else:
            raise TypeError("'origin' should be a valid Point object.")

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, is_visible):
        for artist in self._artists_self:
            artist.set_visible(is_visible)
        for child in self._children:
            child.visible = bool(is_visible)
        self._visible = bool(is_visible)

    @property
    def values(self):
        return [self._values] + [child.values for child in self._children]

    @values.setter
    def values(self, values):
        self._values = values[0]
        for child, vals in zip(self._children, values[1:]):
            child.values = vals

    @abstractmethod
    def _get_expressions_to_evaluate_self(self):
        """Returns a list of the necessary expressions for plotting."""
        pass

    def get_expressions_to_evaluate(self):
        """Returns a list of the necessary expressions for plotting."""
        return (self._get_expressions_to_evaluate_self(),) + tuple(
            child.get_expressions_to_evaluate() for child in self._children)

    @staticmethod
    def _evalf_list(lst, *args, **kwargs):
        if not hasattr(lst, 'evalf'):
            return [PlotBase._evalf_list(expr, *args, **kwargs) for expr in lst]
        if isinstance(lst, MatrixBase):
            return lst.evalf(*args, **kwargs)
        return lst.evalf(*args, **kwargs)

    def evalf(self, *args, **kwargs):
        """
        Evaluates the expressions describing the object, using the `evalf` function from
        sympy.

        Parameters
        ----------
        *args : tuple, optional
            Arguments that are passed to the :func:`sympy.core.evalf` function.
        **kwargs: dict, optional
            Kwargs that are passed to the :func:`sympy.core.evalf` function.

        """
        expressions = self._get_expressions_to_evaluate_self()
        self._values = self._evalf_list(expressions, *args, **kwargs)
        for child in self._children:
            child.evalf(*args, **kwargs)
        return self.values

    def plot(self, ax=None):
        """
        Adds the object artists to the matplotlib `Axes`. Note that the object should be
        evaluated before plotting with for example the `evalf` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            Axes on which the artist should be added. The default is the active axes.

        """
        if ax is None:
            ax = gca()
        self.update()
        for artist in self._artists_self:
            ax.add_artist(artist)
        artists = self._artists_self
        for child in self._children:
            artists += child.plot(ax)
        return artists

    @abstractmethod
    def _update_self(self):
        pass

    def update(self):
        """Updates the artists parameters, based on a current values."""
        artists = self._update_self()
        for child in self._children:
            artists += child.update()
        return artists

    @property
    @abstractmethod
    def annot_coords(self):
        pass

    def contains(self, event):
        for artist in self.artists:
            if artist.contains(event)[0]:
                return True
        return False

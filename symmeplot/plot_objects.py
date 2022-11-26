from symmeplot.plot_base import PlotBase
from symmeplot.plot_artists import Point3D, Vector3D, Circle3D
from sympy import latex, sympify
from sympy.physics.mechanics import (ReferenceFrame, Vector, Point, Particle,
                                     RigidBody)
from typing import Optional, Union, List, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from sympy import Matrix, Expr
    from matplotlib.pyplot import Artist
    import numpy as np
    TArtist = TypeVar('TArtist', bound=Artist)
    TExpr = TypeVar('TExpr', bound=Expr)

__all__ = ['PlotPoint', 'PlotVector', 'PlotFrame', 'PlotBody']


class PlotPoint(PlotBase):
    """Class for plotting points."""
    point = PlotBase.origin  # Alias of origin

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 point: Optional[Union[Point, Vector]], **kwargs):
        """Initialize a PlotPoint instance.

        Parameters
        ==========
        inertial_frame : ReferenceFrame
            The reference frame with respect to which the object is oriented.
        zero_point : Point
            The absolute origin with respect to which the object is positioned.
        point : Point, Vector, optional
            The point that should be plotted with respect to the ``zero_point``.
            If a ``Vector`` is provided the ``origin`` will be at the tip of the
            vector with respect to the ``zero_point``. The default is the
            ``zero_point``.

        Other Parameters
        ================
        **kwargs : dict, optional
            Kwargs that are parsed to ``mpl_toolkits.mplot3d.art3d.Line3D``, so
            ``color='r'`` will make the plotted point red.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ReferenceFrame, Point
        >>> from symmeplot import PlotPoint
        >>> from matplotlib.pyplot import subplots, pause
        >>> l1, l2, l3 = symbols('l:3')
        >>> subs_zero = {l1: 0, l2: 0, l3: 0}
        >>> subs_move = {l1: 0.2, l2: 0.6, l3: 0.3}
        >>> N, O = ReferenceFrame('N'), Point('O')
        >>> P1 = Point('P1')
        >>> P1.set_pos(O, (l1 * N.x + l2 * N.y + l3 * N.z))
        >>> fig, ax = subplots(subplot_kw={'projection': '3d'})
        >>> P1_plot = PlotPoint(N, O, P1, color='k')
        >>> P1_plot.evalf(subs=subs_zero)
        >>> P1_plot.plot()  # Plot the point
        >>> fig.show()
        >>> pause(2)
        >>> P1_plot.evalf(subs=subs_move)
        >>> P1_plot.update()  # The point will now be on its new position

        """
        super().__init__(inertial_frame, zero_point, point, point.name)
        self._artists_self: Tuple[Point3D] = (Point3D((0, 0, 0), **kwargs),)

    @property
    def point_coords(self) -> 'np.array':
        """Coordinates of the point."""
        return self._values[0]

    @property
    def artist_point(self):
        return self._artists_self[0]

    def _get_expressions_to_evaluate_self(self) -> 'List[Matrix]':
        return [
            self.point.pos_from(self.zero_point).to_matrix(self.inertial_frame)]

    def _update_self(self) -> Tuple[Point3D]:
        self.artist_point.update_data(self.point_coords)
        return self._artists_self

    @property
    def annot_coords(self) -> 'np.array':
        """Coordinates where the annotation text is displayed."""
        return self.point_coords


class PlotVector(PlotBase):
    """Class for plotting vectors."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 vector: Vector, origin: Optional[Union[Point, Vector]] = None,
                 name: Optional[str] = None, style: Optional[str] = 'default',
                 **kwargs):
        """Initialize a PlotVector instance.

        Parameters
        ==========
        inertial_frame : ReferenceFrame
            The reference frame with respect to which the object is oriented.
        zero_point : Point
            The absolute origin with respect to which the object is positioned.
        vector : Vector
            Vector that should be plotted.
        origin : Point, Vector, optional
            The origin of the object itself with respect to the zero_point. If a
            ``Vector`` is provided the ``origin`` will be at the tip of the
            vector with respect to the ``zero_point``. The default is the
            ``zero_point``.
        name : str, optional
            Name of the plot object. The default is the name vector as string.
        style : str, optional
            Reference to what style should be used for plotting the arrow. The
            default style is ``'default'``.
            Available styles:
                None: Default of the Vector3D patch
                'default': Normal black arrow

        Other Parameters
        ================
        **kwargs : dict, optional
            Kwargs that are parsed to ``matplotlib.patches.FancyArrow``, so
            ``color='r'`` will make the plotted arrow red.

        Examples
        ========

        >>> from symmeplot import PlotVector
        >>> from matplotlib.pyplot import subplots
        >>> from sympy.physics.mechanics import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> O_v = O.locatenew('O_v', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        >>> v = 0.4 * N.x + 0.4 * N.y - 0.6 * N.z
        >>> v_plot = PlotVector(N, O, v, O_v, color='r', ls='--')
        >>> fig, ax = subplots(subplot_kw={'projection': '3d'})
        >>> v_plot.evalf()
        >>> v_plot.plot(ax)
        >>> fig.show()

        """
        if name is None:
            name = str(latex(vector))
        super().__init__(inertial_frame, zero_point, origin, name)
        self.vector: Vector = vector
        self._values: list = []  # origin, vector
        self._properties: dict = {}
        self._artists_self: Tuple[Vector3D] = (
            Vector3D((0, 0, 0), (0, 0, 0),
                     **self._get_style_properties(style) | kwargs),)

    @property
    def artist_arrow(self):
        return self._artists_self[0]

    def _get_expressions_to_evaluate_self(self) -> 'List[Matrix]':
        return [self.origin.pos_from(self.zero_point).to_matrix(
            self.inertial_frame), self.vector.to_matrix(self.inertial_frame)]

    def _update_self(self) -> Tuple[Vector3D]:
        self.artist_arrow.update_data(self.origin_coords, self.vector_coords)
        return self._artists_self

    @property
    def vector(self) -> Vector:
        """Returns the internal vector."""
        return self._vector

    @vector.setter
    def vector(self, new_vector):
        """Sets the internal vector."""
        if not isinstance(new_vector, Vector):
            raise TypeError("'vector' should be a valid Vector object.")
        else:
            self._vector = new_vector
            self._values = []

    @property
    def origin_coords(self) -> 'np.array':
        """Coordinates of the vector origin."""
        return self._values[0]

    @property
    def vector_coords(self) -> 'np.array':
        """Coordinates of the vector end point."""
        return self._values[1]

    @property
    def annot_coords(self) -> 'np.array':
        """Coordinates where the annotation text is displayed."""
        return self.origin_coords + self.vector_coords

    def _get_style_properties(self, style: Optional[str]) -> dict:
        """Gets the properties of the vector belonging to a certain style.

        Parameters
        ==========
        style : str, None
            Name of the style or None, if no style should be set.
            Available styles:
                'default' : Normal black arrow
        """
        if style is None:
            return {}
        elif style == 'default':
            return {
                'color': 'k',
                'mutation_scale': 10,
                'arrowstyle': '-|>',
                'shrinkA': 0,
                'shrinkB': 0,
                'picker': 20
            }
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")


class PlotFrame(PlotBase):
    """Class for plotting reference frames."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 frame: ReferenceFrame,
                 origin: Optional[Union[Point, Vector]] = None,
                 style: Optional[str] = 'default', scale: float = 0.1,
                 **kwargs):
        """Initialize a PlotFrame instance.

        Parameters
        ==========
        inertial_frame : ReferenceFrame
            The reference frame with respect to which the object is oriented.
        zero_point : Point
            The absolute origin with respect to which the object is positioned.
        frame : ReferenceFrame
            Reference frame that should be plotted.
        origin : Point, Vector, optional
            The origin of the object itself with respect to the zero_point. If a
            ``Vector`` is provided the ``origin`` will be at the tip of the
            vector with respect to the ``zero_point``. The default is the
            ``zero_point``.
        style : str, optional
            Reference to what style should be used for plotting the frame.
            Styles:
                None: No properties of the vectors will be set
                'default': Nice default frame with as color 'rgb' for xyz
        scale : float, optional
            Lenght of the vectors of the reference frame.

        Other Parameters
        ================
        **kwargs : dict, optional
            Kwargs that are parsed to ``PlotVector``s, which parses them to
            ``matplotlib.patches.FancyArrow``, so ``color='r'`` will make all
            vectors of the reference frame red.

        Examples
        ========

        >>> from symmeplot import PlotFrame
        >>> from matplotlib.pyplot import subplots
        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, N.z, 1)
        >>> N0 = Point('N_0')
        >>> A0 = N0.locatenew('A_0', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        >>> N_plot = PlotFrame(N, N0, N, scale=0.5)
        >>> A_plot = PlotFrame(N, N0, A, A0, scale=0.5, ls='--')
        >>> fig, ax = subplots(subplot_kw={'projection': '3d'})
        >>> N_plot.evalf()
        >>> A_plot.evalf()
        >>> N_plot.plot(ax)
        >>> A_plot.plot(ax)
        >>> fig.show()

        """
        super().__init__(inertial_frame, zero_point, origin, frame.name)
        self.frame: ReferenceFrame = frame
        properties = self._get_style_properties(style)
        for prop in properties:
            prop.update(kwargs)
        for vector, prop in zip(frame, properties):
            self._children.append(PlotVector(
                inertial_frame, zero_point, scale * vector, origin, **prop))

    def _get_expressions_to_evaluate_self(self) -> 'List[TExpr]':
        # Children are handled in PlotBase.get_expressions_to_evaluate_self
        return []

    def _update_self(self) -> 'Tuple[TArtist]':
        return self._artists_self  # Children are handled in PlotBase.update

    @property
    def frame(self) -> ReferenceFrame:
        """Return the internal frame."""
        return self._frame

    @frame.setter
    def frame(self, new_frame):
        if not isinstance(new_frame, ReferenceFrame):
            raise TypeError("'frame' should be a valid ReferenceFrame object.")
        else:
            self._frame = new_frame
            self._values = []

    @property
    def vectors(self) -> List[PlotVector]:
        """Returns the plot vectors out of which the frame consists."""
        return self._children

    @property
    def annot_coords(self) -> 'np.array':
        """Coordinates where the annotation text is displayed."""
        return self.vectors[0].origin_coords + 0.3 * sum([v.vector_coords for v in self.vectors])

    @property
    def x(self) -> PlotVector:
        """Returns PlotVector of the x-vector of the frame."""
        return self.vectors[0]

    @property
    def y(self) -> PlotVector:
        """Returns PlotVector of the y-vector of the frame."""
        return self.vectors[1]

    @property
    def z(self) -> PlotVector:
        """Returns PlotVector of the z-vector of the frame."""
        return self.vectors[2]

    def _get_style_properties(self, style: Optional[str]) -> List[dict]:
        """Gets the properties of the vectors belonging to a certain style.

        Parameters
        ==========
        style : str, None
            Name of the style or None, if no style should be set.
            Available styles:
                'default' : Uses the default vectors, overwriting the colors to
                rgb for xyz

        """
        properties = [{}, {}, {}]
        if style is None:
            return properties
        elif style == 'default':
            colors = 'rgb'
            for color, prop in zip(colors, properties):
                prop.update({
                    'style': 'default',
                    'color': color
                })
            return properties
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")


class PlotBody(PlotBase):
    """Class for plotting bodies."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 body: Union[Particle, RigidBody],
                 style: Optional[str] = 'default', plot_frame_properties=None,
                 plot_point_properties=None, **kwargs):
        """Initialize a PlotFrame instance.

        Parameters
        ==========
        inertial_frame : ReferenceFrame
            The reference frame with respect to which the object is oriented.
        zero_point : Point
            The absolute origin with respect to which the object is positioned.
        frame : ReferenceFrame
            Reference frame that should be plotted.
        origin : Point, Vector, optional
            The origin of the object itself with respect to the zero_point. If a
            ``Vector`` is provided the ``origin`` will be at the tip of the
            vector with respect to the ``zero_point``. The default is the
            ``zero_point``.
        style : str, optional
            Reference to what style should be used for plotting the frame.
            Styles:
                None: No properties of the vectors will be set
                'default': Nice default frame with as color 'rgb' for xyz
        scale : float, optional
            Lenght of the vectors of the reference frame.

        Other Parameters
        ================
        **kwargs : dict, optional
            Kwargs that are parsed to ``PlotVector``s, which parses them to
            ``matplotlib.patches.FancyArrow``, so ``color='r'`` will make all
            vectors of the reference frame red.

        Examples
        ========

        >>> from symmeplot import PlotBody
        >>> from matplotlib.pyplot import subplots
        >>> from sympy.physics.mechanics import Point, ReferenceFrame, RigidBody
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, N.z, 1)
        >>> N0 = Point('N_0')
        >>> A0 = N0.locatenew('A_0', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        >>> ground = RigidBody('ground', N0, N, 1, (N.x.outer(N.x), N0))
        >>> body = RigidBody('body', A0, A, 1, (A.x.outer(A.x), A0))
        >>> ground_plot = PlotBody(N, N0, ground)
        >>> body_plot = PlotBody(N, N0, body)
        >>> body_plot.attach_circle(body.masscenter, 0.3, A.x + A.y + A.z,
        ...                         facecolor='none', edgecolor='k')
        >>> fig, ax = subplots(subplot_kw={'projection': '3d'})
        >>> ground_plot.evalf()
        >>> body_plot.evalf()
        >>> ground_plot.plot(ax)
        >>> body_plot.plot(ax)
        >>> fig.show()

        """
        super().__init__(inertial_frame, zero_point, body.masscenter, str(body))
        self.body: Union[Particle, RigidBody] = body
        properties = self._get_style_properties(style)
        if plot_frame_properties is not None:
            properties[0].update(plot_frame_properties)
        if plot_point_properties is not None:
            properties[1].update(plot_point_properties)
        for prop in properties:
            prop.update(kwargs)
        if hasattr(body, 'frame'):
            self._children.append(PlotFrame(
                inertial_frame, zero_point, body.frame, body.masscenter,
                **properties[0]))
        self._children.append(PlotPoint(
            inertial_frame, zero_point, body.masscenter,
            **properties[1]))
        self._expressions_self: list = []

    def _get_expressions_to_evaluate_self(self) -> list:
        return self._expressions_self

    def _update_self(self) -> 'Tuple[TArtist]':
        for artist, values in zip(self._artists_self, self._values):
            artist.update_data(*values)
        return self._artists_self  # Children are handled in PlotBase.update

    def attach_circle(self, center: Optional[Union[Point, Vector]],
                      radius: float, normal: Optional[Union[Point, Vector]],
                      **kwargs):
        """Attaches a circle to a point to represent the body.

        Parameters
        ==========
        center : Point, Vector
            Center of the circle.
        radius : Sympifyable
            Radius of the circle.
        normal : Vector
            Normal of the circle.

        """
        if isinstance(center, Point):
            center = center.pos_from(self.zero_point)
        if isinstance(center, Vector):
            center = center.to_matrix(self.inertial_frame)
        else:
            raise TypeError(f"'center' should be a {type(Point)}.")
        if isinstance(normal, Vector):
            normal = normal.to_matrix(self.inertial_frame)
        else:
            raise TypeError(f"'center' should be a {type(Vector)}.")
        self._artists_self += (Circle3D((0, 0, 0), 0, (0, 0, 1), **kwargs),)
        self._expressions_self.append((center, sympify(radius), normal))

    @property
    def body(self) -> Union[Particle, RigidBody]:
        """Return the internal body."""
        return self._body

    @body.setter
    def body(self, body):
        if not isinstance(body, (Particle, RigidBody)):
            raise TypeError("'body' should be a sympy body.")
        else:
            self._body = body
            self._values = []

    @property
    def plot_frame(self) -> PlotFrame:
        """Returns the plot frame."""
        if len(self._children) == 2:
            return self._children[0]

    @property
    def plot_masscenter(self) -> PlotPoint:
        """Returns the plot frame."""
        return (self._children[1] if len(self._children) == 2 else
                self._children[2])

    @property
    def annot_coords(self) -> 'np.array':
        """Coordinates where the annotation text is displayed."""
        return self.plot_masscenter.annot_coords

    def _get_style_properties(self, style: Optional[str]) -> List[dict]:
        """Gets the properties of the vectors belonging to a certain style.

        Parameters
        ==========
        style : str, None
            Name of the style or None, if no style should be set.
            Available styles:
                'default' : Uses the default style of all children plot objects.

        """
        properties = [{}, {}]
        if style is None:
            return properties
        elif style == 'default':
            properties[0] = {'style': 'default'}
            properties[1] = {'color': 'k', 'marker': r'$\bigoplus$',
                             'markersize': 8, 'markeredgewidth': .5}
            return properties
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")




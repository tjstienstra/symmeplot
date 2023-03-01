import numpy as np
from sympy import latex, sympify
from sympy.physics.mechanics import Particle, Point, ReferenceFrame, RigidBody, Vector

from symmeplot.plot_artists import Circle3D, Line3D, Vector3D
from symmeplot.plot_base import PlotBase

__all__ = ['PlotPoint', 'PlotLine', 'PlotVector', 'PlotFrame', 'PlotBody']


class PlotPoint(PlotBase):
    """
    A class for plotting a Point in 3D using matplotlib.

    Attributes
    ----------
    point : Point
        The sympy Point, which is being plotted.
    artist_point : Line3D
        Corresponding artist for visualizing the point in matplotlib.
    point_coord : numpy.array
        Coordinate values of the plotted point.

    Other Attributes
    ----------------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`.
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
    point : Point or Vector
        The point or vector that should be plotted with respect to the `zero_point`. If
        a vector is provided, the `origin` will be at the tip of the vector with respect
        to the `zero_point`. If not specified, the default is the `zero_point`.
    style : str, optional
        Reference to what style should be used for plotting the point. The default style
        is `'default'`. Available styles:
        - None: Default of the :class:`mpl_toolkits.mplot3d.art3d.Line3D`.
        - 'default': Normal point.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        `color='r'` will make the plotted point red.

    Examples
    --------

    .. jupyter-execute::

        from sympy import symbols
        from sympy.physics.mechanics import ReferenceFrame, Point
        from symmeplot import PlotPoint
        from matplotlib.pyplot import subplots, pause
        l1, l2, l3 = symbols('l:3')
        subs_zero = {l1: 0, l2: 0, l3: 0}
        subs_move = {l1: 0.2, l2: 0.6, l3: 0.3}
        N, O = ReferenceFrame('N'), Point('O')
        P1 = Point('P1')
        P1.set_pos(O, (l1 * N.x + l2 * N.y + l3 * N.z))
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        P1_plot = PlotPoint(N, O, P1, color='k')
        P1_plot.evalf(subs=subs_zero)
        P1_plot.plot()  # Plot the point
        P1_plot.evalf(subs=subs_move)
        P1_plot.update()  # The point will now be on its new position

    """
    point = PlotBase.origin  # Alias of origin

    def __init__(self, inertial_frame, zero_point, point, style='default', **kwargs):
        super().__init__(inertial_frame, zero_point, point, point.name)
        self._artists_self = (
            Line3D([0], [0], [0], **self._get_style_properties(style) | kwargs),)

    @property
    def point_coord(self):
        return self._values[0]

    @property
    def artist_point(self):
        return self._artists_self[0]

    def _get_expressions_to_evaluate_self(self):
        return tuple(
            self.point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:]),

    def _update_self(self):
        self.artist_point.update_data(*[[c] for c in self.point_coord])
        return self._artists_self

    @property
    def annot_coords(self):
        return self.point_coord

    def _get_style_properties(self, style):
        """Gets the properties of the vector belonging to a certain style."""
        if style is None:
            return {}
        elif style == 'default':
            return {'marker': 'o'}
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")


class PlotLine(PlotBase):
    """
    A class for plotting lines in 3D using matplotlib.

    Attributes
    ----------
    points : list of Point
        The points that spawn the line, plotted with respect to the `zero_point`.
    artist_points : Line3D
        Corresponding artist for visualizing the line in matplotlib.
    point_coords : numpy.array
        Coordinate values of the plotted line.

    Other Attributes
    ----------------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`.
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
    points : list of Point or Vector
        The points or vectors through which the line should be plotted with respect to
        the `zero_point`. If a vector is provided, the `origin` will be at the tip of
        the vector with respect to the `zero_point`.
    name : str, optional
        The name of the line. Default is `None`.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        `color='r'` will make the plotted point red.

    Examples
    --------

    .. jupyter-execute::

        from sympy import symbols
        from sympy.physics.mechanics import ReferenceFrame, Point
        from symmeplot import PlotLine
        from matplotlib.pyplot import subplots, pause
        l1, l2, l3 = symbols('l:3')
        subs_zero = {l1: 0, l2: 0, l3: 0}
        subs_move = {l1: 0.2, l2: 0.6, l3: 0.3}
        N, O = ReferenceFrame('N'), Point('O')
        P1 = Point('P1')
        P1.set_pos(O, (l1 * N.x + l2 * N.y + l3 * N.z))
        P2 = P1.locatenew('P2', -0.3 * N.x)
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        line_plot = PlotLine(N, O, [O, P1, P2], color='k')
        line_plot.evalf(subs=subs_zero)
        line_plot.plot()  # Plot the point
        line_plot.evalf(subs=subs_move)
        line_plot.update()  # The point will now be on its new position

    """

    def __init__(self, inertial_frame, zero_point, points, name=None, **kwargs):
        super().__init__(inertial_frame, zero_point, points[0], name)
        self._artists_self = (Line3D([0], [0], [0], **kwargs),)
        self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if isinstance(points, Point):
            self._points = (points,)
            self._values = ()
            return
        _points = []
        for point in points:
            if not isinstance(point, Point):
                raise TypeError("'points' should be a list of Point objects.")
            _points.append(point)
        self._points = tuple(_points)
        self._values = ()

    @property
    def coordinates(self):
        return self._values

    @property
    def artist_points(self):
        return self._artists_self[0]

    def _get_expressions_to_evaluate_self(self):
        vs = []
        for point in self.points:
            vs.append(point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:])
        arr = np.array(vs, dtype=object).T
        return tuple(map(tuple, arr))

    def _update_self(self):
        self.artist_points.update_data(*self.coordinates)
        return self._artists_self

    @property
    def annot_coords(self):
        return np.array(self._values, dtype=np.float64).mean(axis=1)


class PlotVector(PlotBase):
    """
    A class for plotting a Vector in 3D using matplotlib.

    Attributes
    ----------
    vector : Vector
        The sympy Vector, which is being plotted.
    artist_arrow : Vector3D
        Corresponding artist for visualizing the vector in matplotlib.
    origin_coords : numpy.array
        Coordinate values of the origin of the plotted vector.
    vector_coords : numpy.array
        Coordinate values of the tip of the plotted vector.

    Other Attributes
    ----------------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`.
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
    vector : Vector
        The vector that should be plotted with respect to the `zero_point`.
    origin : Point or Vector, optional
        The origin of the vector with respect to the `zero_point`. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the `origin` is at the
        tip of the vector with respect to the `zero_point`. Default is `zero_point`.
    name : str
        Name of the plot object. Default is the vector as string.
    style : str, optional
        Reference to what style should be used for plotting the vector. The default
        style is `'default'`. Available styles:
        - None: Default of the :class:`mpl_toolkits.mplot3d.art3d.Line3D`.
        - 'default': Normal black arrow.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so
        `color='r'` will make the plotted arrow red.

    Examples
    --------

    .. jupyter-execute::

        from symmeplot import PlotVector
        from matplotlib.pyplot import subplots
        from sympy.physics.mechanics import Point, ReferenceFrame
        N = ReferenceFrame('N')
        O = Point('O')
        O_v = O.locatenew('O_v', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        v = 0.4 * N.x + 0.4 * N.y - 0.6 * N.z
        v_plot = PlotVector(N, O, v, O_v, color='r', ls='--')
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        v_plot.evalf()
        v_plot.plot(ax)

    """

    def __init__(self, inertial_frame, zero_point, vector, origin=None, name=None,
                 style='default', **kwargs):
        if name is None:
            name = str(latex(vector))
        super().__init__(inertial_frame, zero_point, origin, name)
        self.vector = vector
        self._values = []  # origin, vector
        self._properties = {}
        self._artists_self = (
            Vector3D((0, 0, 0), (0, 0, 0),
                     **self._get_style_properties(style) | kwargs),)

    @property
    def artist_arrow(self):
        return self._artists_self[0]

    def _get_expressions_to_evaluate_self(self):
        return (
            tuple(self.origin.pos_from(self.zero_point).to_matrix(self.inertial_frame)[
                  :]),
            tuple(self.vector.to_matrix(self.inertial_frame)[:]))

    def _update_self(self):
        self.artist_arrow.update_data(self.origin_coords, self.vector_coords)
        return self._artists_self

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, new_vector):
        if not isinstance(new_vector, Vector):
            raise TypeError("'vector' should be a valid Vector object.")
        else:
            self._vector = new_vector
            self._values = []

    @property
    def origin_coords(self):
        return self._values[0]

    @property
    def vector_coords(self):
        return self._values[1]

    @property
    def annot_coords(self):
        return self.origin_coords + self.vector_coords

    def _get_style_properties(self, style):
        """
        Gets the properties of the vector belonging to a certain style.

        Parameters
        ----------
        style : str or None
            Name of the style or None, if no style should be set. Available styles:
            - 'default' : Normal black arrow

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
    """
    A class for plotting a ReferenceFrame in 3D using matplotlib.

    Attributes
    ----------
    frame : ReferenceFrame
        The sympy ReferenceFrame, which is being plotted.
    vectors : list of PlotVector
        The :class:`PlotVectors<~.PlotVector>` used to plot the reference frame.
    x : PlotVector
        :class:`~.PlotVector` used for the unit vector in the x direction.
    y : PlotVector
        :class:`~.PlotVector` used for the unit vector in the y direction.
    z : PlotVector
        :class:`~.PlotVector` used for the unit vector in the z direction.

    Other Attributes
    ----------------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`.
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
    frame : ReferenceFrame
        The reference frame that should be plotted.
    origin : Point or Vector, optional
        The origin of the frame with respect to the `zero_point`. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the `origin` is at the
        tip of the vector with respect to the `zero_point`. Default is `zero_point`.
    style : str, optional
        Reference to what style should be used for plotting the frame. The default style
        is `'default'`. Available styles:
        - None: No properties of the vectors will be set.
        - 'default': Nice default frame with as color 'rgb' for xyz.
    scale : float, optional
        Length of the vectors of the reference frame.
    **kwargs : dict, optional
        Kwargs that are parsed to :class:`~.PlotVector`s, which possibly parses them to
        :class:`matplotlib.patches.FancyArrow`, so `color='r'` will make all vectors of
        the reference frame red.

    Examples
    --------

    .. jupyter-execute::

        from symmeplot import PlotFrame
        from matplotlib.pyplot import subplots
        from sympy.physics.vector import Point, ReferenceFrame
        N = ReferenceFrame('N')
        A = ReferenceFrame('A')
        A.orient_axis(N, N.z, 1)
        N0 = Point('N_0')
        A0 = N0.locatenew('A_0', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        N_plot = PlotFrame(N, N0, N, scale=0.5)
        A_plot = PlotFrame(N, N0, A, A0, scale=0.5, ls='--')
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        N_plot.evalf()
        A_plot.evalf()
        N_plot.plot(ax)
        A_plot.plot(ax)

    """

    def __init__(self, inertial_frame, zero_point, frame, origin=None, style='default',
                 scale=0.1, **kwargs):
        super().__init__(inertial_frame, zero_point, origin, frame.name)
        self.frame = frame
        properties = self._get_style_properties(style)
        for prop in properties:
            prop.update(kwargs)
        for vector, prop in zip(frame, properties):
            self._children.append(
                PlotVector(inertial_frame, zero_point, scale * vector, origin, **prop))

    def _get_expressions_to_evaluate_self(self):
        # Children are handled in PlotBase.get_expressions_to_evaluate_self
        return ()

    def _update_self(self):
        return self._artists_self  # Children are handled in PlotBase.update

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, new_frame):
        if not isinstance(new_frame, ReferenceFrame):
            raise TypeError("'frame' should be a valid ReferenceFrame object.")
        else:
            self._frame = new_frame
            self._values = []

    @property
    def vectors(self):
        return self._children

    @property
    def annot_coords(self):
        return self.vectors[0].origin_coords + 0.3 * sum(
            [v.vector_coords for v in self.vectors])

    @property
    def x(self):
        return self.vectors[0]

    @property
    def y(self):
        return self.vectors[1]

    @property
    def z(self):
        return self.vectors[2]

    def _get_style_properties(self, style):
        """
        Gets the properties of the vectors belonging to a certain style.

        style : str, None
            Name of the style or None, if no style should be set. Available styles:
            - 'default' : Uses the default vectors, overwriting the colors to rgb for
            xyz.

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
    """
    A class for plotting a body in 3D using matplotlib.

    Attributes
    ----------
    body : RigidBody or Particle
        The sympy body, which is being plotted.
    plot_frame : ReferenceFrame
        :class:`~.PlotFrame` used for plotting the reference frame of the body.
    plot_masscenter : PlotVector
        :class:`~.PlotPoint` used for plotting the center of mass of the body.

    Other Attributes
    ----------------
    name : str
        Name of the plot object. Default is the name of the object being plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    origin : Point
        The origin of the object with respect to the `zero_point`.
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
    body : RigidBody or Particle
        The body that should be plotted.
    origin : Point or Vector, optional
        The origin of the frame with respect to the `zero_point`. If a
        :class:`sympy.physics.vector.vector.Vector` is provided the `origin` is at the
        tip of the vector with respect to the `zero_point`. Default is `zero_point`.
    style : str, optional
        Reference to what style should be used for plotting the body. The default style
        is `'default'`. Available styles:
        - None: No properties of the vectors will be set.
        - 'default': Uses a special point for the center of mass and a frame with as
        color 'rgb' for xyz.
    plot_frame_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`~.PlotFrame`.
    plot_point_properties : dict, optional
        Dictionary of keyword arguments that should be parsed to the
        :class:`~.PlotPoint` representing the center of mass.
    **kwargs : dict, optional
        Kwargs that are parsed to both internally used plot objects.

    Examples
    --------

    .. jupyter-execute::

        from symmeplot import PlotBody
        from matplotlib.pyplot import subplots
        from sympy.physics.mechanics import Point, ReferenceFrame, RigidBody
        N = ReferenceFrame('N')
        A = ReferenceFrame('A')
        A.orient_axis(N, N.z, 1)
        N0 = Point('N_0')
        A0 = N0.locatenew('A_0', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        ground = RigidBody('ground', N0, N, 1, (N.x.outer(N.x), N0))
        body = RigidBody('body', A0, A, 1, (A.x.outer(A.x), A0))
        ground_plot = PlotBody(N, N0, ground)
        body_plot = PlotBody(N, N0, body)
        body_plot.attach_circle(body.masscenter, 0.3, A.x + A.y + A.z,
                                facecolor='none', edgecolor='k')
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        ground_plot.evalf()
        body_plot.evalf()
        ground_plot.plot(ax)
        body_plot.plot(ax)

    """

    def __init__(self, inertial_frame, zero_point, body, style='default',
                 plot_frame_properties=None,
                 plot_point_properties=None, **kwargs):
        super().__init__(inertial_frame, zero_point, body.masscenter, str(body))
        self.body = body
        properties = self._get_style_properties(style)
        if plot_frame_properties is not None:
            properties[0].update(plot_frame_properties)
        if plot_point_properties is not None:
            properties[1].update(plot_point_properties)
        for prop in properties:
            prop.update(kwargs)
        if hasattr(body, 'frame'):
            self._children.append(
                PlotFrame(inertial_frame, zero_point, body.frame, body.masscenter,
                          **properties[0]))
        self._children.append(
            PlotPoint(inertial_frame, zero_point, body.masscenter, **properties[1]))
        self._expressions_self = ()

    def _get_expressions_to_evaluate_self(self):
        return self._expressions_self

    def _update_self(self):
        for artist, values in zip(self._artists_self, self._values):
            artist.update_data(*values)
        return self._artists_self  # Children are handled in PlotBase.update

    def attach_circle(self, center, radius, normal, **kwargs):
        """
        Attaches a circle to a point to represent the body.

        Parameters
        ----------
        center : Point or Vector
            Center of the circle.
        radius : Sympifyable
            Radius of the circle.
        normal : Vector
            Normal of the circle.

        Returns
        -------
        :class:`symmeplot.plot_artists.Circle3D`
            Circle artist.

        """
        if isinstance(center, Point):
            center = center.pos_from(self.zero_point)
        if isinstance(center, Vector):
            center = tuple(center.to_matrix(self.inertial_frame)[:])
        else:
            raise TypeError(f"'center' should be a {type(Point)}.")
        if isinstance(normal, Vector):
            normal = tuple(normal.to_matrix(self.inertial_frame)[:])
        else:
            raise TypeError(f"'center' should be a {type(Vector)}.")
        self._artists_self += (Circle3D((0, 0, 0), 0, (0, 0, 1), **kwargs),)
        self._expressions_self += ((center, sympify(radius), normal),)
        return self._artists_self[-1]

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, body):
        if not isinstance(body, (Particle, RigidBody)):
            raise TypeError("'body' should be a sympy body.")
        else:
            self._body = body
            self._values = []

    @property
    def plot_frame(self):
        if len(self._children) == 2:
            return self._children[0]

    @property
    def plot_masscenter(self):
        return self._children[1] if len(self._children) == 2 else self._children[2]

    @property
    def annot_coords(self):
        return self.plot_masscenter.annot_coords

    def _get_style_properties(self, style):
        """
        Gets the properties of the vectors belonging to a certain style.

        Parameters
        ----------
        style : str, None
            Name of the style or None, if no style should be set. Available styles:
            - 'default' : Uses the default style of all children plot objects.

        """
        properties = [{}, {}]
        if style is None:
            return properties
        elif style == 'default':
            properties[0] = {'style': 'default'}
            properties[1] = {'color': 'k', 'marker': r'$\bigoplus$', 'markersize': 8,
                             'markeredgewidth': .5,
                             'zorder': 10}
            return properties
        else:
            raise NotImplementedError(f"Style '{style}' is not implemented.")

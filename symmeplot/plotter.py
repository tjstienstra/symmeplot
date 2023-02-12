from sympy import lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Vector, Particle, RigidBody
from mpl_toolkits.mplot3d.proj3d import proj_transform
from symmeplot.plot_objects import PlotPoint, PlotLine, PlotVector, PlotFrame, PlotBody
from symmeplot.plot_base import PlotBase
import numpy as np


class SymMePlotter(PlotBase):
    """Class for plotting sympy mechanics in matplotlib.

    Attributes
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes used by the plotter.
    plot_objects : list of PlotPoint, PlotVector, PlotFrame and PlotBody
        List of all plot objects.

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
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Axes on which the sympy mechanics should be plotted.
    inertial_frame : ReferenceFrame
        The reference frame with respect to which all objects will be oriented.
    origin : Point
        The absolute origin with respect to which all objects will be positioned.
    **inertial_frame_properties : dict, optional
        Keyword arguments are parsed to :class:`~.PlotFrame` representing the inertial reference frame.

    Examples
    --------

    .. jupyter-execute::

        from symmeplot import SymMePlotter
        from matplotlib.pyplot import subplots
        from sympy.physics.vector import Point, ReferenceFrame
        N = ReferenceFrame('N')
        A = ReferenceFrame('A')
        A.orient_axis(N, N.z, 1)
        N0 = Point('N_0')
        v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
        A0 = N0.locatenew('A_0', v)
        fig, ax = subplots(subplot_kw={'projection': '3d'})
        plotter = SymMePlotter(ax, N, N0, scale=0.5)
        plotter.add_vector(v)
        plotter.add_frame(A, A0, ls='--')
        plotter.evalf()
        plotter.plot()

    """

    origin = PlotBase.zero_point

    def __init__(self, ax, inertial_frame, origin, **inertial_frame_properties):
        if not hasattr(ax, 'get_zlim'):
            raise TypeError('The axes should be a 3d axes')
        super().__init__(inertial_frame, origin, origin)
        self._ax = ax
        self.add_frame(inertial_frame, **inertial_frame_properties)
        self.annot = self._ax.text2D(0, 0, '', bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1), transform=None)
        self.annot.set_visible(False)
        self.annot_location = 'mouse'
        self._ax.figure.canvas.mpl_connect("motion_notify_event", self._hover)
        self._lambdified_system = SymMePlotter._system_not_lambdified_error

    @staticmethod
    def _system_not_lambdified_error(*args):
        raise ValueError("System has not been lambdified. Use 'SymMePlotter.lambdify_system' to lambdify the system.")

    def _get_expressions_to_evaluate_self(self):
        # Children are handled in PlotBase.get_expressions_to_evaluate_self
        return ()

    def _update_self(self):
        return []  # Children are handled in PlotBase.update

    @property
    def axes(self):
        return self._ax

    @property
    def plot_objects(self):
        return self.children

    @property
    def annot_location(self):
        return self._annot_location

    @annot_location.setter
    def annot_location(self, new_annot_location):
        if new_annot_location == 'object' or new_annot_location == 'mouse':
            self._annot_location = new_annot_location
        else:
            raise NotImplementedError(
                f"Annotation location '{new_annot_location}' has not been "
                f"implemented.")

    @property
    def annot_coords(self):
        return self.annot.get_position()

    def get_plot_object(self, sympy_object):
        """
        Return the `plot_object` based on a sympy object.
        For example `ReferenceFrame('N')` will give the `PlotFrame` of that reference frame if it is added.
        If the object has not been added, it will return `None`.

        Parameters
        ----------
        sympy_object : Point or Vector or ReferenceFrame or Particle or RigidBody or str
            SymPy object to search for. If it is a string it will search for the name.

        Returns
        -------
        PlotPoint or PlotVector or PlotFrame or PlotBody or None
            Retrieved plot object.

        """
        if isinstance(sympy_object, ReferenceFrame):
            for plot_object in self.plot_objects:
                if (isinstance(plot_object, PlotFrame) and
                        sympy_object is plot_object.frame):
                    return plot_object
        elif isinstance(sympy_object, Point):
            for plot_object in self.plot_objects:
                if (isinstance(plot_object, PlotPoint) and
                        sympy_object is plot_object.point):
                    return plot_object
        elif isinstance(sympy_object, (Particle, RigidBody)):
            for plot_object in self.plot_objects:
                if (isinstance(plot_object, PlotBody) and
                        sympy_object is plot_object.body):
                    return plot_object
        elif isinstance(sympy_object, Vector):
            for plot_object in self.plot_objects:
                if (isinstance(plot_object, PlotVector) and
                        sympy_object == plot_object.vector):
                    return plot_object
        elif isinstance(sympy_object, str):
            for plot_object in self.plot_objects:
                if sympy_object == str(plot_object):
                    return plot_object
        else:
            raise NotImplementedError(
                f'Sympy object of type {type(sympy_object)} has not been '
                f'implemented.')

    def add_point(self, point, **kwargs):
        """
        Add a sympy Vector to the plotter.

        Parameters
        ----------
        point : Point or Vector
            The point or vector that should be plotted with respect to the `zero_point`. If a vector is provided, the
            `origin` will be at the tip of the vector with respect to the `zero_point`. If not specified, the default is
            the `zero_point`.
        **kwargs : dict, optional
            Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so `color='r'` will make the plotted
            point red.

        Returns
        -------
        PlotPoint
            The added plot object.

        """
        self._children.append(PlotPoint(self.inertial_frame, self.zero_point, point, **kwargs))
        return self._children[-1]

    def add_line(self, points, name=None, **kwargs):
        """
        Add a sympy Vector to the plotter.

        Parameters
        ----------
        points : list of Point or Vector
            The points or vectors through which the line should be plotted with respect to the `zero_point`. If a vector
            is provided, the `origin` will be at the tip of the vector with respect to the `zero_point`.
        name : str, optional
            The name of the line. Default is `None`.
        **kwargs : dict, optional
            Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so `color='r'` will make the plotted
            point red.

        Returns
        -------
        PlotLine
            The added plot object.

        """
        self._children.append(PlotLine(self.inertial_frame, self.zero_point, points, name, **kwargs))
        return self._children[-1]

    def add_vector(self, vector, origin=None, name=None, style='default', **kwargs):
        """
        Add a sympy Vector to the plotter.

        Parameters
        ----------
        vector : Vector
            The vector that should be plotted with respect to the `zero_point`.
        origin : Point or Vector, optional
            The origin of the vector with respect to the `zero_point`. If a :class:`sympy.physics.vector.vector.Vector`
            is provided the `origin` is at the tip of the vector with respect to the `zero_point`. Default is
            `zero_point`.
        name : str
            Name of the plot object. Default is the vector as string.
        style : str, optional
            Reference to what style should be used for plotting the vector. The default style is `'default'`.
            Available styles:
            - None: Default of the Line3D
            - 'default': Normal black arrow
        **kwargs : dict, optional
            Kwargs that are parsed to :class:`mpl_toolkits.mplot3d.art3d.Line3D`, so `color='r'` will make the plotted
            arrow red.

        Returns
        -------
        PlotVector
            The added plot object.

        """
        self._children.append(PlotVector(self.inertial_frame, self.zero_point, vector, origin=origin, name=name,
                                         style=style, **kwargs))
        return self._children[-1]

    def add_frame(self, frame, origin=None, style='default', scale=0.1, **kwargs):
        """
        Add a sympy ReferenceFrame to the plotter.

        Parameters
        ----------
        frame : ReferenceFrame
            The reference frame that should be plotted.
        origin : Point or Vector, optional
            The origin of the frame with respect to the `zero_point`. If a :class:`sympy.physics.vector.vector.Vector`
            is provided the `origin` is at the tip of the vector with respect to the `zero_point`. Default is
            `zero_point`.
        style : str, optional
            Reference to what style should be used for plotting the frame. The default style is `'default'`.
            Available styles:
            - None: No properties of the vectors will be set
            - 'default': Nice default frame with as color 'rgb' for xyz
        scale : float, optional
            Length of the vectors of the reference frame.
        **kwargs : dict, optional
            Kwargs that are parsed to :class:`~.PlotVector`s, which possibly parses them to
            :class:`matplotlib.patches.FancyArrow`, so `color='r'` will make all vectors of the reference frame red.

        Returns
        -------
        PlotFrame
            The added plot object.

        """
        self._children.append(PlotFrame(self.inertial_frame, self.zero_point, frame, origin=origin, style=style,
                                        scale=scale, **kwargs))
        return self._children[-1]

    def add_body(self, body, style='default', plot_frame_properties=None, plot_point_properties=None, **kwargs):
        """
        Add a sympy body to the plotter.

        Parameters
        ----------
        body : RigidBody or Particle
            The body that should be plotted.
        style : str, optional
            Reference to what style should be used for plotting the body. The default style is `'default'`.
            Available styles:
            - None: No properties of the vectors will be set
            - 'default': Uses a special point for the center of mass and a frame with as color 'rgb' for xyz
        plot_frame_properties : dict, optional
            Dictionary of keyword arguments that should be parsed to the :class:`~.PlotFrame`.
        plot_point_properties : dict, optional
            Dictionary of keyword arguments that should be parsed to the :class:`~.PlotPoint` representing the center of
            mass.
        **kwargs : dict, optional
            Kwargs that are parsed to both internally used plot objects.

        Returns
        -------
        PlotBody
            The added plot object.

        """
        self._children.append(PlotBody(self.inertial_frame, self.zero_point, body, style=style,
                                       plot_frame_properties=plot_frame_properties,
                                       plot_point_properties=plot_point_properties, **kwargs))
        return self._children[-1]

    def plot(self, prettify=True, ax_scale=1.5):
        """
        Plots all plot objects.

        Parameters
        ----------
        prettify : bool, optional
            If True prettify the axes. Default is True.
        ax_scale : float, optional
            Makes the axes bigger in the figure. This function is part of prettifying the figure and only works nicely
            if it is the only subplot. Disabled if set to 0. Default is 1.5

        Returns
        -------
        tuple of Artist
            Returns the plotted artists

        """
        artists = ()
        for plot_object in self._children:
            artists += plot_object.plot(self._ax)
        if prettify:
            self._ax.autoscale_view()
            for axis in (self._ax.xaxis, self._ax.yaxis, self._ax.zaxis):
                axis.set_ticklabels([])
                axis.set_ticks_position('none')
            if ax_scale:
                self._ax.set_position([-(ax_scale - 1) / 2, -(ax_scale - 1) / 2, ax_scale, ax_scale])
            self.auto_zoom()
            self._ax.set_aspect('equal', adjustable='box')
        return artists

    def auto_zoom(self, scale=1.1):
        """Auto scale the axis."""
        _artists = self.artists
        if not _artists:
            return
        _min = np.min([artist.min() for artist in _artists], axis=0)
        _max = np.max([artist.max() for artist in _artists], axis=0)
        size = scale * np.max(_max - _min)
        extra = (size - (_max - _min)) / 2
        self._ax.set_xlim(_min[0] - extra[0], _max[0] + extra[0])
        self._ax.set_ylim(_min[1] - extra[1], _max[1] + extra[1])
        self._ax.set_zlim(_min[2] - extra[2], _max[2] + extra[2])
        return _min, _max

    def _get_selected_object(self, event):
        """Gets the `plot_object` where the mouseevent is currently on.
        Returns `None` if no object contains the mouseevent.
        """
        for plot_object in self._children:
            if plot_object.contains(event):
                return plot_object
        return None

    def _update_annot(self, plot_object, event):
        """Updates the annotation to the given `plot_object`."""
        self.annot.set_text(f'${plot_object}$')
        if self.annot_location == 'object':
            x, y, _ = proj_transform(*plot_object.annot_coords,
                                     self._ax.get_proj())
            self.annot.set_position(self._ax.transData.transform((x, y)))
            # self.annot.set_position_3d(plot_object.annot_coords)
        elif self.annot_location == 'mouse':
            self.annot.set_position(self._ax.transData.transform(
                (event.xdata, event.ydata)))

    def _hover(self, event):
        """Shows an annotation if the mouse is hovering over a `plot_object`."""
        if event.inaxes == self._ax:
            plot_object = self._get_selected_object(event)
            if plot_object is not None:
                self._update_annot(plot_object, event)
                self.annot.set_visible(True)
                self._ax.figure.canvas.draw_idle()
            elif self.annot.get_visible():
                self.annot.set_visible(False)
                self._ax.figure.canvas.draw_idle()

    def set_visibility(self, sympy_object, is_visible, raise_error=True):
        """Hides or shows a `plot_object` based on a `sympy_object`.

        sympy_object : Point or Vector or ReferenceFrame or Particle or RigidBody or str
            SymPy object to show or hide.
        is_visible : bool
            If True show `plot_object`, otherwise hide plot_object.
        raise_error : bool, optional
            If plot_object not found raise an error. Default is True.

        """
        plot_object = self.get_plot_object(sympy_object)
        if plot_object is not None:
            plot_object.visible = is_visible
            return
        if raise_error:
            raise ValueError(
                f"PlotObject corresponding to '{sympy_object}' not found.")

    def clear(self):
        """Clears the axes removing all artists known by the instance.
        Only the inertial frame is kept in the plot_objects.
        """
        for plot_object in self._children:
            plot_object.set_visible(False)
        self._children = [self._children[0]]

    def lambdify_system(self, args, modules=None, printer=None, use_imps=True,
                        dummify=False, cse=True):
        """
        Lambdifies the system for faster evaluation using `evaluate_system`. The workings are the same as for the
        lambdify function in sympy.
        """
        self._lambdified_system = lambdify(
            args, self.get_expressions_to_evaluate(), modules=modules,
            printer=printer, use_imps=use_imps, dummify=dummify, cse=cse)
        return self.evaluate_system

    def evaluate_system(self, *args):
        """Evaluates the system using the function created with `lambdify_system`."""
        self.values = self._lambdified_system(*args)

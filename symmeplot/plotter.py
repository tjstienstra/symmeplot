from sympy import lambdify
from sympy.physics.vector import ReferenceFrame, Point, Vector
from mpl_toolkits.mplot3d.proj3d import proj_transform
from symmeplot.plot_objects import PlotPoint, PlotVector, PlotFrame, PlotBody
from symmeplot.plot_base import PlotBase
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D
    from sympy.physics.mechanics import Particle, RigidBody


class SymMePlotter(PlotBase):
    """Class for plotting sympy mechanics in matplotlib."""

    origin = PlotBase.zero_point

    def __init__(self, ax: 'Axes3D', inertial_frame: ReferenceFrame,
                 origin: Point, **inertial_frame_properties):
        """Initialize a PlotFrame instance.

        Parameters
        ==========
        ax : mpl_toolkits.mplot3d.Axes3d
            Axes on which the sympy mechanics should be plotted.
        inertial_frame : ReferenceFrame
            The inertia reference frame with respect to all objects will be
            plotted.
        zero_point : sympy.physics.mechanics.Point
            The absolute origin with respect to which all objects will be
            plotted.

        Other Parameters
        ================
        **inertial_frame_properties : dict, optional
            kwargs are passed as properties to PlotFrame for inertial frame.

        Examples
        ========

        >>> from symmeplot import SymMePlotter
        >>> from matplotlib.pyplot import subplots
        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, N.z, 1)
        >>> N0 = Point('N_0')
        >>> v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
        >>> A0 = N0.locatenew('A_0', v)
        >>> fig, ax = subplots(subplot_kw={'projection': '3d'})
        >>> plotter = SymMePlotter(ax, N, N0, scale=0.5)
        >>> plotter.add_vector(v)
        >>> plotter.add_frame(A, A0, ls='--')
        >>> plotter.plot()
        >>> plotter.evalf()
        >>> fig.show()

        """

        def system_not_lambdified_error(*args):
            raise ValueError(
                "System has not been lambdified. Use"
                "'SymMePlotter.lambdify_system' to lambdify the system.")

        if not hasattr(ax, 'get_zlim'):
            raise TypeError('The axes should be a 3d axes')
        super().__init__(inertial_frame, origin, origin)
        self._ax = ax
        self.add_frame(inertial_frame, **inertial_frame_properties)
        self.annot = self._ax.text2D(0, 0, '',
                                     bbox=dict(boxstyle='round4', fc='linen',
                                               ec='k', lw=1), transform=None)
        self.annot.set_visible(False)
        self.annot_location = 'mouse'
        self.picked = False
        self._ax.figure.canvas.mpl_connect("motion_notify_event", self._hover)
        self._lambdified_system = system_not_lambdified_error

    def _get_expressions_to_evaluate_self(self) -> list:
        # Children are handled in PlotBase.get_expressions_to_evaluate_self
        return []

    def _plot_self(self, ax: 'Axes3D') -> list:
        return []  # Redundant see SymMePlotter.plot

    def _update_self(self) -> list:
        return []  # Children are handled in PlotBase.update

    @property
    def axes(self) -> 'Axes3D':
        """Axes used by the plotter"""
        return self._ax

    @property
    def plot_objects(self) -> list:
        """Returns all plot objects."""
        return self.children

    @property
    def annot_location(self) -> str:
        """Current location type of the annotation."""
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
        """Returns the ``plot_object`` based on a sympy object.
        For example ``ReferenceFrame('N')`` will give the ``PlotFrame`` of that
        reference frame if it is added. If the object has not been added, it
        will return ``None``.

        Parameters
        ==========
        sympy_object : ReferenceFrame, Vector, Point, str
            sympy object to search for. If it is a string it will check for the
            name, ``PlotVector``s can be given different names, while
            ``PlotPoint``s and ``PlotFrame``s use the name of the sympy object
            they are representing.

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
        """Add a sympy Vector to the plotter."""
        self._children.append(
            PlotPoint(self.inertial_frame, self.zero_point, point, **kwargs))
        return self._children[-1]

    def add_vector(self, vector: Vector,
                   origin: Optional[Union[Point, Vector]] = None,
                   name: Optional[str]=None, style: Optional[str] = 'default',
                   **kwargs):
        """Add a sympy Vector to the plotter."""
        self._children.append(
            PlotVector(self.inertial_frame, self.zero_point, vector,
                       origin=origin, name=name, style=style, **kwargs))
        return self._children[-1]

    def add_frame(self, frame: ReferenceFrame,
                  origin: Optional[Union[Point, Vector]] = None,
                  style: Optional[str] = 'default', scale: float = 0.1,
                  **kwargs):
        """Add a sympy ReferenceFrame to the plotter."""
        self._children.append(
            PlotFrame(self.inertial_frame, self.zero_point, frame,
                      origin=origin, style=style, scale=scale, **kwargs))
        return self._children[-1]

    def add_body(self, body: 'Union[Particle, RigidBody]',
                 style: Optional[str] = 'default',
                 plot_frame_properties: Optional[dict] = None,
                 plot_point_properties: Optional[dict] = None, **kwargs):
        """Add a sympy body to the plotter."""
        self._children.append(
            PlotBody(self.inertial_frame, self.zero_point, body, style=style,
                     plot_frame_properties=plot_frame_properties,
                     plot_point_properties=plot_point_properties, **kwargs))
        return self._children[-1]

    def plot(self, prettify: bool = True, ax_scale: float = 1.5) -> list:
        """Plots all plot objects.

        Parameters
        ==========
        prettify : bool, optional
            If True prettify the axes.
            TODO Not working too well yet
        ax_scale : float, optional
            Makes the axes bigger in the figure.
            This function is part of prettifying the figure and only works nicely if it is the only subplot.
            Disabled if set to 0.
        """
        if prettify:
            self._ax.autoscale_view()
            for axis in (self._ax.xaxis, self._ax.yaxis, self._ax.zaxis):
                axis.set_ticklabels([])
                axis.set_ticks_position('none')
            if ax_scale:
                self._ax.set_position([-(ax_scale - 1) / 2, -(ax_scale - 1) / 2, ax_scale, ax_scale])
        artists = []
        for plot_object in self._children:
            artists += plot_object.plot(self._ax)
        return artists

    def _get_selected_object(self, event):
        """Gets the ``plot_object`` where the mouseevent is currently on.
        Returns ``None`` if no object contains the mouseevent.
        """
        for plot_object in self._children:
            if plot_object.contains(event):
                return plot_object
        return None

    def _update_annot(self, plot_object, event):
        """Updates the annotation to the given plot_object."""
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
        """Shows an annotation if the mouse is hovering over a plot_object."""
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
        """Hides or shows a plot_object based on a sympy_object.

        Parameters
        ==========
        sympy_object : ReferenceFrame, Vector, Point
            sympy object to show or hide.

        is_visible : bool
            If True show plot_object, otherwise hide plot_object.
        raise_error : bool
            If plot_object not found raise an error.

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
        """Lambdifies the system for faster evaluation using
        ``evaluate_system``. The workings are the same as for the lambdify
        function in sympy.
        """
        self._lambdified_system = lambdify(
            args, self.get_expressions_to_evaluate(), modules=modules,
            printer=printer, use_imps=use_imps, dummify=dummify, cse=cse)
        return self.evaluate_system

    def evaluate_system(self, *args):
        """Evaluates the system using the function created with
        ``lambdify_system``.
        """
        self.values = self._lambdified_system(*args)

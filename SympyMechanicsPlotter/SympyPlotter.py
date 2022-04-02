from sympy.physics.vector.vector import Vector
from sympy.physics.mechanics import ReferenceFrame, Point

from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


class Vector3D(FancyArrowPatch):
    """TODO: Add some documentation"""
    # Source: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    def __init__(self, origin, vector, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._origin = origin
        self._vector = vector

    def do_3d_projection(self, renderer=None):
        # Source: https://github.com/matplotlib/matplotlib/issues/21688
        xs, ys, zs = proj_transform(*[(o, o + d) for o, d in zip(self._origin, self._vector)], self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


class PlotBase:
    """Class with the basic attributes and methods for several plot classes.

    It stores the: inertial frame, the zero point, which is the reference from which the object will be plotted.
    Besides absolute origin it also stores the origin of the object itself and possible children
    For example in case of a PlotFrame the origin of the frame and each axis is stored as a child.
    """

    def __init__(self, inertial_frame, zero_point, origin=None):
        """Initialize a PlotBase instance.

        Parameters
        ==========
        inertial_frame : sympy.physics.mechanics.ReferenceFrame
            The reference frame of the object.
        zero_point : sympy.physics.mechanics.Point
            The absolute origin from which the position of the object will be determined.
        origin : sympy.physics.mechanics.Point, sympy.physics.mechanics.vector.vector.Vector, optional
            The origin of the object itself, which is taken with respect to the zero_point.
            If None the zero_point will be used as origin.
            If a Vector is given a Point is created at the tip of the vector with respect to the zero_point.
        """
        self._children = []
        self.inertial_frame = inertial_frame
        self.zero_point = zero_point
        self.origin = origin
        self.visible = True

    def __repr__(self):
        """Representation showing some basic information of the instance."""
        return f"{self.__class__.__name__}({self.inertial_frame}, {self.zero_point}, {self.origin})"

    @property
    def inertial_frame(self):
        """Returns the inertial frame of the object."""
        return self._inertial_frame

    @inertial_frame.setter
    def inertial_frame(self, new_inertial_frame):
        """Sets the inertial frame of the object."""
        if not isinstance(new_inertial_frame, ReferenceFrame):
            raise TypeError("'inertial_frame' should be a valid ReferenceFrame object.")
        elif hasattr(self, '_inertial_frame'):
            raise NotImplementedError("Inertial frame already set and it cannot be changed.")
        else:
            for child in self._children:
                child.inertial_frame = new_inertial_frame
            self._inertial_frame = new_inertial_frame

    @property
    def zero_point(self):
        """Returns the zero point of the object."""
        return self._zero_point

    @zero_point.setter
    def zero_point(self, new_zero_point):
        """Sets the zero point of the object."""
        if not isinstance(new_zero_point, Point):
            raise TypeError("'zero_point' should be a valid Point object.")
        else:
            for child in self._children:
                child.zero_point = new_zero_point
            self._zero_point = new_zero_point

    @property
    def origin(self):
        """Returns the origin of the object."""
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        """Sets the origin of the object.

        Parameters
        ==========
        new_origin : sympy.physics.mechanics.Point, sympy.physics.mechanics.vector.vector.Vector
            The origin of the object itself, which is taken with respect to the zero_point.
            If None the zero_point will be used as origin.
            If a Vector is given a Point is created at the tip of the vector with respect to the zero_point.

        """
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
        """Returns if the object is visible in the plot."""
        return self._visible

    @visible.setter
    def visible(self, is_visible):
        """Changes the visibility of the object, including the visibility of its, children."""
        for child in self._children:
            child._visible = bool(is_visible)
        self._visible = bool(is_visible)


class PlotVector(PlotBase):
    """Class for plotting vectors."""

    def __init__(self, inertial_frame, zero_point, vector, origin=None, style='default', name=None, **kwargs):
        """Initialize a PlotVector instance.

        Parameters
        ==========
        inertial_frame : sympy.physics.mechanics.ReferenceFrame
            The reference frame of the vector.
        zero_point : sympy.physics.mechanics.Point
            The absolute origin from which the position of the vector will be determined.
        vector : sympy.physics.mechanics.vector.vector.Vector
            The vector that will be plotted.
        origin : sympy.physics.mechanics.Point, sympy.physics.mechanics.vector.vector.Vector, optional
            The Point on which the vector is attached, which is taken with respect to the zero_point.
            If None the zero_point will be used as origin.
            If a Vector is given a Point is created at the tip of the vector with respect to the zero_point.
        style : str, optional
            Reference to what style should be used for plotting the arrow.
            Styles:
                None: No properties of the matplotlib.patches.FancyArrowPatch will be set
                'default': Normal black arrow

        Other Parameters
        ================
        **kwargs : dict, optional
            kwargs are passed to the matplotlib.patches.FancyArrowPatch,
            so if you for example give color='r' the plotted vector will be red
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html

        Examples
        ========

        >>> from SympyPlotter import PlotVector
        >>> from matplotlib.pyplot import figure
        >>> from sympy.physics.mechanics import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> v_point = O.locatenew('v_{origin}', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        >>> v = 0.4 * N.x + 0.4 * N.y - 0.6 * N.z
        >>> v_plot = PlotVector(N, O, v, v_point, color='r', ls='--')
        >>> fig = figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> v_plot.plot(ax)
        >>> fig.show()

        """
        super().__init__(inertial_frame, zero_point, origin)
        self.vector = vector  # sympy.physics.mechanics.vector.vector.Vector
        self._vector_coords = None  # tuple of floats
        self._origin_coords = None  # tuple of floats
        self._arrow = None
        self._properties = {}
        self.set_style(style)
        self._properties.update(kwargs)
        self.name = name

    def __str__(self):
        return self.name if self.name is not None else str(self.vector)

    @property
    def vector(self):
        """Returns the internal vector."""
        return self._vector

    @vector.setter
    def vector(self, new_vector):
        """Sets the internal vector."""
        if not isinstance(new_vector, Vector):
            raise TypeError("'vector' should be a valid Vector object.")
        else:
            self._vector = new_vector

    @property
    def vector_coords(self):
        """Returns the coordinates of the tip of the vector.
        Raising an error if there are still free symbols, which have not been evaluated.
        """
        return self._vector_coords if self._vector_coords is not None else self._vector_to_coords(self._vector)

    @property
    def origin_coords(self):
        """Returns the coordinates of the origin of the vector.
        Raising an error if there are still free symbols, which have not been evaluated.
        """
        if self._origin_coords is None:
            self._origin_coords = self._vector_to_coords(self.origin.pos_from(self.zero_point))
        return self._origin_coords

    @property
    def properties(self):
        """Returns the plot properties of the vector."""
        return self._properties

    @property
    def annot_coords(self):
        """Returns coordinates where to display the annotation text, see MechanicsPlotter for more information."""
        return self.origin_coords + self.vector_coords

    def set_property(self, name, value, override=True):
        """Method to change a property of the vector to be plotted.

        Parameters
        ==========
        name : str
            Name of the property, corresponding to one of the kwargs of the matplotlib.patches.FancyArrowPatch
        value : Any
            The value the property should be set to.
        override : bool
            Override the property if already set.

        Examples
        ========

        >>> from SympyPlotter import PlotVector
        >>> from matplotlib.pyplot import figure, pause
        >>> from sympy.physics.mechanics import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> v = 0.6 * (N.x + N.y + N.z)
        >>> v_plot = PlotVector(N, O, v)
        >>> fig = figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> v_plot.plot(ax)
        >>> fig.show()
        >>> pause(3)
        >>> v_plot.set_property('color', 'r')
        >>> v_plot.plot(ax)
        """
        if override or name not in self._properties:
            self._properties[name] = value

    def set_style(self, style):
        """Sets a plot style of the vector.

        Parameters
        ==========
        style : str, None
            Name of the style or None, if no style should be set.
            Available styles:
                'default' : Normal black arrow
        """
        if style is None:
            return
        elif style == 'default':
            self._properties.update({
                'color': 'k',
                'mutation_scale': 10,
                'arrowstyle': '-|>',
                'shrinkA': 0,
                'shrinkB': 0,
                'picker': 20
            })
        else:
            raise ValueError('Unknown style.')

    def _vector_to_coords(self, v, *args, **kwargs):
        """Gets the coordinates of the tip of a vector in the inertial frame.

        Parameters
        ==========
        v : sympy.mechanics.vector.vector.Vector
            Vector of from which the coordinates should be calculated.
        *args : Passed to the SymPy evalf function for evaluating the symbols
        *kwargs : Passed to the SymPy evalf function for evaluating the symbols
        """
        mat = v.express(self.inertial_frame).to_matrix(self.inertial_frame).evalf(*args, **kwargs)
        if mat.free_symbols:
            raise ValueError(f'Free symbols {mat.free_symbols} should be substituted before plotting')
        return mat.__array__(float).flatten()

    def contains(self, event):
        """Returns if the mouseevent is currently on the plotted arrow."""
        if self._arrow is not None:
            return self._arrow.contains(event)[0]

    def evalf(self, *args, **kwargs):
        """Evaluate the vector for plotting, using the evalf function from SymPy.

        Parameters
        ==========
        *args : Arguments that are passed to the SymPy evalf function.
        **kwargs : Kwargs that are passed to the SymPy evalf function.

        """
        self._origin_coords = self._vector_to_coords(self.origin.pos_from(self.zero_point), *args, **kwargs)
        self._vector_coords = self._vector_to_coords(self._vector, *args, **kwargs)

    def plot(self, ax):
        """Plots the vector on an axes.
        Removing the old vector if that is still there.

        Parameters
        ==========
        ax : mpl_toolkits.mplot3d.Axes3d
            Axes on which the vector should be plotted.

        """
        if self._arrow is not None:
            self._arrow.remove()
        self._arrow = Vector3D(self.origin_coords, self.vector_coords, **self._properties)
        if self.visible:
            ax.add_artist(self._arrow)


class PlotFrame(PlotBase):
    """Class for plotting reference frames."""

    def __init__(self, inertial_frame, zero_point, frame, origin=None, style='default', scale=0.1, **kwargs):
        """Initialize a PlotFrame instance.

        Parameters
        ==========
        inertial_frame : sympy.physics.mechanics.ReferenceFrame
            The reference frame of the vector.
        zero_point : sympy.physics.mechanics.Point
            The absolute origin from which the position of the frame will be determined.
        frame : sympy.physics.mechanics.ReferenceFrame
            The frame that will be plotted.
        origin : sympy.physics.mechanics.Point, sympy.physics.mechanics.vector.vector.Vector, optional
            The origin of the frame, which is taken with respect to the zero_point.
            If None the zero_point will be used as origin.
            If a Vector is given a Point is created at the tip of the vector with respect to the zero_point.
        style : str, optional
            Reference to what style should be used for plotting the frame.
            Styles:
                None: No properties of the matplotlib.patches.FancyArrowPatch will be set
                'default': Nice default frame with as color 'rgb' for xyz
        scale : float, optional
            Length of the vectors of the reference frame.

        Other Parameters
        ================
        **kwargs : dict, optional
            kwargs are passed to the matplotlib.patches.FancyArrowPatch.
            https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html
            Note that these properties will be applied to all vectors of the reference frame,
            so if you for example give color='r' the all vectors of the reference frame will be red.

        Examples
        ========

        >>> from SympyPlotter import PlotFrame
        >>> from matplotlib.pyplot import figure
        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, N.z, 1)
        >>> N0 = Point('N_0')
        >>> A0 = N0.locatenew('A_0', 0.2 * N.x + 0.2 * N.y + 0.7 * N.z)
        >>> N_plot = PlotFrame(N, N0, N, scale=0.5)
        >>> A_plot = PlotFrame(N, N0, A, A0, scale=0.5, ls='--')
        >>> fig = figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> N_plot.plot(ax)
        >>> A_plot.plot(ax)
        >>> fig.show()

        """
        super().__init__(inertial_frame, zero_point, origin)
        self.frame = frame  # sympy.physics.mechanics.ReferenceFrame
        self._children = [PlotVector(inertial_frame, zero_point, scale * v, origin) for v in frame]
        self.set_style(style)
        for prop in self.properties:
            prop.update(kwargs)

    def __str__(self):
        return str(self.frame)

    @property
    def frame(self):
        """Return the internal frame."""
        return self._frame

    @frame.setter
    def frame(self, new_frame):
        """Sets the internal frame."""
        if not isinstance(new_frame, ReferenceFrame):
            raise TypeError("'frame' should be a valid ReferenceFrame object.")
        else:
            self._frame = new_frame

    @property
    def vectors(self):
        """Returns the plot vectors out of which the frame consists."""
        return self._children

    @property
    def properties(self):
        """Returns a list with for each child plot vector the set properties."""
        return [v.properties for v in self.vectors]

    @property
    def annot_coords(self):
        """Returns coordinates where to display the annotation text, see MechanicsPlotter for more information."""
        return self.vectors[0].origin_coords + 0.3 * sum([v.vector_coords for v in self.vectors])

    @property
    def x(self):
        """Returns PlotVector of the x-vector of the frame."""
        return self.vectors[0]

    @property
    def y(self):
        """Returns PlotVector of the y-vector of the frame."""
        return self.vectors[1]

    @property
    def z(self):
        """Returns PlotVector of the z-vector of the frame."""
        return self.vectors[2]

    def contains(self, event):
        """Returns if the mouseevent is currently on the plotted frame."""
        for v in self.vectors:
            if v.contains(event):
                return True
        return False

    def set_style(self, style):
        """Sets a plot style of the frame.

        Parameters
        ==========
        style : str, None
            Name of the style or None, if no style should be set.
            Available styles:
                'default' : Uses the default vectors, overwriting the colors to rgb for xyz

        """
        if style is None:
            return
        elif style == 'default':
            colors = 'rgb'
            for color, prop in zip(colors, self.properties):
                prop.update({
                    'color': color
                })
        else:
            raise ValueError('Unknown style.')

    def evalf(self, *args, **kwargs):
        """Evaluate the frame for plotting, using the evalf function from SymPy.

        Parameters
        ==========
        *args : Arguments that are passed to the SymPy evalf function.
        **kwargs : Kwargs that are passed to the SymPy evalf function.

        """
        for v in self.vectors:
            v.evalf(*args, **kwargs)

    def plot(self, ax):
        """Plots the frame on an axes.
        Removing the old frame if that is still there.

        Parameters
        ==========
        ax : mpl_toolkits.mplot3d.Axes3d
            Axes on which the frame should be plotted.

        """
        for v in self.vectors:
            v.plot(ax)


class MechanicsPlotter(PlotBase):
    """Class for plotting SymPy mechanics in matplotlib."""

    def __init__(self, ax: Axes3D, inertial_frame: ReferenceFrame, zero_point: Point, **inertial_frame_properties):
        """Initialize a PlotFrame instance.

        Parameters
        ==========
        ax : mpl_toolkits.mplot3d.Axes3d
            Axes on which the SymPy mechanics should be plotted.
        inertial_frame : sympy.physics.mechanics.ReferenceFrame
            The reference frame of the vector.
        zero_point : sympy.physics.mechanics.Point
            The absolute origin from which all frames and vectors are plotted.

        Other Parameters
        ================
        **inertial_frame_properties : kwargs are passed as properties to PlotFrame for inertial frame.

        Examples
        ========

        >>> from SympyPlotter import MechanicsPlotter
        >>> from matplotlib.pyplot import figure
        >>> from sympy.physics.vector import Point, ReferenceFrame
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, N.z, 1)
        >>> N0 = Point('N_0')
        >>> v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
        >>> A0 = N0.locatenew('A_0', v)
        >>> fig = figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> plotter = MechanicsPlotter(ax, N, N0, scale=0.5)
        >>> plotter.add_vector(v)
        >>> plotter.add_frame(A, A0, ls='--')
        >>> plotter.plot()
        >>> fig.show()

        """
        super().__init__(inertial_frame, zero_point, zero_point)
        self._ax = ax
        self.add_frame(inertial_frame, **inertial_frame_properties)
        self.annot = self._ax.text(0, 0, 0, '', bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1))
        self.annot.set_visible(False)
        self.annot_location = 'object'
        self.picked = False
        self._ax.figure.canvas.mpl_connect("motion_notify_event", self.hover)

    @PlotBase.origin.setter
    def origin(self, new_zero_point):
        """Makes sure that that the origin and zero_point of the overall MechanicsPlotter are always te same.
        TODO Better change it to an alias.
        """
        self.zero_point = new_zero_point

    @property
    def plot_objects(self):
        """Returns all current plot objects in a list."""
        return self._children

    @property
    def annot_location(self):
        """Returns the current coordinate for the annotation."""
        return self._annot_location

    @annot_location.setter
    def annot_location(self, new_annot_location):
        """Sets the current location for the annotation.

        Parameters
        ==========
        new_annot_location : str
            Annotation location:
                'object' : Gets annotation location from the plot_object itself.

        """
        if new_annot_location == 'object':
            self._annot_location = new_annot_location
        else:
            raise NotImplementedError(f"Annotation location '{new_annot_location}' is not been implemented.")

    def get_object(self, sympy_object):
        """Returns the plot_object based on a SymPy object.
        For example ReferenceFrame('N') will give the PlotFrame of that reference frame if it is added.
        If the object has not been added, it will return None.

        Parameters
        ==========
        sympy_object : sympy.physics.mechanics.ReferenceFrame, sympy.mechanics.mechanics.Vector, str
            SymPy object to search for.
            If it is a string it will check for the name, PlotVectors can be given different names, while PlotPoints and
            PlotFrames will just use the name given to the Point or ReferenceFrame they are representing.

        """
        if isinstance(sympy_object, ReferenceFrame):
            for plot_object in self.plot_objects:
                if isinstance(plot_object, PlotFrame) and sympy_object is plot_object.frame:
                    return plot_object
        elif isinstance(sympy_object, Vector):
            for plot_object in self.plot_objects:
                if isinstance(plot_object, PlotVector) and sympy_object == plot_object.vector:
                    return plot_object
        elif isinstance(sympy_object, str):
            for plot_object in self.plot_objects:
                if sympy_object == str(plot_object):
                    return plot_object
        else:
            raise TypeError("Object key should be a valid ReferenceFrame or Vector.")

    def add_vector(self, vector, origin=None, name=None, **kwargs):
        """Add a SymPy Vector to the plotter.
        Origin and other kwargs are passed to the PlotVector constructor."""
        self.plot_objects.append(
            PlotVector(self.inertial_frame, self.zero_point, vector, origin=origin, name=name, **kwargs))

    def add_frame(self, frame, origin=None, **kwargs):
        """Add a SymPy ReferenceFrame to the plotter.
        Origin and other kwargs are passed to the PlotFrame constructor."""
        self.plot_objects.append(PlotFrame(self.inertial_frame, self.zero_point, frame, origin=origin, **kwargs))

    def evalf(self, *args, **kwargs):
        """Evaluate all plotobjects for plotting, using the evalf function from SymPy.

        Parameters
        ==========
        *args : Arguments that are passed to the SymPy evalf function.
        **kwargs : Kwargs that are passed to the SymPy evalf function.

        """
        for plot_object in self.plot_objects:
            plot_object.evalf(*args, **kwargs)

    def plot(self, prettify=True, ax_scale=2):
        """Plots all plotobjects.

        Parameters
        ==========
        prettify : bool, optional
            If True prettify the axes.
            TODO Not working well yet
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
        for plot_object in self.plot_objects:
            plot_object.plot(self._ax)

    def get_selected_object(self, event):
        """Gets the plot_object where the mouseevent is currently on.
        Returns None if no object contains the mouseevent.
        """
        for plot_object in self.plot_objects:
            if plot_object.contains(event):
                return plot_object
        return None

    def update_annot(self, plot_object):
        """Updates the annotation to the given plot_object."""
        self.annot.set_text(f'${plot_object}$')
        if self.annot_location == 'object':
            self.annot.set_position_3d(plot_object.annot_coords)

    def hover(self, event):
        """Shows an annotation if the mouseevent is hovering over the plot_object."""
        if event.inaxes == self._ax:
            plot_object = self.get_selected_object(event)
            if plot_object is not None:
                self.update_annot(plot_object)
                self.annot.set_visible(True)
                self._ax.figure.canvas.draw_idle()
            elif self.annot.get_visible():
                self.annot.set_visible(False)
                self._ax.figure.canvas.draw_idle()

    def set_visibility(self, sympy_object, is_visible, raise_error=True):
        """Hides or shows a plot_object based on a sympy_object.

        Parameters
        ==========
        sympy_object :sympy.physics.mechanics.ReferenceFrame, sympy.mechanics.mechanics.Vector
            SymPy object to show or hide.

        is_visible : bool
            If True show plot_object, otherwise hide plot_object.
        raise_error : bool
            If plot_object not found raise an error.

        """
        plot_object = self.get_object(sympy_object)
        if plot_object is not None:
            plot_object.visible = is_visible
            return
        if raise_error:
            raise ValueError("PlotObject corresponding to 'sympy_object' not found.")

    def clear(self):
        """Clears the axes removing all artists known by the instance.
        Only the inertial frame is kept in the plot_objects. All others are removed.
        """
        for plot_object in self.plot_objects:
            plot_object.set_visible(False)
        self._children = [self.plot_objects[0]]

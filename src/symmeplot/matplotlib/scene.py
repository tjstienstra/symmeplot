"""Definition of the scene for the matplotlib backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.proj3d import proj_transform

from symmeplot.core import SceneBase
from symmeplot.matplotlib.plot_objects import (
    PlotBody,
    PlotFrame,
    PlotLine,
    PlotPoint,
    PlotVector,
)

__all__ = ["Scene3D"]

from symmeplot.utilities.utilities import calculate_euler_angles

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.backend_bases import MouseEvent
    from sympy.physics.vector import Point, ReferenceFrame

    from symmeplot.matplotlib.artists import MplArtistBase
    from symmeplot.matplotlib.plot_base import MplPlotBase


class Scene3D(SceneBase):
    """Class for plotting sympy mechanics in matplotlib.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which all objects will be oriented.
    origin : Point
        The absolute origin with respect to which all objects will be positioned.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Axes on which the sympy mechanics should be plotted.
    **inertial_frame_properties : dict, optional
        Keyword arguments are parsed to the
        :class:`symmeplot.matplotlib.plot_objects.PlotFrame` representing the inertial
        reference frame.

    Examples
    --------
    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import sympy.physics.mechanics as me
        from symmeplot.matplotlib import Scene3D

        N = me.ReferenceFrame("N")
        A = me.ReferenceFrame("A")
        A.orient_axis(N, N.z, 1)
        N0 = me.Point("N_0")
        v = 0.2 * N.x + 0.2 * N.y + 0.7 * N.z
        A0 = N0.locatenew("A_0", v)
        scene = Scene3D(N, N0, scale=0.5)
        scene.add_vector(v, name="v")
        scene.add_frame(A, A0, ls="--")
        scene.lambdify_system(())
        scene.evaluate_system()
        scene.plot()

    """

    _PlotPoint: type[MplPlotBase] = PlotPoint
    _PlotLine: type[MplPlotBase] = PlotLine
    _PlotVector: type[MplPlotBase] = PlotVector
    _PlotFrame: type[MplPlotBase] = PlotFrame
    _PlotBody: type[MplPlotBase] = PlotBody

    def __init__(
        self,
        inertial_frame: ReferenceFrame,
        zero_point: Point,
        ax: plt.Axes | None = None,
        **inertial_frame_properties: object,
    ) -> None:
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        elif not hasattr(ax, "get_zlim"):
            msg = "The axes should be a 3d axes"
            raise TypeError(msg)

        super().__init__(inertial_frame, zero_point, **inertial_frame_properties)
        self._ax = ax
        self.annot: plt.Text = self._ax.text2D(
            0,
            0,
            "",
            bbox={"boxstyle": "round4", "fc": "linen", "ec": "k", "lw": 1},
            transform=None,
        )
        self.annot.set_visible(False)
        self.annot_location = "object"
        self._ax.figure.canvas.mpl_connect("motion_notify_event", self._hover)

    @property
    def axes(self) -> plt.Axes:
        """Axes used by the scene."""
        return self._ax

    @property
    def annot_location(self) -> str:
        """String describing where the annotation should be displayed.

        Explanation
        -----------
        String that is used to determine where the annotation should be displayed.
        Options are:
        - `'object'`: The annotation location is determined based on the `plot_object`
        """
        return self._annot_location

    @annot_location.setter
    def annot_location(self, new_annot_location: str) -> None:
        if new_annot_location == "object":
            self._annot_location = new_annot_location
        else:
            msg = (
                f"Annotation location '{new_annot_location}' has not been implemented."
            )
            raise NotImplementedError(msg)

    @property
    def annot_coords(self) -> tuple[float, float]:
        """Coordinate where the annotation text is displayed."""
        return self.annot.get_position()

    def plot(self, prettify: bool = True, ax_scale: float = 1.5) -> None:
        """Plot all plot objects.

        Parameters
        ----------
        prettify : bool, optional
            If True prettify the axes. Default is True.
        ax_scale : float, optional
            Makes the axes bigger in the figure. This function is part of prettifying
            the figure and only works nicely if it is the only subplot. Disabled if set
            to 0. Default is 1.5

        Returns
        -------
        tuple of artists
            Returns the plotted artists

        """
        self.update()
        for plot_object in self._children:
            plot_object.plot(self.axes)
        if prettify:
            self.axes.autoscale_view()
            for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
                axis.set_ticklabels([])
                axis.set_ticks_position("none")
            if ax_scale:
                self.axes.set_position(
                    [-(ax_scale - 1) / 2, -(ax_scale - 1) / 2, ax_scale, ax_scale]
                )
            self.auto_zoom()
            self.axes.set_aspect("equal", adjustable="box")

    def as_orthogonal_projection_plot(
        self, frame: ReferenceFrame | None = None
    ) -> None:
        """Change the axis to an orthogonal projection making the view seemingly 2D.

        Parameters
        ----------
        frame : ReferenceFrame, optional
            Reference frame w.r.t. which the axis view is oriented aligning the users
            view with its YZ plane. The default is the inertial frame of the scene.

        """
        frame = frame or self.inertial_frame
        self.axes.set_proj_type("ortho")
        self.axes.view_init(**calculate_euler_angles(self.inertial_frame, frame))

    def auto_zoom(self, scale: float = 1.1) -> tuple[float, float] | None:
        """Auto scale the axis."""
        _artists = self.artists
        if not _artists:
            return None
        _min = np.min([artist.min() for artist in _artists], axis=0)
        _max = np.max([artist.max() for artist in _artists], axis=0)
        size = scale * np.max(_max - _min)
        extra = (size - (_max - _min)) / 2
        self.axes.set_xlim(_min[0] - extra[0], _max[0] + extra[0])
        self.axes.set_ylim(_min[1] - extra[1], _max[1] + extra[1])
        self.axes.set_zlim(_min[2] - extra[2], _max[2] + extra[2])
        return _min, _max

    def _get_selected_object(self, event: MouseEvent) -> MplPlotBase | None:
        """Get the `plot_object` where the mouseevent is currently on."""
        for plot_object in self._children:
            if plot_object.contains(event):
                return plot_object
        return None

    def _update_annot(self, plot_object: MplPlotBase, event: MouseEvent) -> None:
        """Update the annotation to the given `plot_object`."""
        self.annot.set_text(str(plot_object))
        if self.annot_location == "object":
            x, y, _ = proj_transform(*plot_object.annot_coords, self._ax.get_proj())
            self.annot.set_position(self._ax.transData.transform((x, y)))
            # self.annot.set_position_3d(plot_object.annot_coords)  # noqa: ERA001
        elif self.annot_location == "mouse":
            self.annot.set_position(
                self._ax.transData.transform((event.xdata, event.ydata))
            )

    def _hover(self, event: MouseEvent) -> None:
        """Show an annotation if the mouse is hovering over a `plot_object`."""
        if event.inaxes == self._ax:
            plot_object = self._get_selected_object(event)
            if plot_object is not None:
                self._update_annot(plot_object, event)
                self.annot.set_visible(True)
                self._ax.figure.canvas.draw_idle()
            elif self.annot.get_visible():
                self.annot.set_visible(False)
                self._ax.figure.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the axes.

        Explanation
        -----------
        Remove all artists known by the instance. Only the inertial frame as plotobject
        in the scene.
        """
        for plot_object in self._children:
            plot_object.visible = False
        self._children: list[MplPlotBase] = [self._children[0]]

    def animate(
        self,
        get_args: Callable[[object], tuple],
        frames: Iterable[object] | int,
        interval: int = 30,
        **kwargs: object,
    ) -> FuncAnimation:
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
            Keyword arguments are parsed to the
            :class:`matplotlib.animation.FuncAnimation`.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object.

        """

        def update(frame: object) -> tuple[MplArtistBase, ...]:
            self.evaluate_system(*get_args(frame))
            self.update()
            return self.artists

        if isinstance(frames, int):
            frames = range(frames)
        return FuncAnimation(
            self.axes.figure,
            update,
            frames=frames,
            interval=interval,
            **{"blit": True, **kwargs},
        )

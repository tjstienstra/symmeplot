from __future__ import annotations

from typing import TYPE_CHECKING

from symmeplot.core import PlotBase

if TYPE_CHECKING:
    import pyqtgraph.opengl as gl

__all__ = ["PgPlotBase"]


class PgPlotBase(PlotBase):
    """Class with the basic attributes and methods for the plot objects.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        The reference frame with respect to which the object is oriented.
    zero_point : Point
        The absolute origin with respect to which the object is positioned.
    name : str, optional
        Name of the plot object. Default is the name of the object being plotted.

    """

    def plot(self, view: gl.GLViewWidget):
        """Plot the associated plot objects."""
        for artist, _ in self._artists:
            artist.plot(view)
        for child in self._children:
            child.plot(view)

    @property
    def visible(self) -> bool:
        """If the object is be visible in the plot."""
        return self._visible

    @visible.setter
    def visible(self, is_visible: bool):
        for artist, _ in self._artists:
            artist.visible = is_visible
        for child in self._children:
            child.visible = bool(is_visible)
        self._visible = bool(is_visible)

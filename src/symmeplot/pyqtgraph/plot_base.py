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

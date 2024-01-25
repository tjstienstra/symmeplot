from __future__ import annotations

from abc import abstractmethod

from matplotlib.pyplot import gca

from symmeplot.core import PlotBase

__all__ = ["MplPlotBase"]


class MplPlotBase(PlotBase):
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

    def plot(self, ax=None):
        """Plot the associated plot objects.

        Explanation
        -----------
        Add the objects artists to the matplotlib `Axes`. Note that the object should be
        evaluated before plotting with for example the :meth:`PlotBase.evalf` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            Axes on which the artist should be added. The default is the active axes.

        """
        if ax is None:
            ax = gca()
        self.update()
        for artist, _ in self._artists:
            ax.add_artist(artist)
        for child in self._children:
            child.plot(ax)

    @property
    @abstractmethod
    def annot_coords(self):
        """Coordinate where the annotation text is displayed."""
        pass

    def contains(self, event):
        """Boolean whether one of the artists contains the event."""
        return any(artist.contains(event)[0] for artist in self.artists)

"""Artists of the matplotlib backend."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3D as _Line3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection as _Path3DCollection
from mpl_toolkits.mplot3d.art3d import PathPatch3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

from symmeplot.core import ArtistBase
from symmeplot.utilities import dcm_to_align_vectors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.path import Path

__all__ = ["Circle3D", "Line3D", "Scatter3D", "Vector3D"]


class MplArtistBase(ArtistBase):
    """Base class for artists used in matplotlib scene."""

    @abstractmethod
    def min(self) -> np.ndarray[np.float64]:
        """Return the minimum values of the bounding box of the artist data."""

    @abstractmethod
    def max(self) -> np.ndarray[np.float64]:
        """Return the maximum values of the bounding box of the artist data."""


class Line3D(_Line3D, MplArtistBase):
    """Artist to plot 3D lines."""

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        z: Sequence[float],
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            *args,
            **kwargs,
        )

    def update_data(
        self, x: Sequence[float], y: Sequence[float], z: Sequence[float]
    ) -> None:
        """Update the data of the artist."""
        self.set_data_3d(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        )

    def min(self) -> np.ndarray[np.float64]:
        """Return the minimum values of the bounding box of the artist data."""
        return np.array([axes.min() for axes in self.get_data_3d()])

    def max(self) -> np.ndarray[np.float64]:
        """Return the maximum values of the bounding box of the artist data."""
        return np.array([axes.max() for axes in self.get_data_3d()])


class Scatter3D(_Path3DCollection, MplArtistBase):
    """Artist to plot 3D scatter points with varying alpha values.

    This class wraps matplotlib's Path3DCollection.

    """

    def __init__(
        self,
        marker: str | None = None,
        s: float | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the Scatter3D.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to Path3DCollection. Common options:
            - facecolors, edgecolors: Colors for the markers
            - linewidths: Edge width of markers

        """
        # Parse marker and size like ax.scatter does.
        if marker is None:
            marker = mpl.rcParams["scatter.marker"]
        marker_obj = marker if isinstance(marker, MarkerStyle) else MarkerStyle(marker)
        self._marker_path = marker_obj.get_path().transformed(
            marker_obj.get_transform()
        )
        if s is None:
            s = (
                20
                if mpl.rcParams["_internal.classic_mode"]
                else mpl.rcParams["lines.markersize"] ** 2.0
            )
        s = np.ma.ravel(s)

        super().__init__((self._marker_path,), s, **kwargs)

        # Critical: set transform to Identity (scatter does this)
        # The marker paths are in points, offset_transform handles positioning
        self.set_transform(mpl.transforms.IdentityTransform())
        self._offset_transform_set = "transform" in kwargs

    def update_data(
        self,
        points: Sequence[Sequence[float]],
        alphas: Sequence[float] | None = None,
    ) -> None:
        """Update the data of the artist.

        Parameters
        ----------
        points : sequence of sequences of float
            Points in the form [[x0, y0, z0], [x1, y1, z1], ...].
        alphas : sequence of float, optional
            Alpha values for each point.

        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(-1, 3)

        if len(points) == 0:
            self.set_paths([])
            return

        # Set paths for each point
        self.set_paths([self._marker_path] * len(points))
        self._offsets3d = tuple(points.T)
        if self.axes is not None and not self._offset_transform_set:
            self.set_offset_transform(self.axes.transData)

        # Set colors with alpha values
        if alphas is not None:
            alphas = np.asarray(alphas, dtype=np.float64)
            facecolors = np.resize(self.get_facecolors(), (len(points), 4))
            facecolors[:, 3] = alphas
            self.set_facecolors(facecolors)
            edgecolors = np.resize(self.get_edgecolors(), (len(points), 4))
            edgecolors[:, 3] = alphas
            self.set_edgecolors(edgecolors)

    def min(self) -> np.ndarray[np.float64]:
        """Return the minimum values of the bounding box of the artist data."""
        if len(self._offsets3d) == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.min(self._offsets3d, axis=1)

    def max(self) -> np.ndarray[np.float64]:
        """Return the maximum values of the bounding box of the artist data."""
        if len(self._offsets3d) == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.max(self._offsets3d, axis=1)


class Vector3D(FancyArrowPatch, MplArtistBase):
    """Artist to plot 3D vectors.

    Notes
    -----
    This class is inspired by
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

    """

    def __init__(
        self,
        origin: Sequence[float],
        vector: Sequence[float],
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._origin = np.array(origin, dtype=np.float64)
        self._vector = np.array(vector, dtype=np.float64)

    def do_3d_projection(self, renderer: object = None) -> float:  # noqa: ARG002
        """Project the artist to the 3D axes."""
        # https://github.com/matplotlib/matplotlib/issues/21688
        xs, ys, zs = proj_transform(
            *[(o, o + d) for o, d in zip(self._origin, self._vector, strict=True)],
            self.axes.M,
        )
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def update_data(self, origin: Sequence[float], vector: Sequence[float]) -> None:
        """Update the data of the artist."""
        self._origin = np.array(origin, dtype=np.float64)
        self._vector = np.array(vector, dtype=np.float64)

    def min(self) -> np.ndarray[np.float64]:
        """Return the minimum values of the bounding box of the artist data."""
        return np.min([self._origin, self._origin + self._vector], axis=0)

    def max(self) -> np.ndarray[np.float64]:
        """Return the maximum values of the bounding box of the artist data."""
        return np.max([self._origin, self._origin + self._vector], axis=0)


class Circle3D(PathPatch3D, MplArtistBase):
    """Artist to plot 3D circles.

    Notes
    -----
    This class is inspired by https://stackoverflow.com/a/18228967/20185124

    """

    def __init__(
        self,
        center: Sequence[float],
        radius: float,
        normal: Sequence[float] = (0, 0, 1),
        **kwargs: object,
    ) -> None:
        path_2d = self._get_2d_path(np.float64(radius))
        super().__init__(path_2d, **{"zs": 0, **kwargs})
        self._segment3d = self._get_segment3d(
            path_2d,
            np.array(center, dtype=np.float64),
            np.array(normal, dtype=np.float64),
        )

    @staticmethod
    def _get_2d_path(radius: np.float64) -> Path:
        circle_2d = Circle((0, 0), radius)
        path = circle_2d.get_path()  # Get the path and the associated transform
        trans = circle_2d.get_patch_transform()
        return trans.transform_path(path)  # Apply the transform

    @staticmethod
    def _get_segment3d(
        path_2d: Path, center: np.ndarray[np.float64], normal: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        verts = path_2d.vertices  # Get the vertices in 2D
        rot_mat = dcm_to_align_vectors((0, 0, 1), normal)
        segment3d = np.array([np.dot(rot_mat, (x, y, 0)) for x, y in verts])
        for i, offset in enumerate(center):
            segment3d[:, i] += offset
        return segment3d

    def update_data(
        self, center: Sequence[float], radius: float, normal: Sequence[float]
    ) -> None:
        """Update the data of the artist."""
        self._path2d = self._get_2d_path(np.float64(radius))
        self._segment3d = self._get_segment3d(
            self._path2d,
            np.array(center, dtype=np.float64),
            np.array(normal, dtype=np.float64),
        )

    def min(self) -> np.array[np.float64]:
        """Return the minimum values of the bounding box of the artist data."""
        return self._segment3d.min(axis=0)

    def max(self) -> np.array[np.float64]:
        """Return the maximum values of the bounding box of the artist data."""
        return self._segment3d.max(axis=0)

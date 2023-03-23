from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3D as _Line3D
from mpl_toolkits.mplot3d.art3d import PathPatch3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib import Path

__all__ = ["Line3D", "Vector3D", "Circle3D"]


class ArtistBase(ABC):
    """Base class for artists used in SymMePlotter."""

    @abstractmethod
    def update_data(self, *args):
        pass

    @abstractmethod
    def min(self) -> np.array:
        pass

    @abstractmethod
    def max(self) -> np.array:
        pass


class Line3D(_Line3D, ArtistBase):
    """Artist to plot 3D lines."""

    def __init__(self, x: Sequence[float], y: Sequence[float], z: Sequence[float],
                 *args, **kwargs):
        super().__init__(np.array(x, dtype=np.float64),
                         np.array(y, dtype=np.float64),
                         np.array(z, dtype=np.float64), *args, **kwargs)

    def update_data(self, x: Sequence[float], y: Sequence[float],
                    z: Sequence[float]):
        self.set_data_3d(np.array(x, dtype=np.float64),
                         np.array(y, dtype=np.float64),
                         np.array(z, dtype=np.float64))

    def min(self) -> np.array:
        return np.array([axes.min() for axes in self.get_data_3d()])

    def max(self) -> np.array:
        return np.array([axes.max() for axes in self.get_data_3d()])


class Vector3D(FancyArrowPatch, ArtistBase):
    """Artist to plot 3D vectors.

    Notes
    -----
    This class is inspired by
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c

    """

    def __init__(self, origin: Sequence[float], vector: Sequence[float], *args,
                 **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._origin = np.array(origin, dtype=np.float64)
        self._vector = np.array(vector, dtype=np.float64)

    def do_3d_projection(self, renderer=None):
        # https://github.com/matplotlib/matplotlib/issues/21688
        xs, ys, zs = proj_transform(
            *[(o, o + d) for o, d in zip(self._origin, self._vector)], self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def update_data(self, origin: Sequence[float], vector: Sequence[float]):
        self._origin = np.array(origin, dtype=np.float64)
        self._vector = np.array(vector, dtype=np.float64)

    def min(self) -> np.array:
        return np.min([self._origin, self._origin + self._vector], axis=0)

    def max(self) -> np.array:
        return np.max([self._origin, self._origin + self._vector], axis=0)


class Circle3D(PathPatch3D, ArtistBase):
    """Artist to plot 3D circles.

    Notes
    -----
    This class is inspired by https://stackoverflow.com/a/18228967/20185124

    """

    def __init__(self, center: Sequence[float], radius: float,
                 normal: Sequence[float] = (0, 0, 1), **kwargs):
        path_2d = self._get_2d_path(np.float64(radius))
        super().__init__(path_2d, **{"zs": 0, **kwargs})
        self._segment3d = self._get_segment3d(
            path_2d,
            np.array(center, dtype=np.float64),
            np.array(normal, dtype=np.float64))

    @staticmethod
    def _get_2d_path(radius: np.float64):
        circle_2d = Circle((0, 0), radius)
        path = circle_2d.get_path()  # Get the path and the associated transform
        trans = circle_2d.get_patch_transform()
        return trans.transform_path(path)  # Apply the transform

    @staticmethod
    def _get_segment3d(path_2d: "Path", center: "npt.NDArray[np.float64]",
                       normal: "npt.NDArray[np.float64]"):
        normal /= np.linalg.norm(normal)
        verts = path_2d.vertices  # Get the vertices in 2D
        rot_mat = Circle3D._rotation_matrix(normal)  # Get the rotation matrix
        segment3d = np.array([np.dot(rot_mat, (x, y, 0)) for x, y in verts])
        for i, offset in enumerate(center):
            segment3d[:, i] += offset
        return segment3d

    @staticmethod
    def _rotation_matrix(normal: np.array):
        """Calculate rotation matrix based a normal vector.

        Notes
        -----
        Calculation is based on https://math.stackexchange.com/a/476311

        """
        v = np.cross((0, 0, 1), normal)
        sin_angle = np.linalg.norm(v)
        if sin_angle == 0:
            return np.identity(3)
        skew = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + skew + (skew @ skew) * (1 / (1 + normal[2]))

    def update_data(self, center: Sequence[float], radius: float,
                    normal: Sequence[float]):
        self._segment3d = self._get_segment3d(self._get_2d_path(np.float64(radius)),
                                              np.array(center, dtype=np.float64),
                                              np.array(normal, dtype=np.float64))

    def min(self) -> np.array:
        return self._segment3d.min(axis=0)

    def max(self) -> np.array:
        return self._segment3d.max(axis=0)

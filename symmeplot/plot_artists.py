import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import PathPatch3D
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d.art3d import Line3D
from typing import Sequence
from matplotlib import Path

__all__ = ['Point3D', 'Vector3D', 'Circle3D']


class Point3D(Line3D):
    def __init__(self, position: Sequence[float], *args, **kwargs):
        super().__init__(*([position[i]] for i in range(3)), *args,
                         **{'marker': 'o'} | kwargs)

    def update_data(self, position: Sequence[float]):
        self.set_data_3d(*([position[i]] for i in range(3)))


class Vector3D(FancyArrowPatch):
    # Source: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    def __init__(self, origin: Sequence[float], vector: Sequence[float], *args,
                 **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._origin = origin
        self._vector = vector

    def do_3d_projection(self, renderer=None):
        # https://github.com/matplotlib/matplotlib/issues/21688
        xs, ys, zs = proj_transform(
            *[(o, o + d) for o, d in zip(self._origin, self._vector)],
            self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

    def update_data(self, origin: Sequence[float], vector: Sequence[float]):
        self._origin, self._vector = origin, vector


class Circle3D(PathPatch3D):
    """Patch to plot 3D circles
    Inpired by: https://stackoverflow.com/a/18228967/20185124
    """
    def __init__(self, center: Sequence[float], radius: float,
                 normal: Sequence[float] = (0, 0, 1), **kwargs):
        path_2d = self._get_2d_path(radius)
        super().__init__(path_2d, **{'zs': 0} | kwargs)
        self._segment3d = self._get_segment3d(path_2d, center, normal)

    @staticmethod
    def _get_2d_path(radius: float):
        circle_2d = Circle((0, 0), radius)
        path = circle_2d.get_path()  # Get the path and the associated transform
        trans = circle_2d.get_patch_transform()
        return trans.transform_path(path)  # Apply the transform

    @staticmethod
    def _get_segment3d(path_2d: Path, center: Sequence[float],
                       normal: Sequence[float]):
        if isinstance(normal, str):  # Translate strings to normal vectors
            index = "xyz".index(normal)
            normal = np.roll((1.0, 0, 0), index)
        else:
            normal = np.array(normal, dtype=np.float64)
            normal /= np.linalg.norm(normal)
        verts = path_2d.vertices  # Get the vertices in 2D
        d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
        M = Circle3D._rotation_matrix(d)  # Get the rotation matrix
        segment3d = np.array([np.dot(M, (x, y, 0)) for x, y in verts])
        for i, offset in enumerate(center):
            segment3d[:, i] += offset
        return segment3d

    @staticmethod
    def _rotation_matrix(d: np.array):
        """
        Calculates a rotation matrix given a vector d. The direction of d
        corresponds to the rotation axis. The length of d corresponds to
        the sin of the angle of rotation.
        """
        sin_angle = np.linalg.norm(d)
        if sin_angle == 0:
            return np.identity(3)
        d /= sin_angle
        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[0, d[2], -d[1]],
                         [-d[2], 0, d[0]],
                         [d[1], -d[0], 0]], dtype=np.float64)
        M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
        return M

    def update_data(self, center: Sequence[float], radius: float,
                    normal: Sequence[float]):
        self._segment3d = self._get_segment3d(self._get_2d_path(radius), center,
                                              normal)

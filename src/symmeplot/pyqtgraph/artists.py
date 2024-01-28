from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pyqtgraph.opengl as gl
from OpenGL.GL import *  # noqa

from symmeplot.core import ArtistBase
from symmeplot.utilities import dcm_to_align_vectors

__all__ = ["PgArtistBase", "Point3D", "Line3D", "Vector3D"]


def create_tube_mesh_data(
    lengths: Sequence[float], radii: Sequence[float],
    position: Sequence[float], direction: Sequence[float],
    mesh_resolution: int = 20) -> gl.MeshData:
    """Create a mesh data for a radial varying tube.

    Parameters
    ----------
    lengths : Sequence[float]
        Positions of the sections where the radius is specified.
    radii : Sequence[float]
        Radii the tube should have at the specified positions.
    position : Sequence[float]
        Three dimensional position of the start of the tube.
    direction : Sequence[float]
        Three dimensional direction of the tube.
    mesh_resolution : int, optional
        Number of points in the circle, by default 20

    Returns
    -------
    gl.MeshData
        Mesh data for the tube.

    Examples
    --------
    The following would create a hollow tube with a radius of 0.1 and a length of 1
    along the z-axis.

    >>> from symmeplot.pyqtgraph.artists import create_tube_mesh_data
    >>> create_tube_mesh_data((0, 1), (0, 0.3), (0, 0, 0), (0, 0, 1));

    One can also close both sides by setting the radii to zero.

    >>> create_tube_mesh_data((0, 1, 1, 0), (0, 0, 0.3, 0.3), (0, 0, 0), (0, 0, 1));

    You can also create a cone by simulateously changing the length in a segment and
    setting the radius to zero.

    >>> create_tube_mesh_data((0, 1), (0.3, 0), (0, 0, 0), (0, 0, 1));
    """
    if len(lengths) != len(radii) or len(lengths) < 2:
        raise ValueError(
            "Lengths and radii must have the same length and at least two elements.")
    rres = mesh_resolution
    n_nonzero_radii = np.count_nonzero(radii)
    n_zero_radii = len(radii) - n_nonzero_radii

    verts = np.empty((rres * n_nonzero_radii + n_zero_radii, 3), dtype=np.float64)
    idx = 0
    for li, ri in zip(lengths, radii):
        if ri == 0:
            verts[idx, 2] = li
            verts[idx, :2] = 0
            idx += 1
        else:
            verts[idx:idx + rres, 2] = li
            verts[idx:idx + rres, 0] = ri * np.cos(
                np.linspace(0, 2 * np.pi, rres + 1)[:-1])
            verts[idx:idx + rres, 1] = ri * np.sin(
                np.linspace(0, 2 * np.pi, rres + 1)[:-1])
            idx += rres

    n_faces = 0
    for r1, r2 in zip(radii[:-1], radii[1:]):
        if r1 == 0 and r2 == 0:  # Infitenly thin tube
            continue
        elif r1 == 0 or r2 == 0:  # Cone
            n_faces += rres
        else:  # Tube
            n_faces += 2 * rres
    faces = np.empty((n_faces, 3), dtype=np.uint32)
    fidx, ridx = 0, 0
    for r1, r2 in zip(radii[:-1], radii[1:]):
        if r1 == 0 and r2 == 0:
            continue
        elif r1 == 0:
            faces[fidx:fidx + rres, 0] = ridx
            faces[fidx:fidx + rres, 1] = np.arange(ridx + 1, ridx + rres + 1)
            faces[fidx:fidx + rres, 2] = np.roll(faces[fidx:fidx + rres, 1], -1)
            fidx += rres
            ridx += 1
        elif r2 == 0:
            faces[fidx:fidx + rres, 0] = np.arange(ridx, ridx + rres)
            faces[fidx:fidx + rres, 1] = np.roll(faces[fidx:fidx + rres, 0], -1)
            faces[fidx:fidx + rres, 2] = ridx + rres
            fidx += rres
            ridx += rres
        else:
            faces[fidx:fidx + rres, 0] = np.arange(ridx, ridx + rres)
            faces[fidx:fidx + rres, 1] = np.roll(faces[fidx:fidx + rres, 0], -1)
            faces[fidx:fidx + rres, 2] = np.arange(ridx + rres, ridx + 2 * rres)
            fidx += rres
            ridx += rres
            faces[fidx:fidx + rres, 0] = np.arange(ridx, ridx + rres)
            faces[fidx:fidx + rres, 1] = np.roll(faces[fidx:fidx + rres, 0], -1)
            faces[fidx:fidx + rres, 2] = np.roll(np.arange(ridx - rres, ridx), -1)
            fidx += rres

    # Transform the tube to the correct orientation and position
    verts = (np.dot(verts, dcm_to_align_vectors((0, 0, 1), direction).T)
             + np.array(position, dtype=np.float64))
    return gl.MeshData(vertexes=verts, faces=faces)


class PgArtistBase(ArtistBase):
    """Base class for artists used in pyqtgraph scene."""

    def __init__(self, *gl_items: gl.GLGraphicsItem):
        super().__init__()
        self._gl_items = gl_items

    @property
    def gl_items(self) -> tuple[gl.GLGraphicsItem, ...]:
        """The pyqtgraph item."""
        return self._gl_items

    def plot(self, view: gl.GLViewWidget):
        """Add the artist to the view."""
        for gl_item in self.gl_items:
            view.addItem(gl_item)

    @property
    def visible(self) -> bool:
        """If the artist is visible."""
        return all(gl_item.visible() for gl_item in self.gl_items)

    @visible.setter
    def visible(self, is_visible: bool):
        """Set the visibility of the artist."""
        for gl_item in self.gl_items:
            gl_item.setVisible(is_visible)


class Point3D(PgArtistBase):
    """Artist to plot 3D lines."""

    def __init__(self, x: float, y: float, z: float, **kwargs):
        super().__init__(gl.GLScatterPlotItem(
            pos=np.array([x, y, z], dtype=np.float64).reshape(1, 3), **kwargs))

    def update_data(self, x: float, y: float, z: float):
        """Update the data of the artist."""
        self.gl_items[0].setData(
            pos=np.array([x, y, z], dtype=np.float64).reshape(1, 3))


class Line3D(PgArtistBase):
    """Artist to plot 3D lines."""

    def __init__(self, x: Sequence[float], y: Sequence[float], z: Sequence[float],
                 **kwargs):
        super().__init__(gl.GLLinePlotItem(
            pos=np.array([x, y, z], dtype=np.float64).T, **kwargs))

    def update_data(self, x: Sequence[float], y: Sequence[float],
                    z: Sequence[float]):
        """Update the data of the artist."""
        self.gl_items[0].setData(pos=np.array([x, y, z], dtype=np.float64).T)


class Vector3D(PgArtistBase):
    """Artist to plot 3D vectors.

    Parameters
    ----------
    origin : Sequence[float]
        The origin of the vector.
    vector : Sequence[float]
        The vector to plot.
    as_mesh : bool, optional
        If True, the vector is plotted as a mesh, by default False.
    vector_radius : float, optional
        The radius of the vector if plotted as mesh, by default defined as
        classattribute.
    head_width : float, optional
        The width of the head of the vector if plotted as mesh, by default defined as
        classattribute.
    head_length : float, optional
        The length of the head of the vector if plotted as mesh, by default defined as
        classattribute.
    mesh_resolution : int, optional
        The number of points in the circle if plotted as mesh, by default defined as
        classattribute.
    """

    vector_radius: float | None = None
    head_width: float | None = None
    head_length: float | None = None
    mesh_resolution: int = 20

    def __init__(self, origin: Sequence[float], vector: Sequence[float],
                 as_mesh: bool = False,**kwargs):
        origin = np.array(origin, dtype=np.float64)
        vector = np.array(vector, dtype=np.float64)
        self._as_mesh = bool(as_mesh)
        for prop in ("vector_radius", "head_width", "head_length", "mesh_resolution"):
            if prop in kwargs:
                setattr(self, prop, kwargs.pop(prop))
        if self._as_mesh:
            if "color" in kwargs:
                kwargs["edgeColor"] = kwargs["color"]
                kwargs["faceColor"] = kwargs["edgeColor"]
            super().__init__(self._create_vector_as_mesh(origin, vector, **kwargs))
        else:
            super().__init__(self._create_vector_as_line(origin, vector, **kwargs))

    def update_data(self, origin: Sequence[float], vector: Sequence[float]):
        """Update the data of the artist."""
        origin = np.array(origin, dtype=np.float64)
        vector = np.array(vector, dtype=np.float64)
        if self._as_mesh:
            # The vector may change size, therefore we need to recompute the mesh data.
            self.gl_items[0].setMeshData(
                meshdata=self._compute_mesh_data(origin, vector)
            )
        else:
            self.gl_items[0].setData(pos=np.array([origin, origin + vector]))

    def _create_vector_as_line(self, origin: npt.NDArray[np.float64],
                               vector: npt.NDArray[np.float64], **kwargs
                               ) -> gl.GLLinePlotItem:
        return gl.GLLinePlotItem(pos=np.array([origin, origin + vector]), **kwargs)

    def _compute_mesh_data(self, origin: npt.NDArray[np.float64],
                            vector: npt.NDArray[np.float64]) -> gl.MeshData:
        length = np.linalg.norm(vector)
        if not length:
            return gl.MeshData()
        if self.vector_radius is None:
            Vector3D.vector_radius = length / 50
        if self.head_width is None:
            Vector3D.head_width = self.vector_radius * 3
        if self.head_length is None:
            Vector3D.head_length = self.vector_radius * 5
        shaft_length = length - self.head_length
        return create_tube_mesh_data(
            lengths=(0, 0, shaft_length, shaft_length, length),
            radii=(0, self.vector_radius, self.vector_radius, self.head_width, 0),
            position=origin, direction=vector, mesh_resolution=self.mesh_resolution)

    def _create_vector_as_mesh(self, origin: npt.NDArray[np.float64],
                               vector: npt.NDArray[np.float64], **kwargs):
        return gl.GLMeshItem(meshdata=self._compute_mesh_data(origin, vector), **kwargs)

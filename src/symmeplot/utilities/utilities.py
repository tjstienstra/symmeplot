"""Utility functions for SymmePlot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.physics.vector import ReferenceFrame


def dcm_to_align_vectors(
    v1: Sequence[float], v2: Sequence[float]
) -> npt.NDArray[np.float64]:
    """Calculate rotation matrix to align v1 with v2.

    Notes
    -----
    Calculation is based on https://math.stackexchange.com/a/476311

    """
    v1 = np.array(v1, dtype=np.float64) / np.linalg.norm(v1)
    v2 = np.array(v2, dtype=np.float64) / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)  # cosine of angle between v1 and v2
    if c == -1:
        return np.identity(3)
    skew = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
    )
    return np.eye(3) + skew + (skew @ skew) * (1 / (1 + c))


def calculate_euler_angels(
    normal_frame: ReferenceFrame,
    projection_frame: ReferenceFrame,
) -> dict[str, float]:
    """Get the Euler angles of the given frame.

    Parameters
    ----------
    normal_frame : ReferenceFrame
        Reference frame for which the Euler angles should be calculated.
    projection_frame : ReferenceFrame
        Reference frame for which the Euler angles should be calculated.

    Returns
    -------
    tuple of float
        The Euler angles in the order of (elev, azim, roll).

    """
    direction_matrix = projection_frame.dcm(normal_frame)
    direction_matrix = np.array(direction_matrix).astype(np.float64)

    elevation = np.arcsin(-direction_matrix[2, 0])
    azimuth = np.arctan2(direction_matrix[1, 0], direction_matrix[0, 0])
    roll = np.arctan2(direction_matrix[2, 1], direction_matrix[2, 2])

    return {
        "elev": np.rad2deg(elevation),
        "azim": np.rad2deg(azimuth),
        "roll": np.rad2deg(roll),
    }

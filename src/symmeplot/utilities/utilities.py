from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


def dcm_to_align_vectors(v1: Sequence[float], v2: Sequence[float]
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
    skew = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + skew + (skew @ skew) * (1 / (1 + c))

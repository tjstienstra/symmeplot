import numpy as np
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from sympy import Matrix


def vector_to_numpy(matrix: 'Optional[Union[Matrix, np.array]]') -> np.array:
    """Converts an evaluated sympy matrix representing a coordinate to a
    ``numpy.array``."""
    return matrix.__array__(np.float64).flatten()

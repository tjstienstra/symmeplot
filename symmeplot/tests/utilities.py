import functools

import numpy as np
from matplotlib.testing.decorators import image_comparison
from sympy.utilities.iterables import iterable

mpl3d_image_comparison = functools.partial(image_comparison, remove_text=True,
                                           style="default")


def equalize_axis_limits(ax_ref, ax_to_equalize):
    ax_to_equalize.set_xlim(ax_ref.get_xlim())
    ax_to_equalize.set_ylim(ax_ref.get_ylim())
    if hasattr(ax_ref, "get_zlim"):
        ax_to_equalize.set_zlim(ax_ref.get_zlim())


def flatten(lst):
    # Source: https://stackoverflow.com/a/10824420/20185124
    for i in lst:
        if iterable(i):
            yield from flatten(i)
        else:
            yield i


def compare_values(vals1, vals2):
    flattened1 = np.array(list(flatten(vals1)), dtype=np.float64)
    flattened2 = np.array(list(flatten(vals2)), dtype=np.float64)
    assert np.allclose(flattened1, flattened2)

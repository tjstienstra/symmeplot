from __future__ import annotations


def test_lambdify_patch() -> None:
    from sympy import lambdify, symbols

    import symmeplot.utilities.sympy_patches  # noqa: F401

    a = symbols("a")
    lambdify((a,), ((), (a,)))

from __future__ import annotations


def test_lambdify_patch() -> None:
    from sympy import lambdify, symbols  # noqa: PLC0415

    import symmeplot.utilities.sympy_patches  # noqa: F401, PLC0415

    a = symbols("a")
    lambdify((a,), ((), (a,)))

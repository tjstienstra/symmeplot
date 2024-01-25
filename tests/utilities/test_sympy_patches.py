def test_lambdify_patch():
    import symmeplot.utilities.sympy_patches  # noqa: F401
    from sympy import lambdify, symbols
    a = symbols("a")
    lambdify((a,), ((), (a,)))

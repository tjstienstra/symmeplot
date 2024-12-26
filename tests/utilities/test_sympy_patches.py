def test_lambdify_patch():
    from sympy import lambdify, symbols

    import symmeplot.utilities.sympy_patches  # noqa: F401

    a = symbols("a")
    lambdify((a,), ((), (a,)))

"""Patches for sympy.

List of patches:
- Patch for lambdify to allow for empty tuples, refer to #26119 in sympy.
"""
from __future__ import annotations

from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import _EvaluatorPrinter


def _recursive_to_string(doprint, arg):
    """Functions in lambdify accept both SymPy types and non-SymPy types such as python
    lists and tuples. This method ensures that we only call the doprint method of the
    printer with SymPy types (so that the printer safely can use SymPy-methods)."""
    from sympy.core.basic import Basic
    try:
        from sympy.matrices.matrixbase import MatrixBase as MatrixType
    except ImportError:
        from sympy.matrices.common import MatrixOperations as MatrixType

    if isinstance(arg, (Basic, MatrixType)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            left, right = "[", "]"
        elif isinstance(arg, tuple):
            left, right = "(", ",)"
            if not arg:
                return "()"  # special case for empty tuple
        else:
            raise NotImplementedError("unhandled type: %s, %s" % (type(arg), arg))
        return left +", ".join(_recursive_to_string(doprint, e) for e in arg) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)

def doprint(self, funcname, args, expr, *, cses=()):
    """Returns the function definition code as a string."""
    from sympy.core.symbol import Dummy

    funcbody = []

    if not iterable(args):
        args = [args]

    if cses:
        subvars, subexprs = zip(*cses)
        exprs = [expr] + list(subexprs)
        argstrs, exprs = self._preprocess(args, exprs)
        expr, subexprs = exprs[0], exprs[1:]
        cses = zip(subvars, subexprs)
    else:
        argstrs, expr = self._preprocess(args, expr)

    # Generate argument unpacking and final argument list
    funcargs = []
    unpackings = []

    for argstr in argstrs:
        if iterable(argstr):
            funcargs.append(self._argrepr(Dummy()))
            unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
        else:
            funcargs.append(argstr)

    funcsig = "def {}({}):".format(funcname, ", ".join(funcargs))

    # Wrap input arguments before unpacking
    funcbody.extend(self._print_funcargwrapping(funcargs))

    funcbody.extend(unpackings)

    for s, e in cses:
        if e is None:
            funcbody.append(f"del {s}")
        else:
            funcbody.append(f"{s} = {self._exprrepr(e)}")

    str_expr = _recursive_to_string(self._exprrepr, expr)

    if "\n" in str_expr:
        str_expr = f"({str_expr})"
    funcbody.append(f"return {str_expr}")

    funclines = [funcsig]
    funclines.extend(["    " + line for line in funcbody])

    return "\n".join(funclines) + "\n"

_EvaluatorPrinter.doprint = doprint

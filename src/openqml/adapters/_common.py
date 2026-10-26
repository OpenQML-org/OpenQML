import importlib
from numbers import Real

from ..errors import MissingDependencyError, conversion_error
from ..parameters import Expression, Parameter, expression, node, real


def dependency(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(f"Install openqml[{extra}] for this integration.") from exc


def numeric(value, location="parameter"):
    if getattr(value, "requires_grad", False) or "Tracer" in type(value).__name__:
        raise conversion_error(
            "AUTODIFF_GRAPH",
            "Conversion cannot preserve this gradient graph.",
            location,
            "Use native OpenQML layers or explicitly bind constants.",
        )
    try:
        if not isinstance(value, Real):
            if hasattr(value, "item"):
                value = value.item()
            elif not isinstance(value, (str, bytes, complex, bool)):
                value = float(value)
        return real(value)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise conversion_error(
            "NONREAL_PARAMETER", "Expected a finite real scalar.", location
        ) from exc


def sympy_expression(value, location="parameter"):
    sympy = dependency("sympy", "symbolic")
    value = sympy.sympify(value)
    if not value.free_symbols:
        return Expression(constant=numeric(value, location))
    if isinstance(value, sympy.Symbol):
        return expression(Parameter(str(value)))
    if value.func in {sympy.Add, sympy.Mul}:
        values = [sympy_expression(v, location) for v in value.args]
        result = expression(0 if value.func is sympy.Add else 1)
        for part in values:
            result = result + part if value.func is sympy.Add else result * part
        return result
    if value.func is sympy.Pow:
        return sympy_expression(value.args[0], location) ** sympy_expression(
            value.args[1], location
        )
    functions = {
        sympy.sin: "sin",
        sympy.cos: "cos",
        sympy.tan: "tan",
        sympy.exp: "exp",
        sympy.log: "log",
    }
    if value.func in functions:
        return node(functions[value.func], sympy_expression(value.args[0], location))
    raise conversion_error(
        "SYMBOLIC_OPERATION", f"Unsupported real symbolic function {value.func}.", location
    )


def to_sympy(expr):
    sympy = dependency("sympy", "symbolic")
    if expr.op is None:
        return expr.constant + sum(c * sympy.Symbol(name, real=True) for name, c in expr.terms)
    values = [to_sympy(v) for v in expr.args]
    if expr.op == "add":
        return values[0] + values[1]
    if expr.op == "mul":
        return values[0] * values[1]
    if expr.op == "div":
        return values[0] / values[1]
    if expr.op == "pow":
        return values[0] ** values[1]
    return getattr(sympy, expr.op)(values[0])


def reverse_local_basis(matrix):
    import numpy as np

    value = np.asarray(matrix, dtype=complex)
    width = int(np.log2(value.shape[0]))
    axes = tuple(reversed(range(width))) + tuple(reversed(range(width, 2 * width)))
    return value.reshape((2,) * (2 * width)).transpose(axes).reshape(value.shape)


def bound_for_export(circuit, bindings):
    return circuit if bindings is None else circuit.bind(bindings)


def check_matrix_width(width, limit, location):
    from ..ir import integer

    integer(limit, "max_matrix_qubits", 1)
    if width > limit:
        raise conversion_error(
            "MATRIX_RESOURCE_LIMIT",
            f"Dense matrix fallback needs {width} qubits; configured limit is {limit}.",
            location,
            "Decompose into local gates or explicitly increase max_matrix_qubits.",
        )

import math
from numbers import Real

from ...errors import ValidationError, conversion_error
from ...ir.core import _require_circuit
from ...ir.parameters import Expression, Parameter, _postorder, expression, node, real
from ...runtime._dependencies import load_dependency as dependency


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
    except (ValueError, TypeError, RuntimeError, ValidationError) as exc:
        raise conversion_error(
            "NONREAL_PARAMETER", "Expected a finite real scalar.", location
        ) from exc


def sympy_expression(value, location="parameter"):
    sympy = dependency("sympy", "symbolic")
    value = sympy.sympify(value)
    functions = {
        sympy.sin: "sin",
        sympy.cos: "cos",
        sympy.tan: "tan",
        sympy.exp: "exp",
        sympy.log: "log",
    }
    values, active, pending = {}, {}, [(value, False)]
    while pending:
        current, ready = pending.pop()
        key = id(current)
        if key in values:
            continue
        if current.args and current.func not in {sympy.Add, sympy.Mul, sympy.Pow, *functions}:
            if not current.free_symbols:
                values[key] = Expression(constant=numeric(current, location))
                active[key] = False
                continue
            raise conversion_error(
                "SYMBOLIC_OPERATION",
                f"Unsupported real symbolic function {current.func}.",
                location,
            )
        if not ready and current.args:
            pending.append((current, True))
            pending.extend((arg, False) for arg in reversed(current.args))
            continue
        if isinstance(current, sympy.Symbol):
            result, has_symbols = expression(Parameter(str(current))), True
        else:
            has_symbols = any(active[id(arg)] for arg in current.args)
            if not has_symbols:
                result = Expression(constant=numeric(current, location))
            elif current.func in {sympy.Add, sympy.Mul}:
                parts = [values[id(arg)] for arg in current.args]
                if current.func is sympy.Add and all(part.is_affine for part in parts):
                    coefficients = {}
                    for part in parts:
                        for name, coefficient in part.terms:
                            coefficients.setdefault(name, []).append(coefficient)
                    result = Expression(
                        tuple((name, math.fsum(items)) for name, items in coefficients.items()),
                        math.fsum(part.constant for part in parts),
                    )
                    values[key], active[key] = result, has_symbols
                    continue
                while len(parts) > 1:
                    parts = [
                        parts[i] + parts[i + 1]
                        if current.func is sympy.Add
                        else parts[i] * parts[i + 1]
                        for i in range(0, len(parts) - 1, 2)
                    ] + (parts[-1:] if len(parts) % 2 else [])
                result = parts[0]
            elif current.func is sympy.Pow:
                result = values[id(current.args[0])] ** values[id(current.args[1])]
            elif current.func in functions:
                result = node(functions[current.func], values[id(current.args[0])])
            else:
                raise conversion_error(
                    "SYMBOLIC_OPERATION",
                    f"Unsupported real symbolic function {current.func}.",
                    location,
                )
        values[key], active[key] = result, has_symbols
    return values[id(value)]


def to_sympy(expr):
    sympy = dependency("sympy", "symbolic")
    values = {}
    for current in _postorder((expr,)):
        if current.op is None:
            result = sympy.Add(
                current.constant,
                *(c * sympy.Symbol(name, real=True) for name, c in current.terms),
            )
        else:
            args = [values[id(arg)] for arg in current.args]
            if current.op == "add":
                result = args[0] + args[1]
            elif current.op == "mul":
                result = args[0] * args[1]
            elif current.op == "div":
                result = args[0] / args[1]
            elif current.op == "pow":
                result = args[0] ** args[1]
            else:
                result = getattr(sympy, current.op)(args[0])
        values[id(current)] = result
    return values[id(expr)]


def reverse_local_basis(matrix):
    import numpy as np

    value = np.asarray(matrix, dtype=complex)
    width = int(np.log2(value.shape[0]))
    axes = tuple(reversed(range(width))) + tuple(reversed(range(width, 2 * width)))
    return value.reshape((2,) * (2 * width)).transpose(axes).reshape(value.shape)


def bound_for_export(circuit, bindings):
    _require_circuit(circuit)
    if bindings is not None:
        return circuit.bind(bindings)
    if circuit.global_phase._raw_constants or any(
        any(value._raw_constants for value in operation.params)
        or any(block.global_phase._raw_constants for block in operation.blocks)
        for _, operation in circuit.walk_operations()
    ):
        return circuit.bind({}, partial=True)
    return circuit


def check_matrix_width(width, limit, location):
    from ...ir.core import integer

    integer(limit, "max_matrix_qubits", 1)
    if width > limit:
        raise conversion_error(
            "MATRIX_RESOURCE_LIMIT",
            f"Dense matrix fallback needs {width} qubits; configured limit is {limit}.",
            location,
            "Decompose into local gates or explicitly increase max_matrix_qubits.",
        )

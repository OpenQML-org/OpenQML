"""Compile a bounded declarative Python circuit without executing source code."""

import ast
import math
import operator
from collections.abc import Mapping

from ..errors import ConversionError, ValidationError, conversion_error
from ..ir.core import GATES, UNITARY_GATES, Circuit, ClassicalCondition
from ..ir.parameters import Parameter
from .inspection import _name, inspect_python

_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_CONSTRUCTORS = {"qiskit.QuantumCircuit", "openqml.Circuit", "openqml.CircuitBuilder"}


class _Compiler:
    def __init__(self, filename, max_operations):
        self.filename = filename
        self.remaining = max_operations
        self.aliases = {}
        self.values = {}
        self.wire_name = None
        self.constructor = None

    def fail(self, node, message, code="PYTHON_CIRCUIT_SYNTAX"):
        location = (
            f"{self.filename}:{getattr(node, 'lineno', 1)}:{getattr(node, 'col_offset', 0) + 1}"
        )
        raise conversion_error(code, message, location)

    def charge(self, node, amount=1):
        self.remaining -= amount
        if self.remaining < 0:
            self.fail(node, "Python circuit exceeds its expansion budget.", "OPERATION_LIMIT")

    def path(self, node):
        name = _name(node)
        start, sep, tail = name.partition(".")
        if start in self.values or start == self.wire_name:
            return ""
        return self.aliases.get(start, start) + (sep + tail if sep else "")

    def value(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float, bool}:
            result = node.value
        elif isinstance(node, ast.Name) and node.id in self.values:
            result = self.values[node.id]
        elif self.path(node) in {"math.pi", "math.tau"}:
            result = math.pi if self.path(node) == "math.pi" else math.tau
        elif isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub, ast.Not}:
            value = self.value(node.operand)
            result = (
                value
                if isinstance(node.op, ast.UAdd)
                else (-value if isinstance(node.op, ast.USub) else not self.truth(value, node))
            )
        elif isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
            left, right = self.value(node.left), self.value(node.right)
            if (
                isinstance(node.op, ast.Pow)
                and type(left) is int
                and type(right) is int
                and abs(left) > 1
                and right > 1024
            ):
                self.fail(node, "Python power exceeds its precision budget.")
            result = _ARITHMETIC[type(node.op)](left, right)
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _COMPARISONS
        ):
            left, right = self.value(node.left), self.value(node.comparators[0])
            if type(left) not in {int, float, bool} or type(right) not in {int, float, bool}:
                self.fail(node, "A Python branch needs concrete comparison operands.")
            result = _COMPARISONS[type(node.ops[0])](left, right)
        elif (
            isinstance(node, ast.Call)
            and self.path(node.func) in {"openqml.Parameter", "qiskit.circuit.Parameter"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return Parameter(node.args[0].value)
        else:
            self.fail(node, "Expected a constant, parameter, or bounded arithmetic expression.")
        if type(result) is int and result.bit_length() > 1024:
            self.fail(node, "Python integer exceeds its precision budget.")
        if type(result) is float and not math.isfinite(result):
            self.fail(node, "Non-finite numbers cannot be stored in a circuit.")
        return result

    def truth(self, value, node):
        if type(value) not in {int, float, bool}:
            self.fail(
                node, "A Python branch needs a concrete value. Use SDK controls for feedback."
            )
        return bool(value)

    def method(self, call):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == self.wire_name
        ):
            return call.func.attr
        return None

    def condition(self, node):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
            self.fail(node, "Use a (circuit.clbits[index], value) condition.")
        bit, value = node.elts
        if isinstance(bit, ast.Subscript) and _name(bit.value) == f"{self.wire_name}.clbits":
            bit = bit.slice
        expected = self.value(value)
        if type(expected) not in {bool, int} or expected not in {0, 1}:
            self.fail(value, "A single-bit condition must compare with zero or one.")
        return ClassicalCondition((self.value(bit),), int(expected))

    def body(self, statements, circuit=None, *, runtime=False):
        skip_else = False
        for index, node in enumerate(statements):
            self.charge(node)
            if skip_else:
                skip_else = False
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name, value = node.targets[0].id, node.value
                if (
                    circuit is None
                    and isinstance(value, ast.Call)
                    and (constructor := self.path(value.func)) in _CONSTRUCTORS
                ):
                    args = [self.value(x) for x in value.args]
                    kwargs = {p.arg: self.value(p.value) for p in value.keywords}
                    qiskit = constructor == "qiskit.QuantumCircuit"
                    if (
                        not 1 <= len(args) <= (2 if qiskit else 1)
                        or set(kwargs) - {"num_clbits"}
                        or len(kwargs) != len(value.keywords)
                        or (len(args) == 2 and kwargs)
                    ):
                        self.fail(node, "Declare explicit quantum and optional classical widths.")
                    circuit = Circuit(
                        args[0],
                        num_clbits=args[1] if len(args) == 2 else kwargs.get("num_clbits", 0),
                    )
                    self.wire_name, self.constructor = name, constructor
                elif circuit is not None and name == self.wire_name and self.method(value):
                    if self.constructor != "openqml.Circuit":
                        self.fail(node, "Apply mutable circuit methods as statements.")
                    circuit = self.operation(circuit, value)
                elif not runtime and name != self.wire_name and name not in self.aliases:
                    self.values[name] = self.value(value)
                else:
                    self.fail(node, "Runtime classical writes belong in the circuit.")
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and (index == 0 and isinstance(node.value.value, str))
            ):
                continue
            elif isinstance(node, ast.Expr) and circuit is not None and self.method(node.value):
                if self.constructor == "openqml.Circuit":
                    self.fail(node, "Assign the result of an immutable Circuit method.")
                circuit = self.operation(circuit, node.value)
            elif isinstance(node, ast.For) and circuit is not None:
                seq = node.iter
                if (
                    not isinstance(node.target, ast.Name)
                    or node.target.id == self.wire_name
                    or node.orelse
                    or not isinstance(seq, ast.Call)
                    or self.path(seq.func) != "range"
                    or seq.keywords
                ):
                    self.fail(node, "Use a bounded for loop over range with no else clause.")
                if "range" in self.values or "range" in self.aliases:
                    self.fail(node, "range must refer to the standard Python constructor.")
                values = range(*(self.value(arg) for arg in seq.args))
                self.charge(node, len(values))
                for value in values:
                    self.values[node.target.id] = value
                    circuit = self.body(node.body, circuit, runtime=runtime)
            elif isinstance(node, ast.If):
                chosen = node.body if self.truth(self.value(node.test), node) else node.orelse
                circuit = self.body(chosen, circuit, runtime=runtime)
            elif isinstance(node, ast.With) and circuit is not None and len(node.items) == 1:
                item = node.items[0]
                call = item.context_expr
                method = self.method(call)
                if (
                    self.constructor != "qiskit.QuantumCircuit"
                    or method not in {"if_test", "while_loop"}
                    or call.keywords
                    or len(call.args) != 1
                ):
                    self.fail(node, "Use an SDK if_test or while_loop circuit context.")
                guard = self.condition(call.args[0])
                empty = Circuit(circuit.num_qubits, num_clbits=circuit.num_clbits)
                values = self.values.copy()
                yes = self.body(node.body, empty, runtime=True)
                self.values = values.copy()
                no = empty
                if item.optional_vars is not None:
                    following = statements[index + 1] if index + 1 < len(statements) else None
                    if (
                        method != "if_test"
                        or not isinstance(item.optional_vars, ast.Name)
                        or not isinstance(following, ast.With)
                        or len(following.items) != 1
                        or _name(following.items[0].context_expr) != item.optional_vars.id
                        or following.items[0].optional_vars is not None
                    ):
                        self.fail(node, "An SDK else context must immediately follow its if_test.")
                    no = self.body(following.body, empty, runtime=True)
                    skip_else = True
                self.values = values
                circuit = (
                    circuit.while_loop(guard, yes)
                    if method == "while_loop"
                    else circuit.if_else(guard, yes, no)
                )
            else:
                self.fail(node, "This statement needs an explicit circuit factory or SDK adapter.")
        return circuit

    def operation(self, circuit, call):
        name = self.method(call)
        if call.keywords:
            self.fail(call, "Circuit operations require positional operands.")
        args = [self.value(arg) for arg in call.args]
        if name in {"break_loop", "continue_loop"} and not args:
            return getattr(circuit, name)()
        if name in {"measure", "reset", "barrier", "classical_set"}:
            return getattr(circuit, name)(*args)
        if name in UNITARY_GATES - {"unitary"}:
            arity, params = GATES[name]
            if len(args) == arity + params:
                return circuit.gate(name, *args[params:], params=args[:params])
        self.fail(call, f"Unsupported circuit operation {name!r} or operand count.")


def from_python(source, *, entrypoint=None, arguments=None, max_operations=100_000, **scan_options):
    """Accept a restricted gate-building entry point with explicit classical values."""
    report = inspect_python(source, **scan_options)
    if not report.ast_parseable or not report.static_scan_complete:
        report.raise_for_errors()
    compiler = _Compiler(report.filename, max_operations)
    module = ast.parse(str.__str__(source), filename=report.filename)
    statements, functions = [], {}
    for item in module.body:
        if isinstance(item, ast.Import):
            if statements:
                compiler.fail(item, "Circuit imports must precede executable statements.")
            for name in item.names:
                if name.name not in {"qiskit", "openqml", "math"}:
                    compiler.fail(item, "Only circuit and mathematical imports are supported.")
                compiler.aliases[name.asname or name.name] = name.name
        elif (
            isinstance(item, ast.ImportFrom)
            and item.level == 0
            and item.module in {"qiskit", "qiskit.circuit", "openqml", "math"}
        ):
            if statements:
                compiler.fail(item, "Circuit imports must precede executable statements.")
            for name in item.names:
                if name.name == "*":
                    compiler.fail(item, "Use explicit imports for circuit constructors.")
                compiler.aliases[name.asname or name.name] = f"{item.module}.{name.name}"
        elif isinstance(item, ast.FunctionDef) and not item.decorator_list:
            functions[item.name] = item
        else:
            statements.append(item)
    if entrypoint is None and not statements and len(functions) == 1:
        entrypoint = next(iter(functions))
    if arguments is not None and not isinstance(arguments, Mapping):
        raise ValidationError("Python entry arguments must be a mapping.")
    try:
        returned = None
        if entrypoint is not None:
            if not isinstance(entrypoint, str) or entrypoint not in functions or statements:
                compiler.fail(module, "Select a single declarative Python entry point.")
            fn = functions[entrypoint]
            params = fn.args
            names = [arg.arg for arg in params.posonlyargs + params.args]
            if (
                params.vararg
                or params.kwarg
                or params.kwonlyargs
                or not fn.body
                or not isinstance(fn.body[-1], ast.Return)
            ):
                compiler.fail(fn, "Use explicit arguments and a final circuit return.")
            values = dict(
                zip(
                    names[len(names) - len(params.defaults) :],
                    (compiler.value(value) for value in params.defaults),
                    strict=True,
                )
            )
            values.update(arguments or {})
            if set(values) != set(names) or any(
                type(value) not in {bool, float, int} for value in values.values()
            ):
                compiler.fail(fn, "Supply concrete scalar values for the entry arguments.")
            compiler.values.update(values)
            statements, returned = fn.body[:-1], fn.body[-1]
        elif functions or arguments:
            report.raise_for_errors()
        circuit = compiler.body(statements)
        if circuit is None:
            report.raise_for_errors()
        if returned is not None and _name(returned.value) != compiler.wire_name:
            if not (
                compiler.method(returned.value) == "build"
                and not returned.value.args
                and not returned.value.keywords
            ):
                compiler.fail(returned, "Return the declared quantum circuit.")
        return circuit
    except ConversionError:
        raise
    except (TypeError, ValueError, OverflowError, ArithmeticError, ValidationError) as exc:
        raise conversion_error("PYTHON_CIRCUIT_VALUE", str(exc), report.filename) from exc

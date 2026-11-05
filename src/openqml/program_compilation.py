"""Structured SDK compilation alongside the strict portable circuit IR path.

Native control flow is not flattened or converted into guessed finite loops.
The SDK imports its supported OpenQASM subset; export preserves scope-local
phase explicitly instead of relying on the SDK's phase-dropping exporter.
"""

import re
from dataclasses import dataclass

from .errors import ConversionError, MissingDependencyError, conversion_error
from .ir import integer


@dataclass(frozen=True)
class ProgramConversion:
    value: object
    source_format: str
    target_format: str
    route: str
    static_work: int
    features: tuple = ()

    @property
    def semantic_scope(self):
        return {
            "unit": "captured_quantum_program",
            "host_python": "executes_before_capture; not_translated",
            "native_control_flow": "retained_on_supported_SDK_route; not_unrolled",
            "global_phase": "explicit_scoped_gphase_on_native_OpenQASM_export",
            "runtime_loop_iterations": "not_bounded_or_executed_by_compilation",
            "external_backend_and_calibrations": "not_converted",
            "verification": "target_syntax_checked; no_whole_program_equivalence_theorem",
        }

    def to_dict(self):
        return {
            "schema": "openqml.program_conversion",
            "version": 1,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "route": self.route,
            "static_work": self.static_work,
            "features": self.features,
            "semantic_scope": self.semantic_scope,
        }


def _native_export(circuit, *, max_operations, max_depth, max_matrix_qubits, max_output_bytes):
    from qiskit import circuit as qiskit_circuit
    from qiskit import qasm3
    from qiskit.circuit import (
        Barrier,
        BreakLoopOp,
        ContinueLoopOp,
        Delay,
        Gate,
        Measure,
        Reset,
    )
    from qiskit.circuit.library import CXGate, GlobalPhaseGate, UGate

    from .adapters._common import dependency

    # Choose an exporter-only basis symbol that cannot collide with any source
    # register, parameter, instruction or classical variable name.
    names, visited, features, work = set(), set(), set(), 0

    def audit(body, depth):
        nonlocal work
        if depth > max_depth:
            raise conversion_error("PROGRAM_DEPTH", "Structured program exceeds max_depth.")
        if id(body) in visited:
            return
        visited.add(id(body))
        if getattr(body, "calibrations", None):
            raise conversion_error(
                "HARDWARE_CALIBRATION_UNSUPPORTED",
                "Native calibration semantics cannot be exported.",
            )
        names.update(register.name for register in (*body.qregs, *body.cregs))
        names.update(p.name for p in body.parameters)
        if callable(getattr(body, "iter_vars", None)):
            names.update(variable.name for variable in body.iter_vars())
        for instruction in body.data:
            work += 1
            if work > max_operations:
                raise conversion_error(
                    "PROGRAM_WORK_LIMIT", "Native program exceeds max_operations."
                )
            operation = instruction.operation
            names.add(operation.name)
            if operation.name in {
                "if_else",
                "while_loop",
                "for_loop",
                "switch_case",
                "break_loop",
                "continue_loop",
                "store",
                "delay",
            }:
                features.add(operation.name)
            for block in getattr(operation, "blocks", ()):
                audit(block, depth + 1)

    audit(circuit, 0)
    marker = "_openqml_scoped_phase"
    while marker in names:
        marker += "_"
    primitives = {UGate, CXGate, Measure, Reset, Barrier, Delay, BreakLoopOp, ContinueLoopOp}
    if hasattr(qiskit_circuit, "Store"):
        primitives.add(qiskit_circuit.Store)

    def lower(body, depth):
        nonlocal work
        if depth > max_depth:
            raise conversion_error("PROGRAM_DEPTH", "Gate/control lowering exceeds max_depth.")
        result = body.copy_empty_like()
        result.global_phase = 0

        def emit_phase(angle):
            if angle != 0:
                result.append(Gate(marker, 0, [angle]), [])
                features.add("scoped_global_phase")

        # A phase belongs to this scope, including a loop's individual body
        # invocation. Keep instruction phases at their original positions.
        emit_phase(body.global_phase)
        for instruction in body.data:
            work += 1
            if work > max_operations:
                raise conversion_error(
                    "PROGRAM_WORK_LIMIT", "Native lowering exceeds max_operations."
                )
            operation = instruction.operation
            blocks = getattr(operation, "blocks", ())
            if blocks:
                operation = operation.replace_blocks(
                    tuple(lower(block, depth + 1) for block in blocks)
                )
                result.append(operation, instruction.qubits, instruction.clbits)
            elif isinstance(operation, GlobalPhaseGate):
                emit_phase(operation.params[0])
            elif operation.base_class in primitives:
                result.append(operation, instruction.qubits, instruction.clbits)
            else:
                if operation.name == "unitary" and (
                    operation.num_qubits > max_matrix_qubits
                    or 3 * operation.num_qubits * 4**operation.num_qubits > max_operations - work
                ):
                    raise conversion_error(
                        "MATRIX_SYNTHESIS_LIMIT", "Unitary synthesis exceeds its width/work budget."
                    )
                definition = operation.definition
                if definition is None:
                    raise conversion_error(
                        "OPAQUE_INSTRUCTION", f"No lossless definition for {operation.name!r}."
                    )
                expanded = lower(definition, depth + 1)
                result.compose(
                    expanded, qubits=instruction.qubits, clbits=instruction.clbits, inplace=True
                )
        return result

    lowered = lower(circuit, 0)
    # SDKs currently discard Circuit.global_phase and emit invalid definitions
    # for zero-wire custom gates. An explicitly declared synthetic basis gate
    # avoids both issues. Rename only that generated token to QASM's primitive.
    text = qasm3.dumps(lowered, basis_gates=("U", marker))
    text = re.sub(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", "gphase", text)
    if len(text.encode("utf-8")) > max_output_bytes:
        raise conversion_error("PROGRAM_OUTPUT_LIMIT", "OpenQASM output exceeds max_output_bytes.")
    parser = dependency("openqasm3", "qasm")
    parser.parse(text)
    return text, work, tuple(sorted(features))


def _import_structured_qasm(
    source, *, max_source_bytes, max_operations, max_depth, max_matrix_qubits
):
    """Stage pure input-angle expressions before the SDK's restricted resolver.

    Host parameters can remain nonlinear ParameterExpressions without becoming
    runtime classical instructions. Expressions involving loop indices/variables
    stay in their scope; no host evaluation or dynamic-loop unrolling occurs.
    """
    from dataclasses import fields

    import openqasm3
    from openqasm3 import ast
    from qiskit.circuit import Parameter

    from ._qasm import clean_source
    from .adapters.openqasm import _angle
    from .adapters.qiskit import to_qiskit
    from .ir import CircuitBuilder
    from .program import QuantumProgram

    tree = openqasm3.parse(clean_source(source, max_source_bytes))
    inputs = {
        node.identifier.name
        for node in tree.statements
        if isinstance(node, ast.IODeclaration)
        and node.io_identifier == ast.IOKeyword.input
        and isinstance(node.type, ast.FloatType)
    }
    names, nodes, pending = set(), 0, [(tree, 0)]
    while pending:
        node, depth = pending.pop()
        if not isinstance(node, ast.QASMNode):
            continue
        nodes += 1
        if depth > max_depth + 16 or nodes > max_operations * 16:
            raise conversion_error(
                "PROGRAM_WORK_LIMIT", "OpenQASM syntax tree exceeds depth/work budget."
            )
        if isinstance(node, ast.Identifier):
            names.add(node.name)
        for field in fields(node):
            if field.name in {"span", "annotations"}:
                continue
            value = getattr(node, field.name)
            pending.extend((x, depth + 1) for x in value) if isinstance(
                value, list
            ) else pending.append((value, depth + 1))
    replacements, cache = {}, {}

    def lift(node, allowed):
        try:
            text = openqasm3.dumps(node)
            value = _angle(text, allowed, "native_parameter_expression")
        except ConversionError:
            return descend(node, True, allowed)
        if value.op is not None and value.parameters:
            if text not in cache:
                name = f"_openqml_angle_{len(replacements)}"
                while name in names:
                    name += "_"
                names.add(name)
                replacements[name], cache[text] = value, name
            return ast.Identifier(cache[text])
        return descend(node, True, allowed)

    def descend(node, angle=False, allowed=None):
        allowed = inputs if allowed is None else allowed
        if not isinstance(node, ast.QASMNode):
            return node
        if isinstance(node, (ast.QuantumGateDefinition, ast.SubroutineDefinition)):
            return node
        if isinstance(node, ast.ForInLoop):
            allowed = allowed - {node.identifier.name}
        if isinstance(node, ast.QuantumGate):
            node.arguments = [lift(value, allowed) for value in node.arguments]
            return node
        if isinstance(node, ast.QuantumPhase):
            node.argument = lift(node.argument, allowed)
            return node
        for field in fields(node):
            if field.name in {"span", "annotations"}:
                continue
            value = getattr(node, field.name)
            if isinstance(value, list):
                available, children = set(allowed), []
                for child in value:
                    children.append(descend(child, angle, available))
                    if isinstance(child, (ast.ClassicalDeclaration, ast.ConstantDeclaration)):
                        available = available - {child.identifier.name}
                setattr(node, field.name, children)
            elif isinstance(value, ast.QASMNode):
                setattr(
                    node,
                    field.name,
                    lift(value, allowed)
                    if angle and isinstance(value, ast.Expression)
                    else descend(value, angle, allowed),
                )
        return node

    descend(tree)
    if replacements:
        declarations = [
            ast.IODeclaration(
                ast.IOKeyword.input, ast.FloatType(ast.IntegerLiteral(64)), ast.Identifier(name)
            )
            for name in replacements
        ]
        # Include statements and other input declarations can follow these;
        # every synthetic declaration precedes its first use.
        tree.statements = declarations + tree.statements
    staged = openqasm3.dumps(tree)
    program = QuantumProgram.from_qasm(
        staged,
        version=3,
        max_source_bytes=max_source_bytes,
        max_operations=max_operations,
        max_depth=max_depth,
        max_matrix_qubits=max_matrix_qubits,
    )
    if replacements:
        lookup = {p.name: p for p in program.native.parameters}
        symbols = {name: lookup.get(name, Parameter(name)) for name in inputs}
        angles = {}
        for name, value in replacements.items():
            # Reuse the portable expression graph's SDK lowering instead of a
            # second symbolic evaluator or unchecked string substitution.
            single = CircuitBuilder(1).rx(value, 0).build()
            angles[lookup[name]] = to_qiskit(single, _symbols=symbols).data[0].operation.params[0]
        program = QuantumProgram.capture(
            program.native.assign_parameters(angles), framework="qiskit"
        )
    return program, bool(replacements)


def compile_program(
    value,
    *,
    target,
    source=None,
    bindings=None,
    strategy="auto",
    max_operations=100_000,
    max_depth=64,
    max_source_bytes=2_000_000,
    max_output_bytes=2_000_000,
    max_matrix_qubits=5,
):
    """Compile a captured program, retaining supported native dynamic structure.

    strategy='ir' requests the existing strict Circuit conversion/certification
    path. 'auto'/'native' use Qiskit's structured route for Qiskit/OpenQASM 3;
    other pairs use portable IR (or fail when native was explicitly required).
    Source text uses bounded syntax frontends; host Python runs before capture.
    """
    from .conversion import _format, convert
    from .program import QuantumProgram, detect_framework

    for name, value_ in (
        ("max_operations", max_operations),
        ("max_source_bytes", max_source_bytes),
        ("max_output_bytes", max_output_bytes),
        ("max_matrix_qubits", max_matrix_qubits),
    ):
        integer(value_, name, 1)
    integer(max_depth, "max_depth")
    if strategy not in {"auto", "native", "ir"}:
        raise conversion_error("PROGRAM_STRATEGY", "strategy must be auto, native or ir.")
    captured, import_options = None, {}
    if isinstance(value, QuantumProgram):
        captured = value
        import_options = dict(value.import_options)
        max_operations = min(
            max_operations,
            integer(import_options.get("max_operations", max_operations), "max_operations", 1),
        )
        max_depth = min(max_depth, integer(import_options.get("max_depth", max_depth), "max_depth"))
        max_matrix_qubits = min(
            max_matrix_qubits,
            integer(
                import_options.get("max_matrix_qubits", max_matrix_qubits), "max_matrix_qubits", 1
            ),
        )
        source = value.framework
        value = value.native
    if source is None:
        source = (
            "openqasm3"
            if isinstance(value, str) and re.match(r"\s*OPENQASM\s+3", value)
            else detect_framework(value)
        )
    source, target = _format(source), _format(target)
    native = (
        source in {"qiskit", "openqasm3"} and target in {"qiskit", "openqasm3"} and strategy != "ir"
    )
    if not native:
        if strategy == "native":
            raise conversion_error(
                "NATIVE_ROUTE_UNAVAILABLE", "This format pair requires the portable IR path."
            )
        result = convert(
            captured if captured is not None else value,
            source=source,
            target=target,
            bindings=bindings,
            max_operations=max_operations,
            max_depth=max_depth,
        )
        return ProgramConversion(
            result.value,
            source,
            target,
            "circuit_ir",
            sum(1 for _ in result.circuit.walk_operations()),
        )
    if set(import_options) - {"max_operations", "max_depth", "max_matrix_qubits"}:
        raise conversion_error(
            "NATIVE_SOURCE_OPTIONS", "Native compilation cannot preserve these source options."
        )
    try:
        lifted = False
        if source == "openqasm3":
            program, lifted = _import_structured_qasm(
                value,
                max_source_bytes=max_source_bytes,
                max_operations=max_operations,
                max_depth=max_depth,
                max_matrix_qubits=max_matrix_qubits,
            )
        else:
            from qiskit import QuantumCircuit

            if not isinstance(value, QuantumCircuit):
                raise conversion_error("SOURCE_TYPE", "Expected a native Qiskit QuantumCircuit.")
            program = QuantumProgram.capture(value, framework="qiskit")
        if bindings is not None:
            program = program.bind(bindings)
        if target == "qiskit":
            # Import has already parsed the SDK's supported structured subset.
            # Audit/export also checks scope depth and phase-bearing definitions.
            _, work, features = _native_export(
                program.native,
                max_operations=max_operations,
                max_depth=max_depth,
                max_matrix_qubits=max_matrix_qubits,
                max_output_bytes=max_output_bytes,
            )
            features = tuple(
                sorted(set(features) | ({"symbolic_angle_lifting"} if lifted else set()))
            )
            return ProgramConversion(
                program.native.copy(), source, target, "native_qiskit", work, features
            )
        text, work, features = _native_export(
            program.native,
            max_operations=max_operations,
            max_depth=max_depth,
            max_matrix_qubits=max_matrix_qubits,
            max_output_bytes=max_output_bytes,
        )
        features = tuple(sorted(set(features) | ({"symbolic_angle_lifting"} if lifted else set())))
        return ProgramConversion(text, source, target, "native_openqasm3", work, features)
    except (ConversionError, MissingDependencyError):
        raise
    except ImportError as exc:
        raise MissingDependencyError("Structured compilation requires openqml[qasm].") from exc
    except Exception as exc:
        raise conversion_error("NATIVE_PROGRAM_COMPILATION", str(exc)) from exc

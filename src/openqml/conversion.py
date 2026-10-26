"""Explicit import -> IR -> export with diagnostics and no silent lossy mode."""

import re
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    ConversionError,
    Diagnostic,
    MissingDependencyError,
    ValidationError,
    conversion_error,
)
from .frontends import BUILTIN_FORMATS, get_adapter, import_native, list_formats
from .ir import CHANNEL_GATES, UNITARY_GATES, Circuit

FORMATS = BUILTIN_FORMATS
_ALIASES = {"qasm2": "openqasm2", "qasm3": "openqasm3"}
_QASM_RESERVED = set(
    "q c qubit bit input output float int uint angle bool complex const "
    "gate def extern if else for while switch case default return let "
    "measure reset barrier include OPENQASM gphase duration stretch "
    "array readonly mutable ctrl negctrl inv pow pi tau".split()
)
# stdgates.inc shares the global namespace with input declarations. Include the
# QASM 3.0 names and documented QASM-2 compatibility aliases used by SDK parsers.
_QASM_RESERVED.update(
    "p x y z h s sdg t tdg sx rx ry rz cx cy cz cp crx cry crz ch swap ccx cswap "
    "cu CX phase cphase id u1 u2 u3 U".split()
)


def _format(value):
    value = _ALIASES.get(value, value)
    if value not in list_formats():
        raise conversion_error("UNKNOWN_FORMAT", f"Supported formats: {', '.join(list_formats())}.")
    return value


def diagnose(circuit: Circuit, target: str) -> tuple[Diagnostic, ...]:
    """Check representability without importing any optional framework."""
    target = _format(target)
    if target not in BUILTIN_FORMATS:
        adapter = get_adapter(target)
        diagnostics = () if adapter.diagnose is None else tuple(adapter.diagnose(circuit))
        if any(not isinstance(item, Diagnostic) for item in diagnostics):
            from .errors import ValidationError

            raise ValidationError("Adapter diagnostics must contain Diagnostic objects.")
        return diagnostics
    diagnostics = []
    if target == "openqasm2":
        if circuit.parameters:
            diagnostics.append(
                Diagnostic(
                    "UNBOUND_PARAMETER",
                    "OpenQASM 2 requires bound angles.",
                    hint="Bind circuit.parameters before exporting.",
                )
            )
        if circuit.global_phase.parameters or circuit.global_phase.constant:
            diagnostics.append(
                Diagnostic(
                    "GLOBAL_PHASE_UNSUPPORTED",
                    "OpenQASM 2 cannot preserve an explicit global phase.",
                    "global_phase",
                    hint="Use OpenQASM 3 or a framework export.",
                )
            )
    if target == "openqasm3":
        for name in circuit.parameters:
            if name in _QASM_RESERVED or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                diagnostics.append(
                    Diagnostic(
                        "RESERVED_IDENTIFIER",
                        f"Parameter {name!r} conflicts with an OpenQASM identifier.",
                        "parameters",
                        hint="Rename the parameter before exporting.",
                    )
                )
    if target == "pennylane":
        if circuit.parameters:
            diagnostics.append(
                Diagnostic(
                    "UNBOUND_PARAMETER",
                    "PennyLane tape export requires bound angles.",
                    hint="Supply bindings.",
                )
            )
        if circuit.num_clbits:
            diagnostics.append(
                Diagnostic(
                    "CLASSICAL_REGISTER_UNSUPPORTED",
                    "Operation-only PennyLane tapes cannot preserve classical bit registers.",
                )
            )
    written = []
    for path, op in circuit.walk_operations():
        location = f"operations[{path[0]}]" + "".join(
            f".blocks[{block}].operations[{index}]"
            for block, index in zip(path[1::2], path[2::2], strict=True)
        )
        if op.name in CHANNEL_GATES:
            if target.startswith("openqasm"):
                diagnostics.append(
                    Diagnostic(
                        "CHANNEL_UNSUPPORTED",
                        "OpenQASM cannot preserve this quantum channel.",
                        location,
                    )
                )
            elif target in {"qiskit", "pennylane", "cirq"} and any(p.parameters for p in op.params):
                diagnostics.append(
                    Diagnostic(
                        "UNBOUND_CHANNEL",
                        "Bind channel probabilities before native export.",
                        location,
                    )
                )
        if op.name == "if_else" and (
            target in {"openqasm2", "pennylane"}
            or (target == "cirq" and any(b.has_measurements for b in op.blocks))
        ):
            diagnostics.append(
                Diagnostic(
                    "CONTROL_FLOW_UNSUPPORTED",
                    f"{target} cannot preserve this atomic if/else block"
                    + (" containing measurements." if target == "cirq" else "."),
                    location,
                )
            )
        if target in {"cirq", "pennylane"} and op.name == "barrier":
            diagnostics.append(
                Diagnostic(
                    "BARRIER_UNSUPPORTED",
                    f"{target} cannot preserve this scheduling barrier.",
                    location,
                )
            )
        if target == "pennylane" and op.name == "measure":
            diagnostics.append(
                Diagnostic(
                    "MEASUREMENT_UNSUPPORTED",
                    "A classical bit write cannot be represented by an operation-only tape.",
                    location,
                )
            )
        if target == "cirq" and op.name == "measure":
            target_bit = op.clbits[0]
            if target_bit in written:
                diagnostics.append(
                    Diagnostic(
                        "REPEATED_MEASUREMENT_KEY",
                        "Repeated bit writes would change Cirq measurement record semantics.",
                        location,
                    )
                )
            written.append(target_bit)
    if target == "cirq" and circuit.num_clbits != (max(written) + 1 if written else 0):
        diagnostics.append(
            Diagnostic(
                "CLASSICAL_REGISTER_UNSUPPORTED",
                "Cirq cannot preserve unused trailing classical register bits.",
            )
        )
    return tuple(diagnostics)


@dataclass(frozen=True)
class ConversionResult:
    value: Any
    circuit: Circuit
    source_format: str
    target_format: str
    diagnostics: tuple[Diagnostic, ...] = ()
    parameter_map: dict[str, str] = field(default_factory=dict)
    optimization: Any = None


def convert(
    value,
    *,
    source: str | None = None,
    target: str,
    bindings=None,
    source_options=None,
    optimize=False,
    optimization_options=None,
) -> ConversionResult:
    from .program import QuantumProgram, detect_framework

    if type(optimize) is not bool:
        raise ValidationError("optimize must be a boolean.")
    try:
        optimization_options = {} if optimization_options is None else dict(optimization_options)
    except (TypeError, ValueError) as exc:
        raise ValidationError("optimization_options must be a mapping of pass settings.") from exc
    if optimization_options and not optimize:
        raise ValidationError("optimization_options requires optimize=True.")
    if set(optimization_options) - {"max_operations", "max_depth", "max_passes"}:
        raise ValidationError("Unknown optimization option.")

    if isinstance(value, QuantumProgram):
        program = value if bindings is None else value.bind(bindings)
        value, source, bindings = program.native, program.framework, None
        source_options = dict(program.import_options) | dict(source_options or {})
    if source is None:
        source = detect_framework(value)
    source, target = _format(source), _format(target)
    source_options = {} if source_options is None else dict(source_options)
    frontend_notes = ()
    program = None
    if source == "ir":
        if not isinstance(value, Circuit):
            raise conversion_error("SOURCE_TYPE", "Expected an OpenQML Circuit.")
        circuit = value
    elif source == "json":
        circuit = Circuit.from_json(value)
    elif source.startswith("openqasm"):
        from .adapters.openqasm import from_openqasm

        parser = source_options.pop("parser", "auto")
        if parser not in {"auto", "builtin", "qiskit"}:
            raise conversion_error("QASM_FRONTEND", "parser must be auto, builtin or qiskit.")
        allowed = {
            "max_source_bytes",
            "max_operations",
            "max_register_bits",
            "max_depth",
            "include_paths",
            "max_matrix_qubits",
            "max_include_depth",
        }
        if set(source_options) - allowed:
            raise conversion_error(
                "QASM_FRONTEND",
                f"Unknown frontend options: {sorted(set(source_options) - allowed)}",
            )
        native_only = bool(source_options.get("include_paths"))
        if parser == "builtin" and native_only:
            raise conversion_error(
                "QASM_FRONTEND", "Custom includes require the qiskit or auto frontend."
            )
        if "max_depth" in source_options:
            from .ir import integer

            integer(source_options["max_depth"], "max_depth")
        if "max_matrix_qubits" in source_options:
            from .ir import integer

            integer(source_options["max_matrix_qubits"], "max_matrix_qubits", 1)
        if "max_include_depth" in source_options:
            from .ir import integer

            integer(source_options["max_include_depth"], "max_include_depth")
        if parser == "qiskit" or native_only:
            program = QuantumProgram.from_qasm(value, version=int(source[-1]), **source_options)
            circuit = program.to_ir()
            frontend_notes = (
                Diagnostic(
                    "SDK_FRONTEND",
                    "Parsed with Qiskit's QASM frontend and dialect.",
                    severity="info",
                ),
            )
        else:
            try:
                builtin_options = {
                    k: v
                    for k, v in source_options.items()
                    if k not in {"include_paths", "max_matrix_qubits", "max_include_depth"}
                }
                circuit = from_openqasm(value, expected_version=int(source[-1]), **builtin_options)
            except ConversionError as original:
                retryable = {
                    "UNSUPPORTED_STATEMENT",
                    "SYNTAX",
                    "ANGLE_EXPRESSION",
                    "QASM2_U_PHASE_CONVENTION",
                }
                if parser == "builtin" or any(
                    d.code not in retryable for d in original.diagnostics
                ):
                    raise
                try:
                    program = QuantumProgram.from_qasm(
                        value, version=int(source[-1]), **source_options
                    )
                except MissingDependencyError:
                    raise original from None
                except ConversionError as error:
                    if any(
                        d.code.endswith("_LIMIT") or d.code in {"QASM_VERSION", "INCLUDE_PATH"}
                        for d in error.diagnostics
                    ):
                        raise
                    raise original from None
                # The source parsed successfully: report actual representability
                # failures rather than hiding them behind the fast parser's error.
                circuit = program.to_ir()
                frontend_notes = (
                    Diagnostic(
                        "SDK_FRONTEND",
                        "Parsed with Qiskit's QASM frontend and dialect.",
                        severity="info",
                    ),
                )
    else:
        circuit = import_native(value, source, **source_options)
    if source == "qiskit" or (program is not None and program.framework == "qiskit"):
        from .adapters.qiskit import static_loop_notes

        frontend_notes += static_loop_notes(value if source == "qiskit" else program.native)
    if source == "cirq" and source_options.get("measurement_map"):
        frontend_notes += tuple(
            Diagnostic(
                "MEASUREMENT_KEY_MAPPED",
                f"Cirq key {key!r} maps to IR bits {tuple(bits)} in measurement qubit order.",
                "measurement_map",
                severity="info",
            )
            for key, bits in source_options["measurement_map"].items()
        )
    if bindings is not None:
        circuit = circuit.bind(bindings)
    optimization = None
    if optimize:
        from .optimization import optimize as optimize_circuit

        optimization = optimize_circuit(circuit, **optimization_options)
        circuit = optimization.circuit
        frontend_notes += optimization.diagnostics
    original_circuit = circuit
    parameter_map = {}
    if target == "openqasm3":
        from .compiler import rename_parameters

        used = set(circuit.parameters) | _QASM_RESERVED
        for index, name in enumerate(circuit.parameters):
            if name in _QASM_RESERVED or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                alias = f"parameter_{index}"
                while alias in used:
                    alias += "_"
                used.add(alias)
                parameter_map[name] = alias
        if parameter_map:
            circuit = rename_parameters(circuit, parameter_map)
    diagnostics = diagnose(circuit, target)
    if any(item.severity == "error" for item in diagnostics):
        raise ConversionError(diagnostics)
    from .compiler import PORTABLE_BASIS, QASM_BASIS

    basis = (
        PORTABLE_BASIS
        if target == "cirq"
        else QASM_BASIS
        if target.startswith("openqasm")
        else None
    )
    notes = (
        ()
        if basis is None
        else tuple(
            Diagnostic(
                "EXACT_GATE_DECOMPOSITION",
                f"{op.name} is lowered to the target basis with full phase.",
                f"operations[{index}]",
                severity="info",
            )
            for index, op in enumerate(circuit.operations)
            if op.name not in basis and op.name in UNITARY_GATES
        )
    )
    if target == "ir":
        result = circuit
    elif target == "json":
        result = circuit.to_json()
    elif target.startswith("openqasm"):
        from .adapters.openqasm import to_openqasm

        result = to_openqasm(circuit, version=int(target[-1]))
    else:
        result = get_adapter(target).exporter(circuit)
    notes += tuple(
        Diagnostic(
            "PARAMETER_RENAMED",
            f"{name!r} is represented by {alias!r}.",
            "parameters",
            severity="info",
        )
        for name, alias in parameter_map.items()
    )
    return ConversionResult(
        result,
        original_circuit,
        source,
        target,
        frontend_notes + diagnostics + notes,
        parameter_map,
        optimization,
    )

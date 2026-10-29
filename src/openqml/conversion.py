"""Explicit import -> IR -> export with diagnostics and no silent lossy mode."""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .compiler import _check_circuit_limits, _operation_location
from .errors import (
    ConversionError,
    Diagnostic,
    MissingDependencyError,
    ValidationError,
    conversion_error,
)
from .frontends import BUILTIN_FORMATS, get_adapter, import_native, list_formats
from .ir import CHANNEL_GATES, UNITARY_GATES, Circuit, integer

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


def _qasm_statement_count(circuit, version, cap):
    """Count emitted lines without expanding repeated false branches into strings."""

    def count(body):
        size = int(
            version == 3 and bool(body.global_phase.parameters or body.global_phase.constant)
        )
        for op in body.operations:
            if op.name == "if_else":
                size += count(op.blocks[0]) + len(op.condition.bits) * (count(op.blocks[1]) + 3)
            else:
                size += 1
            if size > cap:
                return cap + 1
        return size

    headers = 3 + int(bool(circuit.num_clbits))
    if version == 3:
        headers += len(circuit.parameters)
    return min(headers + count(circuit), cap + 1)


def _qasm_output_diagnostics(circuit, target, max_output_statements):
    if (
        target.startswith("openqasm")
        and _qasm_statement_count(circuit, int(target[-1]), max_output_statements)
        > max_output_statements
    ):
        return (
            Diagnostic(
                "QASM_EXPORT_LIMIT",
                "Expanded OpenQASM exceeds max_output_statements.",
                hint="Increase max_output_statements or simplify classical branches.",
            ),
        )
    return ()


def diagnose(
    circuit: Circuit,
    target: str,
    *,
    max_operations=100_000,
    max_depth=64,
    max_output_statements=100_000,
) -> tuple[Diagnostic, ...]:
    """Check representability without importing any optional framework."""
    if not isinstance(circuit, Circuit):
        raise ValidationError("diagnose requires an OpenQML Circuit.")
    target = _format(target)
    integer(max_output_statements, "max_output_statements", 1)
    try:
        _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    except ConversionError as exc:
        return exc.diagnostics
    if target not in BUILTIN_FORMATS:
        adapter = get_adapter(target)
        diagnostics = () if adapter.diagnose is None else tuple(adapter.diagnose(circuit))
        if any(not isinstance(item, Diagnostic) for item in diagnostics):
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
        location = _operation_location(path)
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
        if target == "openqasm3" and op.name == "if_else" and len(op.condition.bits) > 64:
            diagnostics.append(
                Diagnostic(
                    "QASM_EXPORT_LIMIT",
                    "A conditional OpenQASM guard is limited to 64 bits.",
                    location,
                    hint="Split the classical condition into smaller nested conditions.",
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
    return tuple(diagnostics) + _qasm_output_diagnostics(circuit, target, max_output_statements)


@dataclass(frozen=True)
class ConversionResult:
    value: Any
    circuit: Circuit
    source_format: str
    target_format: str
    diagnostics: tuple[Diagnostic, ...] = ()
    parameter_map: dict[str, str] = field(default_factory=dict)
    optimization: Any = None


@dataclass(frozen=True)
class ConversionPlan:
    """Prepared IR and representability report without loading the target SDK.

    Import, binding and optimization errors still raise immediately. Target
    incompatibilities remain in diagnostics until export() is requested.
    """

    circuit: Circuit
    source_format: str
    target_format: str
    diagnostics: tuple[Diagnostic, ...] = ()
    parameter_map: dict[str, str] = field(default_factory=dict)
    optimization: Any = None
    prepared_circuit: Circuit | None = field(default=None, repr=False)
    max_operations: int = 100_000
    max_depth: int = 64
    max_output_statements: int = 100_000

    @property
    def preparation_complete(self):
        """False when an error or deferred SDK synthesis prevents final resource estimates."""
        return self.compatible and not any(
            d.code == "MATRIX_SYNTHESIS_REQUIRED" for d in self.diagnostics
        )

    @property
    def required_extras(self):
        """Compatibility field: built-in integrations are default dependencies."""
        return ()

    @property
    def required_integrations(self):
        """Integrations used at export, without importing or probing their packages."""
        extras = (
            {self.target_format} if self.target_format in {"qiskit", "cirq", "pennylane"} else set()
        )
        if any(d.code == "MATRIX_SYNTHESIS_REQUIRED" for d in self.diagnostics):
            extras.add("qiskit")
        return tuple(sorted(extras))

    @property
    def compatible(self):
        return not any(d.severity == "error" for d in self.diagnostics)

    @property
    def semantic_scope(self):
        """A compatible plan covers circuit IR, not its surrounding Python runtime."""
        return {
            "unit": "circuit",
            "python_program": "not_converted",
            "dynamic_control": "representable_IR_subset_only",
            "custom_instructions": "supported_definition_matrix_or_adapter_required",
            "noise": "explicit_supported_IR_channels_only",
            "external_backend_configuration": "not_preserved",
            "hardware_calibrations": "not_preserved",
            "numerical_verification": "not_run_by_planning",
        }

    def raise_for_errors(self):
        if not self.compatible:
            raise ConversionError(self.diagnostics)

    def to_dict(self):
        from .compiler import circuit_stats

        return {
            "schema": "openqml.conversion_plan",
            "version": 1,
            "source_format": self.source_format,
            "target_format": self.target_format,
            "compatible": self.compatible,
            "semantic_scope": self.semantic_scope,
            "preparation_complete": self.preparation_complete,
            "required_extras": list(self.required_extras),
            "required_integrations": list(self.required_integrations),
            "parameters": list(self.circuit.parameters),
            "parameter_map": dict(self.parameter_map),
            "resources": asdict(circuit_stats(self.circuit)),
            "prepared_resources": (
                asdict(circuit_stats(self.prepared_circuit or self.circuit))
                if self.preparation_complete
                else None
            ),
            "limits": {
                "max_operations": self.max_operations,
                "max_depth": self.max_depth,
                "max_output_statements": self.max_output_statements,
            },
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def verify(
        self,
        bindings=None,
        *,
        atol=1e-10,
        rtol=1e-10,
        max_qubits=6,
        max_memory_mb=64,
        max_work=10_000_000,
        max_branches=256,
        basis_chunk_size=16,
        source_options=None,
        mode="roundtrip",
    ):
        """Numerically check export/reimport over a complete quantum input basis.

        Unitary checks retain phase. Effectful checks retain final classical
        storage and compare linear maps on every matrix unit, with classical
        input zero. This checks the adapters at one binding through the IR
        simulator; it is not a target-runtime test or an all-parameter proof.
        Basis chunks, work, width and branch counts are explicitly bounded.
        mode='native' executes the exported SDK artifact without reimporting it.
        It supports unitary circuits and channels/reset without recorded
        measurement or feedback. OpenQASM uses Qiskit's official frontend.
        """
        from .verification import _verify_conversion

        return _verify_conversion(
            self,
            bindings,
            atol=atol,
            rtol=rtol,
            max_qubits=max_qubits,
            max_memory_mb=max_memory_mb,
            max_work=max_work,
            max_branches=max_branches,
            basis_chunk_size=basis_chunk_size,
            source_options=source_options,
            mode=mode,
        )

    def export(self):
        """Export exactly this plan, refusing every reported incompatibility."""
        self.raise_for_errors()
        circuit = self.prepared_circuit if self.prepared_circuit is not None else self.circuit
        if self.target_format == "ir":
            value = circuit
        elif self.target_format == "json":
            value = circuit.to_json()
        elif self.target_format.startswith("openqasm"):
            from .adapters.openqasm import to_openqasm

            value = to_openqasm(
                circuit,
                version=int(self.target_format[-1]),
                max_operations=self.max_operations,
                max_depth=self.max_depth,
                max_output_statements=self.max_output_statements,
            )
        elif self.target_format in {"cirq", "pennylane"}:
            value = get_adapter(self.target_format).exporter(
                circuit, max_operations=self.max_operations, max_depth=self.max_depth
            )
        else:
            value = get_adapter(self.target_format).exporter(circuit)
        return ConversionResult(
            value,
            self.circuit,
            self.source_format,
            self.target_format,
            self.diagnostics,
            dict(self.parameter_map),
            self.optimization,
        )


def plan_conversion(
    value,
    *,
    source: str | None = None,
    target: str,
    bindings=None,
    source_options=None,
    optimize=False,
    optimization_options=None,
    max_operations=100_000,
    max_depth=64,
    max_output_statements=100_000,
) -> ConversionPlan:
    """Import and prepare a conversion; report all known target incompatibilities."""
    from .program import QuantumProgram, detect_framework

    max_operations = integer(max_operations, "max_operations", 1)
    max_depth = integer(max_depth, "max_depth")
    max_output_statements = integer(max_output_statements, "max_output_statements", 1)
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
    if source == "python" and "python" not in list_formats():
        from .inspection import inspect_python

        inspect_python(value, **(source_options or {})).raise_for_errors()
    if source is None:
        source = detect_framework(value)
    source, target = _format(source), _format(target)
    source_options = {} if source_options is None else dict(source_options)
    if source in BUILTIN_FORMATS:
        context = set(source_options) & {
            "noise_model",
            "backend",
            "calibrations",
            "execution_context",
            "hardware_options",
        }
        if context:
            raise ConversionError(
                tuple(
                    Diagnostic(
                        "EXTERNAL_EXECUTION_CONTEXT_UNSUPPORTED",
                        "Circuit conversion does not preserve external noise, calibration or "
                        "backend context. Represent supported channels/control explicitly in IR.",
                        f"source_options.{name}",
                    )
                    for name in sorted(context)
                )
            )
        if source == "qiskit" and getattr(value, "calibrations", None):
            raise conversion_error(
                "HARDWARE_CALIBRATION_UNSUPPORTED",
                "Native pulse calibrations cannot be represented by circuit IR.",
                "calibrations",
            )
    # Bound expansion inside built-in frontends, before their returned IR exists.
    # Custom adapters retain their declared option contracts and the final check.
    if source in BUILTIN_FORMATS and source != "ir":
        for name, bound in (("max_operations", max_operations), ("max_depth", max_depth)):
            source_options[name] = min(integer(source_options.get(name, bound), name), bound)
    frontend_notes = ()
    program = None
    if source == "ir":
        if not isinstance(value, Circuit):
            raise conversion_error("SOURCE_TYPE", "Expected an OpenQML Circuit.")
        circuit = value
    elif source == "json":
        allowed = {
            "max_source_bytes",
            "max_operations",
            "max_depth",
            "max_matrix_qubits",
            "max_matrix_entries",
        }
        if set(source_options) - allowed:
            raise conversion_error("IR_FRONTEND", "Unknown JSON IR import option.")
        options = dict(source_options)
        if "max_source_bytes" in options:
            options["max_bytes"] = options.pop("max_source_bytes")
        circuit = Circuit.from_json(value, **options)
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
            integer(source_options["max_depth"], "max_depth")
        if "max_matrix_qubits" in source_options:
            integer(source_options["max_matrix_qubits"], "max_matrix_qubits", 1)
        if "max_include_depth" in source_options:
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
    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
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
    diagnostics = diagnose(
        circuit,
        target,
        max_operations=max_operations,
        max_depth=max_depth,
        max_output_statements=max_output_statements,
    )
    from .compiler import PORTABLE_BASIS, _qasm_basis, decompose

    basis = (
        PORTABLE_BASIS
        if target == "cirq"
        else _qasm_basis(circuit)
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
                _operation_location(path),
                severity="info",
            )
            for path, op in circuit.walk_operations()
            if op.name not in basis and op.name in UNITARY_GATES
        )
    )
    notes += tuple(
        Diagnostic(
            "PARAMETER_RENAMED",
            f"{name!r} is represented by {alias!r}.",
            "parameters",
            severity="info",
        )
        for name, alias in parameter_map.items()
    )
    if target.startswith("openqasm"):
        notes += tuple(
            Diagnostic(
                "MATRIX_SYNTHESIS_REQUIRED",
                "Matrix synthesis uses Qiskit at export; final gate count and phase are deferred.",
                _operation_location(path),
                severity="info",
                hint="Qiskit synthesis is included in the default OpenQML installation.",
            )
            for path, op in circuit.walk_operations()
            if op.name == "unitary" and op.name in basis
        )
    if basis is not None:
        try:
            lowered = decompose(
                circuit, basis=basis, max_operations=max_operations, max_depth=max_depth
            )
        except ConversionError as exc:
            diagnostics += exc.diagnostics
            notes = tuple(d for d in notes if d.code != "EXACT_GATE_DECOMPOSITION")
        else:
            circuit = lowered
            # Source diagnostics retain their original operation locations.
            # Lowering may expose a phase or a larger emitted statement count.
            if (
                target == "openqasm2"
                and (circuit.global_phase.parameters or circuit.global_phase.constant)
                and not any(d.code == "GLOBAL_PHASE_UNSUPPORTED" for d in diagnostics)
            ):
                diagnostics += (
                    Diagnostic(
                        "GLOBAL_PHASE_UNSUPPORTED",
                        "Lowering introduces a phase OpenQASM 2 cannot preserve.",
                        "global_phase",
                        hint="Use OpenQASM 3 or a framework export.",
                    ),
                )
            if not any(d.code == "QASM_EXPORT_LIMIT" for d in diagnostics):
                diagnostics += _qasm_output_diagnostics(circuit, target, max_output_statements)
    return ConversionPlan(
        original_circuit,
        source,
        target,
        frontend_notes + diagnostics + notes,
        parameter_map,
        optimization,
        circuit,
        max_operations,
        max_depth,
        max_output_statements,
    )


def convert(
    value,
    *,
    source: str | None = None,
    target: str,
    bindings=None,
    source_options=None,
    optimize=False,
    optimization_options=None,
    max_operations=100_000,
    max_depth=64,
    max_output_statements=100_000,
) -> ConversionResult:
    """Import -> prepare -> export, rejecting unsupported semantics explicitly."""
    return plan_conversion(
        value,
        source=source,
        target=target,
        bindings=bindings,
        source_options=source_options,
        optimize=optimize,
        optimization_options=optimization_options,
        max_operations=max_operations,
        max_depth=max_depth,
        max_output_statements=max_output_statements,
    ).export()

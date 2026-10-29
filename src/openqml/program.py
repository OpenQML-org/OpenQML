"""Native quantum programs and callable factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import ConversionError, MissingDependencyError, conversion_error
from .ir import Circuit


def detect_framework(value):
    if isinstance(value, Circuit):
        return "ir"
    module = type(value).__module__
    if module.startswith("qiskit"):
        return "qiskit"
    if module.startswith("cirq"):
        return "cirq"
    if module.startswith("pennylane"):
        return "pennylane"
    from .frontends import detect_custom

    custom = detect_custom(value)
    if custom is not None:
        return custom
    raise conversion_error("SOURCE_FRAMEWORK", "Specify the framework for this native object.")


@dataclass(frozen=True)
class QuantumProgram:
    native: Any
    framework: str
    import_options: tuple = field(default=(), repr=False)

    @classmethod
    def capture(cls, value, *, framework=None):
        return cls(value, framework or detect_framework(value))

    @classmethod
    def from_callable(cls, factory: Callable, *args, framework=None, **kwargs):
        """Call caller-provided Python code normally; source strings are never eval'ed."""
        if not callable(factory):
            raise TypeError("factory must be a callable, not source text")
        return cls.capture(factory(*args, **kwargs), framework=framework)

    @classmethod
    def from_qasm(
        cls,
        source: str,
        *,
        version=None,
        include_paths=(),
        max_source_bytes=2_000_000,
        max_operations=100_000,
        max_register_bits=100_000,
        max_depth=64,
        max_matrix_qubits=10,
        max_include_depth=16,
    ):
        from ._qasm import check_literal_registers, clean_source, expand_includes, validate_header
        from .adapters._common import dependency
        from .ir import integer

        integer(max_operations, "max_operations", 1)
        integer(max_depth, "max_depth")
        integer(max_matrix_qubits, "max_matrix_qubits", 1)
        source = clean_source(source, max_source_bytes)
        version = validate_header(source, version)
        source = expand_includes(
            source, include_paths, version, max_source_bytes, max_include_depth
        )
        check_literal_registers(source, max_register_bits)
        dependency("qiskit", "qasm")
        try:
            if version == 2:
                from qiskit import qasm2

                native = qasm2.loads(source, include_path=())
            elif version == 3:
                from qiskit import qasm3

                native = qasm3.loads(source)
            else:
                raise conversion_error("QASM_VERSION", "Expected QASM 2 or 3.")
        except (MissingDependencyError, ConversionError):
            raise
        except ImportError as exc:
            raise MissingDependencyError(
                "Install openqml[qasm] for the official QASM frontend."
            ) from exc
        except Exception as exc:
            raise conversion_error("SDK_QASM_PARSE", str(exc)) from exc
        if native.num_qubits + native.num_clbits > max_register_bits:
            raise conversion_error(
                "REGISTER_LIMIT", "SDK register sizes exceed the configured limit."
            )
        if len(native.data) > max_operations:
            raise conversion_error(
                "OPERATION_LIMIT", "SDK instruction count exceeds the configured limit."
            )
        return cls(
            native,
            "qiskit",
            (
                ("max_operations", max_operations),
                ("max_depth", max_depth),
                ("max_matrix_qubits", max_matrix_qubits),
            ),
        )

    def bind(self, bindings):
        if self.framework == "qiskit":
            lookup = {p.name: p for p in self.native.parameters}
            unknown = set(bindings) - set(lookup)
            if unknown:
                raise conversion_error("PARAMETER_BINDING", f"Unknown parameters: {unknown}")
            return replace(
                self,
                native=self.native.assign_parameters({lookup[k]: v for k, v in bindings.items()}),
            )
        if self.framework == "cirq":
            import cirq

            return replace(self, native=cirq.resolve_parameters(self.native, bindings))
        if self.framework == "ir":
            return replace(self, native=self.native.bind(bindings))
        if self.framework != "pennylane":
            from .frontends import get_adapter

            adapter = get_adapter(self.framework)
            native = (
                adapter.binder(self.native, bindings)
                if adapter.binder
                else adapter.exporter(self.to_ir().bind(bindings))
            )
            return replace(self, native=native)
        raise conversion_error(
            "NATIVE_BINDING",
            "Bind a PennyLane callable when creating its tape or use a native QNode layer.",
        )

    def to_ir(self, bindings=None, **options):
        program = self if bindings is None else self.bind(bindings)
        if program.framework == "ir":
            return program.native
        from .frontends import import_native

        return import_native(
            program.native, program.framework, **(dict(program.import_options) | options)
        )

    def convert(
        self,
        target,
        *,
        bindings=None,
        source_options=None,
        optimize=False,
        optimization_options=None,
        max_operations=100_000,
        max_depth=64,
        max_output_statements=100_000,
    ):
        from .conversion import convert

        return convert(
            self,
            target=target,
            bindings=bindings,
            source_options=source_options,
            optimize=optimize,
            optimization_options=optimization_options,
            max_operations=max_operations,
            max_depth=max_depth,
            max_output_statements=max_output_statements,
        )

    def run(self, *, shots=1024, seed=None, **options):
        """Retain native execution semantics/result objects, including non-IR features."""
        if self.framework == "qiskit":
            from .adapters._common import dependency

            aer = dependency("qiskit_aer", "aer")
            from qiskit import transpile

            backend = aer.AerSimulator(**options)
            circuit = self.native
            if not circuit.num_clbits:
                circuit = circuit.copy()
                circuit.measure_all()
            if circuit.num_qubits > backend.num_qubits:
                backend.set_max_qubits(circuit.num_qubits)
            return backend.run(
                transpile(circuit, backend, optimization_level=0), shots=shots, seed_simulator=seed
            ).result()
        if self.framework == "cirq":
            import cirq

            return cirq.Simulator(seed=seed, **options).run(self.native, repetitions=shots)
        if self.framework == "pennylane":
            import pennylane as qml

            device = options.pop("device", None)
            if device is None:
                device = qml.device(
                    "default.qubit", wires=self.native.wires, shots=shots, seed=seed
                )
            return qml.execute((self.native,), device, **options)[0]
        from .simulator import execute

        if self.framework != "ir":
            from .frontends import get_adapter

            runner = get_adapter(self.framework).runner
            if runner is not None:
                return runner(self.native, shots=shots, seed=seed, **options)
        return execute(self.to_ir(), shots=shots, seed=seed, **options)


def as_circuit(value, *, framework=None, bindings=None, **options):
    if isinstance(value, QuantumProgram):
        return value.to_ir(bindings, **options)
    return QuantumProgram.capture(value, framework=framework).to_ir(bindings, **options)

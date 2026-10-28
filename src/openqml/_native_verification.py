"""Independent execution of exported artifacts on complete spanning inputs.

Native SDK state evolution is deliberately separate from IR reimport. Channels
use physical pure-state probes spanning the operator space, so SDKs need not
accept nonphysical matrix units. No sampling or phase alignment is used.

SDK entry points:
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.DensityMatrix
https://quantumai.google/reference/python/cirq/DensityMatrixSimulator
https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html
"""

from itertools import islice

import numpy as np

from .errors import (
    ConversionError,
    MissingDependencyError,
    ResourceError,
    ValidationError,
    conversion_error,
)


def _unsupported(message):
    return conversion_error(
        "NATIVE_VERIFICATION_UNSUPPORTED",
        message,
        hint="Use roundtrip verification for retained measurement records and feedback.",
    )


def _native_scope(circuit, target, source_options):
    if source_options:
        raise ValidationError("source_options apply to roundtrip verification only.")
    if target not in {"qiskit", "cirq", "pennylane", "openqasm2", "openqasm3"}:
        raise _unsupported("Native verification requires a supported SDK or OpenQASM target.")
    if any(op.name in {"measure", "if_else"} for _, op in circuit.walk_operations()):
        raise _unsupported(
            "Native complete-input checks do not support recorded measurement or feedback."
        )


def _probes(width, channel):
    dimension = 1 << width
    for index in range(dimension):
        vector = np.zeros(dimension, dtype=complex)
        vector[index] = 1
        yield vector
    if channel:
        for first in range(dimension):
            for second in range(first + 1, dimension):
                for phase in (1, 1j):
                    vector = np.zeros(dimension, dtype=complex)
                    vector[first], vector[second] = 1 / np.sqrt(2), phase / np.sqrt(2)
                    yield vector


def _profile(arities, width, instrument, max_operations):
    work = 1
    for count, arity in enumerate(arities, 1):
        if count > max_operations:
            raise ResourceError("Native verification exceeds max_operations.")
        if not 0 <= arity <= width:
            raise ValidationError("Native artifact uses an unexpected quantum register.")
        work += 16 * (1 << (2 * arity)) if instrument else 1 << arity
    return work


def _qiskit(value, circuit, instrument, max_operations):
    from qiskit.circuit import Gate
    from qiskit.quantum_info import DensityMatrix, Statevector

    if (value.num_qubits, value.num_clbits) != (circuit.num_qubits, circuit.num_clbits):
        raise ValidationError("Native artifact changed quantum or classical register width.")
    if value.parameters:
        raise ValidationError("Native artifact contains unbound parameters.")

    def arities():
        for item in value.data:
            if item.operation.name == "measure" or getattr(item.operation, "blocks", ()):
                raise _unsupported("Native artifact contains recorded measurement or control flow.")
            if item.clbits:
                raise _unsupported("Native artifact contains a classical instruction.")
            if (
                not instrument
                and not isinstance(item.operation, Gate)
                and item.operation.name != "barrier"
            ):
                raise _unsupported("Native artifact is not unitary.")
            yield len(item.qubits)

    work = _profile(arities(), circuit.num_qubits, instrument, max_operations)
    width = circuit.num_qubits
    permutation = (
        np.arange(1 << width)
        .reshape((2,) * width)
        .transpose(tuple(reversed(range(width))))
        .reshape(-1)
    )

    def run(probes):
        outputs = []
        for vector in probes:
            state = (DensityMatrix if instrument else Statevector)(vector[permutation]).evolve(
                value
            )
            output = np.asarray(state.data)
            outputs.append(
                output[np.ix_(permutation, permutation)].reshape(-1)
                if instrument
                else output[permutation]
            )
        return np.stack(outputs)

    return run, work


def _cirq(value, circuit, instrument, max_operations):
    import cirq

    order = tuple(cirq.LineQubit.range(circuit.num_qubits))
    if value.all_qubits() != frozenset(order):
        raise ValidationError("Native artifact changed canonical quantum wires.")

    def arities():
        for op in value.all_operations():
            if (
                isinstance(op, cirq.CircuitOperation)
                or cirq.is_measurement(op)
                or op.classical_controls
            ):
                raise _unsupported("Native artifact contains measurement or nested control flow.")
            if cirq.is_parameterized(op):
                raise ValidationError("Native artifact contains unbound parameters.")
            if not instrument and not cirq.has_unitary(op):
                raise _unsupported(
                    "Native artifact is not unitary; stochastic verification is refused."
                )
            yield len(op.qubits)

    work = _profile(arities(), circuit.num_qubits, instrument, max_operations)
    simulator = (
        cirq.DensityMatrixSimulator(dtype=np.complex128, split_untangled_states=False)
        if instrument
        else cirq.Simulator(dtype=np.complex128)
    )

    def run(probes):
        outputs = []
        for vector in probes:
            result = simulator.simulate(value, initial_state=vector, qubit_order=order)
            output = result.final_density_matrix if instrument else result.final_state_vector
            outputs.append(np.asarray(output).reshape(-1))
        return np.stack(outputs)

    return run, work


def _pennylane(value, circuit, instrument, max_operations):
    import pennylane as qml

    if value.measurements:
        raise _unsupported("Native PennyLane verification requires an operation-only tape.")
    order = tuple(range(circuit.num_qubits))

    def arities():
        for op in value.operations:
            if not set(op.wires).issubset(order):
                raise ValidationError("Native artifact changed canonical quantum wires.")
            if not op.has_matrix and not isinstance(op, qml.operation.Channel):
                raise _unsupported("Native tape contains an unsupported dynamic operation.")
            if not instrument and isinstance(op, qml.operation.Channel):
                raise _unsupported("Native artifact is not unitary.")
            yield len(op.wires)

    work = _profile(arities(), circuit.num_qubits, instrument, max_operations)
    device = qml.device("default.mixed" if instrument else "default.qubit", wires=order)

    @qml.qnode(device, interface=None, diff_method=None)
    def program(vector):
        qml.StatePrep(vector, wires=order)
        for op in value.operations:
            qml.apply(op)
        return qml.density_matrix(wires=order) if instrument else qml.state()

    def run(probes):
        return np.stack([np.asarray(program(vector)).reshape(-1) for vector in probes])

    return run, work


def _verify_native(
    plan, value, instrument, records, peak, size, work, max_work, budget, chunk, atol, rtol
):
    from .verification import _basis_capacity, _Comparison, _evolve_basis

    circuit, target = plan.circuit, plan.target_format
    backend = "qiskit" if target.startswith("openqasm") else target
    try:
        if target.startswith("openqasm"):
            from .program import QuantumProgram

            value = QuantumProgram.from_qasm(
                value,
                version=int(target[-1]),
                max_operations=plan.max_operations,
                max_depth=plan.max_depth,
                max_register_bits=circuit.num_qubits + circuit.num_clbits,
                max_matrix_qubits=circuit.num_qubits,
            ).native
        runner, native_work = {"qiskit": _qiskit, "cirq": _cirq, "pennylane": _pennylane}[backend](
            value, circuit, instrument, plan.max_operations
        )
    except ImportError as exc:
        extra = "qasm" if target.startswith("openqasm") else backend
        raise MissingDependencyError(
            f"Cannot load {extra} for verification. Run: python -m pip install openqml."
        ) from exc
    except (ConversionError, ResourceError, ValidationError):
        raise
    except Exception as exc:
        raise conversion_error(
            "NATIVE_VERIFICATION_PREPARATION", f"{backend} artifact preparation failed: {exc}"
        ) from exc
    if work + size * size * native_work > max_work:
        raise ResourceError("Native verification exceeds max_work.")
    capacity = _basis_capacity((circuit,), (peak, 1), size, budget)
    chunk, probes, comparison = (
        min(chunk, capacity, size),
        _probes(circuit.num_qubits, instrument),
        _Comparison(),
    )
    for start in range(0, size, chunk):
        vectors = np.stack(tuple(islice(probes, min(chunk, size - start))))
        basis = (
            (vectors[..., :, None] * vectors[..., None, :].conj()).reshape(len(vectors), -1)
            if instrument
            else vectors
        )
        expected = _evolve_basis(circuit, basis, instrument, records)
        try:
            actual = {frozenset(): runner(vectors)}
        except (ConversionError, ResourceError, ValidationError, MissingDependencyError):
            raise
        except Exception as exc:
            raise conversion_error(
                "NATIVE_VERIFICATION_EXECUTION", f"{backend} execution failed: {exc}"
            ) from exc
        comparison.add(expected, actual, start, atol, rtol)
        del vectors, basis, expected, actual
    return comparison.result(
        circuit,
        target,
        "quantum_channel" if instrument else "unitary_including_global_phase",
        size,
        atol,
        rtol,
        "native",
        backend,
    )

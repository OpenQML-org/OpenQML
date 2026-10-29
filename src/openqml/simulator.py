"""Dense simulation with NumPy, PyTorch and JAX."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from .errors import (
    BackendError,
    MissingDependencyError,
    ResourceError,
    ValidationError,
)
from .ir import UNITARY_GATES, Circuit, integer
from .parameters import real

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_FIXED = {
    "id": _I,
    "x": _X,
    "y": _Y,
    "z": _Z,
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]),
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2,
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
}
for _name, _matrix in (("cx", _X), ("cy", _Y), ("cz", _Z)):
    _controlled = np.eye(4, dtype=complex)
    _controlled[2:, 2:] = _matrix
    _FIXED[_name] = _controlled
_FIXED["ccx"] = np.eye(8, dtype=complex)
_FIXED["ccx"][6:, 6:] = _X
_FIXED["sxdg"] = _FIXED["sx"].conj().T
_FIXED["ch"] = np.eye(4, dtype=complex)
_FIXED["ch"][2:, 2:] = _FIXED["h"]
_FIXED["cswap"] = np.eye(8, dtype=complex)
_FIXED["cswap"][4:, 4:] = _FIXED["swap"]
_FIXED["iswap"] = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])


class _TensorOps:
    """Keep framework tensors on their device and inside their differentiation graph."""

    def __init__(self, backend="numpy", like=None):
        self.backend, self.device = backend, None
        try:
            if backend == "numpy":
                self.xp, self.dtype, self.real_dtype = np, np.complex128, np.float64
            elif backend == "torch":
                import torch

                self.xp = torch
                self.real_dtype = like.dtype if like is not None else torch.get_default_dtype()
                if self.real_dtype not in {torch.float32, torch.float64}:
                    raise ValidationError("PyTorch simulation requires float32 or float64 angles.")
                self.dtype = (
                    torch.complex128 if self.real_dtype == torch.float64 else torch.complex64
                )
                self.device = like.device if like is not None else torch.device("cpu")
            elif backend == "jax":
                import jax.numpy as jnp

                self.xp = jnp
                self.real_dtype = jnp.asarray(like if like is not None else 0.0).dtype
                if self.real_dtype not in {jnp.dtype("float32"), jnp.dtype("float64")}:
                    raise ValidationError("JAX simulation requires float32 or float64 angles.")
                self.dtype = jnp.complex128 if self.real_dtype == jnp.float64 else jnp.complex64
            else:
                raise ValidationError(f"Unknown tensor backend: {backend!r}.")
        except ImportError as exc:
            raise MissingDependencyError(
                f'Cannot load {backend}. Run: python -m pip install "openqml[{backend}]".'
            ) from exc

    def array(self, value, *, real_value=False):
        dtype = self.real_dtype if real_value else self.dtype
        if self.backend == "torch":
            return self.xp.as_tensor(value, dtype=dtype, device=self.device)
        return self.xp.asarray(value, dtype=dtype)

    def stack(self, values, axis=0):
        arrays = tuple(self.array(value) for value in values)
        arrays = (
            self.xp.broadcast_tensors(*arrays)
            if self.backend == "torch"
            else self.xp.broadcast_arrays(*arrays)
        )
        return (
            self.xp.stack(tuple(arrays), dim=axis)
            if self.backend == "torch"
            else self.xp.stack(tuple(arrays), axis=axis)
        )

    def matrix(self, rows):
        return self.stack([self.stack(row, axis=-1) for row in rows], axis=-2)

    def transpose(self, value, axes):
        return value.permute(tuple(axes)) if self.backend == "torch" else value.transpose(axes)

    def conj(self, value):
        return self.xp.conj(value)

    def real(self, value):
        return self.xp.real(value)


def _gate_matrix(name, params, ops, matrix=None):
    if name == "unitary":
        return ops.array(matrix)
    if name in _FIXED:
        return ops.array(_FIXED[name])
    angle = ops.array(params[0], real_value=True)
    c, s = ops.xp.cos(angle / 2), ops.xp.sin(angle / 2)
    if name == "rx":
        return ops.matrix([[c, -1j * s], [-1j * s, c]])
    if name == "ry":
        return ops.matrix([[c, -s], [s, c]])
    if name == "rz":
        return ops.matrix([[ops.xp.exp(-0.5j * angle), 0], [0, ops.xp.exp(0.5j * angle)]])
    if name == "p":
        return ops.matrix([[1, 0], [0, ops.xp.exp(1j * angle)]])
    if name == "u":
        phi, lam = (ops.array(p, real_value=True) for p in params[1:])
        return ops.matrix(
            [
                [c, -ops.xp.exp(1j * lam) * s],
                [ops.xp.exp(1j * phi) * s, ops.xp.exp(1j * (phi + lam)) * c],
            ]
        )
    if name == "cp":
        return ops.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, ops.xp.exp(1j * angle)]]
        )
    if name in {"crx", "cry", "crz"}:
        local = _gate_matrix(name[1:], params, ops)
        return ops.matrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, local[..., 0, 0], local[..., 0, 1]],
                [0, 0, local[..., 1, 0], local[..., 1, 1]],
            ]
        )
    if name in {"rxx", "ryy", "rzz"}:
        pauli = {"rxx": _X, "ryy": _Y, "rzz": _Z}[name]
        return c[..., None, None] * ops.array(np.eye(4)) - 1j * s[..., None, None] * ops.array(
            np.kron(pauli, pauli)
        )
    raise ValidationError(f"No matrix for gate {name!r}.")


def _apply_numpy(state, matrix, qubits, num_qubits):
    """Apply local matrices through bounded gather/matmul/scatter tiles.

    One output state is allocated. Work tiles target one MiB, with a minimum
    of one local column; no whole-state permutation or index table is built.
    Input/matrix storage and backend BLAS workspaces are separate.
    """
    dimension = 1 << len(qubits)
    columns = 1 << (num_qubits - len(qubits))
    batch = max(1, state.size // state.shape[-1])
    dtype = np.result_type(state, matrix)
    index_bytes = np.dtype(np.intp).itemsize
    per_column = 3 * dimension * batch * dtype.itemsize + (dimension + 4) * index_bytes
    tile = max(1, min(4096, (1 << 20) // per_column))
    local = np.arange(dimension, dtype=np.intp)
    offsets = sum(
        ((local >> (len(qubits) - 1 - i)) & 1) << (num_qubits - 1 - q) for i, q in enumerate(qubits)
    )
    positions = sorted(num_qubits - 1 - q for q in qubits)
    result = np.empty(state.shape, dtype=dtype)
    block = np.empty(state.shape[:-1] + (dimension, tile), dtype=state.dtype)
    evolved = np.empty(state.shape[:-1] + (dimension, tile), dtype=dtype)
    indices = np.empty((dimension, tile), dtype=np.intp)
    for start in range(0, columns, tile):
        base = np.arange(start, min(columns, start + tile), dtype=np.intp)
        for bit in positions:
            mask = (1 << bit) - 1
            base = ((base & ~mask) << 1) | (base & mask)
        size = len(base)
        selected = indices[:, :size]
        np.bitwise_or(offsets[:, None], base[None, :], out=selected)
        # Every generated index is in range. clip avoids take's mandatory
        # temporary out buffer in raise mode; buffers are reused across tiles.
        np.take(state, selected, axis=-1, out=block[..., :size], mode="clip")
        np.matmul(matrix, block[..., :size], out=evolved[..., :size])
        result[..., selected] = evolved[..., :size]
    return result


def _apply_matrix(state, matrix, qubits, num_qubits, ops):
    """Apply only a local 2**k matrix, without constructing a 2**n square unitary."""
    if (
        ops.backend == "numpy"
        and state.size >= 65_536
        and tuple(qubits) != tuple(range(len(qubits)))
    ):
        return _apply_numpy(state, matrix, qubits, num_qubits)
    batch_shape = tuple(state.shape[:-1])
    rank = len(batch_shape)
    local = tuple(qubits) + tuple(q for q in range(num_qubits) if q not in qubits)
    axes = tuple(range(rank)) + tuple(rank + q for q in local)
    inverse = tuple(range(rank)) + tuple(rank + local.index(q) for q in range(num_qubits))
    block = ops.transpose(state.reshape(batch_shape + (2,) * num_qubits), axes).reshape(
        batch_shape + (2 ** len(qubits), 2 ** (num_qubits - len(qubits)))
    )
    evolved = (matrix @ block).reshape(batch_shape + (2,) * num_qubits)
    return ops.transpose(evolved, inverse).reshape(batch_shape + (2**num_qubits,))


def _resource_check(circuit, max_qubits, max_memory_mb, batch_size=1):
    if max_qubits is not None:
        integer(max_qubits, "max_qubits", 1)
    budget = real(max_memory_mb)
    if budget <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    # Four state buffers; autodiff graphs may retain additional memory.
    numerator, denominator = budget.as_integer_ratio()
    capacity = ((numerator << 20) // denominator) >> (circuit.num_qubits + 6)
    if max(1, batch_size) > capacity or (
        max_qubits is not None and circuit.num_qubits > max_qubits
    ):
        raise ResourceError(
            f"Dense simulation of {circuit.num_qubits} qubits exceeds "
            f"max_qubits={max_qubits} or working budget={budget:g} MiB. "
            "Configure explicit limits or use a different simulator."
        )


def _checkpoint_interval(circuit, backend, value):
    if value is None:
        return None
    value = integer(value, "checkpoint_interval", 1)
    if not (isinstance(backend, str) and backend in {"torch", "jax"}):
        raise ValidationError("checkpoint_interval requires a native torch or jax backend.")
    if circuit.has_effects:
        raise ValidationError("checkpoint_interval requires a unitary circuit.")
    return value


def _evolve_unitary(state, operations, bindings, num_qubits, ops, checkpoint_interval):
    if checkpoint_interval is None:
        for operation in operations:
            if operation.name != "barrier":
                matrix = _gate_matrix(
                    operation.name,
                    [p.evaluate(bindings) for p in operation.params],
                    ops,
                    operation.matrix,
                )
                state = _apply_matrix(state, matrix, operation.qubits, num_qubits, ops)
        return state

    gates = tuple(op for op in operations if op.name != "barrier")
    for start in range(0, len(gates), checkpoint_interval):
        segment = gates[start : start + checkpoint_interval]
        names = tuple(sorted(set().union(*(p.parameters for op in segment for p in op.params))))
        values = tuple(bindings[name] for name in names)

        def evolve(current, *args, segment=segment, names=names):
            local = dict(zip(names, args, strict=True))
            for operation in segment:
                matrix = _gate_matrix(
                    operation.name,
                    [p.evaluate(local) for p in operation.params],
                    ops,
                    operation.matrix,
                )
                current = _apply_matrix(current, matrix, operation.qubits, num_qubits, ops)
            return current

        if ops.backend == "jax":
            import jax

            state = jax.checkpoint(evolve)(state, *values)
        else:
            import torch
            from torch.utils.checkpoint import checkpoint

            # PyTorch's checkpoint input saver has no forward-AD rule. Keep
            # directional derivatives on the ordinary native tensor graph.
            compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
            dual = not compiling and any(
                isinstance(v, torch.Tensor)
                and torch.autograd.forward_ad.unpack_dual(v).tangent is not None
                for v in (state, *values)
            )
            if torch.is_grad_enabled() and not dual:
                state = checkpoint(
                    evolve, state, *values, use_reentrant=False, preserve_rng_state=False
                )
            else:
                state = evolve(state, *values)
    return state


def _initial_state(num_qubits, shape, ops):
    """Allocate a zero input after the caller has checked its working budget."""
    if ops.backend == "torch":
        state = ops.xp.zeros(2**num_qubits, dtype=ops.dtype, device=ops.device)
        state[0] = 1
    elif ops.backend == "jax":
        state = ops.xp.zeros(2**num_qubits, dtype=ops.dtype).at[0].set(1)
    else:
        state = np.zeros(2**num_qubits, dtype=complex)
        state[0] = 1
    return ops.xp.broadcast_to(state, shape + (2**num_qubits,))


def statevector(
    circuit: Circuit,
    bindings: Mapping[str, Any] | None = None,
    *,
    backend: str = "numpy",
    like=None,
    max_qubits: int | None = None,
    max_memory_mb: float = 256,
    batch_shape=None,
    checkpoint_interval=None,
):
    """Return a pure state, q0-most-significant. Measurement requires sample/execute.

    The torch and jax paths use native tensor operations throughout, allowing
    differentiation with respect to every bound rotation and the global phase.
    checkpoint_interval recomputes native unitary gate segments during backward
    to retain fewer activations; it does not change the circuit or state.
    """
    checkpoint_interval = _checkpoint_interval(circuit, backend, checkpoint_interval)
    _resource_check(circuit, max_qubits, max_memory_mb)
    if circuit.has_effects:
        raise ValidationError(
            "Quantum effects have no single pure output state. Use density_matrix()."
        )
    native = isinstance(backend, str) and backend in {"numpy", "torch", "jax"}
    if not native:
        if like is not None:
            raise ValidationError(
                "External backends do not support a native autodiff tensor context."
            )
        result = execute(
            circuit, bindings, backend=backend, max_qubits=max_qubits, max_memory_mb=max_memory_mb
        ).statevector
        if result is None:
            raise ValidationError("This backend provides mixed states. Use density_matrix().")
        return result
    bindings = {} if bindings is None else bindings
    if set(bindings) != set(circuit.parameters):
        raise ValidationError(f"Expected exactly these parameter bindings: {circuit.parameters}.")
    if backend == "numpy":
        clean = {}
        for k, v in bindings.items():
            value = np.asarray(v)
            if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
                raise ValidationError("Bindings must be finite real scalars or arrays.")
            clean[k] = value.astype(float, copy=False)
        bindings = clean
    if like is None and backend != "numpy":
        like = next((v for v in bindings.values() if hasattr(v, "dtype")), None)
    ops = _TensorOps(backend, like)
    try:
        inferred = np.broadcast_shapes(*(tuple(getattr(v, "shape", ())) for v in bindings.values()))
        shape = (
            inferred if batch_shape is None else np.broadcast_shapes(inferred, tuple(batch_shape))
        )
    except ValueError as exc:
        raise ValidationError("Parameter batch shapes cannot be broadcast together.") from exc
    _resource_check(circuit, max_qubits, max_memory_mb, math.prod(shape))
    state = _initial_state(circuit.num_qubits, shape, ops)
    state = _evolve_unitary(
        state, circuit.operations, bindings, circuit.num_qubits, ops, checkpoint_interval
    )
    phase = ops.array(circuit.global_phase.evaluate(bindings), real_value=True)
    return state * ops.xp.exp(1j * phase)[..., None]


@lru_cache(maxsize=32)
def _cached_pauli_plan(paulis):
    size = 2 ** len(paulis)
    indices = np.arange(size, dtype=np.int64)
    flip = sum(1 << (len(paulis) - q - 1) for q, p in enumerate(paulis) if p in "XY")
    source = indices ^ flip
    phase = np.ones(size, dtype=np.complex128) * (1j ** paulis.count("Y"))
    for q, p in enumerate(paulis):
        if p in "YZ":
            phase *= 1 - 2 * ((source >> (len(paulis) - q - 1)) & 1)
    return source, phase


def _pauli_plan(paulis):
    # At most three MiB of NumPy payloads, independent of simulation width.
    factory = _cached_pauli_plan if len(paulis) <= 12 else _cached_pauli_plan.__wrapped__
    return factory(paulis)


@lru_cache(maxsize=32)
def _torch_cached_pauli_plan(paulis, device, dtype):
    import torch

    index, phase = _pauli_plan(paulis)
    return torch.as_tensor(index, device=device), torch.as_tensor(phase, dtype=dtype, device=device)


def _torch_pauli_plan(paulis, device, dtype):
    import torch

    # Function transforms can wrap even constant factories. Never let those
    # wrappers escape into a global device cache (including cold-cache Hessians).
    # Unknown PyTorch internals fall back to uncached tensor construction.
    active = getattr(torch._C, "_are_functorch_transforms_active", None)
    if len(paulis) > 12 or active is None or active():
        return _torch_cached_pauli_plan.__wrapped__(paulis, device, dtype)
    return _torch_cached_pauli_plan(paulis, device, dtype)


def _numpy_pauli_expectation(state, paulis):
    """Contract a wide word through approximately one MiB of transient tiles."""
    width, size = len(paulis), state.shape[-1]
    rows = max(1, math.prod(state.shape[:-1]))
    tile = max(1, min(size, (1 << 20) // (64 * rows + 48)))
    flip = sum(1 << (width - q - 1) for q, p in enumerate(paulis) if p in "XY")
    sign = sum(1 << (width - q - 1) for q, p in enumerate(paulis) if p in "YZ")
    result = np.zeros(state.shape[:-1], dtype=state.real.dtype)
    for start in range(0, size, tile):
        end = min(size, start + tile)
        source = np.arange(start, end, dtype=np.int64) ^ flip
        parity = source & sign
        for shift in (32, 16, 8, 4, 2, 1):
            parity ^= parity >> shift
        phase = (1 - 2 * (parity & 1)) * (1j ** paulis.count("Y"))
        product = np.conj(state[..., start:end]) * state[..., source]
        product *= phase
        result += np.real(product.sum(axis=-1))
    return result


def _pauli_expectation(state, paulis, num_qubits, ops):
    if ops.backend == "numpy" and num_qubits > 12:
        return _numpy_pauli_expectation(state, paulis)
    capture = False
    if ops.backend == "torch":
        from torch._dynamo import is_compiling

        capture = is_compiling()
    if ops.backend != "numpy" and (capture or num_qubits > 12):
        kwargs = {"device": ops.device} if ops.backend == "torch" else {}
        index = ops.xp.arange(state.shape[-1], **kwargs)
        flip = sum(1 << (num_qubits - q - 1) for q, p in enumerate(paulis) if p in "XY")
        index = index ^ flip
        phase = ops.xp.ones(state.shape[-1], dtype=ops.dtype, **kwargs) * (1j ** paulis.count("Y"))
        for q, p in enumerate(paulis):
            if p in "YZ":
                phase = phase * (1 - 2 * ((index >> (num_qubits - q - 1)) & 1))
    elif ops.backend == "torch":
        index, phase = _torch_pauli_plan(paulis, ops.device, ops.dtype)
    else:
        index, phase = _pauli_plan(paulis)
        index, phase = ops.xp.asarray(index), ops.array(phase)
    transformed = (
        ops.xp.index_select(state, -1, index) if ops.backend == "torch" else state[..., index]
    ) * phase
    return ops.real((ops.conj(state) * transformed).sum(axis=-1))


def _expectations_from_state(state, observables, num_qubits, ops):
    from ._readouts import _evaluate_readouts

    return _evaluate_readouts(
        observables,
        num_qubits,
        ops,
        state[..., 0],
        lambda: state.real * state.real + state.imag * state.imag,
        lambda word: _pauli_expectation(state, word, num_qubits, ops),
    )


def _expectation_from_state(state, observable, num_qubits, ops):
    return _expectations_from_state(state, (observable,), num_qubits, ops)[..., 0]


def expectation(
    circuit,
    observable,
    bindings=None,
    *,
    backend="numpy",
    like=None,
    max_branches=256,
    checkpoint_interval=None,
    **limits,
):
    checkpoint_interval = _checkpoint_interval(circuit, backend, checkpoint_interval)
    if checkpoint_interval is not None:
        limits["checkpoint_interval"] = checkpoint_interval
    integer(max_branches, "max_branches", 1)
    if circuit.has_effects and isinstance(backend, str) and backend in {"numpy", "torch", "jax"}:
        return expectations(
            circuit,
            (observable,),
            bindings,
            backend=backend,
            like=like,
            max_branches=max_branches,
            **limits,
        )[..., 0]
    if not (isinstance(backend, str) and backend in {"numpy", "torch", "jax"}):
        from .backends import get_backend

        device = get_backend(backend)
        if device.capabilities.native_expectations:
            return expectations(
                circuit,
                (observable,),
                bindings,
                backend=backend,
                max_branches=max_branches,
                **limits,
            )[0]
        if circuit.has_effects or (
            device.capabilities.density_matrix and not callable(getattr(device, "run", None))
        ):
            return expectations(
                circuit,
                (observable,),
                bindings,
                backend=backend,
                max_branches=max_branches,
                **limits,
            )[0]
    state = statevector(circuit, bindings, backend=backend, like=like, **limits)
    tensor_backend = (
        backend if isinstance(backend, str) and backend in {"torch", "jax"} else "numpy"
    )
    if like is None and tensor_backend != "numpy":
        like = state.real
    return _expectation_from_state(
        state, observable, circuit.num_qubits, _TensorOps(tensor_backend, like)
    )


def expectations(
    circuit,
    observables,
    bindings=None,
    *,
    backend="numpy",
    like=None,
    max_branches=256,
    checkpoint_interval=None,
    **limits,
):
    checkpoint_interval = _checkpoint_interval(circuit, backend, checkpoint_interval)
    if checkpoint_interval is not None:
        limits["checkpoint_interval"] = checkpoint_interval
    integer(max_branches, "max_branches", 1)
    observables = tuple(observables)
    if not observables:
        raise ValidationError("At least one observable is required.")
    if circuit.has_effects and isinstance(backend, str) and backend in {"numpy", "torch", "jax"}:
        from .density import _expectations_from_density, density_matrix

        rho = density_matrix(
            circuit, bindings, backend=backend, like=like, max_branches=max_branches, **limits
        )
        ops = _TensorOps(backend, rho.real if like is None else like)
        return _expectations_from_density(rho, observables, circuit.num_qubits, ops)
    if not (isinstance(backend, str) and backend in {"numpy", "torch", "jax"}):
        from .backends import get_backend

        device = get_backend(backend)
        if device.capabilities.native_expectations:
            from .batch import expectations_batch

            return expectations_batch(
                circuit,
                observables,
                [bindings or {}],
                backend=backend,
                max_branches=max_branches,
                **limits,
            )[0]
        if circuit.has_effects or (
            device.capabilities.density_matrix and not callable(getattr(device, "run", None))
        ):
            from .density import _expectations_from_density, density_matrix

            rho = density_matrix(
                circuit, bindings, backend=backend, max_branches=max_branches, **limits
            )
            return _expectations_from_density(rho, observables, circuit.num_qubits, _TensorOps())
    state = statevector(circuit, bindings, backend=backend, like=like, **limits)
    tensor_backend = (
        backend if isinstance(backend, str) and backend in {"torch", "jax"} else "numpy"
    )
    ops = _TensorOps(tensor_backend, state.real if like is None else like)
    return _expectations_from_state(state, observables, circuit.num_qubits, ops)


@dataclass(frozen=True)
class ExecutionResult:
    backend: str
    probabilities: np.ndarray | None
    statevector: np.ndarray | None
    counts: dict[str, int] | None = None
    density_matrix: np.ndarray | None = None
    classical_bits: tuple[int, ...] | None = None


def _validate_counts(counts, circuit, shots):
    from numbers import Integral

    width = circuit.num_clbits if circuit.has_measurements else circuit.num_qubits
    if (
        not isinstance(counts, dict)
        or any(
            not isinstance(bits, str)
            or len(bits) != width
            or set(bits) - {"0", "1"}
            or isinstance(count, bool)
            or not isinstance(count, Integral)
            or count < 0
            for bits, count in counts.items()
        )
        or sum(counts.values()) != shots
    ):
        raise BackendError("Native sampler returned invalid canonical shot counts.")


def _external_state(circuit, backend):
    """Independent framework execution. Endianness is normalized at this boundary."""
    if backend == "qiskit":
        from .adapters.qiskit import to_qiskit

        native = to_qiskit(circuit)
        from qiskit.quantum_info import Statevector

        state = np.asarray(Statevector.from_instruction(native).data)
        return (
            state.reshape((2,) * circuit.num_qubits)
            .transpose(tuple(reversed(range(circuit.num_qubits))))
            .reshape(-1)
        )
    if backend == "cirq":
        from .adapters.cirq import to_cirq

        native = to_cirq(circuit)
        import cirq

        return np.asarray(
            cirq.Simulator(dtype=np.complex128)
            .simulate(native, qubit_order=cirq.LineQubit.range(circuit.num_qubits))
            .final_state_vector
        )
    if backend == "pennylane":
        from .adapters.pennylane import to_pennylane

        tape = to_pennylane(circuit)
        import pennylane as qml

        device = qml.device("default.qubit", wires=circuit.num_qubits)

        @qml.qnode(device)
        def program():
            for op in tape.operations:
                qml.apply(op)
            return qml.state()

        return np.asarray(program())
    raise ValidationError(f"Unknown execution backend: {backend!r}.")


def execute(
    circuit: Circuit,
    bindings=None,
    *,
    backend="numpy",
    shots=None,
    seed=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    classical_bits=None,
) -> ExecutionResult:
    """Execute locally, optionally sampling final classical storage.

    Counts use c0-first strings for measured programs, or the explicit order in
    classical_bits. Unmeasured circuits use q0-first strings. Probabilities, when
    returned, describe the final quantum computational basis; mixed execution
    can also return an unconditional density matrix.
    """
    from ._validation import (
        invoke,
        project_counts,
        validate_classical_bits,
        validate_shots,
        validate_state,
    )
    from .backends import ExecutionOptions, get_backend

    device = get_backend(backend)
    if shots is not None:
        validate_shots(shots)
    classical_bits = validate_classical_bits(circuit, classical_bits)
    if classical_bits is not None and shots is None:
        raise ValidationError("Set shots when selecting classical count bits.")
    if circuit.has_measurements and shots is None:
        raise ValidationError("Set shots for a measured circuit.")
    bound = circuit.bind({} if bindings is None else bindings)
    if (
        device.capabilities.max_qubits is not None
        and circuit.num_qubits > device.capabilities.max_qubits
    ):
        raise ResourceError(
            f"Backend {device.name!r} supports at most {device.capabilities.max_qubits} qubits."
        )
    ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    from .backends import check_effects

    check_effects(bound, device)
    if shots is not None and device.capabilities.native_sampling:
        if max_qubits is not None and circuit.num_qubits > max_qubits:
            raise ResourceError("Circuit exceeds the requested qubit limit.")
        results = invoke(
            device,
            "sample_batch",
            [bound],
            shots,
            options=ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches),
        )
        if not isinstance(results, (list, tuple)) or len(results) != 1:
            raise BackendError("Native sampler returned an invalid batch length.")
        counts = results[0]
        _validate_counts(counts, bound, shots)
        return ExecutionResult(
            device.name,
            None,
            None,
            project_counts(counts, classical_bits),
            classical_bits=classical_bits,
        )
    mixed = any(op.name not in UNITARY_GATES | {"measure", "barrier"} for op in bound.operations)
    if not mixed and bound.has_measurements:
        try:
            bound.unitary_part()
        except ValidationError:
            mixed = True
    density_only = device.capabilities.density_matrix and (
        not callable(getattr(device, "run", None))
        or getattr(device, "method", None) == "density_matrix"
    )
    if mixed or density_only:
        from .density import density_matrix, execute_density

        if device is get_backend("numpy"):
            return execute_density(
                bound,
                shots=shots,
                seed=seed,
                max_qubits=20 if max_qubits is None else max_qubits,
                max_memory_mb=max_memory_mb,
                max_branches=max_branches,
                classical_bits=classical_bits,
            )
        rho = density_matrix(
            bound,
            backend=device,
            max_qubits=20 if max_qubits is None else max_qubits,
            max_memory_mb=max_memory_mb,
            max_branches=max_branches,
        )
        probabilities = np.maximum(0, np.real(np.diag(rho)))
        counts = None
        if shots is not None:
            measured = False
            for op in bound.operations:
                if op.name == "if_else" or (measured and op.name not in {"measure", "barrier"}):
                    raise ValidationError(
                        "Use numpy or a native sampler for feedback/mid-measurement counts."
                    )
                measured |= op.name == "measure"
            counts = _sample_probabilities(
                bound, probabilities, shots, seed, max_memory_mb, classical_bits
            )
        return ExecutionResult(device.name, probabilities, None, counts, rho, classical_bits)
    _resource_check(circuit, max_qubits, max_memory_mb)
    pure = bound.unitary_part()
    from .compiler import decompose

    if (
        device.capabilities.max_qubits is not None
        and pure.num_qubits > device.capabilities.max_qubits
    ):
        raise ResourceError(
            f"Backend {device.name!r} supports at most {device.capabilities.max_qubits} qubits."
        )
    # Barriers affect scheduling, not statevector evolution.
    pure = replace(pure, operations=tuple(op for op in pure.operations if op.name != "barrier"))
    pure = decompose(pure, basis=device.capabilities.gates)
    options = ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    state = validate_state(invoke(device, "run", pure, options=options), pure)
    with np.errstate(over="ignore", invalid="ignore"):
        probabilities = np.abs(state) ** 2
    if not np.all(np.isfinite(probabilities)) or not np.isclose(
        probabilities.sum(), 1, atol=1e-6, rtol=0
    ):
        raise BackendError(f"Backend {device.name!r} returned an unnormalized state.")
    counts = None
    if shots is not None:
        counts = _sample_probabilities(
            bound, probabilities, shots, seed, max_memory_mb, classical_bits
        )
    return ExecutionResult(
        device.name,
        probabilities,
        None if bound.has_measurements else state,
        counts,
        classical_bits=classical_bits,
    )


def _sample_probabilities(circuit, probabilities, shots, seed, max_memory_mb, classical_bits=None):
    """Joint computational sampling with terminal classical storage writes."""
    from .parameters import real

    measurements = [op for op in circuit.operations if op.name == "measure"]
    width = circuit.num_clbits if classical_bits is None else len(classical_bits)
    if measurements and width > int(real(max_memory_mb) * 2**20):
        raise ResourceError("Classical count strings exceed the working-memory budget.")
    histogram = np.random.default_rng(seed).multinomial(shots, probabilities / probabilities.sum())
    counts = {}
    writes = {op.clbits[0]: op.qubits[0] for op in measurements}
    for index in np.flatnonzero(histogram):
        bits = format(int(index), f"0{circuit.num_qubits}b")
        if measurements:
            bits = "".join(
                bits[writes[c]] if c in writes else "0"
                for c in (range(width) if classical_bits is None else classical_bits)
            )
        counts[bits] = counts.get(bits, 0) + int(histogram[index])
    return counts


def sample(circuit, shots=1024, bindings=None, **kwargs) -> dict[str, int]:
    return execute(circuit, bindings, shots=shots, **kwargs).counts

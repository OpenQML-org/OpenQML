"""Explicit quantum memory, fresh-environment channels and associative scans.

The carry is an ensemble density operator, never an implicitly cached state.
Native simulation is differentiable; exporting a zero-state unroll is a
separate physical circuit contract with no noninvasive intermediate readout.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .density import _conjugate, _resource_check
from .errors import ResourceError, ValidationError
from .ir import Circuit, integer
from .layers import LayerSpec
from .observables import PauliString
from .parameters import real
from .quantum_cells import FeatureCircuitPlan
from .simulator import _apply_matrix, _gate_matrix, _TensorOps


def _checked(value, valid, ops, message):
    if ops.backend == "numpy":
        eager = True
    elif ops.backend == "torch":
        eager = not getattr(ops.xp.compiler, "is_compiling", lambda: False)()
    else:
        import jax

        eager = not isinstance(valid, jax.core.Tracer)
    if eager and not bool(ops.xp.all(valid)):
        raise ValidationError(message)
    return ops.xp.where(valid, value, float("nan"))


def _real_tensor(value, ops, *, finite=True):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    if getattr(dtype, "kind", None) in {"c", "b", "O", "U", "S", "V"} or str(dtype).replace(
        "torch.", ""
    ).startswith(("complex", "bool", "object", "str")):
        raise ValidationError("Controls and trainable parameters must be real numbers.")
    result = ops.array(value, real_value=True)
    if finite and ops.backend == "numpy" and not np.all(np.isfinite(result)):
        raise ValidationError("Expected finite real values.")
    return result


def _adjoint(value, ops):
    return ops.conj(value).swapaxes(-1, -2)


def _roundoff_density(rho, ops):
    """Restore analytically exact invariants after native floating-point maps.

    This only applies to computed CPTP outputs, never to unchecked caller
    states. There is no eigenvalue clipping, projection or rank truncation.
    """
    rho = 0.5 * (rho + _adjoint(rho, ops))
    trace = ops.real(ops.xp.einsum("...ii->...", rho))
    return rho / trace[..., None, None]


def _validate_density(rho, dimension, ops):
    rho = ops.array(rho)
    if rho.shape[-2:] != (dimension, dimension):
        raise ValidationError("Density shape must match the memory Hilbert space.")
    xp = ops.xp
    tolerance = (
        64 * np.finfo(np.float64 if "64" in str(ops.real_dtype) else np.float32).eps * dimension
    )
    finite = (
        xp.all(xp.isfinite(rho), dim=(-1, -2))
        if ops.backend == "torch"
        else xp.all(xp.isfinite(rho), axis=(-1, -2))
    )
    safe = xp.where(finite[..., None, None], rho, ops.array(np.eye(dimension) / dimension))
    error = xp.abs(safe - _adjoint(safe, ops))
    hermitian = error.amax(dim=(-1, -2)) if ops.backend == "torch" else xp.max(error, axis=(-1, -2))
    trace = xp.einsum("...ii->...", safe)
    eigen = xp.linalg.eigvalsh((safe + _adjoint(safe, ops)) * 0.5)[..., 0]
    valid = (
        finite & (hermitian <= tolerance) & (xp.abs(trace - 1) <= tolerance) & (eigen >= -tolerance)
    )
    return _checked(
        rho,
        valid[..., None, None],
        ops,
        "Carry must be a finite, positive, trace-one density matrix.",
    )


@dataclass(frozen=True)
class QuantumChannelPlan:
    memory_qubits: int = 1
    environment_qubits: int = 1
    depth: int = 1
    num_outputs: int = 3
    reupload: bool = True
    max_operations: int = 100_000
    max_memory_mb: float = 256
    visible_qubits: int = 0
    spec: LayerSpec = field(init=False, repr=False)

    def __post_init__(self):
        for name in (
            "memory_qubits",
            "environment_qubits",
            "depth",
            "num_outputs",
            "max_operations",
        ):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        object.__setattr__(
            self, "visible_qubits", integer(self.visible_qubits, "visible_qubits", 0)
        )
        object.__setattr__(self, "max_memory_mb", real(self.max_memory_mb))
        m, e = self.memory_qubits, self.environment_qubits
        if self.max_memory_mb <= 0 or self.visible_qubits > m:
            raise ValidationError("Invalid memory budget or visible register width.")
        if (self.num_outputs.bit_length() > 2 * m) and not self.visible_qubits:
            raise ValidationError("Readouts exceed the memory Pauli basis.")
        n = m + e
        base = FeatureCircuitPlan(
            n, self.depth, 1, self.reupload, self.max_operations, True, n
        ).to_spec()
        circuit = base.circuit.substitute({f"feature_{i}": 0 for i in range(m)})
        for q in range(m, n):
            circuit = circuit.reset(q)
        if len(circuit.operations) > self.max_operations:
            raise ResourceError("Channel and reset operations exceed max_operations.")
        count = (1 << self.visible_qubits) - 1 if self.visible_qubits else self.num_outputs
        if 64 * count * n > self.max_memory_mb * 2**20:
            raise ResourceError("Channel readout descriptors exceed max_memory_mb.")
        if self.visible_qubits:
            words = tuple(
                "".join(
                    "Z" if i & (1 << (self.visible_qubits - q - 1)) else "I"
                    for q in range(self.visible_qubits)
                )
                + "I" * (n - self.visible_qubits)
                for i in range(1, count + 1)
            )
        else:
            words = tuple(
                "".join("IXYZ"[(i // 4**q) % 4] for q in range(m)) + "I" * e
                for i in range(1, count + 1)
            )
        object.__setattr__(self, "num_outputs", count)
        object.__setattr__(
            self,
            "spec",
            LayerSpec(
                circuit,
                base.input_parameters[m:],
                base.weight_parameters,
                tuple(PauliString(w) for w in words),
            ),
        )

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "fresh_environment_quantum_memory",
            "memory_qubits": self.memory_qubits,
            "environment_qubits": self.environment_qubits,
            "depth": self.depth,
            "reupload": self.reupload,
            "visible_qubits": self.visible_qubits,
            "parameters": len(self.spec.weight_parameters),
            "readouts": self.num_outputs,
            "state_contract": "explicit_ensemble_density; fresh_environment; no_postselection",
        }

    def _admit(self, rows, outputs=0):
        _resource_check(self.spec.circuit, None, self.max_memory_mb, rows)
        if (
            64 * rows * 4 ** (self.memory_qubits + self.environment_qubits) + 8 * outputs
            > self.max_memory_mb * 2**20
        ):
            raise ResourceError("Channel primal workspace exceeds max_memory_mb.")

    def _bindings(self, weights, controls):
        return {
            **{n: weights[i] for i, n in enumerate(self.spec.weight_parameters)},
            **{
                n: controls[i] if controls.ndim == 1 else controls[..., i]
                for i, n in enumerate(self.spec.input_parameters)
            },
        }

    def _step(self, weights, controls, rho, ops):
        m, e = 1 << self.memory_qubits, 1 << self.environment_qubits
        fresh = np.zeros((e, e), complex)
        fresh[0, 0] = 1
        joint = ops.xp.einsum("...ab,cd->...acbd", rho, ops.array(fresh)).reshape(
            rho.shape[:-2] + ((m * e) ** 2,)
        )
        bindings = self._bindings(weights, controls)
        for op in self.spec.circuit.operations:
            if op.name not in {"reset", "barrier"}:
                matrix = _gate_matrix(op.name, tuple(p.evaluate(bindings) for p in op.params), ops)
                joint = _conjugate(
                    joint, matrix, op.qubits, self.memory_qubits + self.environment_qubits, ops
                )
        return ops.xp.einsum("...aibi->...ab", joint.reshape(joint.shape[:-1] + (m, e, m, e)))

    def apply(self, weights, controls, rho, *, backend="numpy", validate_state=True):
        ops = _TensorOps(backend, weights)
        weights, controls = _real_tensor(weights, ops), _real_tensor(controls, ops)
        if weights.shape != (len(self.spec.weight_parameters),) or controls.shape[-1:] != (
            self.environment_qubits,
        ):
            raise ValidationError("Channel weights or controls have an invalid shape.")
        self._admit(math.prod(np.broadcast_shapes(controls.shape[:-1], rho.shape[:-2])))
        rho = (
            _validate_density(rho, 1 << self.memory_qubits, ops)
            if validate_state
            else ops.array(rho)
        )
        return _roundoff_density(self._step(weights, controls, rho, ops), ops)

    def _read(self, rho, ops, visible, depolarization, povm, matrices=None):
        if povm is not None:
            return povm.probabilities(rho)
        if visible:
            d, k = 1 << self.memory_qubits, 1 << visible
            diagonal = ops.real(ops.xp.einsum("...ii->...i", rho)).reshape(
                rho.shape[:-2] + (k, d // k)
            )
            value = diagonal.sum(dim=-1) if ops.backend == "torch" else diagonal.sum(axis=-1)
            return (1 - depolarization) * value + depolarization / k
        matrices = ops.array(self._readout_matrices()) if matrices is None else matrices
        return ops.real(ops.xp.einsum("...ab,kba->...k", rho, matrices))

    def _readout_matrices(self):
        paulis = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1, -1]),
        }
        matrices = []
        for ob in self.spec.observables:
            value = np.array([[1]])
            for p in ob.paulis[: self.memory_qubits]:
                value = np.kron(value, paulis[p])
            matrices.append(value)
        return np.stack(matrices)

    def scan(
        self,
        weights,
        controls,
        *,
        state=None,
        mask=None,
        reset_mask=None,
        mixing=None,
        backend="numpy",
        visible_qubits=None,
        depolarization=0,
        povm=None,
        method="sequential",
        max_work=1_000_000_000,
    ):
        ops = _TensorOps(backend, weights)
        xp = ops.xp
        weights = _real_tensor(weights, ops)
        # Mask inactive NaNs before inspecting or using the controls.
        controls = _real_tensor(controls, ops, finite=False)
        if (
            controls.ndim != 3
            or controls.shape[-1] != self.environment_qubits
            or min(controls.shape[:2]) < 1
            or weights.shape != (len(self.spec.weight_parameters),)
        ):
            raise ValidationError("scan requires weights[P] and controls[B,T,E] with nonempty B,T.")
        batch, tokens, _ = controls.shape
        if method not in {"sequential", "parallel"}:
            raise ValidationError("scan method must be sequential or parallel.")
        max_work = integer(max_work, "max_work", 1)
        visible = (
            self.visible_qubits
            if visible_qubits is None
            else integer(visible_qubits, "visible_qubits", 0)
        )
        if visible > self.memory_qubits:
            raise ValidationError("Visible register cannot exceed the memory width.")
        width = (
            povm.weights.shape[0]
            if povm is not None
            else (1 << visible)
            if visible
            else self.num_outputs
        )
        self._admit(batch, batch * tokens * width)
        depolarization = real(depolarization)
        if not 0 <= depolarization < 1:
            raise ValidationError("depolarization must lie in [0,1).")

        def boolean(value, default):
            value = np.full((batch, tokens), default, bool) if value is None else value
            result = (
                xp.as_tensor(value, device=ops.device, dtype=xp.bool)
                if backend == "torch"
                else xp.asarray(value, dtype=bool)
            )
            if result.shape != (batch, tokens):
                raise ValidationError("Sequence masks must have shape [B,T].")
            return result

        valid, reset = boolean(mask, True), boolean(reset_mask, False)
        controls = xp.where(valid[..., None], controls, 0)
        controls = _checked(controls, xp.isfinite(controls), ops, "Active controls must be finite.")
        alpha = (
            xp.ones_like(controls[..., 0])
            if mixing is None
            else _real_tensor(mixing, ops, finite=False)
        )
        if alpha.shape != (batch, tokens):
            raise ValidationError("mixing must have shape [B,T].")
        alpha = xp.where(valid, alpha, 1)
        alpha = _checked(
            alpha,
            xp.isfinite(alpha) & (alpha >= 0) & (alpha <= 1),
            ops,
            "mixing must lie in [0,1].",
        )
        d = 1 << self.memory_qubits
        zero = np.zeros((d, d), complex)
        zero[0, 0] = 1
        initial = ops.array(zero)
        rho = (
            ops.array(np.broadcast_to(zero, (batch, d, d)).copy())
            if state is None
            else _validate_density(state, d, ops)
        )
        if rho.shape != (batch, d, d):
            raise ValidationError("Carry must have shape [B,2**m,2**m].")
        if method == "parallel":
            return self._parallel(
                weights,
                controls,
                rho,
                initial,
                valid,
                reset,
                alpha,
                ops,
                visible,
                depolarization,
                povm,
                max_work,
            )

        matrices = ops.array(self._readout_matrices()) if not visible and povm is None else None

        def step(carry, row):
            control, active, restart, fraction = row
            before = xp.where((active & restart)[..., None, None], initial, carry)
            after = self._step(weights, control, before, ops)
            after = fraction[..., None, None] * after + (1 - fraction[..., None, None]) * before
            after = _roundoff_density(after, ops)
            after = xp.where(active[..., None, None], after, carry)
            output = self._read(after, ops, visible, depolarization, povm, matrices)
            return after, xp.where(active[..., None], output, 0)

        if backend == "jax":
            import jax

            rho, output = jax.lax.scan(
                step, rho, tuple(x.swapaxes(0, 1) for x in (controls, valid, reset, alpha))
            )
            return output.swapaxes(0, 1), rho
        output = []
        for t in range(tokens):
            rho, row = step(rho, (controls[:, t], valid[:, t], reset[:, t], alpha[:, t]))
            output.append(row)
        return (xp.stack(output, dim=1) if backend == "torch" else xp.stack(output, axis=1)), rho

    def _parallel(
        self,
        weights,
        controls,
        rho,
        initial,
        valid,
        reset,
        alpha,
        ops,
        visible,
        noise,
        povm,
        max_work,
    ):
        xp, batch, tokens = ops.xp, controls.shape[0], controls.shape[1]
        d, e = 1 << self.memory_qubits, 1 << self.environment_qubits
        v, joint = d * d, d * e
        if (
            8 * batch * tokens * v**3 * max(1, (tokens - 1).bit_length()) > max_work
            or 128 * batch * tokens * (v * v + joint * joint) > self.max_memory_mb * 2**20
        ):
            raise ResourceError(
                "Parallel channel scan exceeds max_work or max_memory_mb; use sequential scan."
            )
        vectors = ops.array(np.eye(joint)) + ops.array(np.zeros((batch, tokens, joint, joint)))
        bindings = self._bindings(weights, controls[..., None, :])
        for op in self.spec.circuit.operations:
            if op.name not in {"reset", "barrier"}:
                vectors = _apply_matrix(
                    vectors,
                    _gate_matrix(op.name, tuple(p.evaluate(bindings) for p in op.params), ops),
                    op.qubits,
                    self.memory_qubits + self.environment_qubits,
                    ops,
                )
        unitary = vectors.swapaxes(-1, -2).reshape((batch, tokens, d, e, d, e))
        kraus = unitary[..., 0]
        transfer = xp.einsum("...aec,...bed->...abcd", kraus, ops.conj(kraus)).reshape(
            (batch, tokens, v, v)
        )
        identity = ops.array(np.eye(v))
        transfer = alpha[..., None, None] * transfer + (1 - alpha[..., None, None]) * identity
        reset_map = initial.reshape((v, 1)) * ops.array(np.eye(d).reshape((1, v)))
        transfer = xp.where(reset[..., None, None], transfer @ reset_map, transfer)
        transfer = xp.where(valid[..., None, None], transfer, identity)
        if ops.backend == "jax":
            import jax

            prefix = jax.lax.associative_scan(lambda a, b: b @ a, transfer, axis=1)
        else:
            prefix, offset = transfer, 1
            while offset < tokens:
                updated = prefix[:, offset:] @ prefix[:, :-offset]
                prefix = (
                    xp.cat((prefix[:, :offset], updated), dim=1)
                    if ops.backend == "torch"
                    else xp.concatenate((prefix[:, :offset], updated), axis=1)
                )
                offset *= 2
        states = _roundoff_density(
            (prefix @ rho.reshape((batch, 1, v, 1))).reshape((batch, tokens, d, d)), ops
        )
        outputs = self._read(states, ops, visible, noise, povm)
        return xp.where(valid[..., None], outputs, 0), states[:, -1]

    def time_controls(self, times, *, backend="numpy", like=None):
        ops = _TensorOps(backend, like)
        times = _real_tensor(times, ops)
        times = _checked(
            times, (times >= 0) & (times <= 1), ops, "Diffusion times must lie in [0,1]."
        )
        return (
            math.pi
            * times[..., None]
            * ops.array(np.arange(1, self.environment_qubits + 1), real_value=True)
        )

    def unroll(self, weights, controls, *, max_operations=None):
        weights, controls = np.asarray(weights), np.asarray(controls)
        if (
            weights.shape != (len(self.spec.weight_parameters),)
            or controls.ndim != 2
            or controls.shape[1] != self.environment_qubits
        ):
            raise ValidationError("Unroll needs bound weights[P] and controls[T,E].")
        limit = (
            self.max_operations
            if max_operations is None
            else integer(max_operations, "max_operations", 1)
        )
        if len(controls) * len(self.spec.circuit.operations) > limit:
            raise ResourceError("Unrolled circuit exceeds max_operations.")
        operations = tuple(
            op
            for row in controls
            for op in self.spec.circuit.bind(self._bindings(weights, row)).operations
        )
        return Circuit(self.spec.circuit.num_qubits, operations)

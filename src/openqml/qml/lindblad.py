"""Native, completely positive splitting of controlled Lindblad dynamics."""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..compute.kernel import _array_backend, compute_operation, get_compute_kernel, memory_budget
from ..errors import ResourceError, ValidationError
from ..ir.core import integer
from ..ir.observables import PauliString, z_observables
from ..ir.parameters import _checked, real
from ..sim.dense import _apply_matrix, _gate_matrix, _initial_state, _TensorOps
from ..sim.density import _expectations_from_density


class LindbladResult(NamedTuple):
    outputs: object
    state: object
    error_bound: object


class LindbladUncertainty(NamedTuple):
    state_error_bound: object
    observable_error_bounds: object


def _words(values, width):
    result = tuple(PauliString(x) if isinstance(x, str) else x for x in values)
    if any(not isinstance(x, PauliString) or len(x.paulis) != width for x in result):
        raise ValidationError("Lindblad Pauli terms must match the declared register width.")
    return result


@dataclass(frozen=True, eq=False)
class LindbladPlan:
    """Piecewise-constant Pauli Hamiltonians, Pauli jumps and local energy decay.

    Coefficients have shape [...,time,H], nonnegative rates [...,time,J],
    with Pauli jumps followed by amplitude-damping sites. Symmetric splitting
    composes exact CPTP maps; its trace-norm bound excludes floating roundoff.
    State is caller-owned. Intermediate expectations are simulator readouts.
    """

    num_qubits: int
    hamiltonians: tuple = ()
    pauli_jumps: tuple = ()
    damping_qubits: tuple = ()
    observables: tuple | None = None
    substeps: int = 4
    max_memory_mb: float = 256
    max_work: int = 10_000_000_000
    _components: tuple = field(init=False, repr=False)
    _commuting: bool = field(init=False, repr=False)

    def __post_init__(self):
        width = integer(self.num_qubits, "num_qubits", 1)
        integer(self.substeps, "substeps", 1)
        integer(self.max_work, "max_work", 1)
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("Lindblad memory budget must be positive.")
        ham, jumps = _words(self.hamiltonians, width), _words(self.pauli_jumps, width)
        sites = tuple(integer(q, "damping qubit") for q in self.damping_qubits)
        if len(set(sites)) != len(sites) or any(q >= width for q in sites):
            raise ValidationError("Damping sites must be unique and within the register.")
        observables = (
            z_observables(width) if self.observables is None else _words(self.observables, width)
        )
        if not observables or not (ham or jumps or sites):
            raise ValidationError("A Lindblad plan needs generator terms and readouts.")
        components = tuple(
            (kind, i, term.paulis, term.coefficient)
            for kind, terms in (("hamiltonian", ham), ("pauli", jumps))
            for i, term in enumerate(terms)
            if term.coefficient != 0 and set(term.paulis) != {"I"}
        ) + tuple(("damping", len(jumps) + i, q, 1) for i, q in enumerate(sites))
        words = tuple(c[2] for c in components if c[0] != "damping")
        commuting = len(components) <= 1 or (
            not sites
            and all(
                sum(a != b and a != "I" and b != "I" for a, b in zip(x, y, strict=True)) % 2 == 0
                for i, x in enumerate(words)
                for y in words[i + 1 :]
            )
        )
        for name, value in (
            ("num_qubits", width),
            ("hamiltonians", ham),
            ("pauli_jumps", jumps),
            ("damping_qubits", sites),
            ("observables", observables),
            ("max_memory_mb", budget),
            ("_components", components),
            ("_commuting", commuting),
        ):
            object.__setattr__(self, name, value)
        self.resource_profile(1)

    @property
    def num_rates(self):
        return len(self.pauli_jumps) + len(self.damping_qubits)

    def resource_profile(self, intervals, *, batch_size=1):
        t, b = integer(intervals, "intervals", 1), integer(batch_size, "batch_size", 1)
        size = 1 << (2 * self.num_qubits)
        applications = sum(
            len(c[2].replace("I", "")) if c[0] != "damping" else 1 for c in self._components
        )
        work = 64 * b * size * t * self.substeps * max(1, applications)
        elements = b * (
            24 * size + t * (len(self.hamiltonians) + self.num_rates + len(self.observables) + 4)
        )
        if work > self.max_work:
            raise ResourceError("Lindblad evolution exceeds max_work.")
        get_compute_kernel().admit(elements, self.max_memory_mb, itemsize=16)
        return dict(
            intervals=t,
            batch_size=b,
            primal_workspace_bytes=16 * elements,
            arithmetic_work_estimate=work,
            generator_matrix_materialized=False,
            excludes="autodiff_tapes_compiler_workspaces_and_allocator_reserves",
        )

    def to_dict(self):
        return dict(
            num_qubits=self.num_qubits,
            hamiltonians=[(p.paulis, p.coefficient) for p in self.hamiltonians],
            pauli_jumps=[(p.paulis, p.coefficient) for p in self.pauli_jumps],
            damping_qubits=list(self.damping_qubits),
            substeps=self.substeps,
            observables=[(p.paulis, p.coefficient) for p in self.observables],
            method="CPTP_symmetric_splitting; matrix_free_Liouvillian",
            commuting_generators=self._commuting,
        )

    def _real(self, value, ops):
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        if _array_backend(value) not in {"numpy", ops.backend}:
            raise ValidationError("Lindblad controls must use one native framework.")
        result = ops.array(_checked(value), real_value=True)
        return ops.check(result, ops.xp.isfinite(result), "Lindblad controls must be finite.")

    @compute_operation
    def uncertainty_bound(
        self,
        coefficient_errors,
        rate_errors,
        durations=1,
        *,
        initial_error=0,
        numerical_error=0,
        mask=None,
        backend="numpy",
    ):
        """Propagate absolute generator-error envelopes to final observable bounds.

        Both generators must be CPTP and use this plan's operator family.
        Envelopes have the same [...,time,term] layout as evolve controls.
        This Duhamel/contraction bound is deterministic conditional on those
        envelopes; calibrating statistical coverage remains the caller's job.
        Supply evolve.error_bound as numerical_error to include splitting.
        Initial and state errors use trace norm, without the half factor.
        """
        ops = _TensorOps(backend, coefficient_errors if backend != "numpy" else None)
        h, r, dt, start, numerical = (
            self._real(value, ops)
            for value in (
                coefficient_errors,
                rate_errors,
                durations,
                initial_error,
                numerical_error,
            )
        )
        if (
            h.ndim < 2
            or r.ndim < 2
            or (h.shape[-1] != len(self.hamiltonians) or r.shape[-1] != self.num_rates)
        ):
            raise ValidationError("Uncertainty envelopes require [...,time,term] control shapes.")
        from ..sim.dense import _broadcast_shape

        time = _broadcast_shape((h.shape[-2:-1], r.shape[-2:-1], dt.shape[-1:]))
        if not time or time[0] < 1:
            raise ValidationError("Uncertainty envelopes need at least one interval.")
        active = ops.array(True if mask is None else mask, real_value=True)
        prefix = _broadcast_shape(
            (
                h.shape[:-2],
                r.shape[:-2],
                dt.shape[:-1],
                start.shape,
                numerical.shape,
                active.shape[:-1],
            )
        )
        xp, shape = ops.xp, prefix + time
        elements = math.prod(shape) * (len(self.hamiltonians) + self.num_rates + 16)
        get_compute_kernel().admit(
            8 * elements,
            self.max_memory_mb,
            itemsize=16,
            work=8 * elements,
            max_work=self.max_work,
        )
        h = xp.broadcast_to(h, shape + (len(self.hamiltonians),))
        r = xp.broadcast_to(r, shape + (self.num_rates,))
        dt, active = xp.broadcast_to(dt, shape), xp.broadcast_to(active, shape)
        start, numerical = xp.broadcast_to(start, prefix), xp.broadcast_to(numerical, prefix)
        h = ops.check(h, h >= 0, "Coefficient error envelopes must be nonnegative.")
        r = ops.check(r, r >= 0, "Rate error envelopes must be nonnegative.")
        dt = ops.check(dt, dt >= 0, "Uncertainty durations must be nonnegative.")
        active = ops.check(active, (active == 0) | (active == 1), "Use a boolean interval mask.")
        start = ops.check(
            start, (start >= 0) & (start <= 2), "Initial trace-norm error is in [0,2]."
        )
        numerical = ops.check(numerical, numerical >= 0, "Numerical error must be nonnegative.")
        bound = dt * 0
        safe_dt = xp.where(dt > 0, dt, 1)
        for kind, index, _, scale in self._components:
            envelope = h[..., index] if kind == "hamiltonian" else r[..., index]
            safe_error = xp.where(envelope > 0, envelope, 1)
            power = 1 if kind == "hamiltonian" else 2
            logarithm = (
                math.log(2) + power * math.log(abs(scale)) + xp.log(safe_dt) + xp.log(safe_error)
            )
            term = xp.exp(xp.minimum(logarithm, ops.array(math.log(2), real_value=True)))
            bound = bound + xp.where((envelope > 0) & (dt > 0) & (active != 0), term, 0)
        total = bound.sum(dim=-1) if ops.backend == "torch" else bound.sum(axis=-1)
        two = ops.array(2, real_value=True)
        total = xp.minimum(start + xp.minimum(numerical, two) + total, two)
        norms = ops.array([abs(p.coefficient) for p in self.observables], real_value=True)
        return LindbladUncertainty(total, total[..., None] * norms)

    def _state(self, value, prefix, ops):
        n, xp = self.num_qubits, ops.xp
        dimension = 1 << n
        if value is None:
            return _initial_state(2 * n, prefix, ops)
        value = ops.array(value)
        if value.shape[-2:] != (dimension, dimension):
            raise ValidationError("Lindblad initial state must match the density dimension.")
        value = xp.broadcast_to(value, prefix + (dimension, dimension))
        axes = (-2, -1)
        finite = (
            xp.all(xp.isfinite(value), dim=axes)
            if ops.backend == "torch"
            else xp.all(xp.isfinite(value), axis=axes)
        )
        safe = xp.where(
            finite[..., None, None],
            value,
            ops.array(np.eye(dimension) / dimension),
        )
        adjoint = safe.conj().swapaxes(-2, -1)
        defect = xp.abs(safe - adjoint)
        defect = defect.amax(dim=axes) if ops.backend == "torch" else xp.max(defect, axis=axes)
        trace = ops.einsum("...ii->...", safe)
        eigen = xp.linalg.eigvalsh((safe + adjoint) * 0.5)[..., 0]
        epsilon = 2.220446049250313e-16 if ops.real_dtype == xp.float64 else 1.1920928955078125e-7
        tolerance = 64 * epsilon * dimension
        valid = (
            finite
            & (defect <= tolerance)
            & (xp.abs(trace - 1) <= tolerance)
            & (eigen >= -tolerance)
        )
        return ops.check(
            value,
            valid[..., None, None],
            "Initial density must be Hermitian, positive and trace one.",
        ).reshape(prefix + (dimension * dimension,))

    def _pauli(self, rho, word, ops, *, bra=False):
        for q, symbol in enumerate(word):
            if symbol != "I":
                matrix = _gate_matrix(symbol.lower(), (), ops)
                rho = _apply_matrix(
                    rho,
                    matrix.conj() if bra else matrix,
                    (q + (self.num_qubits if bra else 0),),
                    2 * self.num_qubits,
                    ops,
                )
        return rho

    def _map(self, rho, component, hamiltonian, rates, dt, ops):
        kind, index, payload, scale = component
        xp = ops.xp
        if kind == "damping":
            exposure = rates[..., index] * dt
            survival = xp.exp(-exposure * 0.5)
            probability = -xp.expm1(-exposure)
            matrix = ops.matrix(
                (
                    (1, 0, 0, probability),
                    (0, survival, 0, 0),
                    (0, 0, survival, 0),
                    (0, 0, 0, xp.exp(-exposure)),
                )
            )
            return _apply_matrix(
                rho, matrix, (payload, self.num_qubits + payload), 2 * self.num_qubits, ops
            )
        left = self._pauli(rho, payload, ops)
        both = self._pauli(left, payload, ops, bra=True)
        if kind == "pauli":
            probability = -xp.expm1(-2 * rates[..., index] * (scale * scale) * dt) * 0.5
            return (1 - probability)[..., None] * rho + probability[..., None] * both
        angle = hamiltonian[..., index] * scale * dt
        cosine, sine = xp.cos(angle)[..., None], xp.sin(angle)[..., None]
        right = self._pauli(rho, payload, ops, bra=True)
        return (
            cosine * cosine * rho
            + sine * sine * both
            - ops.array(1j) * cosine * sine * (left - right)
        )

    def _advance(self, rho, hamiltonian, rates, duration, ops):
        dt = duration / self.substeps
        for _ in range(self.substeps):
            for component in self._components[:-1]:
                rho = self._map(rho, component, hamiltonian, rates, dt * 0.5, ops)
            if self._components:
                rho = self._map(rho, self._components[-1], hamiltonian, rates, dt, ops)
            for component in reversed(self._components[:-1]):
                rho = self._map(rho, component, hamiltonian, rates, dt * 0.5, ops)
        return rho

    def _error(self, hamiltonian, rates, duration, mask, ops):
        total = duration * 0
        for kind, index, _, scale in self._components:
            total = total + (
                2 * ops.xp.abs(hamiltonian[..., index] * scale)
                if kind == "hamiltonian"
                else 2 * rates[..., index] * scale * scale
            )
        if self._commuting:
            return total[..., 0] * 0
        z = duration * total / self.substeps
        safe = ops.xp.where(z > 0, z, 1)
        log_error = math.log(self.substeps / 3) + z + 3 * ops.xp.log(safe)
        term = ops.xp.exp(ops.xp.minimum(log_error, ops.array(math.log(2), real_value=True)))
        term = ops.xp.where((z > 0) & mask, term, 0)
        error = term.sum(dim=-1) if ops.backend == "torch" else term.sum(axis=-1)
        return ops.xp.minimum(error, ops.array(2, real_value=True))

    @compute_operation
    def evolve(
        self,
        coefficients,
        rates,
        durations=1,
        *,
        state=None,
        backend="numpy",
        mask=None,
        checkpoint=True,
        tolerance=None,
    ):
        ops = _TensorOps(backend, coefficients if backend != "numpy" else None)
        h, r, t = (self._real(v, ops) for v in (coefficients, rates, durations))
        if type(checkpoint) is not bool:
            raise ValidationError("checkpoint must be a boolean.")
        state = None if state is None else ops.array(state)
        active = ops.array(1 if mask is None else mask, real_value=True)
        if (
            h.ndim < 2
            or h.shape[-1] != len(self.hamiltonians)
            or h.shape[-2] < 1
            or r.ndim < 2
            or r.shape[-2:] != (h.shape[-2], self.num_rates)
        ):
            raise ValidationError("Lindblad controls require matching time and generator axes.")
        count = h.shape[-2]
        prefix = ops.broadcast_shape(
            h.shape[:-2],
            r.shape[:-2],
            t.shape[:-1] if t.ndim else (),
            getattr(state, "shape", ())[:-2],
            active.shape[:-1],
        )
        self.resource_profile(count, batch_size=math.prod(prefix))
        h = ops.xp.broadcast_to(h, prefix + (count, len(self.hamiltonians)))
        r = ops.xp.broadcast_to(r, prefix + (count, self.num_rates))
        t = ops.xp.broadcast_to(t, prefix + (count,))
        r = ops.check(r, r >= 0, "Lindblad rates must be nonnegative.")
        t = ops.check(t, t >= 0, "Lindblad durations must be nonnegative.")
        active = ops.xp.broadcast_to(active, prefix + (count,))
        active = (
            ops.check(active, (active == 0) | (active == 1), "Time masks must contain zero or one.")
            > 0
        )
        error = self._error(h, r, t, active, ops)
        if tolerance is not None:
            tolerance = real(tolerance)
            if tolerance <= 0:
                raise ValidationError("Lindblad tolerance must be positive.")
            h = ops.check(
                h,
                (error <= tolerance)[..., None, None],
                "Lindblad splitting bound exceeds tolerance; increase substeps.",
                error=ResourceError,
            )
        rho = self._state(state, prefix, ops) + (h * 0).sum() + (r * 0).sum()
        dimension = 1 << self.num_qubits

        def step(current, h, r, dt, active):
            updated = self._advance(current, h, r, dt, ops)
            updated = ops.xp.where(active[..., None], updated, current)
            output = _expectations_from_density(
                updated.reshape(prefix + (dimension, dimension)),
                self.observables,
                self.num_qubits,
                ops,
            )
            return updated, output

        if backend == "jax":
            import jax

            axis = len(prefix)

            def scan_step(current, values):
                return step(current, *values)

            native_step = jax.checkpoint(scan_step) if checkpoint else scan_step
            values = tuple(ops.xp.moveaxis(v, axis, 0) for v in (h, r, t, active))
            rho, outputs = jax.lax.scan(native_step, rho, values)
            outputs = ops.xp.moveaxis(outputs, 0, axis)
        else:
            outputs = []
            for index in range(count):
                values = (
                    rho,
                    h[..., index, :],
                    r[..., index, :],
                    t[..., index],
                    active[..., index],
                )
                if backend == "torch" and checkpoint:
                    from torch.utils.checkpoint import checkpoint as rematerialize

                    rho, output = rematerialize(step, *values, use_reentrant=False)
                else:
                    rho, output = step(*values)
                outputs.append(output)
            outputs = (
                ops.xp.stack(tuple(outputs), dim=-2)
                if backend == "torch"
                else ops.xp.stack(tuple(outputs), axis=-2)
            )
        rho = rho.reshape(prefix + (dimension, dimension))
        rho = ops.check(rho, ops.xp.isfinite(rho), "Lindblad state exceeds finite precision.")
        return LindbladResult(outputs, rho, error)

    def torch_layer(self, input_features, **options):
        from .interfaces._torch_lindblad import LindbladLayer

        return LindbladLayer(self, input_features, **options)

    def flax_layer(self, **options):
        from .interfaces._flax_lindblad import LindbladLayer

        return LindbladLayer(self, **options)

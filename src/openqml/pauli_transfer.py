"""Exact, batched sparse Pauli transfers with shared observable frontiers.

Inspired by the structure/parameter separation in PADO-Pauli (2610.17755).
This implementation compiles shrinking per-step gather maps, not a dense
2**n state or a fixed 4**n Pauli basis. No term truncation is performed.
"""

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

import numpy as np

from .compiler import _check_circuit_limits, decompose
from .errors import ResourceError, ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString
from .parameters import expression, real
from .pauli import (
    _BASIS,
    _CLIFFORD,
    _ROTATIONS,
    _clifford,
    _clifford_table,
    _masks,
    _prepare_bindings,
    _rotation_sign,
)


@dataclass(frozen=True, eq=False)
class PauliTransferPlan:
    """Compile once; evaluate all readouts/bindings with native higher derivatives.

    Resource bounds cover compilation and conservative primal workspace, not
    autodiff tapes or allocator reserves. Exact frontier growth can still be
    exponential. Zero-state pruning is structural, never data dependent.
    """

    circuit: object
    observables: tuple
    max_terms: int = 4096
    max_nodes: int = 100_000
    max_work: int = 1_000_000
    max_operations: int = 100_000
    max_memory_mb: float = 256
    active_parameters: tuple = field(init=False)
    peak_terms: int = field(init=False)
    propagation_work: int = field(init=False)
    _steps: tuple = field(init=False, repr=False)
    _routes: tuple = field(init=False, repr=False)
    _leaves: int = field(init=False, repr=False)
    _metadata_bytes: int = field(init=False, repr=False)
    _torch_prepared: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for name in ("max_terms", "max_nodes", "max_work", "max_operations"):
            integer(getattr(self, name), name, 1)
        budget = real(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        object.__setattr__(self, "max_memory_mb", budget)
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        if self.circuit.has_effects:
            raise ValidationError("Pauli transfers require a unitary circuit without effects.")
        if isinstance(self.observables, (str, bytes, Mapping, Set)):
            raise ValidationError("Pauli observables require an ordered iterable.")
        try:
            iterator = iter(self.observables)
        except TypeError as exc:
            raise ValidationError("Pauli observables require an ordered iterable.") from exc
        collected, words, terms = [], {}, 0
        for ob in iterator:
            collected.append(ob)
            if len(collected) > self.max_nodes:
                raise ResourceError("Pauli transfer outputs exceed max_nodes.")
            for term in ob.terms if isinstance(ob, Hamiltonian) else (ob,):
                terms += 1
                if terms > self.max_nodes:
                    raise ResourceError("Pauli transfer source terms exceed max_nodes.")
                if not isinstance(term, PauliString) or len(term.paulis) != self.circuit.num_qubits:
                    raise ValidationError("Pauli observable width must match the circuit.")
                if term.coefficient:
                    words.setdefault(_masks(term.paulis), len(words))
        if not collected:
            raise ValidationError("Pauli transfers require observables.")
        roots = tuple(
            tuple(
                (t.coefficient, words[_masks(t.paulis)])
                for t in (ob.terms if isinstance(ob, Hamiltonian) else (ob,))
                if t.coefficient
            )
            for ob in collected
        )
        initial = tuple(words)
        lowered = decompose(self.circuit, basis=_BASIS, max_operations=self.max_operations)
        raw, work, metadata, peak = [], 0, 256 * (terms + len(collected)), len(words)
        if peak > self.max_terms or metadata > budget * 2**20:
            raise ResourceError("Pauli transfer source exceeds max_terms/max_memory_mb.")
        for op in reversed(lowered.operations):
            if op.name == "barrier" or (op.name in _ROTATIONS and op.params[0] == expression(0)):
                continue
            work += len(words)
            if work > self.max_work:
                raise ResourceError("Pauli transfer propagation exceeds max_work.")
            following, a, b, signs, rotating = {}, [], [], [], []

            def index(word, following=following):
                if word not in following:
                    if len(following) >= self.max_terms:
                        raise ResourceError("Pauli transfer frontier exceeds max_terms.")
                    following[word] = len(following)
                return following[word]

            clifford = op.name in _CLIFFORD
            angle = None if clifford else op.params[0]
            if not clifford:
                mask = sum(1 << q for q in op.qubits)
                axis = _ROTATIONS[op.name]
                gx, gz = (mask if axis != "Z" else 0), (mask if axis != "X" else 0)
            for x, z in words:
                if clifford:
                    mapped, sign = _clifford(x, z, op.qubits, _clifford_table(op.name))
                    a.append(index(mapped))
                    b.append(a[-1])
                    signs.append(sign)
                    rotating.append(False)
                else:
                    anti = bool(((gx & z).bit_count() + (gz & x).bit_count()) % 2)
                    a.append(index((x, z)))
                    b.append(index((x ^ gx, z ^ gz)) if anti else a[-1])
                    signs.append(_rotation_sign(gx, gz, x, z) if anti else 1)
                    rotating.append(anti)
            metadata += 512 * len(words) + 256
            if work > self.max_nodes or metadata > budget * 2**20:
                raise ResourceError("Pauli transfer descriptors exceed max_nodes/max_memory_mb.")
            raw.append((tuple(a), tuple(b), tuple(signs), tuple(rotating), angle))
            frontier_bytes = sum(
                256 + (x.bit_length() + z.bit_length() + 7) // 8 for x, z in following
            )
            if metadata + 2 * frontier_bytes > budget * 2**20:
                raise ResourceError("Pauli transfer frontiers exceed max_memory_mb.")
            words = following
            peak = max(peak, len(words))
        # Pull zero-state expectation values back through the sparse maps.
        # Sentinel slot 0 represents a structurally vanishing coefficient.
        live = {i: j + 1 for j, i in enumerate(i for i, (x, _) in enumerate(words) if x == 0)}
        leaves, steps, active = len(live), [], set()
        for a, b, signs, rotating, angle in reversed(raw):
            selected = [i for i in range(len(a)) if a[i] in live or (rotating[i] and b[i] in live)]
            aa, bb = (
                tuple(live.get(a[i], 0) for i in selected),
                tuple(live.get(b[i], 0) for i in selected),
            )
            rr, ss = tuple(rotating[i] for i in selected), tuple(signs[i] for i in selected)
            if angle is not None and any(rr):
                active.update(angle.parameters)
            else:
                angle = None
            steps.append((aa, bb, ss, rr, angle))
            live = {old: new + 1 for new, old in enumerate(selected)}
        routes = tuple(
            tuple((coefficient, live.get(index, 0)) for coefficient, index in row) for row in roots
        )
        # An empty circuit still has its original word set as the final frontier.
        if not raw:
            live = {i: j + 1 for j, i in enumerate(i for i, (x, _) in enumerate(initial) if x == 0)}
            routes = tuple(tuple((c, live.get(i, 0)) for c, i in row) for row in roots)
        for name, value in (
            ("observables", tuple(collected)),
            ("_steps", tuple(steps)),
            ("_routes", routes),
            ("_leaves", leaves),
            ("_metadata_bytes", metadata),
            ("active_parameters", tuple(sorted(active))),
            ("peak_terms", peak),
            ("propagation_work", work),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self):
        return dict(
            representation="exact_sparse_pauli_transfers",
            initial_state="all_zero",
            readouts=len(self.observables),
            transfer_steps=len(self._steps),
            peak_terms=self.peak_terms,
            propagation_work=self.propagation_work,
            active_parameters=list(self.active_parameters),
            metadata_bytes=self._metadata_bytes,
            approximation="none",
            memory_scope="conservative_primal_workspace_excludes_autodiff_tapes",
        )

    def prepare_torch(self, like):
        """Materialize index/sign constants once, outside tracing, on one device.

        Native layers call this during construction. Replaces at most one
        previous device's metadata; no bindings, outputs or gradients are kept.
        Moving a layer remains valid, with ordinary constant conversion until
        this optional preparation is repeated for the new device/dtype.
        """
        from .simulator import _TensorOps

        ops = _TensorOps("torch", like)
        xp = ops.xp
        constants = tuple(
            (
                xp.as_tensor(a, device=like.device, dtype=xp.int64),
                xp.as_tensor(b, device=like.device, dtype=xp.int64),
                xp.as_tensor(signs, device=like.device, dtype=ops.real_dtype),
                xp.as_tensor(rotating, device=like.device, dtype=xp.bool),
            )
            for a, b, signs, rotating, _ in self._steps
        )
        object.__setattr__(self, "_torch_prepared", (str(like.device), ops.real_dtype, constants))
        return self

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        ops, local, shape = _prepare_bindings(
            bindings,
            backend,
            like,
            batch_shape,
            frozenset(self.circuit.parameters),
            self.active_parameters,
        )
        volume = 8 * max(1, math.prod(shape)) * (12 * (self.peak_terms + 1) + len(self.observables))
        if self._metadata_bytes + volume > self.max_memory_mb * 2**20:
            raise ResourceError("Pauli transfer batch exceeds max_memory_mb; reduce the batch.")
        xp = ops.xp
        prepared = self._torch_prepared
        constants = (
            prepared[2]
            if backend == "torch"
            and prepared is not None
            and prepared[:2] == (str(ops.device), ops.real_dtype)
            else None
        )

        def array(value, *, indices=False):
            if backend == "torch":
                return xp.as_tensor(
                    value,
                    device=ops.device,
                    dtype=xp.int64 if indices else ops.real_dtype,
                )
            return xp.asarray(value, dtype="int32" if indices else None)

        def cat(a, b):
            return xp.cat((a, b), dim=-1) if backend == "torch" else xp.concatenate((a, b), axis=-1)

        zero = xp.broadcast_to(ops.array(0, real_value=True), shape + (1,))
        # Preserve zero derivatives for structurally inactive source symbols.
        for value in local.values():
            zero = zero + (xp.broadcast_to(value, shape)[..., None] * 0) ** 2
        values = cat(zero, xp.broadcast_to(ops.array(1, real_value=True), shape + (self._leaves,)))
        for index, (a, b, signs, rotating, angle) in enumerate(self._steps):
            aa, bb, ss, flags = (
                constants[index]
                if constants is not None
                else (
                    array(a, indices=True),
                    array(b, indices=True),
                    array(signs),
                    xp.as_tensor(rotating, device=ops.device, dtype=xp.bool)
                    if backend == "torch"
                    else xp.asarray(rotating, dtype=bool),
                )
            )
            base = values[..., aa]
            if angle is None:
                result = base * ss
            else:
                theta = angle.evaluate(local)
                theta = xp.broadcast_to(ops.array(theta, real_value=True), shape)[..., None]
                mixed = xp.cos(theta) * base + xp.sin(theta) * ss * values[..., bb]
                result = xp.where(flags, mixed, base)
            values = cat(zero, result)
        outputs = [
            sum((coefficient * values[..., i] for coefficient, i in row), zero[..., 0])
            for row in self._routes
        ]
        result = xp.stack(outputs, dim=-1) if backend == "torch" else xp.stack(outputs, axis=-1)
        if backend == "numpy" and not np.isfinite(result).all():
            raise ValidationError("Pauli transfer expectations exceed finite precision.")
        return result

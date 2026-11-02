"""Compiled exact circuit tensor contractions, with bounded paths and index slicing.

Ket/bra networks preserve arbitrary unitary circuit readouts. Their cost depends
on contraction width; highly entangled networks can still require exponential
work. No numerical tensor rank, singular-value cutoff or postselection is used.
"""

import heapq
import math
from dataclasses import dataclass, field

from ._gradient_common import _observables
from .compiler import _check_circuit_limits
from .errors import MissingDependencyError, ResourceError, ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString
from .parameters import real
from .pauli import _prepare_bindings


def _step(left, right):
    shared = tuple(x for x in left if x in right)
    a, b = tuple(x for x in left if x not in shared), tuple(x for x in right if x not in shared)
    return shared, a + b


def _greedy(labels, charge):
    live = {i: tuple(x) for i, x in enumerate(labels)}
    owners = {}
    for i, indices in live.items():
        for x in indices:
            owners.setdefault(x, set()).add(i)
    heap, steps = [], []

    def add(a, b):
        if a > b:
            a, b = b, a
        charge(1)
        shared, out = _step(live[a], live[b])
        heapq.heappush(heap, (len(out), len(set(live[a]) | set(live[b])), a, b))

    for sites in owners.values():
        if len(sites) != 2:
            raise ValidationError("Circuit tensor indices must have exactly two incidences.")
        a, b = sorted(sites)
        add(a, b)
    next_id = len(live)
    while len(live) > 1:
        while heap and (heap[0][2] not in live or heap[0][3] not in live):
            heapq.heappop(heap)
        if heap:
            _, _, a, b = heapq.heappop(heap)
        else:
            # Disconnected factors are multiplied only after their local edges.
            a, b = sorted(live, key=lambda i: (len(live[i]), i))[:2]
        shared, out = _step(live[a], live[b])
        steps.append((a, b, live[a], live[b], shared, out))
        for x in shared:
            owners.pop(x, None)
        for x in out:
            owners[x].difference_update((a, b))
            owners[x].add(next_id)
        del live[a], live[b]
        live[next_id] = out
        neighbors = set().union(*(owners[x] for x in out)) if out else set()
        for other in sorted(neighbors - {next_id}):
            add(other, next_id)
        next_id += 1
    return tuple(steps)


def _cotengra(labels, trials, charge):
    try:
        import cotengra as ctg
    except ImportError as exc:
        raise MissingDependencyError(
            'Install contraction search with: python -m pip install "openqml[tensor]".'
        ) from exc
    charge(trials * len(labels))
    optimizer = ctg.HyperOptimizer(
        methods=["greedy"], max_repeats=trials, optlib="random", parallel=False, progbar=False
    )
    tree = optimizer.search(tuple(labels), (), {x: 2 for row in labels for x in row})
    live = list(range(len(labels)))
    indices = {i: tuple(x) for i, x in enumerate(labels)}
    steps = []
    for positions in tree.get_path():
        if (
            len(positions) != 2
            or len(set(positions)) != 2
            or any(i < 0 or i >= len(live) for i in positions)
        ):
            raise ValidationError("Contraction optimizer returned a nonbinary/invalid path.")
        a, b = (live[i] for i in positions)
        shared, out = _step(indices[a], indices[b])
        steps.append((a, b, indices[a], indices[b], shared, out))
        for i in sorted(positions, reverse=True):
            live.pop(i)
        node = len(labels) + len(steps) - 1
        indices[node] = out
        live.append(node)
    if len(live) != 1:
        raise ValidationError("Contraction optimizer returned an incomplete path.")
    return tuple(steps)


def _profile(labels, steps):
    live = {i: 1 << len(row) for i, row in enumerate(labels)}
    total = sum(live.values())
    peak, flops, largest = total, 0, max(live.values(), default=1)
    for k, (a, b, la, lb, shared, out) in enumerate(steps):
        size = 1 << len(out)
        peak = max(peak, total + 2 * (live[a] + live[b]) + 3 * size)
        flops += 8 << (len(la) + len(lb) - len(shared))
        total += size - live.pop(a) - live.pop(b)
        live[len(labels) + k] = size
        largest = max(largest, size)
    return peak, flops, largest


def _network(circuit, observable, max_operations):
    from .readout import plan_readout

    reduced = plan_readout(circuit, (PauliString(observable),), max_operations=max_operations)
    c = reduced.circuit
    descriptors, labels = [], []
    next_index = 0
    ends = []
    for bra in (False, True):
        current = list(range(next_index, next_index + c.num_qubits))
        next_index += c.num_qubits
        for index in current:
            descriptors.append(("zero", None, bra))
            labels.append((index,))
        for op in c.operations:
            if op.name == "barrier":
                continue
            outgoing = tuple(range(next_index, next_index + len(op.qubits)))
            next_index += len(op.qubits)
            descriptors.append(("gate", op, bra))
            labels.append(outgoing + tuple(current[q] for q in op.qubits))
            for q, index in zip(op.qubits, outgoing, strict=True):
                current[q] = index
        ends.append(current)
    word = reduced.observables[0].paulis
    for q, p in enumerate(word):
        descriptors.append(("pauli", p, False))
        labels.append((ends[1][q], ends[0][q]))
    return c, tuple(descriptors), tuple(labels)


def _contract(left, right, la, lb, shared, ops):
    a, b = tuple(x for x in la if x not in shared), tuple(x for x in lb if x not in shared)
    na, nb = left.ndim - len(la), right.ndim - len(lb)
    pa = tuple(range(na)) + tuple(na + la.index(x) for x in a + shared)
    pb = tuple(range(nb)) + tuple(nb + lb.index(x) for x in shared + b)
    left = ops.transpose(left, pa).reshape(left.shape[:na] + (1 << len(a), 1 << len(shared)))
    right = ops.transpose(right, pb).reshape(right.shape[:nb] + (1 << len(shared), 1 << len(b)))
    value = left @ right
    return value.reshape(value.shape[:-2] + (2,) * (len(a) + len(b)))


@dataclass(frozen=True, eq=False)
class TensorNetworkPlan:
    circuit: object
    observables: tuple
    max_memory_mb: float = 256
    max_work: int = 1_000_000
    max_operations: int = 100_000
    max_slices: int = 64
    max_flops: int = 1_000_000_000
    optimizer: str = "greedy"
    search_trials: int = 16
    rematerialize: bool = False
    reuse_intermediates: bool = False
    _graphs: tuple = field(init=False, repr=False)
    _routes: tuple = field(init=False, repr=False)
    _metadata_bytes: int = field(init=False, repr=False)
    active_parameters: tuple = field(init=False)
    working_bytes: int = field(init=False)
    planning_work: int = field(init=False)
    _reuse_bytes: int = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.rematerialize) is not bool:
            raise ValidationError("rematerialize must be a boolean.")
        if type(self.reuse_intermediates) is not bool:
            raise ValidationError("reuse_intermediates must be a boolean.")
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        if self.circuit.has_effects:
            raise ValidationError("Circuit tensor contractions currently require unitary circuits.")
        if self.optimizer not in {"greedy", "cotengra"}:
            raise ValidationError("optimizer must be greedy or cotengra.")
        for name in ("max_work", "max_operations", "max_slices", "max_flops", "search_trials"):
            integer(getattr(self, name), name, 1)
        budget = real(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        obs = _observables(self.observables, self.circuit.num_qubits, label="Tensor network")
        work = 0

        def charge(value):
            nonlocal work
            work += value
            if work > self.max_work:
                raise ResourceError("Tensor contraction planning exceeds max_work.")

        terms = [o.terms if isinstance(o, Hamiltonian) else (o,) for o in obs]
        charge(sum(len(t) for t in terms) * (1 + self.circuit.num_qubits))
        words = dict.fromkeys(
            t.paulis for ts in terms for t in ts if t.coefficient and set(t.paulis) != {"I"}
        )
        raw, metadata = [], 256 * work + 128 * len(obs)
        for word in words:
            c, desc, labels = _network(self.circuit, word, self.max_operations)
            charge(len(labels) + sum(len(row) for row in labels))
            metadata += 512 * len(labels) + 128 * sum(map(len, labels))
            if metadata >= budget * 2**20:
                raise ResourceError("Tensor contraction descriptors exceed max_memory_mb.")
            raw.append((c, desc, labels))
        capacity = int(budget * 2**20) - metadata
        reuse_bytes = capacity // 5 if self.reuse_intermediates else 0
        capacity -= reuse_bytes
        if capacity <= 0:
            raise ResourceError("Tensor contraction descriptors exceed max_memory_mb.")
        graphs, active, maximum, total_flops = [], set(), 1, 0
        for c, desc, original in raw:
            sliced = []
            raw_storage = sum(1 << len(row) for row in original)
            while True:
                labels = tuple(tuple(x for x in row if x not in sliced) for row in original)
                steps = (
                    _greedy(labels, charge)
                    if self.optimizer == "greedy"
                    else _cotengra(labels, self.search_trials, charge)
                )
                peak, flops, largest = _profile(labels, steps)
                needed = 16 * (peak + raw_storage)
                if needed <= capacity:
                    break
                if (1 << (len(sliced) + 1)) > self.max_slices:
                    raise ResourceError("No contraction path fits max_memory_mb within max_slices.")
                candidates = list(labels) + [s[-1] for s in steps]
                worst = max(candidates, key=len)
                if not worst:
                    raise ResourceError("Tensor input/slicing workspace exceeds the memory budget.")
                # Slice a wire occurring in the largest live separator. Every
                # binary assignment is summed; no amplitude is pruned.
                selected = max(
                    worst, key=lambda x: sum(1 << len(row) for row in candidates if x in row)
                )
                sliced.append(selected)
            total_flops += flops * (1 << len(sliced))
            if total_flops > self.max_flops:
                raise ResourceError("Tensor contraction/slicing exceeds max_flops.")
            active.update(c.parameters)
            graphs.append((desc, original, labels, steps, tuple(sliced), needed, flops, largest))
            maximum = max(maximum, needed)
        indices = {word: i for i, word in enumerate(words)}
        routes = tuple(
            tuple(
                (t.coefficient, None if set(t.paulis) == {"I"} else indices[t.paulis])
                for t in ts
                if t.coefficient
            )
            for ts in terms
        )
        for name, value in (
            ("observables", obs),
            ("_graphs", tuple(graphs)),
            ("_routes", routes),
            ("_metadata_bytes", metadata),
            ("working_bytes", metadata + maximum + reuse_bytes),
            ("_reuse_bytes", reuse_bytes),
            ("active_parameters", tuple(sorted(active))),
            ("planning_work", work),
            ("max_memory_mb", budget),
        ):
            object.__setattr__(self, name, value)

    @property
    def batch_capacity(self):
        return max(
            1,
            int(self.max_memory_mb * 2**20 - self._metadata_bytes - self._reuse_bytes)
            // max(1, self.working_bytes - self._metadata_bytes - self._reuse_bytes),
        )

    def to_dict(self):
        return {
            "representation": "exact_circuit_tensor_network",
            "optimizer": self.optimizer,
            "rematerialize": self.rematerialize,
            "reuse_intermediates": self.reuse_intermediates,
            "reuse_cache_bytes": self._reuse_bytes,
            "num_qubits": self.circuit.num_qubits,
            "readouts": len(self.observables),
            "working_bytes": self.working_bytes,
            "planning_work": self.planning_work,
            "batch_capacity": self.batch_capacity,
            "contractions": [
                {
                    "tensors": len(g[0]),
                    "sliced_indices": len(g[4]),
                    "slices": 1 << len(g[4]),
                    "estimated_flops_per_slice": g[6],
                    "largest_intermediate_elements": g[7],
                }
                for g in self._graphs
            ],
            "exactness": "sum_all_slices; no_intentional_truncation; roundoff_remains",
            "memory_scope": "representation_admission; excludes_autodiff_tapes_and_process_RSS",
        }

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        if self.rematerialize and bindings and backend in {"torch", "jax"}:
            keys, values = tuple(bindings), tuple(bindings.values())

            def forward(*arguments):
                return self._evaluate(
                    dict(zip(keys, arguments, strict=True)),
                    backend=backend,
                    like=like,
                    batch_shape=batch_shape,
                )

            if backend == "torch":
                from torch.utils.checkpoint import checkpoint

                return checkpoint(forward, *values, use_reentrant=False, preserve_rng_state=False)
            import jax

            return jax.checkpoint(forward)(*values)
        return self._evaluate(bindings, backend=backend, like=like, batch_shape=batch_shape)

    def _evaluate(
        self, bindings=None, *, backend="numpy", like=None, batch_shape=None, _executor=None
    ):
        from .simulator import _I, _X, _Y, _Z, _gate_matrix

        ops, local, shape = _prepare_bindings(
            bindings,
            backend,
            like,
            batch_shape,
            frozenset(self.circuit.parameters),
            self.active_parameters,
        )
        rows = max(1, math.prod(shape))
        if (
            rows > self.batch_capacity
            or 8 * rows * (len(local) + len(self.observables))
            > self.max_memory_mb * 2**20 - self._metadata_bytes
        ):
            raise ResourceError("Tensor network batch exceeds max_memory_mb; reduce the batch.")
        zero = ops.xp.broadcast_to(ops.array(0, real_value=True), shape)
        for value in local.values():
            zero = zero + (value * 0) ** 2
        constants = {p: ops.array(m) for p, m in zip("IXYZ", (_I, _X, _Y, _Z), strict=True)}
        constants["zero"] = ops.array((1, 0))
        from ._contraction_reuse import ContractionReuse

        reuse = (
            ContractionReuse(self._reuse_bytes)
            if self.reuse_intermediates and _executor is None
            else None
        )
        readouts = []
        for graph_index, (desc, original, labels, steps, sliced, _, _, _) in enumerate(
            self._graphs
        ):
            # Gate tensors are pure functions of this call's bindings. Reuse
            # ket tensors for their conjugates and across all exact slices.
            gates, raw, origins = {}, [], []
            for j, (kind, op, bra) in enumerate(desc):
                if kind == "gate":
                    key = (op.name, op.params, op.matrix)
                    if key not in gates:
                        gate_id = reuse.node(("gate", key)) if reuse else None
                        matrix = reuse.get(gate_id) if reuse else None
                        if matrix is None:
                            angles = tuple(p.evaluate(local) for p in op.params)
                            matrix = _gate_matrix(op.name, angles, ops, op.matrix)
                            if reuse:
                                reuse.put(gate_id, matrix)
                        gates[key] = matrix
                    matrix = gates[key]
                    value = ops.conj(matrix) if bra else matrix
                    value = value.reshape(value.shape[:-2] + (2,) * len(original[j]))
                else:
                    value = constants["zero" if kind == "zero" else op]
                raw.append(value)
                origins.append(
                    reuse.node(
                        (
                            "input",
                            kind,
                            key if kind == "gate" else op,
                            bra if kind == "gate" else False,
                            len(original[j]),
                        )
                    )
                    if reuse
                    else None
                )
            total = zero
            for assignment in range(1 << len(sliced)):
                fixed = {index: (assignment >> j) & 1 for j, index in enumerate(sliced)}
                values, identities = {}, {}
                for j, value in enumerate(raw):
                    prefix = value.ndim - len(original[j])
                    select = (slice(None),) * prefix + tuple(
                        fixed.get(index, slice(None)) for index in original[j]
                    )
                    selected = value[select]
                    values[j] = (
                        selected if _executor is None else _executor.input(selected, labels[j])
                    )
                    identities[j] = (
                        reuse.node(
                            ("slice", origins[j], tuple(fixed.get(index) for index in original[j]))
                        )
                        if reuse and origins[j] is not None
                        else None
                    )
                for k, (a, b, la, lb, shared, _) in enumerate(steps):
                    left, right = values.pop(a), values.pop(b)
                    ia, ib = identities.pop(a), identities.pop(b)
                    key = (
                        reuse.node(
                            (
                                "contract",
                                ia,
                                ib,
                                tuple(la.index(i) for i in shared),
                                tuple(lb.index(i) for i in shared),
                            )
                        )
                        if reuse and ia is not None and ib is not None
                        else None
                    )
                    cached = reuse.get(key) if reuse else None
                    result = (
                        cached
                        if cached is not None
                        else (
                            _contract(left, right, la, lb, shared, ops)
                            if _executor is None
                            else _executor.contract(left, right, la, lb, shared, graph_index, k)
                        )
                    )
                    values[len(desc) + k], identities[len(desc) + k] = result, key
                    if reuse and cached is None:
                        reuse.put(key, result)
                result = next(iter(values.values()))
                if _executor is not None:
                    result = _executor.finish(result)
                total = total + ops.real(result)
            readouts.append(total)
        outputs = []
        for route in self._routes:
            result = zero
            for coefficient, index in route:
                result = result + coefficient * (1 if index is None else readouts[index])
            outputs.append(result)
        return ops.real(ops.stack(outputs, axis=-1))

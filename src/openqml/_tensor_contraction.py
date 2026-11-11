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
from .compute import compute_context, compute_operation, memory_budget
from .errors import MissingDependencyError, ResourceError, ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString
from .pauli import _prepare_bindings


def _step(left, right):
    shared = tuple(x for x in left if x in right)
    a, b = tuple(x for x in left if x not in shared), tuple(x for x in right if x not in shared)
    return shared, a + b


def _greedy(labels, charge, *, objective="size"):
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
        union = len(live[a]) + len(live[b]) - len(shared)
        score = (
            (union, len(out))
            if objective == "flops"
            else (max(len(out), len(live[a]), len(live[b])), union)
            if objective == "balanced"
            else (len(out), union)
        )
        heapq.heappush(heap, (*score, a, b))

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
            'Install contraction search with: python -m pip install "openqml[compute]".'
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
    _programs: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.rematerialize) is not bool:
            raise ValidationError("rematerialize must be a boolean.")
        if type(self.reuse_intermediates) is not bool:
            raise ValidationError("reuse_intermediates must be a boolean.")
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        if self.circuit.has_effects:
            raise ValidationError("Circuit tensor contractions currently require unitary circuits.")
        if self.optimizer not in {"greedy", "portfolio", "cotengra"}:
            raise ValidationError("optimizer must be greedy, portfolio or cotengra.")
        for name in ("max_work", "max_operations", "max_slices", "max_flops", "search_trials"):
            integer(getattr(self, name), name, 1)
        budget = memory_budget(self.max_memory_mb)
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
            metadata += 768 * len(labels) + 128 * sum(map(len, labels))
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
                from ._contraction_path import depth_first, portfolio

                if self.optimizer == "portfolio":
                    steps = portfolio(
                        labels,
                        charge,
                        _greedy,
                        _step,
                        _profile,
                        max(0, capacity // 16 - raw_storage),
                        self.search_trials,
                    )
                elif self.optimizer == "cotengra":
                    steps = _cotengra(labels, self.search_trials, charge)
                else:
                    steps = _greedy(labels, charge)
                if self.reuse_intermediates:
                    steps = depth_first(labels, steps, _step)
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
        from ._contraction_program import ContractionProgram

        intern = {}
        programs = tuple(ContractionProgram(g, i, intern) for i, g in enumerate(graphs))
        for name, value in (
            ("observables", obs),
            ("_graphs", tuple(graphs)),
            ("_routes", routes),
            ("_programs", programs),
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
            int(
                memory_budget(self.max_memory_mb) * 2**20 - self._metadata_bytes - self._reuse_bytes
            )
            // max(1, self.working_bytes - self._metadata_bytes - self._reuse_bytes),
        )

    def to_dict(self):
        return {
            "representation": "exact_circuit_tensor_network",
            "optimizer": self.optimizer,
            "rematerialize": self.rematerialize,
            "reuse_intermediates": self.reuse_intermediates,
            "reuse_cache_bytes": self._reuse_bytes,
            "execution": "compiled_gemm_with_demand_driven_subtrees",
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

    @compute_operation
    def evaluate(
        self,
        bindings=None,
        *,
        backend="numpy",
        like=None,
        batch_shape=None,
        return_report=False,
        batch_chunk_size=None,
    ):
        """Evaluate readouts; optionally return forward execution/cache counters."""
        if type(return_report) is not bool:
            raise ValidationError("return_report must be a boolean.")
        if batch_chunk_size is not None:
            return self._evaluate_chunks(
                bindings,
                backend=backend,
                like=like,
                batch_shape=batch_shape,
                return_report=return_report,
                batch_chunk_size=batch_chunk_size,
            )
        statistics = (
            {
                "contractions_executed": 0,
                "subtree_cache_hits": 0,
                "slices_evaluated": 0,
                "cache_peak_bytes": 0,
                "cache_budget_bytes": self._reuse_bytes,
            }
            if return_report
            else None
        )
        if self.rematerialize and bindings and backend in {"torch", "jax"}:
            keys, values = tuple(bindings), tuple(bindings.values())

            def forward(*arguments):
                return self._evaluate(
                    dict(zip(keys, arguments, strict=True)),
                    backend=backend,
                    like=like,
                    batch_shape=batch_shape,
                    _statistics=statistics,
                )

            if backend == "torch":
                from torch.utils.checkpoint import checkpoint

                result = checkpoint(forward, *values, use_reentrant=False, preserve_rng_state=False)
            else:
                import jax

                result = jax.checkpoint(forward)(*values)
        else:
            result = self._evaluate(
                bindings,
                backend=backend,
                like=like,
                batch_shape=batch_shape,
                _statistics=statistics,
            )
        return (result, dict(statistics)) if return_report else result

    def iter_evaluate(
        self,
        bindings=None,
        *,
        backend="numpy",
        like=None,
        batch_shape=None,
        batch_chunk_size=None,
        return_report=False,
    ):
        """Yield row-major native readout chunks while reusing the compiled plan.

        Broadcast parameter grids are gathered per chunk without constructing
        the full broadcasted grid. Each yield has shape [chunk_rows,readouts].
        Consumers control retained outputs; native gradients remain connected.
        """
        from ._tensor_batches import chunk_capacity, gather_rows

        if type(return_report) is not bool:
            raise ValidationError("return_report must be a boolean.")
        ops, local, shape = _prepare_bindings(
            bindings,
            backend,
            like,
            batch_shape,
            frozenset(self.circuit.parameters),
            self.active_parameters,
            numpy_copy=False,
        )
        with compute_context(kernel=ops.kernel):
            capacity = chunk_capacity(self, local, shape)
        size = (
            capacity
            if batch_chunk_size is None
            else min(capacity, integer(batch_chunk_size, "batch_chunk_size", 1))
        )
        total = math.prod(shape)
        for start in range(0, max(1, total), size):
            stop = min(start + size, total)
            chunk = gather_rows(local, shape, start, stop, ops)
            with compute_context(kernel=ops.kernel):
                result = self.evaluate(
                    chunk,
                    backend=backend,
                    like=like,
                    batch_shape=(stop - start,),
                    return_report=return_report,
                )
            yield result

    def _evaluate_chunks(
        self,
        bindings,
        *,
        backend,
        like,
        batch_shape,
        return_report,
        batch_chunk_size,
    ):
        size = integer(batch_chunk_size, "batch_chunk_size", 1)
        from ._tensor_batches import chunk_capacity

        ops, local, shape = _prepare_bindings(
            bindings,
            backend,
            like,
            batch_shape,
            frozenset(self.circuit.parameters),
            self.active_parameters,
            numpy_copy=False,
        )
        output_bytes = 16 * math.prod(shape) * len(self.observables)
        capacity = chunk_capacity(self, local, shape, output_bytes)
        values, reports = [], []
        for chunk in self.iter_evaluate(
            bindings,
            backend=backend,
            like=like,
            batch_shape=batch_shape,
            batch_chunk_size=min(size, capacity),
            return_report=return_report,
        ):
            value, report = chunk if return_report else (chunk, None)
            values.append(value)
            if report is not None:
                reports.append(report)
        result = (
            ops.xp.cat(values, dim=0) if backend == "torch" else ops.xp.concatenate(values, axis=0)
        ).reshape(shape + (len(self.observables),))
        if not return_report:
            return result
        report = {
            key: max(r[key] for r in reports)
            if key in {"cache_peak_bytes", "cache_budget_bytes"}
            else sum(r[key] for r in reports)
            for key in reports[0]
        }
        report["chunks"] = len(reports)
        return result, report

    def _evaluate(
        self,
        bindings=None,
        *,
        backend="numpy",
        like=None,
        batch_shape=None,
        _executor=None,
        _statistics=None,
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
            > memory_budget(self.max_memory_mb) * 2**20 - self._metadata_bytes
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
            gates, raw = {}, []
            for j, (kind, op, bra) in enumerate(desc):
                if kind == "gate":
                    key = (op.name, op.params, op.matrix)
                    if key not in gates:
                        gate_id = ("gate", key) if reuse else None
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
            total = zero
            program = self._programs[graph_index]
            for assignment in range(1 << len(sliced)):
                if _statistics is not None:
                    _statistics["slices_evaluated"] += 1
                if reuse is not None:
                    result = program.run(raw, assignment, reuse, ops, _statistics)
                    total = total + ops.real(result)
                    continue
                if _executor is None:
                    result = program.run_flat(raw, assignment, ops, _statistics)
                    total = total + ops.real(result)
                    continue
                fixed = {index: (assignment >> j) & 1 for j, index in enumerate(sliced)}
                values = {}
                for j, value in enumerate(raw):
                    prefix = value.ndim - len(original[j])
                    select = (slice(None),) * prefix + tuple(
                        fixed.get(index, slice(None)) for index in original[j]
                    )
                    selected = value[select]
                    values[j] = (
                        selected if _executor is None else _executor.input(selected, labels[j])
                    )
                for k, (a, b, la, lb, shared, _) in enumerate(steps):
                    left, right = values.pop(a), values.pop(b)
                    result = _executor.contract(left, right, la, lb, shared, graph_index, k)
                    values[len(desc) + k] = result
                    if _statistics is not None:
                        _statistics["contractions_executed"] += 1
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
        if _statistics is not None and reuse is not None:
            _statistics["cache_peak_bytes"] = reuse.peak_bytes
        return ops.real(ops.stack(outputs, axis=-1))

"""Exact product-state execution across quantum AND classical dependency factors.

Factors are conservative: both arms of a conditional and every shared classical
bit join the graph. This representation evaluates zero-state local readouts;
entangled input states and full-state outputs use their dedicated executors.
"""

from dataclasses import dataclass, field

from ._gradient_common import _observables
from ._structure import split_circuit
from .compiler import _check_circuit_limits
from .compute import memory_budget
from .errors import ResourceError, ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString


@dataclass(frozen=True, eq=False)
class ComponentPlan:
    circuit: object
    observables: tuple
    max_qubits: int | None = None
    max_memory_mb: float = 256
    max_branches: int = 256
    max_work: int = 1_000_000
    max_operations: int = 100_000
    components: tuple = field(init=False, repr=False)
    working_bytes: int = field(init=False)
    planning_work: int = field(init=False)
    _queries: tuple = field(init=False, repr=False)
    _routes: tuple = field(init=False, repr=False)
    _metadata_bytes: int = field(init=False, repr=False)

    @property
    def active_parameters(self):
        return self.circuit.parameters

    def __post_init__(self):
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        for name in ("max_work", "max_branches"):
            integer(getattr(self, name), name, 1)
        if self.max_qubits is not None:
            integer(self.max_qubits, "max_qubits", 1)
            if self.circuit.num_qubits > self.max_qubits:
                raise ResourceError("Component execution exceeds the explicit max_qubits limit.")
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        obs = _observables(self.observables, self.circuit.num_qubits, label="Component")
        work = 0

        def charge(amount):
            nonlocal work
            work += amount
            if work > self.max_work:
                raise ResourceError("Component analysis exceeds max_work.")

        terms = [ob.terms if isinstance(ob, Hamiltonian) else (ob,) for ob in obs]
        charge(sum(len(ts) for ts in terms) * (1 + self.circuit.num_qubits))
        components = split_circuit(self.circuit, charge)
        active = {q for wires, _, _ in components for q in wires}
        queries, routes = [{} for _ in components], []
        for output in terms:
            route = []
            for term in output:
                if not term.coefficient or any(
                    p in "XY" and q not in active for q, p in enumerate(term.paulis)
                ):
                    continue
                factors = []
                for i, (wires, _, _) in enumerate(components):
                    word = "".join(term.paulis[q] for q in wires)
                    if set(word) != {"I"}:
                        index = queries[i].setdefault(word, len(queries[i]))
                        factors.append((i, index))
                route.append((term.coefficient, tuple(factors)))
            routes.append(tuple(route))
        metadata = 256 * sum(len(q) for q in queries) + 128 * work
        metadata += 16 + 32 * sum(len(q) for q in queries) + 64 * len(obs)
        capacity = int(budget * 2**20)
        if metadata >= capacity:
            raise ResourceError("Component descriptors exceed max_memory_mb.")
        local_budget = (capacity - metadata) / 2**20
        peak = 0
        for (_, _, child), query in zip(components, queries, strict=True):
            if not query:
                continue
            if child.has_effects:
                from .density import _planning_limit, _resource_check
                from .planning import plan_density

                limit = _planning_limit(child, None, local_budget, max_branches=self.max_branches)
                records = plan_density(child, max_branches=limit)
                _resource_check(child, None, local_budget, branches=records.peak_branches)
                required = 64 * records.peak_branches * (1 << (2 * child.num_qubits))
            else:
                from .simulator import _resource_check

                _resource_check(child, None, local_budget)
                required = 64 * (1 << child.num_qubits)
            peak = max(peak, required)
        for name, value in (
            ("observables", obs),
            ("components", components),
            ("_queries", tuple(tuple(PauliString(w) for w in q) for q in queries)),
            ("_routes", tuple(routes)),
            ("planning_work", work),
            ("working_bytes", metadata + peak),
            ("max_memory_mb", budget),
            ("_metadata_bytes", metadata),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {
            "scope": "declared_expectations_from_zero_product_state",
            "num_qubits": self.circuit.num_qubits,
            "active_qubits": sum(len(wires) for wires, _, _ in self.components),
            "components": [
                {"qubits": list(w), "clbits": list(c), "effects": b.has_effects}
                for w, c, b in self.components
            ],
            "max_component_qubits": max((len(w) for w, _, _ in self.components), default=0),
            "planning_work": self.planning_work,
            "working_bytes": self.working_bytes,
            "exactness": "no_intentional_approximation; floating_point_roundoff_remains",
            "memory_scope": "representation_model_excludes_autodiff_tapes_and_process_RSS",
        }

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        import math

        from .pauli import _prepare_bindings
        from .simulator import expectations

        names = frozenset(self.circuit.parameters)
        ops, values, shape = _prepare_bindings(
            bindings, backend, like, batch_shape, names, tuple(names)
        )
        if like is None and backend != "numpy":
            like = next((v for v in values.values() if hasattr(v, "dtype")), None)
        rows = max(1, math.prod(shape))
        output_bytes = 16 + 32 * sum(len(q) for q in self._queries) + 64 * len(self.observables)
        capacity = (
            int(memory_budget(self.max_memory_mb) * 2**20)
            - self._metadata_bytes
            - output_bytes * (rows - 1)
        )
        if capacity <= 0:
            raise ResourceError("Component batched readouts exceed max_memory_mb.")
        local_budget = capacity / 2**20
        zero = ops.array(0, real_value=True)
        for value in values.values():
            zero = zero + (ops.array(value, real_value=True) * 0) ** 2
        outputs = {}
        for i, ((_, _, child), queries) in enumerate(
            zip(self.components, self._queries, strict=True)
        ):
            if queries:
                outputs[i] = expectations(
                    child,
                    queries,
                    {p: values[p] for p in child.parameters},
                    backend=backend,
                    like=like,
                    max_memory_mb=local_budget,
                    max_branches=self.max_branches,
                    batch_shape=shape,
                )
        result = []
        for route in self._routes:
            total = zero
            for coefficient, factors in route:
                value = zero + coefficient
                for group, index in factors:
                    value = value * outputs[group][..., index]
                total = total + value
            result.append(total)
        return ops.real(ops.stack(result, axis=-1))


def plan_components(circuit, observables, **limits):
    """Prepare exact quantum/classical component readouts without a total-width ceiling."""
    return ComponentPlan(circuit, observables, **limits)

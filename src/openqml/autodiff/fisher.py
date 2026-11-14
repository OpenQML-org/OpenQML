"""Factorwise regularized SLD geometry for noisy and dynamic density circuits."""

from dataclasses import dataclass, field
from typing import NamedTuple

from ..compute.kernel import get_compute_kernel, memory_budget
from ..errors import ResourceError, ValidationError
from ..ir._structure import split_circuit
from ..ir.compiler import _check_circuit_limits
from ..ir.core import _require_circuit, integer
from ..ir.parameters import real
from ..sim.density import _compiled_density, _density_program, _planning_limit, _record_plan
from ._common import _names
from ._sylvester import sld_solve


class _DensityFactor(NamedTuple):
    circuit: object
    program: object


@dataclass(frozen=True, eq=False)
class QuantumFisherPlan:
    """Apply J* (L_rho + state_damping I)^-1 J without a parameter matrix.

    L_rho(X)=(rho X+X rho)/2. Independent connected factors contribute
    additively. Positive damping regularizes each factor's SLD equation;
    the zero-damping limit is the final marginal-state QFIM. Classical
    measurement records are marginalized, rather than treated as readouts.
    """

    circuit: object
    parameters: tuple | None = None
    state_damping: float = 1e-4
    max_qubits: int | None = None
    max_memory_mb: float = 256
    max_branches: int = 256
    max_work: int = 1_000_000
    _plans: tuple = field(init=False, repr=False)
    _native_workspace: int = field(init=False, repr=False)
    _metric_kind = "sld"

    def __post_init__(self):
        _require_circuit(self.circuit)
        _check_circuit_limits(self.circuit)
        damping = real(self.state_damping)
        budget = memory_budget(self.max_memory_mb)
        if damping <= 0 or budget <= 0:
            raise ValidationError("SLD state damping and memory budget must be positive.")
        if self.max_qubits is not None:
            integer(self.max_qubits, "max_qubits", 1)
            if self.circuit.num_qubits > self.max_qubits:
                raise ResourceError("Density geometry exceeds the explicit max_qubits limit.")
        integer(self.max_branches, "max_branches", 1)
        integer(self.max_work, "max_work", 1)
        names = _names(self.parameters, self.circuit.parameters)
        work = 0

        def charge(amount):
            nonlocal work
            work += amount
            if work > self.max_work:
                raise ResourceError("Density geometry dependency analysis exceeds max_work.")

        components = split_circuit(self.circuit, charge)
        positions = {name: i for i, name in enumerate(names)}
        plans, workspace = [], 0
        for _, _, child in components:
            indices = tuple(sorted(positions[p] for p in child.parameters if p in positions))
            if not indices:
                continue
            limit = _planning_limit(child, None, budget, max_branches=self.max_branches)
            records = _record_plan(child, max_branches=limit)
            count = sum(1 for _ in child.walk_operations())
            workspace = max(
                workspace,
                16 * (1 << (2 * child.num_qubits)) * (32 + 12 * count * records.peak_branches),
            )
            get_compute_kernel().admit(workspace + 256 * work, budget, itemsize=1)
            plans.append((_DensityFactor(child, _density_program(child, records)), indices))
        get_compute_kernel().admit(workspace + 256 * work + 128 * len(names), budget, itemsize=1)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "state_damping", damping)
        object.__setattr__(self, "max_memory_mb", budget)
        object.__setattr__(self, "_plans", tuple(plans))
        object.__setattr__(self, "_native_workspace", workspace + 256 * work)

    def to_dict(self):
        return dict(
            scope="factorwise_regularized_SLD_of_final_unconditional_density",
            parameters=list(self.parameters),
            state_damping=self.state_damping,
            differentiated_components=len(self._plans),
            max_component_qubits=max((p.circuit.num_qubits for p, _ in self._plans), default=0),
            working_mb=self._native_workspace / 2**20,
            classical_records="marginalized",
            derivative="implicit_Sylvester; no_eigenvector_derivatives",
            density_execution="prepared_static_register_program",
        )

    def native_matvec(self, bindings, vector, *, backend="torch", **options):
        from .native_geometry import native_matvec

        return native_matvec(self, bindings, vector, backend=backend, **options)

    def native_natural_gradient(self, bindings, gradient, *, backend="torch", **options):
        from .native_geometry import native_natural_gradient

        return native_natural_gradient(self, bindings, gradient, backend=backend, **options)

    def spectral_step(self, bindings, gradient, *, block_shapes, backend="torch", **options):
        from .native_geometry import spectral_step

        return spectral_step(
            self, bindings, gradient, block_shapes=block_shapes, backend=backend, **options
        )


def quantum_fisher(circuit, *, parameters=None, **options):
    """Prepare resident Torch/JAX regularized density-state Fisher geometry."""
    return QuantumFisherPlan(circuit, parameters, **options)


def density_product(plan, local, vector, mass, ops, backend):
    xp = ops.xp
    if backend == "torch":
        jvp, vjp = xp.func.jvp, xp.func.vjp
    else:
        import jax

        jvp, vjp = jax.jvp, jax.vjp

    def cat(values):
        return xp.cat(values, dim=-1) if backend == "torch" else xp.concatenate(values, axis=-1)

    def stack(values):
        return xp.stack(values, dim=-1) if backend == "torch" else xp.stack(values, axis=-1)

    result = [vector[i] * 0 for i in range(len(plan.parameters))]
    for child, indices in plan._plans:
        names = tuple(plan.parameters[i] for i in indices)
        per_sample = any(local[name].ndim for name in names)
        coordinates = stack(
            tuple(
                xp.broadcast_to(local[name], mass.shape) if per_sample else local[name]
                for name in names
            )
        )
        tangent = xp.broadcast_to(stack(tuple(vector[i] for i in indices)), coordinates.shape)

        def state(values, names=names, circuit=child.circuit, program=child.program):
            selected = {name: values[..., j] for j, name in enumerate(names)}
            rho = _compiled_density(
                circuit,
                program,
                {name: selected.get(name, local[name]) for name in circuit.parameters},
                ops,
                None,
                plan.max_memory_mb,
            )
            rho = xp.broadcast_to(rho, mass.shape + rho.shape[-2:])
            return cat((rho.real, rho.imag))

        packed, velocity = jvp(state, (coordinates,), (tangent,))
        size = packed.shape[-1] // 2
        imaginary = ops.array(1j) if backend == "torch" else 1j
        rho = packed[..., :size] + imaginary * packed[..., size:]
        drho = velocity[..., :size] + imaginary * velocity[..., size:]
        sld = sld_solve(rho, drho, plan.state_damping, ops)
        _, pullback = vjp(state, coordinates)
        product = pullback(cat((sld.real, sld.imag)) * mass[..., None, None])[0]
        if product.ndim > 1:
            product = product.sum(dim=0) if backend == "torch" else product.sum(axis=0)
        for j, index in enumerate(indices):
            result[index] = result[index] + product[j]
    return xp.stack(tuple(result)) if result else vector * 0

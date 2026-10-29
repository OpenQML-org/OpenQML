"""Bounded, inspectable selection of exact floating-point readout algorithms."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .backends import BackendCapabilities
from .compiler import _check_circuit_limits
from .errors import ConversionError, ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .semantics import _CLIFFORD


def _available_memory():
    """Observe host availability and the remaining container limit when present."""
    import psutil

    available = int(psutil.virtual_memory().available)
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            limit = int(Path(limit_path).read_text().strip())
            used = int(Path(used_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 1 << 60:
            available = min(available, max(0, limit - used))
    return available


def _budget(value):
    if value is None:
        available = _available_memory()
        budget = min(4096, available // (4 * 2**20))
        if budget < 1:
            raise ResourceError("Available memory cannot support the automatic working budget.")
        return float(budget), "quarter_available_host_or_container_memory_capped_at_4_GiB"
    value = real(value)
    if value <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    return value, "explicit"


def _fits(exponent, budget, coefficient=1):
    numerator, denominator = budget.as_integer_ratio()
    capacity = (numerator << 20) // denominator
    return coefficient <= capacity >> exponent


def _dense_bytes(width, density=False):
    exponent = (2 if density else 1) * width + 6
    return (1 << exponent) if exponent < 63 else None


def _gpu_status():
    """Check the installed native tensor runtime; a CPU wheel needs no GPU extra."""
    from .adapters._common import dependency

    torch = dependency("torch", "torch")
    if not torch.cuda.is_available():
        return False, 0
    try:
        free, _ = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        return False, 0
    return True, int(free)


@dataclass(frozen=True)
class ExecutionCandidate:
    method: str
    eligible: bool
    reason: str
    working_bytes: int | None = None


@dataclass(frozen=True)
class AdaptivePlan:
    circuit: object = field(repr=False)
    observables: tuple = field(repr=False)
    selected_method: str | None
    max_memory_mb: float
    memory_policy: str
    max_qubits: int | None
    max_branches: int
    candidates: tuple[ExecutionCandidate, ...]
    mps_profile: dict | None
    _pauli: object = field(default=None, repr=False, compare=False)
    _runner: object = field(default=None, repr=False, compare=False)
    dense_device: str = "CPU"

    @property
    def supported(self):
        return self.selected_method is not None

    def to_dict(self):
        return {
            "scope": "declared_expectations_from_zero_state",
            "selected_method": self.selected_method,
            "device": None
            if not self.supported
            else "GPU"
            if self.selected_method == "statevector-gpu"
            else "CPU",
            "requested_dense_device": self.dense_device,
            "supported": self.supported,
            "num_qubits": self.circuit.num_qubits,
            "max_qubits": self.max_qubits,
            "max_memory_mb": self.max_memory_mb,
            "memory_policy": self.memory_policy,
            "exactness": "no_intentional_truncation_or_shot_approximation; "
            "floating_point_roundoff_remains",
            "memory_guarantee": "representation_admission_model_not_process_RSS_limit",
            "mps_profile": self.mps_profile,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def raise_for_errors(self):
        if not self.supported:
            details = "; ".join(f"{c.method}: {c.reason}" for c in self.candidates)
            raise ResourceError(f"No admitted exact execution method. {details}")

    def evaluate(self, bindings=None, *, backend="numpy", like=None):
        self.raise_for_errors()
        if backend not in {"numpy", "torch", "jax"}:
            raise ValidationError("Evaluation backend must be numpy, torch or jax.")
        if self.selected_method == "pauli":
            return self._pauli.evaluate(bindings, backend=backend, like=like)
        if self.selected_method == "density":
            from .density import _expectations_from_density, density_matrix
            from .simulator import _TensorOps

            rho = density_matrix(
                self.circuit,
                bindings,
                backend=backend,
                like=like,
                max_qubits=self.circuit.num_qubits,
                max_memory_mb=self.max_memory_mb,
                max_branches=self.max_branches,
            )
            ops = _TensorOps(backend, rho.real if like is None else like)
            return _expectations_from_density(rho, self.observables, self.circuit.num_qubits, ops)
        if self.selected_method == "statevector":
            from .simulator import expectations

            return expectations(
                self.circuit,
                self.observables,
                bindings,
                backend=backend,
                like=like,
                max_qubits=self.circuit.num_qubits if self.max_qubits is None else self.max_qubits,
                max_memory_mb=self.max_memory_mb,
                max_branches=self.max_branches,
            )
        if self.selected_method == "statevector-gpu":
            import torch

            from .simulator import expectations

            if backend not in {"numpy", "torch"}:
                raise ValidationError(
                    "CUDA dense plans support numpy output or native torch autodiff."
                )
            if like is None:
                like = torch.empty((), dtype=torch.float64, device="cuda")
            if not isinstance(like, torch.Tensor) or like.device.type != "cuda":
                raise ValidationError("CUDA plans require a CUDA tensor context.")
            available, free = _gpu_status()
            if not available or _dense_bytes(self.circuit.num_qubits) > free // 2:
                raise ResourceError(
                    "Free CUDA memory changed after planning; re-plan the execution."
                )
            values = expectations(
                self.circuit,
                self.observables,
                bindings,
                backend="torch",
                like=like,
                max_qubits=self.circuit.num_qubits,
                max_memory_mb=self.max_memory_mb,
            )
            return values if backend == "torch" else values.detach().cpu().numpy()
        if backend != "numpy" or like is not None:
            raise ValidationError(
                "The selected MPS/stabilizer executor returns numerical readouts; "
                "use a numerical gradient bridge, or choose pauli/statevector for native autodiff."
            )
        from .batch import expectations_batch

        return expectations_batch(
            self.circuit,
            self.observables,
            [bindings or {}],
            backend=self._runner,
            max_qubits=self.max_qubits,
            max_memory_mb=self.max_memory_mb,
            max_branches=self.max_branches,
        )[0]


def plan_execution(
    circuit,
    observables,
    *,
    method="auto",
    max_memory_mb=None,
    max_qubits=None,
    max_branches=256,
    max_operations=100_000,
    max_work=1_000_000,
    max_pauli_terms=4096,
    max_bond_dimension=256,
    device="CPU",
):
    """Plan before allocating quantum states; selection is inspectable and bounded.

    Automatic memory uses one quarter of observed host/container availability,
    capped at 4 GiB. No fixed source-width ceiling applies unless requested.
    MPS admission uses structural upper bounds and disables truncation. It can
    reject circuits whose actual bond ranks would be lower. Selection never
    retries an execution failure with an unbudgeted algorithm.
    """
    from .batch import _observables
    from .pauli import PauliPlan
    from .planning import plan_density

    allowed = {"auto", "pauli", "stabilizer", "mps", "statevector", "density"}
    if method not in allowed or device not in {"CPU", "GPU", "auto"}:
        raise ValidationError("Unknown execution method or device (CPU, GPU, auto).")
    _check_circuit_limits(circuit, max_operations=max_operations)
    integer(max_work, "max_work", 1)
    integer(max_pauli_terms, "max_pauli_terms", 1)
    integer(max_bond_dimension, "max_bond_dimension", 1)
    integer(max_branches, "max_branches", 1)
    if max_qubits is not None:
        integer(max_qubits, "max_qubits", 1)
    observables = _observables(observables, [circuit])
    from .observables import Hamiltonian

    terms = sum(
        len(observable.terms) if isinstance(observable, Hamiltonian) else 1
        for observable in observables
    )
    if terms * (1 + circuit.num_qubits) > max_work:
        raise ResourceError("Readout descriptions exceed max_work.")
    explicit_width = max_qubits is not None and circuit.num_qubits > max_qubits
    budget, policy = (
        (0.0, "not_queried_after_explicit_width_rejection")
        if explicit_width and max_memory_mb is None
        else _budget(max_memory_mb)
    )
    candidates, selected, pauli, runner, mps_profile = [], None, None, None, None
    deferred_stabilizer = False
    if explicit_width:
        candidates.append(
            ExecutionCandidate(method, False, "Source exceeds the explicit max_qubits limit.")
        )
    else:
        unitary = not circuit.has_effects
        clifford = unitary and all(op.name in _CLIFFORD | {"barrier"} for op in circuit.operations)
        if method in {"auto", "stabilizer"}:
            # A packed tableau needs O(n^2) bits. Reserve a deliberately larger
            # byte-per-entry workspace model for native snapshots and readouts.
            workspace = (
                16 * circuit.num_qubits * circuit.num_qubits
                + 128 * len(observables) * circuit.num_qubits
            )
            eligible = clifford and workspace <= budget * 2**20
            candidates.append(
                ExecutionCandidate(
                    "stabilizer",
                    eligible,
                    "Named Clifford gates and bounded tableau workspace."
                    if eligible
                    else "Requires named Clifford gates and sufficient tableau workspace.",
                    workspace,
                )
            )
            if eligible:
                if method == "stabilizer" or circuit.num_qubits >= 64 or len(observables) >= 16:
                    selected = "stabilizer"
                else:
                    # Avoid importing/compiling an SDK for tiny Clifford
                    # queries that the local symbolic engine can answer.
                    deferred_stabilizer = True
        if selected is None and method in {"auto", "pauli"}:
            try:
                if not unitary:
                    raise ResourceError("Pauli propagation requires a unitary circuit.")
                numerator, denominator = budget.as_integer_ratio()
                available_bytes = (numerator << 20) // denominator
                nodes = min(max_work, 100_000, available_bytes // 1024)
                term_limit = min(
                    max_pauli_terms,
                    available_bytes // (2 * (512 + 2 * ((circuit.num_qubits + 7) // 8))),
                )
                if nodes < 2 or term_limit < 1:
                    raise ResourceError("Pauli graph/frontier bookkeeping exceeds the budget.")
                pauli = PauliPlan(
                    circuit,
                    observables,
                    max_terms=term_limit,
                    max_nodes=nodes,
                    max_work=max_work,
                    max_operations=max_operations,
                    max_memory_mb=budget,
                )
                selected = "pauli"
                candidates.append(
                    ExecutionCandidate(
                        "pauli", True, "Bounded exact symbolic Pauli propagation completed."
                    )
                )
            except (ConversionError, ResourceError) as error:
                candidates.append(ExecutionCandidate("pauli", False, str(error)))
        if selected is None and deferred_stabilizer:
            selected = "stabilizer"
        if selected is None and method in {"auto", "mps"}:
            from .mps import _layout

            mps_profile, reason = (
                _layout(circuit, max_bond_dimension, max_work)
                if unitary
                else (None, "MPS admission currently requires a unitary circuit.")
            )
            eligible = bool(
                mps_profile
                and mps_profile["analysis_complete"]
                and mps_profile["max_bond"] <= max_bond_dimension
                and mps_profile["workspace_bytes"] <= budget * 2**20
            )
            candidates.append(
                ExecutionCandidate(
                    "mps",
                    eligible,
                    reason
                    if eligible or mps_profile is None
                    else "Full-SVD MPS bond/workspace bound exceeds policy.",
                    None if mps_profile is None else mps_profile["workspace_bytes"],
                )
            )
            if eligible:
                selected = "mps"
        if selected is None and method in {"auto", "statevector", "density"}:
            density = not unitary or method == "density"
            chosen = "density" if density else "statevector"
            fits = _fits((2 if density else 1) * circuit.num_qubits + 6, budget)
            reason = "Four-buffer dense working model fits the memory budget."
            if fits and density:
                try:
                    records = plan_density(circuit, max_branches=max_branches)
                    from .density import _resource_check

                    _resource_check(circuit, max_qubits, budget, branches=records.peak_branches)
                except (ResourceError, ConversionError) as error:
                    fits, reason = False, str(error)
            candidates.append(
                ExecutionCandidate(
                    chosen,
                    fits,
                    reason
                    if fits
                    else "Dense storage or classical branches exceed the budget. " + reason,
                    _dense_bytes(circuit.num_qubits, density),
                )
            )
            if fits:
                selected = chosen
                if not density and device in {"GPU", "auto"}:
                    available, free = _gpu_status()
                    needed = _dense_bytes(circuit.num_qubits)
                    gpu_ok = available and needed is not None and needed <= free // 2
                    candidates.append(
                        ExecutionCandidate(
                            "statevector-gpu",
                            gpu_ok,
                            "CUDA is available; buffers fit half of free device memory."
                            if gpu_ok
                            else "CUDA unavailable or insufficient free device memory.",
                            needed,
                        )
                    )
                    if gpu_ok:
                        selected = "statevector-gpu"
                    elif device == "GPU":
                        selected = None
        if selected == "stabilizer":
            from .aer import AerBackend

            runner = AerBackend(
                name="adaptive-" + selected,
                method="stabilizer",
                parameter_batching=True,
                mps_truncation_threshold=0.0,
            )
        elif selected == "mps":
            from .mps import MPSBackend

            runner = MPSBackend(
                max_bond_dimension=max_bond_dimension,
                max_operations=max_operations,
                max_work=max_work,
            )
    return AdaptivePlan(
        circuit,
        observables,
        selected,
        budget,
        policy,
        max_qubits,
        max_branches,
        tuple(candidates),
        mps_profile,
        pauli,
        runner,
        device,
    )


@dataclass(frozen=True)
class AdaptiveBackend:
    """Numeric adaptive readouts through backend='auto'; no hidden approximation."""

    name: str = "auto"
    capabilities: BackendCapabilities = BackendCapabilities(
        native_expectations=True,
        channels=True,
        reset=True,
        classical_control=True,
        mid_circuit_measurements=True,
        measurement_expectations=True,
        deterministic_expectations=True,
    )

    def expectations_batch(self, circuits, observables, *, options):
        return np.asarray(
            [
                plan_execution(
                    circuit,
                    observables,
                    max_memory_mb=options.max_memory_mb,
                    max_qubits=options.max_qubits,
                    max_branches=options.max_branches,
                ).evaluate()
                for circuit in circuits
            ]
        )

    def run(self, circuit, *, options):
        from .simulator import statevector

        # A full-state request necessarily has exponential output size. It
        # cannot inherit the capacity of a structured readout representation.
        return statevector(
            circuit, max_qubits=options.max_qubits, max_memory_mb=options.max_memory_mb
        )

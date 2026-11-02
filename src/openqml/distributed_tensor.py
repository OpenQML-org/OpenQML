"""Communication-aware intermediate tensor distribution for exact contractions.

One rank owns one CUDA device. A binary DeviceMesh maps quantum mode bits to
DTensor shards. Contractions become distributed GEMMs with explicit layouts;
only scalar/readout outputs are replicated. CPU/Gloo is a correctness path and
is never reported as CUDA/NCCL performance evidence.
"""

import hashlib
import math
import time
from dataclasses import dataclass, field

from .errors import MissingDependencyError, ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .tensor_contraction import TensorNetworkPlan


@dataclass(frozen=True)
class Interconnect:
    """Sustained, caller-calibrated rates in bytes/s and complex multiply-adds/s.

    These values guide planning. They do not prove hardware throughput.
    """

    compute_flops: float
    memory_bandwidth: float
    network_bandwidth: float
    collective_latency: float

    def __post_init__(self):
        for name in ("compute_flops", "memory_bandwidth", "network_bandwidth"):
            if real(getattr(self, name)) <= 0:
                raise ValidationError(f"{name} must be positive.")
        if real(self.collective_latency) < 0:
            raise ValidationError("collective_latency must be nonnegative.")


def _local_elements(labels, layout):
    return 1 << (len(labels) - sum(x in labels for x in layout if x is not None))


def _layouts(la, lb, shared, old_a, old_b, bits, limit):
    retained = tuple(x for x in la + lb if x not in shared)
    reduced = tuple(shared)
    orders = (retained + reduced, reduced + retained, tuple(reversed(retained)) + reduced)
    result = []

    def add(order):
        chosen = list(dict.fromkeys(x for x in order if x is not None))[:bits]
        value = tuple(chosen + [None] * (bits - len(chosen)))
        if value not in result:
            result.append(value)

    # Keeping a child's layout avoids communication at large intermediates.
    add(())
    add(old_a + old_b)
    add(old_b + old_a)
    for order in orders:
        for start in range(min(len(order), limit)):
            add(order[start:] + order[:start])
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def _redistribution(labels, old, new, hardware):
    size = 16 * max(_local_elements(labels, old), _local_elements(labels, new))
    changes = sum(a != b and (a in labels or b in labels) for a, b in zip(old, new, strict=True))
    return (
        changes * (hardware.collective_latency + size / hardware.network_bandwidth),
        changes * size,
    )


def _plan_graph(graph, bits, hardware, budget, beam_width, max_layouts, charge):
    desc, _, labels, steps, _, _, _, _ = graph
    empty = (None,) * bits
    sizes = {i: row for i, row in enumerate(labels)}
    # Beam states hold only mode layouts and costs, never tensor values.
    states = [(0.0, dict.fromkeys(range(len(desc)), empty), (), 0, 0)]
    for k, (a, b, la, lb, shared, out) in enumerate(steps):
        next_states = {}
        for elapsed, live, schedule, peak, traffic in states:
            for layout in _layouts(la, lb, shared, live[a], live[b], bits, max_layouts):
                charge()
                new_a = tuple(x if x in la else None for x in layout)
                new_b = tuple(x if x in lb else None for x in layout)
                new_out = tuple(x if x in out else None for x in layout)
                comm_a, bytes_a = _redistribution(la, live[a], new_a, hardware)
                comm_b, bytes_b = _redistribution(lb, live[b], new_b, hardware)
                old_size = sum(
                    _local_elements(sizes[i], placement) for i, placement in live.items()
                )
                ea, eb = _local_elements(la, new_a), _local_elements(lb, new_b)
                eo = _local_elements(out, new_out)
                # Conservatively reserve input permutation/redistribution,
                # GEMM output and collective buffers, excluding autodiff tapes.
                workspace = 16 * (
                    old_size
                    + 2 * sum(1 << len(row) for row in graph[1])
                    + 4
                    * (_local_elements(la, live[a]) + _local_elements(lb, live[b]) + ea + eb + eo)
                )
                if workspace > budget:
                    continue
                reduced = sum(x in shared for x in layout if x is not None)
                partial_bytes = reduced * 16 * eo
                comm = comm_a + comm_b + reduced * hardware.collective_latency
                comm += partial_bytes / hardware.network_bandwidth
                work = (1 << (len(out) + len(shared))) / (1 << sum(x is not None for x in layout))
                gemm = max(
                    work / hardware.compute_flops, 16 * (ea + eb + eo) / hardware.memory_bandwidth
                )
                updated = {i: v for i, v in live.items() if i not in (a, b)}
                updated[len(desc) + k] = new_out
                key = tuple(updated.items())
                candidate = (
                    elapsed + comm + gemm,
                    updated,
                    schedule + (layout,),
                    max(peak, workspace),
                    traffic + bytes_a + bytes_b + partial_bytes,
                )
                if key not in next_states or candidate[0] < next_states[key][0]:
                    next_states[key] = candidate
        if not next_states:
            raise ResourceError("No distributed layout fits the per-rank memory budget.")
        states = sorted(next_states.values(), key=lambda s: s[0])[:beam_width]
        sizes[len(desc) + k] = out
        sizes.pop(a)
        sizes.pop(b)
    best = min(states, key=lambda s: s[0])
    return best[2], best[3], best[4], best[0]


@dataclass(frozen=True, eq=False)
class DistributedTensorPlan:
    circuit: object
    observables: tuple
    world_size: int
    interconnect: Interconnect
    max_memory_mb: float = 256
    max_layout_work: int = 100_000
    beam_width: int = 4
    max_layouts: int = 12
    tensor_options: object = field(default_factory=dict)
    rematerialize: bool = True
    _plan: object = field(init=False, repr=False)
    _schedules: tuple = field(init=False, repr=False)
    working_bytes_per_rank: int = field(init=False)
    estimated_communication_bytes: int = field(init=False)
    estimated_seconds: float = field(init=False)
    layout_work: int = field(init=False)

    def __post_init__(self):
        size = integer(self.world_size, "world_size", 2)
        if size & (size - 1):
            raise ValidationError("Binary-mode distribution requires a power-of-two world_size.")
        if not isinstance(self.interconnect, Interconnect):
            raise ValidationError("Provide calibrated Interconnect rates.")
        if type(self.rematerialize) is not bool:
            raise ValidationError("rematerialize must be a boolean.")
        for name in ("max_layout_work", "beam_width", "max_layouts"):
            integer(getattr(self, name), name, 1)
        budget = real(self.max_memory_mb) * 2**20
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        options = dict(self.tensor_options)
        if {"max_memory_mb", "rematerialize"} & options.keys():
            raise ValidationError("Use the distributed plan's memory/rematerialize arguments.")
        plan = TensorNetworkPlan(
            self.circuit,
            tuple(self.observables),
            max_memory_mb=self.max_memory_mb * size,
            **options,
        )
        available = budget - plan._metadata_bytes
        if available <= 0:
            raise ResourceError("Distributed descriptors exceed the per-rank memory budget.")
        work = 0

        def charge():
            nonlocal work
            work += 1
            if work > self.max_layout_work:
                raise ResourceError("Distributed layout search exceeds max_layout_work.")

        schedules, peak, traffic, seconds = [], 0, 0, 0.0
        for graph in plan._graphs:
            schedule, local, comm, elapsed = _plan_graph(
                graph,
                size.bit_length() - 1,
                self.interconnect,
                available,
                self.beam_width,
                self.max_layouts,
                charge,
            )
            schedules.append(schedule)
            peak = max(peak, local)
            slices = 1 << len(graph[4])
            traffic += comm * slices
            seconds += elapsed * slices
        for name, value in (
            ("_plan", plan),
            ("_schedules", tuple(schedules)),
            ("working_bytes_per_rank", plan._metadata_bytes + max(peak, 16)),
            ("estimated_communication_bytes", traffic),
            ("estimated_seconds", seconds),
            ("layout_work", work),
            ("observables", plan.observables),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {
            "representation": "exact_distributed_intermediate_tensors",
            "world_size": self.world_size,
            "working_bytes_per_rank": self.working_bytes_per_rank,
            "estimated_communication_bytes": self.estimated_communication_bytes,
            "estimated_seconds": self.estimated_seconds,
            "layout_work": self.layout_work,
            "distributed_contractions": sum(
                any(x is not None for x in layout) for graph in self._schedules for layout in graph
            ),
            "exactness": "all_modes_and_slices_retained; floating_point_roundoff",
            "memory_scope": "forward_admission; excludes_autodiff_tapes_and_allocator_reserves",
            "performance_evidence": "planning_estimates_only",
        }

    def evaluate(self, bindings=None, *, mesh, like, batch_shape=None, return_report=False):
        """Collective call with identical inputs and loss on every tensor rank.

        Replicated parameters receive the full derivative once. Use a separate
        data-parallel process group for DDP; do not average these tensor ranks.
        All ranks must also participate in backward/checkpoint recomputation.
        """
        try:
            import torch
            import torch.distributed as dist
            from torch.distributed.tensor import DTensor, Replicate, Shard
        except ImportError as exc:
            raise MissingDependencyError(
                "Distributed contractions require PyTorch >= 2.5."
            ) from exc
        if not dist.is_initialized():
            raise ValidationError(
                "Initialize the torch process group before distributed evaluation."
            )
        bits = self.world_size.bit_length() - 1
        if mesh.size() != self.world_size or tuple(mesh.mesh.shape) != (2,) * bits:
            raise ValidationError(
                "DeviceMesh must have binary shape (2, ..., 2) matching world_size."
            )
        if not isinstance(like, torch.Tensor) or like.dtype not in (torch.float32, torch.float64):
            raise ValidationError("like must be a float32/float64 Torch tensor on the rank device.")
        if mesh.device_type != like.device.type:
            raise ValidationError("DeviceMesh and input tensor device types differ.")
        if like.device.type not in {"cpu", "cuda"}:
            raise ValidationError("Distributed contractions support CPU/Gloo or CUDA/NCCL.")
        from .pauli import _prepare_bindings

        _, _, shape = _prepare_bindings(
            bindings,
            "torch",
            like,
            batch_shape,
            frozenset(self.circuit.parameters),
            self._plan.active_parameters,
        )
        rows = max(1, math.prod(shape))
        metadata = self._plan._metadata_bytes
        if metadata + rows * (self.working_bytes_per_rank - metadata) > self.max_memory_mb * 2**20:
            raise ResourceError("Distributed batch exceeds the per-rank memory budget.")
        cuda = like.device.type == "cuda"
        if cuda and (not torch.cuda.is_available() or dist.get_backend() != "nccl"):
            raise ResourceError("CUDA distributed contractions require actual CUDA/NCCL.")
        if cuda and like.device.index != torch.cuda.current_device():
            raise ValidationError("Set the current CUDA device to this rank's input device.")
        # Verify collective ordering and replicated inputs before launching any
        # layout-specific collective. No parameter values enter the report.
        fingerprint = hashlib.sha256(
            repr(
                (self._plan.circuit, self._plan.observables, self._schedules, batch_shape)
            ).encode()
        )
        for key, value in sorted((bindings or {}).items()):
            fingerprint.update(key.encode())
            if not isinstance(value, torch.Tensor) or value.device != like.device:
                raise ValidationError("Distributed bindings must be tensors on the rank device.")
            fingerprint.update(str((tuple(value.shape), value.dtype)).encode())
            fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
        digest = fingerprint.digest()
        signature = torch.tensor(list(digest), device=like.device, dtype=torch.uint8)
        signatures = [torch.empty_like(signature) for _ in range(self.world_size)]
        mismatch = torch.zeros((), dtype=torch.int32, device=like.device)
        for axis in range(mesh.ndim):
            pair = signatures[:2]
            dist.all_gather(pair, signature, group=mesh.get_group(axis))
            if any(not torch.equal(signature, other) for other in pair):
                mismatch.fill_(1)
        for axis in range(mesh.ndim):
            dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=mesh.get_group(axis))
        if mismatch.item():
            raise ValidationError(
                "Tensor ranks must use identical plans, bindings and batch shapes."
            )
        executor = _DTensorExecutor(self, mesh, DTensor, Replicate, Shard)
        keys, values = tuple(bindings or {}), tuple((bindings or {}).values())

        def forward(*arguments):
            return self._plan._evaluate(
                dict(zip(keys, arguments, strict=True)),
                backend="torch",
                like=like,
                batch_shape=batch_shape,
                _executor=executor,
            )

        if cuda:
            torch.cuda.synchronize(like.device)
        start = time.perf_counter()
        if self.rematerialize and any(v.requires_grad for v in values):
            from torch.utils.checkpoint import checkpoint

            result = checkpoint(forward, *values, use_reentrant=False, preserve_rng_state=False)
        else:
            result = forward(*values)
        if not return_report:
            return result
        if cuda:
            torch.cuda.synchronize(like.device)
        report = {
            "execution_kind": "cuda_nccl" if cuda else "cpu_correctness_only",
            "rank": sum(bit << (bits - 1 - axis) for axis, bit in enumerate(mesh.get_coordinate())),
            "world_size": self.world_size,
            "device": str(like.device),
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(like.device) if cuda else "CPU",
            "wall_seconds": time.perf_counter() - start,
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(like.device)
            if cuda
            else None,
            "plan": self.to_dict(),
        }
        return result, report


class _DTensorExecutor:
    def __init__(self, plan, mesh, tensor_type, replicate, shard):
        self.plan, self.mesh = plan, mesh
        self.tensor_type, self.replicate, self.shard = tensor_type, replicate, shard

    def _placements(self, labels, layout, prefix):
        return tuple(
            self.shard(prefix + labels.index(x)) if x in labels else self.replicate()
            for x in layout
        )

    def input(self, value, labels):
        return self.tensor_type.from_local(
            value, self.mesh, [self.replicate()] * self.mesh.ndim, run_check=False
        )

    def _matrix(self, value, labels, first, second, layout):
        prefix = value.ndim - len(labels)
        placements = self._placements(labels, layout, prefix)
        value = value.redistribute(placements=placements)
        local = value.to_local(grad_placements=placements)
        axes = tuple(range(prefix)) + tuple(prefix + labels.index(x) for x in first + second)
        local = local.permute(axes)
        shape = tuple(value.shape[:prefix]) + (1 << len(first), 1 << len(second))
        local_first = 1 << (len(first) - sum(x in first for x in layout if x is not None))
        local_second = 1 << (len(second) - sum(x in second for x in layout if x is not None))
        local = local.reshape(tuple(local.shape[:prefix]) + (local_first, local_second))
        matrix_layout = tuple(
            self.shard(len(shape) - 2)
            if x in first
            else self.shard(len(shape) - 1)
            if x in second
            else self.replicate()
            for x in layout
        )
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        return self.tensor_type.from_local(
            local, self.mesh, matrix_layout, run_check=False, shape=shape, stride=stride
        )

    def contract(self, left, right, la, lb, shared, graph, step):
        layout = self.plan._schedules[graph][step]

        # Mode reordering keeps sharded bits first within each GEMM dimension.
        def ordered(labels):
            return tuple(x for x in layout if x in labels) + tuple(
                x for x in labels if x not in layout
            )

        a = ordered(tuple(x for x in la if x not in shared))
        b = ordered(tuple(x for x in lb if x not in shared))
        k = ordered(shared)
        value = self._matrix(left, la, a, k, layout) @ self._matrix(right, lb, k, b, layout)
        prefix = value.ndim - 2
        placements = tuple(
            self.shard(prefix) if x in a else self.shard(prefix + 1) if x in b else self.replicate()
            for x in layout
        )
        value = value.redistribute(placements=placements)
        local = value.to_local(grad_placements=placements)
        out = a + b
        local = local.reshape(
            tuple(local.shape[:prefix]) + tuple(1 if x in layout else 2 for x in out)
        )
        shape = tuple(value.shape[:prefix]) + (2,) * len(out)
        full = self.tensor_type.from_local(
            local,
            self.mesh,
            self._placements(out, layout, prefix),
            run_check=False,
            shape=shape,
            stride=tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))),
        )
        original = tuple(x for x in la if x not in shared) + tuple(x for x in lb if x not in shared)
        return full.permute(tuple(range(prefix)) + tuple(prefix + out.index(x) for x in original))

    def finish(self, value):
        return value.full_tensor()

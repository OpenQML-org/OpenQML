# Distributed tensor execution

`DistributedTensorPlan` partitions intermediate circuit tensors across devices,
then contracts them as distributed GEMMs. It preserves every quantum mode and
every exact slice. This adds model parallelism alongside the existing DDP/JAX
data-parallel trainers. Contraction structure determines the total work.

Construct a plan from `circuit`, `observables`, `world_size`, calibrated
`Interconnect` rates and `max_memory_mb` **per rank**. Supply path-search options
through `tensor_options`. Construction performs bounded layout search without
initializing a cluster or executing a circuit. `to_dict()` labels all predicted
time, traffic and memory values as planning estimates.

The layout search considers keeping input shards, redistributing retained modes,
and distributing contracted modes. It compares compute, memory bandwidth,
interconnect bandwidth and collective latency with a bounded beam search.
Sharded bits are ordered within GEMM dimensions. DTensor performs the required
all-to-all, gather and reduction operations; large intermediates remain sharded.
High communication costs can make replicated local execution preferable.

## Execution contract

Use a Torch build exposing `torch.distributed.tensor` (PyTorch 2.5 or later).
Initialize the process group and select the current CUDA device before building
the mesh. One process owns one device. Pass a binary `DeviceMesh` with shape
`(2, ..., 2)` and a matching power-of-two device count. CUDA execution requires
actual CUDA and NCCL; CPU/Gloo is a separate correctness path.

Call `plan.evaluate(bindings, mesh=mesh, like=reference_tensor)` on every tensor
rank with identical plans, parameters, inputs and loss. The reference tensor
sets the device and float32/float64 precision. Input fingerprints are checked
collectively before layout-specific operations. Batched bindings obey per-rank
forward-memory admission; allocator reserves and autodiff tapes are excluded.

Replicated parameters receive one full first-order gradient on each tensor
rank. Higher-order distributed derivatives raise an explicit error. Local
`TensorNetworkPlan` retains native higher derivatives. Rematerialization defaults
to enabled; every tensor rank must participate in backward and recomputation.
Use a separate data-parallel group when combining this with DDP.

`return_report=True` also returns the rank, device name, runtime version,
measured forward wall time and CUDA process allocation high-water mark. Reports
describe that invocation and omit binding values. Tensor-rank timing and
collective overhead are part of execution. QPUs execute independent provider
jobs through the runtime scheduler.

## Hardware validation

CPU integration exercises two- and four-rank execution against independent local
readouts and gradients. The manually dispatched `CUDA NCCL integration` workflow
requires a caller-managed runner labelled `self-hosted, linux, x64, gpu` with at
least two CUDA devices. It verifies real device UUIDs, values and gradients and
uploads per-device execution reports. Missing CUDA/NCCL causes failure in this
hardware workflow, rather than a simulated-device pass.

Record throughput, strong scaling, memory use and complete training time on the
target cluster using the intended topology and workload. Keep these measurements
alongside the workflow's device and runtime reports.

The implementation adopts intermediate distribution and GEMM-oriented mode
ordering from [Pan et al., 2026](https://arxiv.org/abs/2606.01852), using Torch
DTensor and bounded search. OpenQML's planner uses the declared interconnect,
per-rank memory and candidate layouts. DTensor layout/gradient
semantics follow the [PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html).

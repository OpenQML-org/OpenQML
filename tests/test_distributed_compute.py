"""Distributed numerical contracts. GPU tests never emulate CUDA devices."""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

import openqml as q
from openqml._distributed_tensor import DistributedTensorPlan, Interconnect
from openqml._tensor_contraction import TensorNetworkPlan


def circuit():
    x, w = q.Parameter("x"), q.Parameter("w")
    builder = q.CircuitBuilder(5)
    for i in range(5):
        builder.ry(x + w * (i + 1), i).rz(w, i)
    for _ in range(3):
        for i in range(4):
            builder.rxx(w, i, i + 1)
    return builder.build()


def hardware(latency=0):
    # Synthetic planning rates, not a claim about available hardware.
    return Interconnect(1e10, 1e11, 1e13, latency)


def test_communication_cost_changes_layouts_and_search_is_bounded():
    c, obs = circuit(), (q.PauliString("XYZXZ"),)
    fast = DistributedTensorPlan(c, obs, 4, hardware())
    slow = DistributedTensorPlan(c, obs, 4, hardware(10))
    assert fast.to_dict()["distributed_contractions"] > 0
    assert slow.to_dict()["distributed_contractions"] == 0
    assert fast.working_bytes_per_rank < fast.max_memory_mb * 2**20
    assert fast.to_dict()["performance_evidence"] == "planning_estimates_only"
    with pytest.raises(q.ResourceError, match="max_layout_work"):
        DistributedTensorPlan(c, obs, 4, hardware(), max_layout_work=1)
    with pytest.raises(q.ResourceError, match="memory"):
        DistributedTensorPlan(c, obs, 4, hardware(), max_memory_mb=0.001)
    with pytest.raises(q.ValidationError, match="power-of-two"):
        DistributedTensorPlan(c, obs, 3, hardware())


def test_partial_layouts_preserve_mesh_axes_and_charge_only_collective_moves():
    from openqml._distributed_tensor import _layouts, _redistribution

    candidates = _layouts((0, 1), (1, 2), (1,), (None, 0), (None, 2), 2, 12)
    assert (None, 0) in candidates and (None, 2) in candidates
    assert any(sum(x is not None for x in p) == 1 for p in candidates)
    assert _redistribution((0, 1), (None, None), (0, 1), hardware(1)) == (0, 0)
    assert _redistribution((0, 1), (0, None), (None, None), hardware(1))[0] >= 1


def test_latency_aware_search_keeps_replication_as_a_completion_candidate():
    c, obs = circuit(), (q.PauliString("XYZXZ"),)
    for beam in (1, 2):
        plan = DistributedTensorPlan(c, obs, 4, hardware(10), beam_width=beam)
        assert plan.to_dict()["distributed_contractions"] == 0
    partial = DistributedTensorPlan(c, obs, 4, hardware(1e-9))
    assert partial.to_dict()["partially_sharded_contractions"] > 0


def _worker(rank, directory, size, cuda):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=(Path(directory) / "init").as_uri(),
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cuda" if cuda else "cpu", (2,) * (size.bit_length() - 1))
        c, obs = (
            circuit(),
            (
                q.PauliString("XYZXZ"),
                q.PauliString("IIIII"),
                q.Hamiltonian((q.PauliString("ZZZZZ", 0.3), q.PauliString("XIIII", -0.2))),
            ),
        )
        plan = DistributedTensorPlan(c, obs, size, hardware(), rematerialize=True)
        assert plan.to_dict()["distributed_contractions"] > 0
        like = torch.empty((), device=device, dtype=torch.float64)
        bindings = {
            "w": torch.tensor(0.41, device=device, dtype=torch.float64, requires_grad=True),
            "x": torch.tensor([0.2, -0.3], device=device, dtype=torch.float64, requires_grad=True),
        }
        if size == 4:
            bindings["x"] = bindings["x"].reshape(2, 1)
        expected = TensorNetworkPlan(c, obs).evaluate(bindings, backend="torch", like=like)
        actual, report = plan.evaluate(bindings, mesh=mesh, like=like, return_report=True)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        # A replicated loss must give one full gradient on each tensor rank,
        # not a world-size multiple, including across checkpoint recomputation.
        gradient = torch.autograd.grad(actual.square().sum(), tuple(bindings.values()))
        reference = torch.autograd.grad(expected.square().sum(), tuple(bindings.values()))
        for got, wanted in zip(gradient, reference, strict=True):
            torch.testing.assert_close(got, wanted, atol=2e-11, rtol=2e-11)
        if size == 4:
            mixed = DistributedTensorPlan(c, obs, size, hardware(1e-9), rematerialize=True)
            assert mixed.to_dict()["partially_sharded_contractions"] > 0
            mixed_bindings = {k: v.detach().clone().requires_grad_() for k, v in bindings.items()}
            mixed_result = mixed.evaluate(mixed_bindings, mesh=mesh, like=like)
            torch.testing.assert_close(mixed_result, expected.detach(), atol=2e-12, rtol=2e-12)
            mixed_gradients = torch.autograd.grad(
                mixed_result.square().sum(), tuple(mixed_bindings.values())
            )
            for got, wanted in zip(mixed_gradients, reference, strict=True):
                torch.testing.assert_close(got, wanted, atol=2e-11, rtol=2e-11)
        if size == 2:
            architecture = q.HybridArchitecture(
                2,
                1,
                hidden_features=4,
                depth=2,
                quantum=q.QuantumCellSpec(num_qubits=2, num_outputs=2),
                operators=(
                    q.OperatorSpec(),
                    q.OperatorSpec.attention(key_features=2, positional=False),
                ),
            )
            network = architecture.build(
                "torch",
                backend="torch",
                dtype=torch.float64,
                device=device,
                tensor_mesh=mesh,
                interconnect=hardware(),
            )
            with torch.no_grad():
                for i in range(2):
                    network.values[f"block_{i}_gain_q"].fill_(0.2)
                    network.values[f"block_{i}_gain_ff"].fill_(0.1)
            local_network = architecture.build(
                "torch", backend="torch", dtype=torch.float64, device=device
            )
            local_network.load_state_dict(network.state_dict())
            features = torch.tensor([[0.1, 0.3], [-0.2, 0.4]], dtype=torch.float64, device=device)
            features = torch.stack((features, features * 0.7 + 0.1), dim=1)
            distributed_output = network(features)
            local_output = local_network(features)
            torch.testing.assert_close(distributed_output, local_output, atol=2e-11, rtol=2e-11)
            distributed_output.square().sum().backward()
            local_output.square().sum().backward()
            for name in network.values:
                torch.testing.assert_close(
                    network.values[name].grad,
                    local_network.values[name].grad,
                    atol=3e-10,
                    rtol=3e-10,
                )
        fresh = plan.evaluate(bindings, mesh=mesh, like=like)
        with pytest.raises(q.ValidationError, match="first-order"):
            torch.autograd.grad(fresh.sum(), tuple(bindings.values()), create_graph=True)
        assert report["execution_kind"] == ("cuda_nccl" if cuda else "cpu_correctness_only")
        if cuda:
            report["physical_device_uuid"] = str(torch.cuda.get_device_properties(device).uuid)
            report["maximum_readout_error"] = float((actual - expected).abs().max().detach())
            report["maximum_gradient_error"] = max(
                float((g - r).abs().max().detach())
                for g, r in zip(gradient, reference, strict=True)
            )
            (Path(directory) / f"gpu-rank-{rank}.json").write_text(json.dumps(report, indent=2))
        # Divergent replicated parameters fail collectively before the GEMMs.
        wrong = {k: v.detach().clone() for k, v in bindings.items()}
        if rank == 0:
            wrong["w"] += 1
        with pytest.raises(q.ValidationError, match="identical"):
            plan.evaluate(wrong, mesh=mesh, like=like)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux distributed correctness integration")
@pytest.mark.parametrize("size", (2, 4))
def test_real_gloo_tensor_distribution_and_gradients(tmp_path, size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.distributed.tensor")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path), size, False), nprocs=size, join=True)


def test_physical_multi_gpu_tensor_distribution_and_gradients(tmp_path):
    torch = pytest.importorskip("torch")
    required = os.environ.get("OPENQML_REQUIRE_CUDA") == "1"
    if torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available():
        if required:
            pytest.fail("Hardware acceptance requires at least two physical CUDA GPUs and NCCL.")
        pytest.skip("Two physical CUDA GPUs and NCCL are required")
    pytest.importorskip("torch.distributed.tensor")
    directory = Path(os.environ.get("OPENQML_HARDWARE_REPORT_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(_worker, args=(str(directory), 2, True), nprocs=2, join=True)
    reports = [json.loads((directory / f"gpu-rank-{rank}.json").read_text()) for rank in range(2)]
    assert len({r["physical_device_uuid"] for r in reports}) == 2
    assert all(r["execution_kind"] == "cuda_nccl" for r in reports)

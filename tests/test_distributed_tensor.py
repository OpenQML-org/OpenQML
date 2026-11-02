"""Distributed numerical contracts. GPU tests never emulate CUDA devices."""

import json
import os
import sys
from pathlib import Path

import pytest

import openqml as q


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
    return q.Interconnect(1e10, 1e11, 1e13, latency)


def test_communication_cost_changes_layouts_and_search_is_bounded():
    c, obs = circuit(), (q.PauliString("XYZXZ"),)
    fast = q.DistributedTensorPlan(c, obs, 4, hardware())
    slow = q.DistributedTensorPlan(c, obs, 4, hardware(10))
    assert fast.to_dict()["distributed_contractions"] > 0
    assert slow.to_dict()["distributed_contractions"] == 0
    assert fast.working_bytes_per_rank < fast.max_memory_mb * 2**20
    assert fast.to_dict()["performance_evidence"] == "planning_estimates_only"
    with pytest.raises(q.ResourceError, match="max_layout_work"):
        q.DistributedTensorPlan(c, obs, 4, hardware(), max_layout_work=1)
    with pytest.raises(q.ResourceError, match="memory"):
        q.DistributedTensorPlan(c, obs, 4, hardware(), max_memory_mb=0.001)
    with pytest.raises(q.ValidationError, match="power-of-two"):
        q.DistributedTensorPlan(c, obs, 3, hardware())


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
        plan = q.DistributedTensorPlan(c, obs, size, hardware(), rematerialize=True)
        assert plan.to_dict()["distributed_contractions"] > 0
        like = torch.empty((), device=device, dtype=torch.float64)
        bindings = {
            "w": torch.tensor(0.41, device=device, dtype=torch.float64, requires_grad=True),
            "x": torch.tensor([0.2, -0.3], device=device, dtype=torch.float64, requires_grad=True),
        }
        expected = q.TensorNetworkPlan(c, obs).evaluate(bindings, backend="torch", like=like)
        actual, report = plan.evaluate(bindings, mesh=mesh, like=like, return_report=True)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        # A replicated loss must give one full gradient on each tensor rank,
        # not a world-size multiple, including across checkpoint recomputation.
        gradient = torch.autograd.grad(
            actual.square().sum(), tuple(bindings.values()), create_graph=True
        )
        reference = torch.autograd.grad(
            expected.square().sum(), tuple(bindings.values()), create_graph=True
        )
        for got, wanted in zip(gradient, reference, strict=True):
            torch.testing.assert_close(got, wanted, atol=2e-11, rtol=2e-11)
        second = torch.autograd.grad(sum(g.sum() for g in gradient), tuple(bindings.values()))
        second_ref = torch.autograd.grad(sum(g.sum() for g in reference), tuple(bindings.values()))
        for got, wanted in zip(second, second_ref, strict=True):
            torch.testing.assert_close(got, wanted, atol=2e-10, rtol=2e-10)
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

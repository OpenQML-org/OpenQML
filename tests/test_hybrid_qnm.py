"""End-to-end hybrid optimization, microbatch invariance and checkpoint replay."""

import copy
import io
from datetime import timedelta

import numpy as np
import pytest

import openqml as oq


def _model():
    import torch

    from openqml.qml.interfaces.torch import QuantumLayer

    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            x, y, a, b, c, d = (oq.Parameter(n) for n in ("x", "y", "a", "b", "c", "d"))
            circuit = oq.Circuit(2).ry(x, 0).ry(y, 1).ry(a, 0).cx(0, 1).rz(b, 1).rx(c, 0).ry(d, 1)
            self.encoder = torch.nn.Linear(2, 2, dtype=torch.float64)
            self.q = QuantumLayer(
                circuit, ("x", "y"), ("a", "b", "c", "d"), dtype=torch.float64, device="cpu"
            )
            self.head = torch.nn.Linear(2, 1, dtype=torch.float64)

        def forward(self, x):
            return self.head(self.q(self.encoder(x)))

    return Model()


def _trainer(model, microbatch, chunk_size=None):
    from openqml.qml.interfaces.torch import HybridTrainer

    optimizer = oq.hybrid_qnm(
        model,
        quantum_lr=0.03,
        block_shapes={"q" if not hasattr(model, "module") else "module.q": ((2, 2),)},
        quantum_options={
            "damping": 0.1,
            "tolerance": 1e-9,
            "max_iterations": 12,
            "sample_chunk_size": chunk_size,
        },
    )
    return HybridTrainer(model, optimizer, microbatch_size=microbatch, max_grad_norm=None)


def _loss(model, batch):
    return (model(batch[0]) - batch[1]).square().mean()


def test_quantum_and_classical_parameters_train_with_microbatch_invariant_geometry():
    torch = pytest.importorskip("torch")
    first, second = _model(), _model()
    a, b = _trainer(first, 2), _trainer(second, 5)
    x = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 7
    y = x.sum(-1, keepdim=True).sin()
    original = [p.detach().clone() for p in first.parameters()]
    for _ in range(3):
        result_a, result_b = a.step((x, y), _loss), b.step((x, y), _loss)
        assert not result_a["skipped"] and not result_b["skipped"]
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-11)
    assert all(not torch.equal(p, old) for p, old in zip(first.parameters(), original, strict=True))
    assert a.optimizer.state[first.q.weights]["openqml_qnm"]["steps"] == 3
    assert not a.optimizer._capture and a.optimizer._captured_bytes == 0
    a.optimizer.close()
    b.optimizer.close()


def test_chunked_quantum_geometry_matches_full_joint_training_and_checkpoint_identity():
    torch = pytest.importorskip("torch")
    first, second = _model(), _model()
    a, b = _trainer(first, 3), _trainer(second, 3, 2)
    x = torch.arange(14, dtype=torch.float64).reshape(7, 2) / 7
    batch = x, x.sum(-1, keepdim=True).sin()
    for _ in range(2):
        a.step(batch, _loss)
        b.step(batch, _loss)
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)
    a.load_state_dict(b.state_dict())
    a.optimizer.close()
    b.optimizer.close()


def test_full_training_checkpoint_replays_quantum_momentum_and_rejects_wrong_identity():
    torch = pytest.importorskip("torch")
    model = _model()
    trainer = _trainer(model, 2)
    x = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 7
    batch = (x, x.sum(-1, keepdim=True).sin())
    trainer.step(batch, _loss)
    stream = io.BytesIO()
    torch.save(trainer.state_dict(), stream)
    trainer.step(batch, _loss)
    expected = [p.detach().clone() for p in model.parameters()]
    restored = _model()
    other = _trainer(restored, 2)
    stream.seek(0)
    state = torch.load(stream, weights_only=False)
    other.load_state_dict(state)
    other.step(batch, _loss)
    for p, q in zip(restored.parameters(), expected, strict=True):
        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-13)
    bad = copy.deepcopy(state)
    bad["optimizer"]["openqml_geometry"]["schema"] = "wrong"
    before = [p.detach().clone() for p in restored.parameters()]
    with pytest.raises(oq.ValidationError):
        other.load_state_dict(bad)
    assert all(torch.equal(p, q) for p, q in zip(restored.parameters(), before, strict=True))
    trainer.optimizer.close()
    other.optimizer.close()


def test_unused_forward_and_zero_learning_rate_do_not_corrupt_context_or_weights():
    torch = pytest.importorskip("torch")
    model = _model()
    trainer = _trainer(model, 5)
    model(torch.ones((7, 2), dtype=torch.float64))
    assert not trainer.optimizer._capture
    for group in trainer.optimizer.param_groups:
        group["lr"] = 0
    x = torch.ones((5, 2), dtype=torch.float64)
    before = [p.detach().clone() for p in model.parameters()]
    trainer.step((x, x[:, :1] * 0.2), _loss)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
    assert trainer.optimizer.state[model.q.weights]["openqml_qnm"]["steps"] == 1
    trainer.optimizer.close()
    with pytest.raises(oq.ValidationError, match="closed"):
        trainer.optimizer.step()


def _worker(rank, path, output):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo", init_method="file://" + path, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        model = _model()
        ddp = DistributedDataParallel(model)
        trainer = _trainer(ddp, 1)
        x = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 7
        y = x.sum(-1, keepdim=True).sin()
        indices = slice(0, 2) if rank == 0 else slice(2, 5)
        for _ in range(2):
            trainer.step((x[indices], y[indices]), _loss)
        if rank == 0:
            torch.save(model.state_dict(), output)
        trainer.optimizer.close()
    finally:
        dist.destroy_process_group()


def test_distributed_hybrid_qnm_global_metric_matches_unequal_rank_reference(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Distributed Torch is unavailable.")
    import torch.multiprocessing as mp

    output = str(tmp_path / "rank-zero.pt")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), output), nprocs=2, join=True)
    model = _model()
    trainer = _trainer(model, 5)
    x = torch.arange(10, dtype=torch.float64).reshape(5, 2) / 7
    for _ in range(2):
        trainer.step((x, x.sum(-1, keepdim=True).sin()), _loss)
    expected = torch.load(output, weights_only=False)
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[name], rtol=1e-8, atol=2e-10)
    trainer.optimizer.close()


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_scenario_execution_objective_trains_probabilities_and_proposals(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = oq.finance.ScenarioPlan([[-0.12, -0.03], [0.04, -0.08], [0.1, 0.02], [-0.05, 0.01]])
    value = xp.asarray([0.3, -0.4, 0.2, 0.1, 0.4, 0.5], dtype=dtype)

    def objective(v):
        p = xp.exp(v[:4])
        p = p / p.sum()
        result = plan.execution_loss(
            p,
            v[4:],
            xp.asarray([0.02, -0.01], dtype=dtype),
            previous=[0.3, 0.2],
            budget=0.02,
            confidence=0.7,
            radius=0.001,
            impact=0.01,
        )
        return result.loss

    if backend == "torch":
        value.requires_grad_()
        assert lib.autograd.gradcheck(objective, (value,))
        gradient = lib.autograd.grad(objective(value), value)[0].detach().numpy()
    else:
        gradient = np.asarray(lib.jit(lib.grad(objective))(value))
    assert np.isfinite(gradient).all() and (np.abs(gradient) > 1e-8).all()

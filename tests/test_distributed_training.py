import os
import subprocess
import sys
from pathlib import Path

import pytest


def _ddp_worker(rank, directory):
    import torch
    import torch.distributed as dist

    import openqml as oq

    torch.set_num_threads(1)
    torch.manual_seed(43 + rank)
    dist.init_process_group(
        "gloo", init_method=(Path(directory) / "init").as_uri(), rank=rank, world_size=2
    )
    try:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.fill_(0.25)
        model = torch.nn.parallel.DistributedDataParallel(model)
        trainer = oq.torch.HybridTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            microbatch_size=2,
            max_grad_norm=None,
        )
        x = (
            torch.arange(1, 4, dtype=torch.float64)
            if rank == 0
            else torch.arange(4, 9, dtype=torch.float64)
        )
        x = x[:, None]
        target = 2 * x
        result = trainer.step(
            (x, target), lambda network, batch: (network(batch[0]) - batch[1]).square().mean()
        )
        torch.testing.assert_close(
            model.module.weight,
            torch.tensor([[1.1425]], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        assert result["samples"] == 8 and not result["skipped"]
        before = model.module.weight.detach().clone()
        bad = trainer.step(
            (x, target),
            lambda network, batch: (
                network(batch[0]).square().mean() * (float("inf") if rank == 0 else 1)
            ),
        )
        assert bad["skipped"]
        torch.testing.assert_close(model.module.weight, before, atol=0, rtol=0)
        trainer.save(Path(directory) / "ddp.pt")
        expected_random = torch.rand(5)
        dist.barrier()
        trainer.load(Path(directory) / "ddp.pt")
        assert trainer.updates == 1
        torch.testing.assert_close(torch.rand(5), expected_random, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CPU distributed integration")
def test_ddp_unequal_microbatches_are_globally_sample_weighted(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo distributed backend is unavailable")
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux virtual CPU mesh integration")
def test_jax_two_device_sharding_and_psum():
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    script = """
import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
import openqml as oq
assert len(jax.devices())==2
mesh=Mesh(np.asarray(jax.devices()),('batch',))
x=jnp.arange(16,dtype=jnp.float32).reshape(8,2)/10
y=jnp.arange(8,dtype=jnp.float32)/8
parameters={'w':jnp.array([.2,-.3])}
def loss(p,b):return (b['x']@p['w']-b['y'])**2
gradient=jax.grad(lambda p:loss(p,{'x':x,'y':y}).mean())(parameters)
expected=parameters['w']-.1*gradient['w']
optimizer=optax.sgd(.1)
step=oq.jax.make_train_step(optimizer,loss,microbatch_size=3,max_grad_norm=None)
p=jax.device_put(parameters,NamedSharding(mesh,P()))
b=jax.device_put({'x':x,'y':y},NamedSharding(mesh,P('batch')))
actual,_,metrics=step(p,optimizer.init(p),b)
np.testing.assert_allclose(actual['w'],expected,rtol=2e-6)
assert int(metrics['samples'])==8
local=oq.jax.make_train_step(optimizer,loss,microbatch_size=3,max_grad_norm=None,axis_name='batch',jit=False)
try:
    mapper=jax.shard_map
except AttributeError:
    from jax.experimental.shard_map import shard_map as mapper
parallel=jax.jit(mapper(local,mesh=mesh,in_specs=(P(),P(),P('batch')),out_specs=(P(),P(),P())))
actual,_,metrics=parallel(p,optimizer.init(p),b)
np.testing.assert_allclose(actual['w'],expected,rtol=2e-6)
assert int(metrics['samples'])==8
print('Two-device NamedSharding and sample-weighted psum passed.')
"""
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

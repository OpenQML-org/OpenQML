"""Distribution boundaries of the supported public APIs."""

import importlib.util

import pytest

import openqml as oq


def test_retired_modules_and_exports_stay_out_of_the_public_namespace():
    modules = """rl rl_training quantum_tree adaptive_mps truncated_mps tensor_contraction
        tensor_train distributed_tensor mps network_attention network_graph network_operators
        network_training""".split()
    assert set(modules).isdisjoint(dir(oq))
    installed = {
        name for name in modules if importlib.util.find_spec(f"openqml.{name}") is not None
    }
    assert not installed
    exports = """PPOConfig ppo_loss sac_loss ActorCriticOutput TensorNetworkPlan
        DistributedTensorPlan Interconnect QuantumTreePlan TensorTrainMatrix TensorTrainPlan
        MPSBackend AdaptiveMPSBackend TruncatedMPSBackend""".split()
    assert set(exports).isdisjoint(oq.__all__)
    for name in exports:
        with pytest.raises(AttributeError, match=name):
            getattr(oq, name)

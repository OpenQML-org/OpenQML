"""Public distribution boundaries for the three supported application areas."""

import importlib.util

import pytest

import openqml as oq


@pytest.mark.parametrize(
    "name",
    [
        "rl",
        "rl_training",
        "quantum_tree",
        "adaptive_mps",
        "truncated_mps",
        "tensor_contraction",
        "tensor_train",
        "distributed_tensor",
        "mps",
        "network_attention",
        "network_graph",
        "network_operators",
        "network_training",
    ],
)
def test_retired_product_modules_are_not_shipped(name):
    assert importlib.util.find_spec(f"openqml.{name}") is None
    assert name not in dir(oq)


@pytest.mark.parametrize(
    "name",
    [
        "PPOConfig",
        "ppo_loss",
        "sac_loss",
        "ActorCriticOutput",
        "TensorNetworkPlan",
        "DistributedTensorPlan",
        "Interconnect",
        "QuantumTreePlan",
        "TensorTrainMatrix",
        "TensorTrainPlan",
        "MPSBackend",
        "AdaptiveMPSBackend",
        "TruncatedMPSBackend",
    ],
)
def test_retired_root_exports_fail_explicitly(name):
    assert name not in oq.__all__
    with pytest.raises(AttributeError, match=name):
        getattr(oq, name)


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_training_facades_only_own_supported_model_extensions(framework):
    pytest.importorskip(framework)
    facade = getattr(oq, framework)
    assert callable(facade.QuantumLayer)
    assert callable(facade.HybridNetwork)
    assert not {
        "PPOTrainer",
        "QuantumActorCritic",
        "QuantumSurrogate",
        "TreeQuantumLayer",
        "TensorNetworkAdapter",
        "TensorTrainLinear",
    }.intersection(vars(facade))

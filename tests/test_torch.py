import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openqml import Circuit, Parameter, PauliString, expectation, parameter_shift
from openqml.interfaces.torch import QuantumLayer


def test_native_gradients_shared_symbols_and_second_derivative():
    t = Parameter("theta")
    circuit, ob = Circuit(1).ry(t, 0).ry(2 * t + 0.1, 0), PauliString("Z")
    theta = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    value = expectation(circuit, ob, {"theta": theta}, backend="torch")
    (first,) = torch.autograd.grad(value, theta, create_graph=True)
    (second,) = torch.autograd.grad(first, theta)
    assert first.item() == pytest.approx(-3 * np.sin(3 * theta.item() + 0.1), abs=1e-12)
    assert second.item() == pytest.approx(-9 * np.cos(3 * theta.item() + 0.1), abs=1e-12)
    assert first.item() == pytest.approx(
        parameter_shift(circuit, ob, {"theta": theta.item()})["theta"]
    )


def test_layer_input_and_weight_gradcheck_and_gradgradcheck():
    torch.manual_seed(13)
    layer = QuantumLayer.hardware_efficient(2, depth=1, dtype=torch.float64)
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    weights = layer.weights.detach().clone().requires_grad_(True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (x, weights), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradgradcheck(function, (x, weights), eps=1e-6, atol=1e-5)


def test_batch_shape_dtype_optimizer_and_state_dict():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2), QuantumLayer.hardware_efficient(2, 1), torch.nn.Linear(2, 1)
    )
    x = torch.rand(12, 2)
    target = x[:, :1] - 0.4 * x[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.25 * losses[0]
    layer = model[1]
    assert layer(torch.zeros(3, 4, 2)).shape == (3, 4, 2)
    assert layer(torch.zeros(0, 2)).shape == (0, 2)
    copy = QuantumLayer.hardware_efficient(2, 1)
    copy.load_state_dict(layer.state_dict())
    torch.testing.assert_close(copy(x), layer(x))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_constant_and_weight_only_layers():
    constant = QuantumLayer(Circuit(1).h(0), (), ())
    assert constant(torch.empty(4, 0)).shape == (4, 1)
    assert constant(torch.empty(0)).shape == (1,)
    layer = QuantumLayer(Circuit(1).ry(Parameter("w"), 0), (), ("w",))
    layer(torch.empty(3, 0)).sum().backward()
    assert layer.weights.grad is not None


def test_extended_gate_layer_has_correct_input_and_weight_autodiff():
    x, a, b = Parameter("x"), Parameter("a"), Parameter("b")
    circuit = (
        Circuit(3)
        .h(0)
        .u(x, a, b, 1)
        .crx(a + 2 * x, 0, 2)
        .cry(b, 1, 0)
        .crz(x - b, 2, 1)
        .cp(a, 1, 2)
        .ch(2, 0)
        .iswap(1, 2)
        .cswap(0, 2, 1)
        .sxdg(2)
    )
    layer = QuantumLayer(circuit, ("x",), ("a", "b"), dtype=torch.float64)
    inputs = torch.tensor([[0.43]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([-0.27, 0.61], dtype=torch.float64, requires_grad=True)

    def function(inputs, weights):
        return torch.func.functional_call(layer, {"weights": weights}, (inputs,))

    assert torch.autograd.gradcheck(function, (inputs, weights), atol=1e-5)
    assert torch.autograd.gradgradcheck(function, (inputs, weights), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_stays_on_device():
    layer = QuantumLayer.hardware_efficient(2, depth=1, device="cuda", dtype=torch.float64)
    x = torch.randn(3, 2, device="cuda", dtype=torch.float64, requires_grad=True)
    output = layer(x)
    output.sum().backward()
    assert output.is_cuda and layer.weights.grad.is_cuda and x.grad.is_cuda

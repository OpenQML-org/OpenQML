"""Cold-cache native function transforms must not retain transformed constants."""

import numpy as np
import pytest

from openqml import Circuit, ExpectationPlan, Parameter, PauliString, expectations


@pytest.mark.parametrize("engine", ["statevector", "density", "planned"])
def test_repeated_cold_cache_hessians_and_eager_gradients_never_escape_transform(engine):
    torch = pytest.importorskip("torch")
    from openqml.simulator import _torch_cached_pauli_plan

    _torch_cached_pauli_plan.cache_clear()
    c, ob, amplitude = Circuit(1).ry(Parameter("x"), 0), PauliString("X"), 1
    if engine == "density":
        c, amplitude = c.phase_damping(0.2, 0), np.sqrt(0.8)
    elif engine == "planned":
        c, ob = Circuit(2).ry(Parameter("x"), 0).unitary(np.eye(4), 0, 1), PauliString("XI")
    plan = ExpectationPlan(c, (ob,)) if engine == "planned" else None

    def function(value):
        bindings = {"x": value}
        values = (
            plan.evaluate(bindings, backend="torch", like=value)
            if plan is not None
            else expectations(c, (ob,), bindings, backend="torch", like=value)
        )
        return values.sum()

    for angle in (0.3, -0.4, 0.0):
        value = torch.tensor(angle, dtype=torch.float64)
        hessian = torch.func.hessian(function)(value)
        torch.testing.assert_close(hessian, -amplitude * value.sin())
    assert _torch_cached_pauli_plan.cache_info().currsize == 0
    value = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    function(value).backward()
    torch.testing.assert_close(value.grad, amplitude * value.cos())
    assert _torch_cached_pauli_plan.cache_info().currsize == 1
    grid = torch.linspace(-0.3, 0.4, 4, dtype=torch.float64)
    torch.testing.assert_close(
        torch.func.vmap(torch.func.grad(function))(grid), amplitude * grid.cos()
    )
    for _ in range(2):
        torch.testing.assert_close(torch.func.hessian(function)(value), -amplitude * value.sin())
        torch.testing.assert_close(function(value), amplitude * value.sin())
    assert _torch_cached_pauli_plan.cache_info().currsize == 1


def test_ordinary_device_readout_plans_still_reuse_cached_constants():
    torch = pytest.importorskip("torch")
    from openqml.simulator import _torch_cached_pauli_plan, _torch_pauli_plan

    _torch_cached_pauli_plan.cache_clear()
    first = _torch_pauli_plan("XY", torch.device("cpu"), torch.complex128)
    second = _torch_pauli_plan("XY", torch.device("cpu"), torch.complex128)
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert _torch_cached_pauli_plan.cache_info().hits == 1

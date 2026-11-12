"""Native-engine configuration, routing and execution contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize(
    "options",
    [
        {"device": []},
        {"device": "GPU", "method": "matrix_product_state"},
        {"cu_statevec": True},
        {"device": "GPU", "cu_statevec": True, "batched_shots_gpu": True},
        {"device": "GPU", "target_gpus": [0, 0]},
        {"device": "GPU", "target_gpus": [-1]},
        {"device": "GPU", "precision": "half"},
        {"device": "GPU", "blocking_qubits": 0},
    ],
)
def test_invalid_native_gpu_options_fail_before_startup(options):
    with pytest.raises(oq.ValidationError):
        oq.AerBackend(**options)


def test_native_gpu_options_reach_the_cpp_engine_without_cpu_fallback(monkeypatch):
    from openqml.interop.adapters import _common

    configurations = []

    class Simulator:
        devices = ("CPU", "GPU")

        def __init__(self, **options):
            configurations.append(options)

        def available_devices(self):
            return self.devices

        def available_methods(self):
            return ("statevector", "tensor_network", "density_matrix")

    monkeypatch.setattr(
        _common, "dependency", lambda *args: SimpleNamespace(AerSimulator=Simulator)
    )
    backend = oq.AerBackend(
        device="GPU", precision="single", cu_statevec=True, target_gpus=(0, 1), blocking_qubits=8
    )
    assert backend.capabilities.device_type == "GPU" and backend.is_available()
    backend._engine(oq.ExecutionOptions(max_memory_mb=32))
    options = configurations[-1]
    assert options["device"] == "GPU" and options["cuStateVec_enable"]
    assert options["fusion_enable"] and options["precision"] == "single"
    assert options["target_gpus"] == [0, 1]
    assert options["blocking_enable"] and options["blocking_qubits"] == 8
    Simulator.devices = ("CPU",)
    assert not backend.is_available()
    with pytest.raises(oq.ResourceError, match="cannot execute"):
        backend._engine(oq.ExecutionOptions())


def test_native_cpu_readouts_and_parameter_binding_remain_numerically_correct():
    pytest.importorskip("qiskit_aer")
    x = oq.Parameter("x")
    circuit = oq.Circuit(2).ry(x, 0).cx(0, 1).rz(x / 3, 1)
    rows = [{"x": 0.2}, {"x": -0.7}]
    obs = [oq.PauliString("ZI"), oq.PauliString("XX")]
    engine = oq.AerBackend(runtime_parameter_binding=True)
    np.testing.assert_allclose(
        oq.expectations_batch(circuit, obs, rows, backend=engine),
        oq.expectations_batch(circuit, obs, rows, backend="numpy"),
        atol=2e-12,
    )


def test_native_gpu_builtin_is_lazy_and_classified_for_gpu_scheduling(monkeypatch):
    from openqml.runtime import backends

    monkeypatch.setattr(backends, "_REGISTRY", dict(backends._REGISTRY))
    backend = oq.get_backend("aer-gpu")
    assert isinstance(backend, oq.AerBackend)
    assert backend.device == backend.capabilities.device_type == "GPU"

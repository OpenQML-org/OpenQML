"""Native-engine configuration, routing and execution contracts."""

import numpy as np
import pytest

import openqml as oq


def test_engine_provenance_is_configuration_without_runtime_or_hardware_probes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Execution provenance must not probe an engine or device.")

    monkeypatch.setattr(oq.AerBackend, "is_available", forbidden)
    monkeypatch.setattr(oq.CUDABackend, "is_available", forbidden)
    assert oq.AerBackend().execution_info()["device"] == "CPU"
    native = oq.CUDABackend(compile_mode="default").execution_info()
    assert native["algorithm_owner"] == "OpenQML" and native["local_gate_kernel"] == "openqml"
    assert native["compiler"] == "torchinductor" and not native["execution_verified"]
    assert oq.CUDABackend().execution_info()["compiler"] is None


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
    backend = oq.get_backend("gpu")
    assert isinstance(backend, oq.CUDABackend)
    assert backend.capabilities.device_type == "GPU"

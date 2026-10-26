"""Caller-supplied local Qiskit V2 primitives, ISA layout mapping and noisy readouts."""

import cirq
import numpy as np
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

from openqml import (
    Circuit,
    CirqDensityBackend,
    Parameter,
    PauliString,
    QiskitEstimatorBackend,
    QiskitSamplerBackend,
    estimate,
    expectations_batch,
    parameter_shift_vjp,
)

circuit = Circuit(1).ry(Parameter("theta"), 0)
rows = [{"theta": 0.41}, {"theta": -0.63}]
observables = (PauliString("X"), PauliString("Z"))
estimator = QiskitEstimatorBackend(StatevectorEstimator(seed=19))
print("EstimatorV2:", expectations_batch(circuit, observables, rows, backend=estimator))
sampler = QiskitSamplerBackend(StatevectorSampler(seed=19))
report = estimate(circuit, observables, rows, backend=sampler, shots=2000, seed=19)
print("SamplerV2 estimates:", report.values)
print("SamplerV2 standard errors:", report.standard_errors)

noisy = CirqDensityBackend(noise=cirq.amplitude_damp(0.17))
values = expectations_batch(circuit, observables, rows, backend=noisy)
gradient = parameter_shift_vjp(circuit, observables, rows, np.ones((2, 2)), backend=noisy)
print("Exact mixed-state values and reverse-mode gradients:", values, gradient)
np.testing.assert_allclose(values[:, 1], 0.83 * np.cos([0.41, -0.63]) + 0.17, atol=1e-12)

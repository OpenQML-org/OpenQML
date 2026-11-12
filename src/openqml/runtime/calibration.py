"""Layout-aware logarithmic success estimates for scheduling compiled circuits.

This is a ranking heuristic based on independent operation errors and idle-time
coherence. It is not a diamond-norm certificate; correlated noise, leakage and
model double-counting require device-specific measured corrections.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType

from ..errors import ValidationError
from ..ir.core import Circuit, integer
from ..ir.observables import Hamiltonian
from ..ir.parameters import real
from .costs import ExecutionProfile


@dataclass(frozen=True)
class GateCalibration:
    error: float
    seconds: float

    def __post_init__(self):
        if not 0 <= real(self.error) <= 1 or real(self.seconds) < 0:
            raise ValidationError("Gate errors must lie in [0,1] and durations be nonnegative.")


@dataclass(frozen=True)
class NoiseEstimate:
    success_score: float
    shot_seconds: float
    normalized_bias_estimate: float
    model: str = "independent_log_success_with_coherence; not_a_noise_certificate"

    def profile(self, *, queue_seconds, cost_per_second, **kwargs):
        """Include explicit queue/pricing and all measurement/mitigation overheads."""
        return ExecutionProfile(
            circuit_seconds=0,
            shot_seconds=self.shot_seconds,
            noise_bias=self.normalized_bias_estimate,
            queue_seconds=queue_seconds,
            cost_per_second=cost_per_second,
            source=self.model,
            **kwargs,
        )


@dataclass(frozen=True)
class CalibrationModel:
    """Physical gate/layout calibration with optional T1/T2 coherence times.

    gates maps (gate_name, physical_wires) to GateCalibration. readout maps wire
    to GateCalibration; coherence maps wire to (T1_seconds, T2_seconds). Pass a
    compiled/routed IR circuit, not an unrouted logical circuit. Missing used
    calibrations fail rather than being assumed noiseless.
    """

    gates: object
    readout: object
    coherence: object = None

    def __post_init__(self):
        try:
            gates, readout, coherence = (
                dict(self.gates),
                dict(self.readout),
                dict(self.coherence or {}),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("Calibration data must be mappings.") from exc
        for key, value in gates.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[0], str)
                or not isinstance(key[1], tuple)
                or not key[1]
                or not isinstance(value, GateCalibration)
            ):
                raise ValidationError("gates requires (name, wire_tuple): GateCalibration entries.")
            for wire in key[1]:
                integer(wire, "physical wire", 0)
        for wire, value in readout.items():
            integer(wire, "readout wire", 0)
            if not isinstance(value, GateCalibration):
                raise ValidationError("readout requires GateCalibration values.")
        for wire, times in coherence.items():
            integer(wire, "coherence wire", 0)
            if len(times) != 2 or any(real(t) <= 0 for t in times):
                raise ValidationError("Coherence requires positive (T1,T2) seconds.")
            coherence[wire] = tuple(times)
        for name, value in (("gates", gates), ("readout", readout), ("coherence", coherence)):
            object.__setattr__(self, name, MappingProxyType(value))

    def estimate(self, circuit, observables):
        if not isinstance(circuit, Circuit) or circuit.has_effects:
            raise ValidationError(
                "Noise estimation requires a statically scheduled unitary IR circuit."
            )
        measured = set()
        for observable in observables:
            terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            for term in terms:
                if len(term.paulis) != circuit.num_qubits:
                    raise ValidationError("Observable and physical circuit widths differ.")
                if term.coefficient:
                    measured.update(i for i, p in enumerate(term.paulis) if p != "I")
        finish, log_success = {}, 0.0

        def add_error(error):
            nonlocal log_success
            log_success += math.log1p(-error) if error < 1 else -math.inf

        for op in circuit.operations:
            if op.name in {"barrier", "id"}:
                if op.name == "barrier" and op.qubits:
                    end = max((finish.get(q, 0) for q in op.qubits), default=0)
                    finish.update(dict.fromkeys(op.qubits, end))
                continue
            try:
                calibration = self.gates[(op.name, op.qubits)]
            except KeyError as exc:
                raise ValidationError(
                    f"Missing physical gate calibration for {(op.name, op.qubits)}."
                ) from exc
            start = max((finish.get(q, 0) for q in op.qubits), default=0)
            finish.update(dict.fromkeys(op.qubits, start + calibration.seconds))
            add_error(calibration.error)
        circuit_end = max(finish.values(), default=0)
        for wire in measured:
            if wire not in self.readout:
                raise ValidationError(f"Missing readout calibration for physical wire {wire}.")
            calibration = self.readout[wire]
            add_error(calibration.error)
            finish[wire] = circuit_end + calibration.seconds
        for wire, duration in finish.items():
            if wire in self.coherence:
                t1, t2 = self.coherence[wire]
                log_success -= duration * (1 / (6 * t1) + 1 / (3 * t2))
        success = math.exp(log_success)
        return NoiseEstimate(success, max(finish.values(), default=0), 2 * (1 - success))

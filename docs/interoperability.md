# Cross-framework quantum programming

OpenQML imports Qiskit, Cirq, PennyLane and OpenQASM circuits into a shared IR.
Conversion plans separate source import, target preparation and artifact export.
JSON and IR targets work with the core installation. Install the corresponding
extra when importing or exporting a native SDK object.

## Prepare once, bind many

```python
import openqml as oq

angle = oq.Parameter("angle")
circuit = oq.Circuit(2).h(0).ry(angle, 1).cx(0, 1)
plan = oq.plan_conversion(circuit, target="openqasm3")

for artifact in plan.iter_exports([{"angle": 0.1}, {"angle": 0.4}]):
    print(artifact.value)
```

`iter_exports` imports the original source once, retains input order and yields
one independent artifact at a time. `export_many` collects the results into a
tuple. Both accept `max_bindings` (default 1,024) to bound sweep length.
Streaming may yield earlier valid rows before a later row raises an error.
Each row must bind every remaining source parameter to a finite real scalar.

Compatible symbolic plans reuse their target-prepared topology. Each binding
receives a fresh target compatibility check. Targets requiring concrete angles,
such as OpenQASM 2 and PennyLane tapes, are prepared from the bound source IR.
SDK exporters may perform their own final synthesis or validation.

## Partial binding and parameter ownership

```python
import openqml as oq

a, b = oq.Parameter("input"), oq.Parameter("second angle")
circuit = oq.Circuit(1).ry(a + b, 0)
plan = oq.plan_conversion(circuit, target="openqasm3")
part = plan.bind({"input": 0.2}, partial=True)
result = part.bind({"second angle": 0.3}).export()
print(result.value)
```

Bindings always use the source names. OpenQASM aliases remain an internal export
mapping, available through `parameter_map`. Binding returns a new immutable plan;
the original circuit and plan retain their symbolic parameters.

## Compatibility and verification

`plan.compatible`, `plan.diagnostics` and `plan.to_dict()` expose representability,
resource estimates and required integrations before target SDK loading. Explicit
phase, classical storage, measurement, feedback and channels are checked against
each target's circuit semantics. An incompatible export raises `ConversionError`
with structured locations and hints. Provider calibration and execution context
are configured separately on the runtime.

Use `plan.verify(bindings, mode="roundtrip")` to compare export/reimport maps at
a binding. `mode="native"` executes supported native artifacts, and
`plan.certify()` applies bounded exact algebraic checks. See
[semantic kernels](semantic-kernels.md) and [execution](training-and-execution.md)
for the supported verification domains and resource controls.

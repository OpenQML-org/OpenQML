# Cross-framework quantum programming

OpenQML imports Qiskit, Cirq, PennyLane and OpenQASM circuits into a shared IR.
Conversion plans separate source import, target preparation and artifact export.
JSON and IR targets work with the core installation. Install the corresponding
extra when importing or exporting a native SDK object.

## Dynamic control and compilation reuse

| Source feature | Lowering and semantic contract |
| --- | --- |
| Qiskit `if_else` | Atomic equality selection over explicit classical bits |
| Qiskit `switch` | Bit/register selectors, grouped integer labels and a default branch |
| Qiskit finite `for` | Bounded static expansion with explicit parameter binding |
| Repeated Qiskit definitions | One imported body per source object and option set, with fresh wire remapping |
| Qiskit symbolic angles | Per-import expression reuse and iterative DAG export |
| OpenQASM 3 guards | Bit/register/index-set equality and conjunctions, preserving arbitrary bit order |

Switch lowering uses a nested equality decision chain. After a case is selected,
its body runs once; writes to the selector cannot enter another case. Grouped
labels share the imported body. The default body is reached only after every
label fails. Source operation, loop-expansion and control-depth budgets apply
to the resulting decision structure. Switch expression selectors receive a
structured diagnostic unless they directly wrap a bit or register.

OpenQASM 3 emission uses one register equality or ordered index-set equality,
such as `if (c[{2, 0}] == 2)`, with one copy of each branch body. These compact
guards also load through Qiskit's OpenQASM 3 importer. Text size grows with the
represented branch tree and guard width. The built-in parser also accepts
equality conjunctions with nested grouping, combines consistent repeated bits
and diagnoses contradictory terms. Conditions
are evaluated before either body executes. `max_output_statements` admits the
complete output before emission, including declarations, phases and braces.
Sparse guards inspect only their selected bits, including circuits with large
classical register declarations. Boolean Qiskit switch labels map to the
corresponding integer guard values.

`max_output_bytes` (default 2,000,000) bounds UTF-8 OpenQASM and JSON output.
Planning, binding, sweeps and the CLI's `--max-output-bytes` retain the same
limit. `Circuit.to_json(max_bytes=...)` also admits expanded expression trees
before constructing dictionaries. Serialized expressions use a 4,096-node and
64-level budget; QASM angles additionally fit the parser's 16,384-character
and AST budgets. Native evaluation retains shared expression DAGs.

Affine leaves and negative power bases keep their arithmetic grouping during
QASM emission. Shared symbolic imports use a memoized iterative traversal.
Affine sums use compensated aggregation, and nonlinear reductions use balanced
trees; repeated subexpressions are converted once per call.

Definition and expression caches live within one conversion call. They preserve
symbolic parameter identity, global phase, wire mappings and operation budgets;
subsequent calls import the current source object afresh. Numerical bindings and
target objects remain independent across exported artifacts.

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
[compute policies](compute.md) and [runtime dispatch](runtime.md)
for the supported verification domains and resource controls.

Directly constructed parameter-free expression nodes are evaluated at conversion
boundaries, including constant children of symbolic expressions. Immutable node
metadata identifies these subgraphs without rescanning every expression DAG.
The prepared circuit retains source parameter names and global phase, while the
original input circuit remains unchanged. Direct SDK exporters use the same
constant-folding boundary as conversion plans and compatibility diagnostics.

# Cross-framework quantum programming

OpenQML imports Qiskit, Cirq, PennyLane and OpenQASM circuits into a shared IR.
Use `convert` to import, check compatibility and export in one call.
JSON and IR targets work with the core installation. Install the corresponding
extra when importing or exporting a native SDK object.

## Classical variables and assignments

IR v5 adds `Circuit.classical_set(bit, 0|1)`, a classical write with no quantum
operand. Readers retain support for IR versions 1–4. `has_measurements` still
means physical measurement; `has_classical_outputs` also includes assignments.
NumPy counts, native density execution and conversion verification retain
overwritten final storage and merge branches after dead writes.

Qiskit Bool/Uint variables, bit/register assignments, statically addressed
indexed targets and finite loops lower to classical bits and bounded decision
graphs. Every RHS is resolved into a shared snapshot before any destination
bit changes, including overlapping arithmetic assignments. Runtime variables
are checked for definite assignment. A conservative check can reject a variable
initialized only in a loop; initialize it before entering that loop.
Runtime while loops and dynamically addressed assignment targets require
native execution and receive explicit diagnostics.

Original SDK classical bits retain their indices. Standalone variables,
snapshots and finite-loop control flags use additional classical bits, with
two reusable flow flags per active loop depth. No extra quantum wires are
allocated. `CLASSICAL_STORAGE_LOWERED` reports the added storage; use ordered
`classical_bits` to select original outputs when sampling. Input variables
require `source_options={"classical_inputs": {name: integer}}`; unbound inputs,
out-of-range values and undeclared input names are rejected.

Qiskit, JSON, IR and OpenQASM 3 preserve lowered stores. The built-in QASM 3
reader accepts constant assignments to bit/register targets. Rich expressions
enter through the typed Qiskit frontend and export as portable guards/stores.
Cirq, operation-only PennyLane tapes and QASM 2 reject unsupported assignments.
The target hardware must independently support the exported operations.

Assignment-by-copy follows the
[OpenQASM specification](https://openqasm.com/language/classical.html).
The separation of control storage and data follows compiler practice also
discussed in [QuCtrl-BELL, arXiv:2605.22433](https://arxiv.org/abs/2605.22433).
This lowering does not target its real-time controller or promise its timing.

## Convert a circuit

```python
import openqml as oq

circuit = oq.Circuit(2).h(0).cx(0, 1)
result = oq.convert(circuit, target="openqasm3")
print(result.value)
```

`result.value` is the requested native object or text. `target="qiskit"`,
`"cirq"` and `"pennylane"` require their SDKs; `"openqasm2"`, `"openqasm3"`,
`"json"` and `"ir"` use the core. Native circuit objects can be passed directly.
Pass `bindings={"angle": 0.2}` to bind parameters during conversion. Unsupported
semantics raise `ConversionError` with structured diagnostics.

`plan_conversion` is the advanced entry point for reusing one imported source
across parameter sweeps, partial binding and verification.

## Dynamic control and compilation reuse

| Source feature | Lowering and semantic contract |
| --- | --- |
| Qiskit `if_else` | Atomic equality selection over explicit classical bits |
| Qiskit `switch` | Bit/register selectors, grouped integer labels and a default branch |
| Qiskit finite `for` | Bounded static expansion with explicit parameter binding |
| Qiskit `break_loop` / `continue_loop` | Continuations preserve the nearest finite loop's exit or next iteration |
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

## Deferred quantum modifiers

The Qiskit frontend expands `AnnotatedOperation` with ordered inverse,
control and numeric power modifiers. Nested modifiers retain operand order;
open controls and controlled global phases enter the resulting gate semantics.
Integer powers use bounded repetition and inverse operations. Fractional
powers use the SDK's concrete gate operation under the matrix-width admission.
Symbolic rotation parameters remain symbolic when that operation supports it.

`max_operations` and `max_depth` bound expansion, including large powers and
control widths. `source_options={"max_matrix_qubits": 5}` sets the matrix limit.
Unresolved or nonunitary modifiers raise structured conversion diagnostics.
Normalized definitions are reused within one conversion. The source SDK may
allocate its own synthesis workspace. Export then follows the existing IR
routes to OpenQASM, Cirq, Qiskit or PennyLane within each target's supported set.

The source representation follows
[Qiskit's AnnotatedOperation API](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.AnnotatedOperation).

## Typed classical predicates

The Qiskit IR route additionally accepts pure `Bool`/`Uint` expressions over
declared classical bits/registers: `logic_not`, `logic_and`, `logic_or`,
`bit_not`, `bit_and`, `bit_or`, `bit_xor`, equality/inequality and all unsigned
ordered comparisons. Bool/Uint casts implement nonzero truth conversion,
unsigned truncation and zero extension. Uint addition, subtraction and
multiplication use the expression's declared width, including modulo `2**width`
overflow. Logical shifts and bit indexing accept constant selectors or dynamic
selectors whose full bit-vector range is provably inside the operand width.
Barrel-select networks lower dynamic selectors without enumerating valuations.
Out-of-range or unproven selectors retain range diagnostics.

Unsigned division lowers to bounded restoring bit-vector division. The compiler
must symbolically prove the divisor nonzero for **every** classical input;
otherwise it raises `CLASSICAL_DIVISOR_DOMAIN`. Nonconstant divisors such as a
widened two-bit register plus one are admitted when the proof succeeds.
Division by a constant power of two simplifies to a logical shift. General
division charges quadratic bit work before expansion. No undefined zero-divisor
behavior is chosen and no logical short-circuit evaluation is assumed.

Selector bounds are proved on the full symbolic bit vector, retaining
correlations between bits. A selector whose only possible values are 1 and 2
can therefore index a three-bit word even when independent bit bounds would
allow 3. Predicate decisions still complete before either selected payload.

Standalone mutable variables, signed/float arithmetic, stores, timing
and other unsupported expression operations retain
explicit diagnostics or require the existing native program route.

A bit-vector compiler lowers these predicates into reduced ordered binary
decision diagrams, reusing canonical nodes and cached Boolean operations.
Portable IR emits immutable equality decisions with shared branch bodies.
The predicate is fully evaluated before its selected body executes, including
when the body overwrites a measurement bit used by the predicate. No scratch
classical bits or truth-table enumeration are introduced. The emitted equality
subset remains compatible with OpenQASM and Cirq export/reimport paths.

`max_operations` bounds expression/bit-vector work, BDD construction and the
expanded decision/payload cost; `max_depth` bounds expression, guard and payload
nesting. Ripple-carry arithmetic, truncated multiplication and selector networks
charge their Boolean work before expansion, including constant expressions.
Variable order follows declared bit indices. Some Boolean functions
have large decision or expanded output representations and are rejected at the
budget boundary. This is an expanded representable subset, not universal host
Python or general classical-program translation.

The supported source nodes follow [Qiskit's typed classical-expression API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_classical).
Recent work on [structured quantum programs (September 2026)](https://arxiv.org/abs/2609.16171)
motivates retaining classical structure. The existing native program route
continues to retain supported dynamic loops instead of flattening their runtime
iterations; this addition handles bounded pure predicates in the portable IR.

## Finite loops with early exits

Finite Qiskit loops can contain nested `break_loop` and `continue_loop` nodes,
including exits under typed predicates and grouped switch cases. Continuation
lowering attaches each exit to its nearest lexical loop. A branch resolves its
selector before the selected body can overwrite that selector. Loop parameters,
wire maps, global phases and operations following the whole loop are preserved.

The portable output uses existing atomic IR branches, with no mutable scratch
bits. `max_operations` counts expanded branch payloads, including repeated
continuations; `max_depth` bounds the resulting control nesting. Oversized
programs fail before constructing their expanded output. Runtime while loops
continue to require native program execution.

Qiskit and OpenQASM 3 retain nested measurements and feedback. Cirq export is
available when the resulting atomic payload is representable in Cirq; blocks
containing further measurements or repeated record writes retain explicit
compatibility diagnostics.

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

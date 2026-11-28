# Cross-framework quantum programming

OpenQML imports Qiskit, Cirq, PennyLane and OpenQASM circuits into a shared IR.
Use `convert` to import, check compatibility and export in one call.
JSON and IR targets work with the core installation. Install the corresponding
extra when importing or exporting a native SDK object.

Cirq imports retain symbolic FSim angles as commuting XX, YY and conditional
phase rotations. Fractional iSWAP and phased-X powers also retain their
parameters and global phase. The emitted rotations can be bound, differentiated
or exported through the other adapters. Source expansion limits count each
emitted operation, including a control-wire phase correction.

## Readouts in the source framework's measurement order

`program.certified_readout(absolute_error=e)` returns a value and a Schmidt
truncation bound for each terminal classical bit in ascending IR bit order.
`classical_bits=[...]` selects and orders marginal probabilities of bit value
one. Unwritten bits start at zero. The last measurement or unconditional
constant assignment wins. An explicitly supplied quantum observable reads
the postmeasurement state, including dephasing caused by measurements whose
classical result was later overwritten.

Qiskit and parsed OpenQASM use the imported classical register map. For Cirq,
`source_options={"measurement_map": {"key": (2, 0)}}` maps a two-wire key's
native measurement order to explicit bit indices. A PennyLane analytic tape
can supply Pauli expectations and real linear combinations directly. Their
return order and wires used only for measurement are preserved. The built-in
importers accept their usual source options, including explicit wire order.
Sampled tape returns, mid-circuit measurements followed by quantum operations,
feedback and noise require other execution paths.

## Classical variables and assignments

IR v5 adds `Circuit.classical_set(bit, 0|1)`, a classical write with no quantum
operand. Readers retain support for IR versions 1–4. `has_measurements` still
means physical measurement; `has_classical_outputs` also includes assignments.
NumPy counts, native density execution and conversion verification retain
overwritten final storage and merge branches after dead writes.

Qiskit Bool/Uint variables, bit and register assignments, and indexed writes
lower to classical bits and bounded decisions. An indexed store selects its
destination and reads its value from the same pre-write state. Whole-register
stores snapshot overlapping operands before changing any destination bit.
A variable read must have a definite assignment. Initialize it before entering
a loop if the first iteration might read it.

Original SDK classical bits retain their indices. Standalone variables,
snapshots and finite-loop control flags use additional classical bits, with
two reusable flow flags per active loop depth. No extra quantum wires are
allocated. `CLASSICAL_STORAGE_LOWERED` reports the added storage; use ordered
`classical_bits` to select original outputs when sampling. Input variables
require `source_options={"classical_inputs": {name: integer}}`; unbound inputs,
out-of-range values and undeclared input names are rejected.

Qiskit, JSON, IR and OpenQASM 3 preserve lowered stores. The built-in QASM 3
reader accepts constant stores, bit copies and fixed-width unsigned counter
updates. The Qiskit frontend also lowers typed arithmetic into guards and stores.
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

## Affine phase folding

Pass `optimize=True, optimization_options={"phase_folding": True}` to
`convert` or `plan_conversion`. The pass follows XOR parities and bit
complements through X, CX and SWAP, combining P, RZ, RZZ and fixed Z-axis
phases when they act on the same parity. CZ, CP and CRZ preserve these
parities. Complemented parities can leave a global phase, which stays in
its original control-flow block. Each measurement, barrier or unsupported
gate ends the region. The compiler tracks wires touched in each region.
`max_phase_work` bounds parity analysis across blocks and passes.
`result.optimization.folded_phases` counts merged phase gates. Logical guard
folding tracks explicit assignments and invalidates known values on measurement.

The affine analysis follows phase folding in
[Amy and Lunderville, POPL (2025)](https://doi.org/10.1145/3704873).
OpenQML applies it within straight-line regions of each lowered block.

## Dynamic control and compilation reuse

| Source feature | Lowering and semantic contract |
| --- | --- |
| Qiskit `if_else` | Atomic equality selection over explicit classical bits |
| Qiskit `switch` | Bit/register and typed Bool/Uint expression selectors, grouped labels and a default branch |
| Qiskit finite `for` | Bounded static expansion with explicit parameter binding |
| Qiskit and OpenQASM 3 `while` | A loop header reevaluates its condition on each iteration |
| `break` / `continue` | Exit or restart the nearest enclosing loop, including inside a switch |
| Repeated Qiskit definitions | One imported body per source object and option set, with fresh wire remapping |
| Qiskit symbolic angles | Per-import expression reuse and iterative DAG export |
| OpenQASM 3 guards | Bit/register/index-set equality and conjunctions, preserving arbitrary bit order |

Switch lowering uses a nested equality decision chain. After a case is selected,
its body runs once; writes to the selector cannot enter another case. Grouped
labels share the imported body. The default body is reached only after every
label fails. Source operation, loop-expansion and control-depth budgets apply
to the resulting decision structure. OpenQASM 3.1 source can use grouped cases
and a default arm. Its portable OpenQASM 3 output uses ordinary conditionals.

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

The portable predicate language uses Bool and Uint. Signed and floating-point
classical arithmetic and hardware timing need additional frontend support.

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
motivates retaining classical structure. Native and portable compilation both
preserve runtime iteration for supported loop conditions.

## Runtime loops and addressed wires

Qiskit and OpenQASM 3 can exchange runtime loops through OpenQML IR and JSON.
Cirq exports supported measurement-controlled loops as native
`CircuitOperation` repeat-until operations. The Cirq import retains the final
measurement values and checks that the exit condition reads the current
iteration's measurements. PennyLane tapes and OpenQASM 2 accept loops after
their control has been resolved to a finite sequence of operations.

For other control structures, pass `portable=True` to `plan_conversion` or
`convert`. The result carries the requested target and an executable
`PortableProgram`. Its JSON representation retains IR registers, runtime
branches, exits and index errors. Load it with `PortableProgram.from_json` and
call `execute` to run it through OpenQML. Use `to_native` when the target
supports all of its operations.

```python
import openqml as oq

source = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] out;
uint[2] i = 0;
while (i < 2) {
    x q[i];
    i += 1;
}
out = measure q;
"""
plan = oq.plan_conversion(source, target="openqasm3")
print(oq.sample(plan.circuit, shots=32, classical_bits=(0, 1)))  # {'11': 32}
print(plan.verify().matched)  # True
```

A dynamic quantum wire lowers to conditional operations on concrete wires.
The core frontend accepts unsigned register and bit selectors, unsigned casts,
and literal masks. A comparison with a literal can prove a smaller range.
A write to the index invalidates the bound. The same dispatch handles classical
reads, writes and measurement destinations. When a static proof is unavailable,
IR v7 retains a runtime range check. An out-of-range execution raises
`CLASSICAL_INDEX_RANGE` before the addressed operation. Native exports that
cannot represent this error return a diagnostic. The portable form retains it.

NumPy samples measurement-controlled loops until every shot exits.
`max_loop_steps` bounds executed instructions across trajectory groups.
For probability-one loops with an unbounded number of possible iterations,
`density_matrix` evaluates the limit of the quantum instrument on the finite
control graph. `plan.verify()` checks termination and compares the instrument
on a complete operator basis at the selected bindings. Both enforce numerical
convergence and their resource budgets. Programs with nonzero nontermination
probability produce a diagnostic.

Set `backend="torch"` or `backend="jax"` for runtime density calculation
and sampling on those tensor backends. Keep parameters as native tensors
when differentiating the density result. `execute` and `sample` return
classical counts as Python dictionaries.

## Import Python circuit builders

Pass `source="python"` to convert a bounded circuit-building entry point.
The static frontend accepts explicit circuit constructors, scalar arguments,
arithmetic, finite `range` loops, gates, measurements and supported Qiskit
control contexts. Use `source_options={"entrypoint": "build", "arguments": {...}}`
to select a function and its concrete arguments. Source text is compiled
without executing imports or arbitrary statements.

Pass a trusted callable with `source="python"` and explicit `args` or `kwargs`
to capture the circuit it returns. This entry point covers the returned
circuit. Python I/O, callbacks, device services and other surrounding
application behavior stay with their host program.

## Finite loops with early exits

Finite Qiskit loops can contain nested `break_loop` and `continue_loop` nodes,
including exits under typed predicates and grouped switch cases. Continuation
lowering attaches each exit to its nearest lexical loop. A branch resolves its
selector before the selected body can overwrite that selector. Loop parameters,
wire maps, global phases and operations following the whole loop are preserved.

The portable output uses existing atomic IR branches, with no mutable scratch
bits. `max_operations` counts expanded branch payloads, including repeated
continuations; `max_depth` bounds the resulting control nesting. Oversized
programs fail before constructing their expanded output. A runtime while loop
retains a back edge in portable IR v6.

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

"""Isolated-process comparison with a previous local source checkout; no downloads."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RUNNER = r"""
import hashlib, json, platform, sys, tracemalloc
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
import openqml as q
options = json.loads(sys.argv[1])
if options['source']:
    assert Path(q.__file__).resolve().is_relative_to(Path(options['source']).resolve())
length = options['length']
mode = options['mode']
if mode == 'nested':
    c = q.Circuit(8).ry(.01, 0)
    for _ in range(length):
        c = q.Circuit(8).ry(.01, 0).if_else(q.ClassicalCondition((0,), 0), c)
else:
    c = q.Circuit(2)
    for i in range(length):
        c = c.h(0).measure(0, i)
        if mode != 'deferred':
            c = c.if_else(q.ClassicalCondition((i,), 1), q.Circuit(2).x(1))
        c = c.reset(0)
if mode == 'deferred':
    for i in range(20):
        c = c.ry(.11*i, 0).cx(0, 1).amplitude_damping(.01, 1)
    c = c.if_else(q.ClassicalCondition(tuple(range(length)), 2**length-1), q.Circuit(2).x(1))
memory = 4 if mode == 'nested' else .25
def evaluate():
    if mode == 'estimate':
        return q.estimate(c, [q.PauliString('ZI'), q.PauliString('IZ')], shots=4096,
                          seed=7, max_memory_mb=.25).values
    return q.density_matrix(c, max_memory_mb=memory)
result = {'openqml': q.__version__, 'python': platform.python_version(), 'numpy': np.__version__,
          'length': length, 'mode': mode, 'max_memory_mb': memory, 'max_branches': 256}
digest = hashlib.sha256()
for file in sorted(Path(q.__file__).parent.rglob('*.py')):
    digest.update(file.relative_to(Path(q.__file__).parent).as_posix().encode()+b'\0'+file.read_bytes())
result['source_sha256'] = digest.hexdigest()
try:
    first = evaluate()
    if mode == 'estimate':
        np.testing.assert_allclose(first, [[1, 0]], atol=.05)
    else:
        np.testing.assert_allclose(np.trace(first), 1, atol=1e-12)
        if mode == 'nested':
            vector = q.statevector(q.Circuit(8).ry(.01*(length+1), 0))
            np.testing.assert_allclose(first, np.outer(vector, vector.conj()), atol=1e-12)
        elif mode != 'deferred':
            np.testing.assert_allclose(first, np.diag([.5,.5,0,0]), atol=1e-12)
    timings = []
    for _ in range(options['repeats']):
        start = perf_counter(); evaluate(); timings.append(perf_counter()-start)
    tracemalloc.start()
    evaluate()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.update(status='passed', median_ms=median(timings)*1000, traced_peak_bytes=peak)
    if mode == 'nested':
        result['rho_probe'] = [float(first[0,0].real),float(first[128,128].real),
                               float(first[0,128].real),float(first[0,128].imag)]
    elif mode != 'estimate':
        result['rho_real'] = first.real.tolist(); result['rho_imag'] = first.imag.tolist()
    planned = c if mode != 'estimate' else c.measure(0, c.num_clbits).measure(1, c.num_clbits+1)
    if hasattr(q, 'plan_density'):
        if mode == 'estimate':
            plan = q.plan_density(planned, classical_bits=(c.num_clbits,c.num_clbits+1))
        else:
            plan = q.plan_density(c)
        result.update(peak_branches=plan.peak_branches,
                      estimated_state_working_bytes=round(plan.estimated_memory_mb()*2**20))
    else:
        from openqml.density import _branch_plan
        peak_branches = _branch_plan(planned, {frozenset()}, 256)[1]
        result.update(peak_branches=peak_branches,
                      estimated_state_working_bytes=64*4**c.num_qubits*peak_branches)
except q.ResourceError as error:
    result.update(status='resource_error', error=str(error))
print(json.dumps(result))
"""

parser = argparse.ArgumentParser()
parser.add_argument("--baseline-source", type=Path, help="Previous extracted src directory")
parser.add_argument("--output", default="benchmarks/results_liveness.json")
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
current = Path(__file__).resolve().parents[1] / "src"
if args.repeats < 1:
    parser.error("repeats must be positive")
sources = [("current", current)]
if args.baseline_source:
    baseline = args.baseline_source.resolve()
    if not (baseline / "openqml" / "__init__.py").is_file():
        parser.error("baseline-source must contain openqml/__init__.py")
    sources.insert(0, ("baseline", baseline))
cases = [("density", n) for n in (4, 8, 12, 64)] + [
    ("deferred", 5),
    ("estimate", 4),
    ("estimate", 12),
    ("nested", 40),
]
results = []
for mode, length in cases:
    case = {"mode": mode, "length": length}
    for label, source in sources:
        environment = dict(os.environ, PYTHONPATH=str(source))
        settings = {"source": str(source), "mode": mode, "length": length, "repeats": args.repeats}
        output = subprocess.run(
            [sys.executable, "-c", RUNNER, json.dumps(settings)],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        case[label] = json.loads(output.stdout)
    if "baseline" in case and case["baseline"]["status"] == case["current"]["status"] == "passed":
        before, after = case["baseline"], case["current"]
        case["time_ratio"] = before["median_ms"] / after["median_ms"]
        if mode == "nested":
            import numpy as np

            np.testing.assert_allclose(before["rho_probe"], after["rho_probe"], atol=1e-11)
        elif mode != "estimate":
            import numpy as np

            np.testing.assert_allclose(before["rho_real"], after["rho_real"], atol=1e-11)
            np.testing.assert_allclose(before["rho_imag"], after["rho_imag"], atol=1e-11)
    results.append(case)
result = {
    "repeats": args.repeats,
    "statistic": "median",
    "warmup_excluded": True,
    "memory_metric": "tracemalloc peak during a separate warm call; not process RSS",
    "baseline": "previous local source in an isolated Python process; identical public call",
    "cases": results,
}
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

"""Kernel methods and variational training on the same four tasks.

Slower than the other examples (about half a minute): the variational models
pay for every gradient step, and the table reports that cost too.

    python examples/06_variational_vs_kernel.py
"""

from openqml import (
    ClassicalKernelClassifier,
    QuantumKernelClassifier,
    VariationalQuantumClassifier,
    get_suite,
    get_task,
)

suite = get_suite("qml-cls-1")
runs = []
for model in (QuantumKernelClassifier(feature_map="iqp"),
              VariationalQuantumClassifier(),
              ClassicalKernelClassifier()):
    runs.extend(suite.run(model))

print(f"{'model':32s} {'task':38s} {'accuracy':>18s} {'seconds':>9s}")
for run in runs:
    values = run.get_metric_fn("accuracy")
    name = run.flow_name.split("(")[0]
    task = get_task(run.task_id).name
    print(f"{name:32s} {task:38s} {values.mean():>10.4f} +/- {values.std():.4f} "
          f"{run.runtime_seconds:>9.1f}")

print("""
The kernel model has no training loop at all -- one pass to encode, one linear
solve -- so it is orders of magnitude cheaper here and loses nothing on these
tasks. Variational training earns its cost when the state space is too large to
form a Gram matrix over, which is not the case for anything in this suite.""")

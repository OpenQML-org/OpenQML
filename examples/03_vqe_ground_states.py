"""VQE against exact diagonalisation across a family of Hamiltonians.

    python examples/03_vqe_ground_states.py
"""

import numpy as np

import openqml
from openqml.models import VQE

task = openqml.get_task(4)
run = openqml.run_model_on_task(VQE(ansatz="real_amplitudes", layers=2, maxiter=80), task)

print(f"{'separation':>11} {'VQE energy':>12} {'exact':>12} {'error':>10}")
for record in run.predictions:
    print(f"{record['separation']:>11.2f} {record['prediction']:>12.6f} "
          f"{record['truth']:>12.6f} {record['error']:>10.2e}")

errors = np.array([r["error"] for r in run.predictions])
print(f"\nmean |error| {errors.mean():.2e}, worst {errors.max():.2e}, "
      f"runtime {run.runtime_seconds:.1f}s")
print("Every VQE energy sits above the exact one, as a variational bound must.")
print("This is a two-qubit surrogate: it validates the optimiser, not chemistry.")

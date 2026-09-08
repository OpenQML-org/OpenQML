"""The five-line version: pick a task, run a model, publish the result.

    python examples/01_quickstart.py
"""

import openqml
from openqml.models import QuantumKernelClassifier

print("datasets on this instance")
print(openqml.list_datasets().to_string(index=False))

task = openqml.get_task(1)
print("\n", repr(task), sep="")

model = QuantumKernelClassifier(feature_map="zz", n_qubits=2)
run = openqml.run_model_on_task(model, task)
print("\n" + run.summary())

run.publish()
print(f"\npublished as run {run.id}")
print("\nleaderboard for task 1")
print(openqml.leaderboard(1).to_string(index=False))

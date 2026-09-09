"""Publish your own dataset and task, then run against it.

    python examples/04_custom_dataset_and_task.py
"""

import numpy as np

from openqml import (
    ClassicalKernelClassifier,
    QuantumKernelClassifier,
    create_dataset,
    create_task,
    leaderboard,
    run_model_on_task,
)

rng = np.random.default_rng(0)
angles = rng.uniform(0, 2 * np.pi, 160)
radii = rng.uniform(0.2, 1.0, 160)
X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
y = (radii > 0.6).astype(int)

dataset = create_dataset(
    name="annulus-2d",
    description="Points inside a disc, labelled by radius. Rotationally symmetric, "
                "so anything that only sees pairwise distances has an easy time.",
    X=X, y=y, attribute_names=["x0", "x1"],
    qubits=2, suggested_encoding="angle", tags=["synthetic", "demo"],
).publish()
print("dataset id", dataset.id)

task = create_task(
    task_type="supervised_classification",
    dataset_id=dataset.id,
    name="annulus-2d / binary classification",
    evaluation_measure="accuracy",
    estimation_procedure={"type": "crossvalidation", "folds": 5, "stratified": True, "seed": 1},
    qubits=2,
).publish()
print("task id", task.id)

for model in (QuantumKernelClassifier(feature_map="zz", n_qubits=2), ClassicalKernelClassifier()):
    run = run_model_on_task(model, task).publish()
    print(run.summary())

print("\n" + leaderboard(task.id).to_string(index=False))

"""Entities bundled with the package and seeded into the local store.

These are deliberately small and reproducible: every one of them regenerates in
under a second on a laptop, so the examples and the test suite never depend on
a network or on a downloaded blob.
"""

from __future__ import annotations

SEED_DATASETS = [
    {
        "id": 1,
        "name": "moons-2d",
        "version": 1,
        "description": "Two interleaving half-circles with Gaussian noise. Non-linear but "
                       "easy; a QML model that cannot fit this is broken, not interesting.",
        "data_type": "tabular",
        "default_target_attribute": "class",
        "attribute_names": ["x0", "x1"],
        "qubits": 2,
        "suggested_encoding": "angle",
        "n_samples": 200, "n_features": 2, "n_classes": 2,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["synthetic", "binary", "baseline"],
        "generator": {"function": "make_moons", "kwargs": {"n_samples": 200, "noise": 0.12, "seed": 0}},
    },
    {
        "id": 2,
        "name": "circles-2d",
        "version": 1,
        "description": "Concentric circles: no linear separator exists in the input space, "
                       "so any accuracy above chance comes from the feature map.",
        "data_type": "tabular",
        "default_target_attribute": "class",
        "attribute_names": ["x0", "x1"],
        "qubits": 2,
        "suggested_encoding": "angle",
        "n_samples": 200, "n_features": 2, "n_classes": 2,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["synthetic", "binary", "baseline"],
        "generator": {"function": "make_circles", "kwargs": {"n_samples": 200, "noise": 0.06, "seed": 1}},
    },
    {
        "id": 3,
        "name": "blobs-4d",
        "version": 1,
        "description": "Three isotropic Gaussian clusters in four dimensions. Included as a "
                       "control: classical baselines should win here and usually do.",
        "data_type": "tabular",
        "default_target_attribute": "class",
        "attribute_names": ["x0", "x1", "x2", "x3"],
        "qubits": 4,
        "suggested_encoding": "angle",
        "n_samples": 150, "n_features": 4, "n_classes": 3,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["synthetic", "multiclass", "control"],
        "generator": {"function": "make_blobs", "kwargs": {"n_samples": 150, "n_features": 4,
                                                            "centers": 3, "seed": 2}},
    },
    {
        "id": 4,
        "name": "parity-4bit",
        "version": 1,
        "description": "All 16 four-bit strings labelled by parity. A global correlation with "
                       "no local shortcut; the entangling layer has to do real work.",
        "data_type": "bitstring",
        "default_target_attribute": "class",
        "attribute_names": ["b0", "b1", "b2", "b3"],
        "qubits": 4,
        "suggested_encoding": "angle",
        "n_samples": 16, "n_features": 4, "n_classes": 2,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["synthetic", "binary", "structure"],
        "generator": {"function": "make_parity", "kwargs": {"n_bits": 4}},
    },
    {
        "id": 5,
        "name": "tfim-1d-6q-ground-states",
        "version": 1,
        "description": "Exact ground states of the six-site transverse-field Ising chain for a "
                       "range of field strengths, labelled by phase (h < 1 ordered, h > 1 "
                       "disordered), with the critical window left out. Rows are statevectors, "
                       "so this is data that is natively quantum rather than encoded.",
        "data_type": "quantum_state",
        "default_target_attribute": "phase",
        "qubits": 6,
        "suggested_encoding": "amplitude",
        "n_samples": 40, "n_features": 64, "n_classes": 2,
        "licence": "CC0-1.0", "creator": "openqml",
        "citation": "Phase labels follow the exactly known critical point of the 1-D TFIM.",
        "tags": ["physics", "quantum-data", "phase-classification"],
        "generator": {"function": "make_tfim_ground_states",
                      "kwargs": {"n_qubits": 6, "n_samples": 40, "gap": 0.2}},
    },
    {
        "id": 6,
        "name": "h2-surrogate-2q",
        "version": 1,
        "description": "A family of two-qubit Pauli-sum Hamiltonians whose coefficients decay "
                       "smoothly with a separation parameter, giving an energy curve with a "
                       "minimum and a flat tail. The coefficients are a synthetic surrogate, "
                       "NOT an electronic-structure calculation; the reference energies are "
                       "exact diagonalisations of these same operators, which is what makes "
                       "the benchmark meaningful.",
        "data_type": "hamiltonian",
        "default_target_attribute": "ground_state_energy",
        "attribute_names": ["separation"],
        "qubits": 2,
        "suggested_encoding": "none",
        "n_samples": 12, "n_features": 1,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["vqe", "ground-state", "surrogate"],
        "generator": {"function": "make_two_qubit_hamiltonians", "kwargs": {"n_points": 12}},
    },
    {
        "id": 7,
        "name": "periodic-sin3",
        "version": 1,
        "description": "y = sin(3*pi*x) on [-1, 1] with light noise. Included because "
                       "high-frequency periodic targets are one of the few places where a "
                       "data re-uploading circuit has a structural advantage over a "
                       "parameter-matched classical baseline -- and the comparison is on the "
                       "same task, so anyone can check the claim.",
        "data_type": "tabular",
        "default_target_attribute": "target",
        "attribute_names": ["x0"],
        "qubits": 2,
        "suggested_encoding": "angle",
        "n_samples": 120, "n_features": 1,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["regression", "periodic", "fourier"],
        "generator": {"function": "make_periodic", "kwargs": {"n_samples": 120, "frequency": 3.0,
                                                               "seed": 3}},
    },
    {
        "id": 8,
        "name": "bars-and-stripes-3x3",
        "version": 1,
        "description": "The 14 bar/stripe patterns of a 3x3 grid, flattened to nine bits. A "
                       "standard small target distribution for generative circuit models.",
        "data_type": "bitstring",
        "default_target_attribute": "class",
        "qubits": 9,
        "suggested_encoding": "angle",
        "n_samples": 14, "n_features": 9, "n_classes": 2,
        "licence": "CC0-1.0", "creator": "openqml",
        "tags": ["generative", "structure"],
        "generator": {"function": "make_bars_and_stripes", "kwargs": {"size": 3}},
    },
]

_CV5 = {"type": "crossvalidation", "folds": 5, "repeats": 1, "stratified": True, "seed": 7}
_CV4 = {"type": "crossvalidation", "folds": 4, "repeats": 1, "stratified": True, "seed": 7}
_CV5_PLAIN = {"type": "crossvalidation", "folds": 5, "repeats": 1, "stratified": False, "seed": 7}

SEED_TASKS = [
    {
        "id": 1, "name": "moons-2d / binary classification", "task_type": "supervised_classification",
        "dataset_id": 1, "target_name": "class", "evaluation_measure": "accuracy",
        "estimation_procedure": _CV5, "qubits": 2,
        "description": "Baseline binary classification with 5-fold stratified cross-validation.",
        "tags": ["classification", "baseline"],
    },
    {
        "id": 2, "name": "parity-4bit / binary classification", "task_type": "supervised_classification",
        "dataset_id": 4, "target_name": "class", "evaluation_measure": "accuracy",
        "estimation_procedure": _CV4, "qubits": 4,
        "description": "Parity with 4-fold cross-validation. Held-out bitstrings are never "
                       "seen during training, so a smooth interpolator has nothing to go on "
                       "and can land *below* chance -- an RBF kernel scores 0.00 here, "
                       "predicting the opposite label every time. That is anti-learning, not "
                       "a bug: report it as it is. With 16 points in total, always report the "
                       "fold spread alongside the mean.",
        "tags": ["classification", "structure"],
    },
    {
        "id": 3, "name": "tfim-6q / phase classification", "task_type": "state_classification",
        "dataset_id": 5, "target_name": "phase", "evaluation_measure": "accuracy",
        "estimation_procedure": _CV5, "qubits": 6,
        "description": "Classify the phase of a ground state given the state itself. Inputs are "
                       "statevectors, so a fidelity kernel is computed directly with no encoding.",
        "tags": ["classification", "quantum-data", "physics"],
    },
    {
        "id": 4, "name": "h2-surrogate-2q / ground state energy", "task_type": "ground_state_estimation",
        "dataset_id": 6, "target_name": "ground_state_energy",
        "evaluation_measure": "absolute_energy_error",
        "estimation_procedure": {"type": "none"}, "qubits": 2,
        "description": "Estimate the ground-state energy of each Hamiltonian in the family. "
                       "Scored against exact diagonalisation, so chemical-accuracy style "
                       "thresholds are meaningful within the surrogate.",
        "tags": ["vqe", "ground-state"],
    },
    {
        "id": 5, "name": "periodic-sin3 / regression", "task_type": "supervised_regression",
        "dataset_id": 7, "target_name": "target", "evaluation_measure": "mean_squared_error",
        "estimation_procedure": _CV5_PLAIN, "qubits": 2,
        "description": "Fit a high-frequency sine with 5-fold cross-validation. The point of "
                       "the task is the quantum/classical gap, so always run a classical "
                       "baseline alongside.",
        "tags": ["regression", "periodic"],
    },
    {
        "id": 6, "name": "circles-2d / binary classification", "task_type": "supervised_classification",
        "dataset_id": 2, "target_name": "class", "evaluation_measure": "accuracy",
        "estimation_procedure": _CV5, "qubits": 2,
        "description": "Concentric circles with 5-fold stratified cross-validation.",
        "tags": ["classification", "baseline"],
    },
    {
        "id": 7, "name": "blobs-4d / multiclass classification", "task_type": "supervised_classification",
        "dataset_id": 3, "target_name": "class", "evaluation_measure": "accuracy",
        "estimation_procedure": _CV5, "qubits": 4,
        "description": "Three-class control task. Expect classical baselines to win.",
        "tags": ["classification", "control"],
    },
]

SEED_SUITES = [
    {
        "id": 1, "name": "qml-classification-1", "alias": "qml-cls-1",
        "description": "Four classification tasks spanning easy, structured and natively "
                       "quantum data. Report every task, including the ones you lose.",
        "task_ids": [1, 2, 3, 6],
        "tags": ["benchmark", "classification"],
    },
    {
        "id": 2, "name": "quantum-vs-classical-1", "alias": "qvc-1",
        "description": "Paired tasks for honest comparison: a target where a re-uploading "
                       "circuit is expected to win (periodic regression) and a control where "
                       "it is expected to lose (Gaussian blobs).",
        "task_ids": [5, 7],
        "tags": ["benchmark", "comparison"],
    },
]

SEED_ENTITIES = {
    "datasets": SEED_DATASETS,
    "tasks": SEED_TASKS,
    "suites": SEED_SUITES,
}

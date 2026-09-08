import numpy as np
import pytest

import openqml
from openqml.exceptions import OpenQMLServerNoResult


def test_bundled_catalog_is_listed():
    datasets = openqml.list_datasets(output_format="records")
    assert len(datasets) >= 8
    assert {"moons-2d", "tfim-1d-6q-ground-states", "h2-surrogate-2q"} <= {d["name"] for d in datasets}


def test_dataset_by_name_and_id_agree():
    assert openqml.get_dataset(1).name == openqml.get_dataset("moons-2d").name


def test_dataset_arrays_are_reproducible():
    first = openqml.get_dataset("moons-2d").get_data(target="class")[0]
    openqml.config.set_root_cache_directory(openqml.config.get_root_cache_directory())
    second = openqml.get_dataset("moons-2d").get_data(target="class")[0]
    assert np.allclose(first, second)


def test_quantum_state_dataset_is_normalised():
    states = openqml.get_dataset("tfim-1d-6q-ground-states").get_statevectors()
    assert states.shape == (40, 64)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0)


def test_hamiltonian_dataset_reference_energies_are_exact():
    from openqml.hamiltonians import exact_ground_state

    for record in openqml.get_dataset("h2-surrogate-2q").get_hamiltonians():
        exact, _ = exact_ground_state(record["terms"], 2)
        assert exact == pytest.approx(record["reference_energy"], abs=1e-12)


def test_create_and_publish_dataset():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = (X.sum(axis=1) > 0).astype(int)
    dataset = openqml.create_dataset("unit-test-set", "created by the test suite", X, y)
    dataset.publish()
    assert dataset.id is not None
    fetched = openqml.get_dataset(dataset.id)
    assert np.allclose(fetched.get_data(target="class")[0], X)


def test_missing_entity_raises():
    with pytest.raises(OpenQMLServerNoResult):
        openqml.get_dataset(9999)


def test_task_splits_are_deterministic_and_disjoint():
    task = openqml.get_task(1)
    train, test = task.get_train_test_split_indices(fold=0)
    again, _ = task.get_train_test_split_indices(fold=0)
    assert np.array_equal(train, again)
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(task.get_X_and_y()[0])


def test_stratification_preserves_class_balance():
    task = openqml.get_task(1)
    _, y = task.get_X_and_y()
    for _, _, _, test in task.download_split():
        assert set(np.unique(y[test])) == set(np.unique(y))


def test_create_task_round_trip():
    task = openqml.create_task("supervised_classification", dataset_id=2, target_name="class")
    task.publish()
    assert openqml.get_task(task.id).dataset_id == 2


def test_suite_lookup_by_alias():
    suite = openqml.get_suite("qml-cls-1")
    assert len(suite.task_ids) == 4
    assert all(t.id in suite.task_ids for t in suite.tasks)

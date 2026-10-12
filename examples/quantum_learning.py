"""Training and held-out metrics; results are computed, never preset."""

import openqml as q


def main():
    data = q.load_dataset("moons", n_samples=128, seed=7)
    X_train, X_test, y_train, y_test = data.split(seed=7)
    models = [
        q.QuantumKernelClassifier(),
        q.VariationalQuantumClassifier(epochs=60, seed=7),
        q.LogisticClassifier(seed=7),
    ]
    for model in models:
        model.fit(X_train, y_train)
        print(type(model).__name__, "held-out accuracy:", model.score(X_test, y_test))

    # A smooth periodic target; Fourier is an essential classical comparator.
    data = q.load_dataset("sine", frequency=1, n_samples=64, noise=0, seed=7)
    a, b, c, d = data.split(seed=7)
    for model in [
        q.VariationalQuantumRegressor(epochs=100, seed=7),
        q.FourierRegressor(harmonics=3),
    ]:
        model.fit(a, c)
        print(type(model).__name__, "held-out R2:", model.score(b, d))


if __name__ == "__main__":
    main()

"""Where a re-uploading circuit genuinely helps -- and where it does not.

    python examples/05_periodic_advantage.py
"""

from openqml import (
    ClassicalKernelRegressor,
    LinearRegressor,
    VariationalQuantumRegressor,
    compare,
    get_task,
)

task = get_task(5)  # y = sin(3*pi*x)
print(repr(task), "\n")

table = compare(
    [
        VariationalQuantumRegressor(),
        LinearRegressor(n_frequencies=0),
        LinearRegressor(n_frequencies=4),
        ClassicalKernelRegressor(kernel="rbf"),
    ],
    [task],
)
print(table.to_string(index=False))

print("""
Lower is better (mean squared error, five folds).

The circuit beats ridge regression on the raw feature by two orders of
magnitude, and the reason is structural rather than mysterious: a data
re-uploading circuit is a truncated Fourier series in its input, and the target
is a sine. Hand the classical model the same basis -- four cosine/sine pairs --
and it wins on accuracy and on runtime, by a lot.

The useful claim is therefore narrow: the circuit found the right basis without
being told it. That is worth something, and it is not the same as an advantage.""")

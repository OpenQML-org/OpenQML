"""Small composable algorithm blocks and variational workflows."""

import numpy as np

from ._validation import integer, real, vector
from .circuit import Circuit
from .hamiltonian import Hamiltonian
from .optimization import minimize


def ghz(n_qubits=3):
    circuit = Circuit(n_qubits).h(0)
    for q in range(1, circuit.n_qubits):
        circuit.cx(0, q)
    return circuit


def bell():
    return ghz(2)


def qft(n_qubits, *, inverse=False):
    return Circuit(n_qubits).qft(inverse=inverse)


def grover(n_qubits, marked, *, iterations=None):
    """Search explicit marked basis indices, with a phase oracle and diffuser.

    Classical oracle construction knows the marked states. This toy example is
    a correctness exercise, not a black-box search speedup demonstration.
    """
    n_qubits = integer(n_qubits, "n_qubits", 1, 20)
    marked = tuple(integer(m, "marked index", 0, (1 << n_qubits) - 1) for m in marked)
    if not marked or len(set(marked)) != len(marked):
        raise ValueError("marked indices must be nonempty and unique")
    if iterations is None:
        theta = np.arcsin(np.sqrt(len(marked) / (1 << n_qubits)))
        iterations = max(0, int(np.round(np.pi / (4 * theta) - 0.5)))
    iterations = integer(iterations, "iterations", 0, 10_000)
    circuit = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit.h(q)

    def phase(index):
        for q in range(n_qubits):
            if not ((index >> q) & 1):
                circuit.x(q)
        circuit.unitary([[1, 0], [0, -1]], [n_qubits - 1], controls=range(n_qubits - 1))
        for q in range(n_qubits):
            if not ((index >> q) & 1):
                circuit.x(q)

    for _ in range(iterations):
        for index in marked:
            phase(index)
        for q in range(n_qubits):
            circuit.h(q)
        phase(0)
        for q in range(n_qubits):
            circuit.h(q)
        circuit.global_phase(np.pi)
    return circuit


def hardware_efficient_ansatz(parameters, n_qubits, *, layers=2):
    n_qubits = integer(n_qubits, "n_qubits", 1, 40)
    layers = integer(layers, "layers", 1)
    parameters = vector(parameters)
    if parameters.size != 2 * n_qubits * layers:
        raise ValueError("ansatz needs 2 * n_qubits * layers angles")
    circuit = Circuit(n_qubits)
    for layer in range(layers):
        for q in range(n_qubits):
            i = 2 * (layer * n_qubits + q)
            circuit.ry(parameters[i], q).rz(parameters[i + 1], q)
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)
    return circuit


def vqe(hamiltonian, ansatz, initial, *, steps=100, learning_rate=0.05, gradient_fn=None):
    """Optimize a user's ansatz; finite differences are the safe default gradient."""
    if not isinstance(hamiltonian, Hamiltonian):
        hamiltonian = Hamiltonian(hamiltonian)

    def energy(parameters):
        circuit = ansatz(parameters)
        if not isinstance(circuit, Circuit) or circuit.n_qubits != hamiltonian.n_qubits:
            raise ValueError("ansatz must return a Circuit matching the Hamiltonian")
        return circuit.run().expectation(hamiltonian)

    return minimize(
        energy, initial, gradient_fn=gradient_fn, steps=steps, learning_rate=learning_rate
    )


def qaoa_maxcut(n_qubits, edges, gammas, betas):
    """Return a QAOA circuit for C=sum_edges(1-Z_i Z_j)/2.

    Each cost block is exp(-i gamma C), each mixer exp(-i beta sum X).
    Edges can be (i, j) or (i, j, nonnegative_weight).
    """
    n_qubits = integer(n_qubits, "n_qubits", 1, 40)
    gammas, betas = vector(gammas), vector(betas)
    if gammas.size != betas.size or not gammas.size:
        raise ValueError("gammas and betas must have the same nonzero length")
    validated = []
    for edge in edges:
        if len(edge) not in (2, 3):
            raise ValueError("edges must be (first, second) or (first, second, weight)")
        first, second = [integer(q, "edge wire", 0, n_qubits - 1) for q in edge[:2]]
        weight = real(edge[2], "edge weight") if len(edge) == 3 else 1.0
        if first == second or weight < 0:
            raise ValueError("edges need distinct wires and nonnegative weights")
        validated.append((first, second, weight))
    circuit = Circuit(n_qubits)
    for q in range(n_qubits):
        circuit.h(q)
    for gamma, beta in zip(gammas, betas):
        for first, second, weight in validated:
            label = ["I"] * n_qubits
            label[-1 - first] = label[-1 - second] = "Z"
            circuit.pauli_rotation(-gamma * weight, "".join(label))
            circuit.global_phase(-gamma * weight / 2)
        for q in range(n_qubits):
            circuit.rx(2 * beta, q)
    return circuit

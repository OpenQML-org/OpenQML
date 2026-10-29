"""Installation advice only; optional integrations never install themselves."""

INTEGRATION_EXTRAS = {
    "torch": "torch",
    "jax": "jax",
    "flax": "flax",
    "optax": "flax",
    "qiskit": "qiskit",
    "qiskit_aer": "aer",
    "pennylane": "pennylane",
    "cirq": "cirq",
    "openqasm3": "qasm",
    "sympy": "symbolic",
}


def install_hint(name):
    extra = INTEGRATION_EXTRAS.get(name, name)
    package = "openqml" if extra == "numpy" else f"openqml[{extra}]"
    return f'python -m pip install "{package}"'

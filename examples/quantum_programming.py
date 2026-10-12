"""Bell/GHZ, custom blocks, measurement feedback, noise, and VQE."""

import numpy as np

import openqml as q


def main():
    state = q.bell().run(seed=7)
    print("Bell counts:", state.sample(1024))
    print("Bell entropy S(q0):", state.entropy([0]))
    print("40-qubit GHZ marginal:", q.ghz(40).run().probabilities([0, 39]))

    # Teleport a user's prepared qubit with actual measurement branches.
    circuit = q.Circuit(3).ry(0.713, 0).rz(0.441, 0).h(1).cx(1, 2)
    circuit.cx(0, 1).h(0).measure(0, "phase").measure(1, "flip")
    circuit.x(2, condition=("flip", 1)).z(2, condition=("phase", 1))
    result = circuit.run(seed=7)
    expected = q.Circuit(1).ry(0.713, 0).rz(0.441, 0).run().reduced_density()
    print("Teleportation branch:", result.classical)
    print("Teleportation matches:", np.allclose(result.reduced_density([2]), expected))

    noisy = q.bell().amplitude_damping(0.2, 0).run()
    print("Noisy density eigenvalues:", np.linalg.eigvalsh(noisy.reduced_density()))
    print("QASM 2 circuit:\n", q.bell().to_qasm())

    H = q.Hamiltonian({"ZZ": -1, "XI": -0.5, "IX": -0.5})

    def ansatz(p):
        return q.Circuit(2).ry(p[0], 0).ry(p[1], 1).cx(0, 1)

    fit = q.vqe(H, ansatz, [0.4, 0.3], steps=100, learning_rate=0.08)
    print("VQE energy / exact reference:", fit.value, H.ground_energy())


if __name__ == "__main__":
    main()

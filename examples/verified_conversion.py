"""Check complete input maps through adapters and diagnose unsupported targets."""

import argparse

from openqml import Circuit, ClassicalCondition, Parameter, plan_conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("json", "openqasm2", "openqasm3", "qiskit", "cirq", "pennylane", "all"),
        default="json",
    )
    args = parser.parse_args()
    targets = (
        ("json", "openqasm2", "openqasm3", "qiskit", "cirq", "pennylane")
        if args.target == "all"
        else (args.target,)
    )
    unitary = Circuit(2).ry(Parameter("theta"), 1).cx(1, 0).with_global_phase(0.2)
    noisy = Circuit(2).ry(0.3, 1).cx(1, 0).amplitude_damping(0.2, 0).reset(1)
    feedback = (
        Circuit(2).h(0).measure(0, 0).if_else(ClassicalCondition((0,), 1), Circuit(2).x(1)).reset(0)
    )
    for target in targets:
        for name, circuit in (("unitary", unitary), ("channel", noisy), ("feedback", feedback)):
            plan = plan_conversion(
                circuit, target=target, bindings={"theta": 0.4} if name == "unitary" else None
            )
            if not plan.compatible:
                codes = ", ".join(d.code for d in plan.diagnostics if d.severity == "error")
                print(f"{target}: {name}: unsupported ({codes})")
                continue
            check = plan.verify(basis_chunk_size=3)
            check.raise_for_errors()
            print(
                f"{target}: {check.semantics}: checked {check.checked_inputs} inputs, "
                f"{check.checked_entries} entries"
            )


if __name__ == "__main__":
    main()

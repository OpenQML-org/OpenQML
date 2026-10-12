"""python -m openqml: offline demo, catalog, or seeded local benchmark."""

import argparse
import json

from . import compare_models, ghz, list_datasets, list_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenQML quantum programming and benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a Bell/GHZ state without any account")
    demo.add_argument("--qubits", type=int, default=3)
    demo.add_argument("--shots", type=int, default=1024)
    demo.add_argument("--seed", type=int, default=1)
    commands.add_parser("datasets", help="List offline dataset generators")
    commands.add_parser("models", help="List built-in learning models")
    benchmark = commands.add_parser("benchmark", help="Compare quantum and classical models")
    benchmark.add_argument("--task", choices=[d["id"] for d in list_datasets()], default="moons")
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    benchmark.add_argument("--epochs", type=int, default=40)
    benchmark.add_argument("--output", help="Save all metrics and provenance to JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            circuit = ghz(args.qubits)
            state = circuit.run(seed=args.seed)
            result = {
                "circuit": circuit.resources(),
                "backend": state.backend,
                "counts": state.sample(args.shots),
                "first_qubit_bloch": state.bloch_vector(0).tolist(),
            }
        elif args.command == "datasets":
            result = list_datasets()
        elif args.command == "models":
            result = list_models()
        else:
            comparison = compare_models(args.task, seeds=args.seeds, epochs=args.epochs)
            if args.output:
                comparison.save(args.output)
            result = comparison.to_dict()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

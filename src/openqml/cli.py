"""Offline package diagnostics and static circuit conversion."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import (
    OpenQMLError,
    __version__,
    circuit_stats,
    convert,
    decompose,
    list_backends,
    optimize,
    plan_conversion,
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openqml")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report installed extras without importing them")
    command = commands.add_parser(
        "convert", help="Convert JSON IR and the supported OpenQASM subset"
    )
    command.add_argument("input", type=Path)
    command.add_argument(
        "--from", dest="source", choices=("json", "openqasm2", "openqasm3"), required=True
    )
    command.add_argument(
        "--to", dest="target", choices=("json", "openqasm2", "openqasm3"), required=True
    )
    command.add_argument("-o", "--output", type=Path)
    command.add_argument("--force", action="store_true", help="Replace an existing output file")
    command.add_argument(
        "--diagnostics", action="store_true", help="Write JSON diagnostics to stderr"
    )
    command.add_argument(
        "--check", action="store_true", help="Print a JSON conversion plan without exporting"
    )
    command.add_argument("--max-operations", type=int, default=100_000)
    command.add_argument("--max-depth", type=int, default=64)
    command.add_argument("--max-output-statements", type=int, default=100_000)
    inspection = commands.add_parser(
        "inspect", help="Report circuit width, dependencies and gate counts"
    )
    inspection.add_argument("input", type=Path)
    inspection.add_argument(
        "--from", dest="source", choices=("json", "openqasm2", "openqasm3"), required=True
    )
    optimization = commands.add_parser(
        "optimize", help="Exact logical feedback optimization and basis lowering"
    )
    optimization.add_argument("input", type=Path)
    optimization.add_argument(
        "--from", dest="source", choices=("json", "openqasm2", "openqasm3"), required=True
    )
    optimization.add_argument(
        "--to", dest="target", choices=("json", "openqasm2", "openqasm3"), default="json"
    )
    optimization.add_argument("--basis", help="Comma-separated unitary gates, e.g. p,ry,cx")
    optimization.add_argument("-o", "--output", type=Path)
    optimization.add_argument("--force", action="store_true")
    optimization.add_argument("--diagnostics", action="store_true")
    optimization.add_argument("--max-operations", type=int, default=100_000)
    optimization.add_argument("--max-depth", type=int, default=64)
    optimization.add_argument("--max-passes", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            packages = {"openqml": __version__}
            for package in (
                "numpy",
                "torch",
                "jax",
                "qiskit",
                "qiskit-aer",
                "pennylane",
                "cirq-core",
                "flax",
                "optax",
            ):
                try:
                    packages[package] = version(package)
                except PackageNotFoundError:
                    packages[package] = None
            packages["registered_backends"] = list_backends()
            print(json.dumps(packages, indent=2))
        else:
            if args.input.stat().st_size > 8_000_000:
                raise ValueError("Input file exceeds 8 MB.")
            text = args.input.read_text(encoding="utf-8")
            if args.command == "convert" and args.check:
                if args.output or args.force:
                    raise ValueError("--check cannot be combined with --output or --force.")
                plan = plan_conversion(
                    text,
                    source=args.source,
                    target=args.target,
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                    max_output_statements=args.max_output_statements,
                )
                print(plan.to_json())
                return 0 if plan.compatible else 2
            if args.command == "inspect":
                circuit = convert(text, source=args.source, target="ir").circuit
                print(json.dumps(asdict(circuit_stats(circuit)), indent=2))
                return 0
            if args.command == "optimize":
                circuit = convert(
                    text,
                    source=args.source,
                    target="ir",
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                ).circuit
                profile = optimize(
                    circuit,
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                    max_passes=args.max_passes,
                )
                circuit = profile.circuit
                if args.basis is not None:
                    circuit = decompose(
                        circuit,
                        basis=tuple(name.strip() for name in args.basis.split(",")),
                        max_operations=args.max_operations,
                        max_depth=args.max_depth,
                    )
                result = convert(
                    circuit,
                    source="ir",
                    target=args.target,
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                )
                result = replace(
                    result,
                    optimization=profile,
                    diagnostics=profile.diagnostics + result.diagnostics,
                )
            else:
                result = convert(
                    text,
                    source=args.source,
                    target=args.target,
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                    max_output_statements=args.max_output_statements,
                )
            if args.diagnostics:
                print(
                    json.dumps([asdict(diagnostic) for diagnostic in result.diagnostics]),
                    file=sys.stderr,
                )
            if args.output:
                with args.output.open("w" if args.force else "x", encoding="utf-8") as stream:
                    stream.write(result.value)
            else:
                print(result.value, end="" if result.value.endswith("\n") else "\n")
        return 0
    except (OpenQMLError, OSError, ValueError) as exc:
        print(f"openqml: {exc}", file=sys.stderr)
        return 2

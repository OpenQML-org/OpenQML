"""Offline package diagnostics and static circuit conversion."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import (
    OpenQMLError,
    circuit_stats,
    convert,
    decompose,
    environment_report,
    estimate_resources,
    inspect_python,
    list_backends,
    optimize,
    plan_conversion,
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openqml")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Report integration availability and compatibility")
    doctor.add_argument(
        "--report", action="store_true", help="Include dependency and platform details"
    )
    doctor.add_argument(
        "--check-imports", action="store_true", help="Probe imports in isolated subprocesses"
    )
    doctor.add_argument(
        "--sizes", action="store_true", help="Measure logical installed dependency files"
    )
    doctor.add_argument("--output", type=Path)
    doctor.add_argument("--force", action="store_true")
    python = commands.add_parser(
        "inspect-python", help="Inspect source without executing or translating it"
    )
    python.add_argument("input", type=Path)
    python.add_argument("--max-source-bytes", type=int, default=8_000_000)
    python.add_argument("--max-tokens", type=int, default=200_000)
    python.add_argument("--max-nodes", type=int, default=100_000)
    resources = commands.add_parser(
        "resources", help="Estimate storage separately from circuit size"
    )
    resources.add_argument("input", type=Path)
    resources.add_argument(
        "--from", dest="source", choices=("json", "openqasm2", "openqasm3"), required=True
    )
    resources.add_argument("--max-qubits", type=int, default=20)
    resources.add_argument("--max-memory-mb", type=float, default=256)
    resources.add_argument("--batch-size", type=int, default=1)
    resources.add_argument("--max-operations", type=int, default=100_000)
    resources.add_argument("--max-depth", type=int, default=64)
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
            if args.report or args.check_imports or args.sizes or args.output:
                report = environment_report(
                    check_imports=args.check_imports, include_sizes=args.sizes
                )
                text = json.dumps(report, indent=2) + "\n"
                if args.output:
                    with args.output.open("w" if args.force else "x", encoding="utf-8") as stream:
                        stream.write(text)
                else:
                    print(text, end="")
                return 0 if report["status"] == "ok" else 2
            packages = {"openqml": "installed"}
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
                "openqasm3",
                "sympy",
                "packaging",
            ):
                try:
                    version(package)
                    packages[package] = "installed"
                except PackageNotFoundError:
                    packages[package] = "missing"
            packages["registered_backends"] = list_backends()
            print(json.dumps(packages, indent=2))
        elif args.command == "inspect-python":
            if args.max_source_bytes < 1 or args.input.stat().st_size > args.max_source_bytes:
                raise ValueError("Python source exceeds --max-source-bytes.")
            report = inspect_python(
                args.input.read_text(encoding="utf-8"),
                filename=str(args.input),
                max_source_bytes=args.max_source_bytes,
                max_tokens=args.max_tokens,
                max_nodes=args.max_nodes,
            )
            print(report.to_json())
            return 0 if report.static_scan_complete else 2
        else:
            if args.input.stat().st_size > 8_000_000:
                raise ValueError("Input file exceeds 8 MB.")
            text = args.input.read_text(encoding="utf-8")
            if args.command == "resources":
                circuit = plan_conversion(
                    text,
                    source=args.source,
                    target="ir",
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                ).circuit
                report = estimate_resources(
                    circuit,
                    batch_size=args.batch_size,
                    max_qubits=args.max_qubits,
                    max_memory_mb=args.max_memory_mb,
                    max_operations=args.max_operations,
                    max_depth=args.max_depth,
                )
                print(report.to_json())
                return 0
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

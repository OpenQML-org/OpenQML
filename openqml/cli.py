"""A small command line front end: ``python -m openqml ...``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, config, datasets, evaluations, study, tasks


def _print(table) -> None:
    if hasattr(table, "to_string"):
        print(table.to_string(index=False))
    else:
        print(json.dumps(table, indent=2, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="openqml", description="OpenQML command line")
    parser.add_argument("--version", action="version", version=f"openqml {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="show the active configuration")
    subparsers.add_parser("datasets", help="list datasets")
    subparsers.add_parser("tasks", help="list tasks")
    subparsers.add_parser("suites", help="list benchmark suites")

    board = subparsers.add_parser("leaderboard", help="show the leaderboard for a task")
    board.add_argument("task_id", type=int)

    show = subparsers.add_parser("show", help="show one entity")
    show.add_argument("entity", choices=["dataset", "task", "suite"])
    show.add_argument("identifier")

    args = parser.parse_args(argv)

    if args.command == "info":
        print(f"openqml {__version__}")
        for key, value in config.get_config_as_dict().items():
            print(f"  {key}: {value}")
    elif args.command == "datasets":
        _print(datasets.list_datasets())
    elif args.command == "tasks":
        _print(tasks.list_tasks())
    elif args.command == "suites":
        _print(study.list_suites())
    elif args.command == "leaderboard":
        _print(evaluations.leaderboard(args.task_id))
    elif args.command == "show":
        getter = {"dataset": datasets.get_dataset, "task": tasks.get_task,
                  "suite": study.get_suite}[args.entity]
        identifier = int(args.identifier) if args.identifier.isdigit() else args.identifier
        entity = getter(identifier)
        print(repr(entity))
        if getattr(entity, "description", ""):
            print()
            print(entity.description)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

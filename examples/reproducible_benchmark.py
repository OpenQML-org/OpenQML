"""Evaluate a local suite and optionally save result JSON with --output."""

import argparse
import json

import openqml as q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional path for one full comparison JSON")
    args = parser.parse_args()
    for dataset_id in q.get_suite("qml-cls-1"):
        comparison = q.compare_models(q.get_task(dataset_id), seeds=(1, 2, 3), epochs=30)
        print(dataset_id, json.dumps(comparison.summary(), indent=2))
        if args.output and dataset_id == "moons":
            comparison.save(args.output)


if __name__ == "__main__":
    main()

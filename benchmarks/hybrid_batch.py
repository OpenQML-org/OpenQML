"""Reproducible batch forward/backward benchmark against the same kernels in a scalar loop."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from openqml import __version__
from openqml.interfaces.torch import QuantumLayer

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="benchmarks/results_cpu.json")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--qubits", type=int, default=6)
parser.add_argument("--depth", type=int, default=2)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(19)
layer = QuantumLayer.hardware_efficient(args.qubits, args.depth, dtype=torch.float64)
x = torch.randn(args.batch, args.qubits, dtype=torch.float64, requires_grad=True)


def scalar():
    return torch.stack([layer(row) for row in x])


def vector():
    return layer(x)


def forward(fn):
    with torch.no_grad():
        return fn()


def train(fn):
    layer.zero_grad(set_to_none=True)
    x.grad = None
    loss = fn().square().mean()
    loss.backward()
    return layer.weights.grad.detach().clone(), x.grad.detach().clone()


def time_fn(fn):
    fn()
    durations = []
    for _ in range(args.repeats):
        start = perf_counter()
        fn()
        durations.append(perf_counter() - start)
    return median(durations)


torch.testing.assert_close(scalar(), vector(), atol=1e-11, rtol=1e-11)
g_scalar, x_scalar = train(scalar)
g_vector, x_vector = train(vector)
torch.testing.assert_close(g_scalar, g_vector, atol=1e-11, rtol=1e-11)
torch.testing.assert_close(x_scalar, x_vector, atol=1e-11, rtol=1e-11)
f_scalar = time_fn(lambda: forward(scalar))
f_vector = time_fn(lambda: forward(vector))
t_scalar = time_fn(lambda: train(scalar))
t_vector = time_fn(lambda: train(vector))
result = {
    "openqml": __version__,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": "cpu",
    "torch_threads": 1,
    "dtype": "float64",
    "batch": args.batch,
    "qubits": args.qubits,
    "depth": args.depth,
    "repeats": args.repeats,
    "statistic": "median",
    "baseline": "sequential scalar loop using the same current kernels",
    "scalar_forward_ms": f_scalar * 1000,
    "batched_forward_ms": f_vector * 1000,
    "forward_speedup": f_scalar / f_vector,
    "scalar_forward_backward_ms": t_scalar * 1000,
    "batched_forward_backward_ms": t_vector * 1000,
    "training_speedup": t_scalar / t_vector,
    "max_output_error": (scalar() - vector()).abs().max().item(),
    "max_weight_gradient_error": (g_scalar - g_vector).abs().max().item(),
    "max_input_gradient_error": (x_scalar - x_vector).abs().max().item(),
}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

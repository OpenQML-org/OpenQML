"""Compare saved complex tensor volume for ordinary and recomputed hybrid training.

This measures tensors retained by autograd, not process or device peak memory.
Run: python benchmarks/activation_memory.py
"""

import json

import torch

from openqml.interfaces.torch import QuantumLayer


def measure(layer, inputs):
    sizes = []

    def pack(value):
        if value.is_complex():
            sizes.append(value.numel() * value.element_size())
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = layer(inputs)
    gradients = torch.autograd.grad(output.square().mean(), (layer.weights, inputs))
    return output.detach(), tuple(g.detach() for g in gradients), sum(sizes)


if __name__ == "__main__":
    torch.manual_seed(7)
    ordinary = QuantumLayer.data_reuploading(6, 8, dtype=torch.float64, batch_chunk_size=4)
    recomputed = QuantumLayer.from_checkpoint(
        ordinary.checkpoint(), checkpoint_interval=8, batch_chunk_size=4
    )
    inputs = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    output, gradient, base_bytes = measure(ordinary, inputs)
    actual, actual_gradient, saved_bytes = measure(recomputed, inputs)
    torch.testing.assert_close(output, actual)
    for left, right in zip(gradient, actual_gradient, strict=True):
        torch.testing.assert_close(left, right)
    print(
        json.dumps(
            {
                "saved_complex_tensor_bytes": {"ordinary": base_bytes, "recomputed": saved_bytes},
                "reduction_fraction": 1 - saved_bytes / base_bytes,
                "outputs_and_gradients_match": True,
            },
            indent=2,
        )
    )

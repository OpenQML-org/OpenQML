"""Reuse a prepared imported circuit with named batches and derivative products."""

import numpy as np

from openqml import AdjointPlan, LayerSpec

spec = LayerSpec.from_program(
    "OPENQASM 3.0; input float feature; input float weight; qubit q; "
    "ry(sin(feature)+feature*weight+weight*weight) q;",
    ("feature",),
    framework="openqasm3",
    parser="builtin",
)
plan = AdjointPlan(spec.circuit, spec.observables, parameters=("feature", "weight"))
features, weight = np.linspace(-0.6, 0.6, 64), 0.3
bindings = {"feature": features, "weight": weight}
angle = np.sin(features) + features * weight + weight**2

values = plan.expectations(bindings, batch_chunk_size=16)
weight_gradients = plan.vjp(bindings, np.ones((64, 1)), parameters=("weight",))
directions = plan.jvp(bindings, np.tile([0.2, 0.1], (64, 1)))

np.testing.assert_allclose(values[:, 0], np.cos(angle), atol=1e-12)
np.testing.assert_allclose(
    weight_gradients[:, 0], -np.sin(angle) * (features + 2 * weight), atol=1e-12
)
np.testing.assert_allclose(
    directions[:, 0],
    -np.sin(angle) * (0.2 * (np.cos(features) + weight) + 0.1 * (features + 2 * weight)),
    atol=1e-12,
)

print("Readouts:", values.shape)
print("Selected weight gradients:", weight_gradients.shape)
print("Directional derivatives:", directions.shape)
print("All products match independent analytic derivatives.")

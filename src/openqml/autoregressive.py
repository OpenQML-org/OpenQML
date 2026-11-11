"""Streaming autoregression with explicit quantum carries and caller RNG state."""

from contextlib import nullcontext

import numpy as np

from .compute import memory_budget
from .errors import ResourceError, ValidationError
from .ir import integer
from .quantum_channels import _checked
from .quantum_generative import shifted_tokens
from .simulator import _TensorOps


def detach_quantum_state(state, *, backend="torch"):
    if isinstance(state, dict):
        return {name: detach_quantum_state(value, backend=backend) for name, value in state.items()}
    if backend == "torch":
        return state.detach()
    if backend == "jax":
        import jax

        return jax.lax.stop_gradient(state)
    if backend != "numpy":
        raise ValidationError("Unknown state backend.")
    return np.array(state, copy=True)


def sample_tokens(
    predict,
    classes,
    steps,
    *,
    batch_size=1,
    prompt=None,
    backend="numpy",
    like=None,
    seed=0,
    key=None,
    max_memory_mb=256,
):
    classes, steps = integer(classes, "classes", 2), integer(steps, "steps", 1)
    batch_size = integer(batch_size, "batch_size", 1)
    ops = _TensorOps(backend, like)
    xp = ops.xp
    if prompt is not None:
        prompt = (
            xp.as_tensor(prompt, device=ops.device) if backend == "torch" else xp.asarray(prompt)
        )
        if prompt.ndim != 2 or min(prompt.shape) < 1:
            raise ValidationError("Prompt requires nonempty token IDs[B,T].")
        batch_size = prompt.shape[0]
    if 16 * batch_size * (steps + classes) > memory_budget(max_memory_mb) * 2**20:
        raise ResourceError("Autoregressive output/features exceed max_memory_mb.")
    if backend == "jax" and key is None:
        raise ValidationError("JAX sampling requires an explicit random key.")
    if backend == "torch":
        generator = xp.Generator(device=ops.device).manual_seed(integer(seed, "seed", 0))
    elif backend == "numpy":
        generator = np.random.default_rng(seed)

    def features(ids):
        vocabulary = (
            xp.arange(classes, device=ops.device) if backend == "torch" else xp.arange(classes)
        )
        result = (
            (ids[..., None] == vocabulary).to(ops.real_dtype)
            if backend == "torch"
            else (ids[..., None] == vocabulary).astype(ops.real_dtype)
        )
        return result[:, None, :]

    def draw(probability, random_key):
        if probability.shape != (batch_size, 1, classes):
            raise ValidationError("Streaming predictor must return probabilities[B,1,K].")
        p = probability[:, 0]
        finite = (
            xp.all(xp.isfinite(p) & (p >= 0), dim=-1)
            if backend == "torch"
            else xp.all(xp.isfinite(p) & (p >= 0), axis=-1)
        )
        total = p.sum(dim=-1) if backend == "torch" else p.sum(axis=-1)
        checked = _checked(
            p,
            (finite & (xp.abs(total - 1) < 1e-4))[..., None],
            ops,
            "Sampling requires finite normalized nonnegative probabilities.",
        )
        if backend == "torch":
            return xp.multinomial(checked, 1, generator=generator)[:, 0]
        if backend == "numpy":
            return np.asarray([generator.choice(classes, p=row / row.sum()) for row in checked])
        import jax

        selected = jax.random.categorical(random_key, xp.log(checked), axis=-1)
        return xp.where(finite & (xp.abs(total - 1) < 1e-4), selected, -1)

    manager = xp.no_grad() if backend == "torch" else nullcontext()
    with manager:
        state = None
        initial = ops.array(np.zeros((batch_size, 1, classes)), real_value=True)
        if prompt is not None:
            _, state = predict(
                shifted_tokens(
                    prompt, classes, backend=backend, like=like, max_memory_mb=max_memory_mb
                ),
                state,
            )
            initial = features(prompt[:, -1])
        if backend == "jax":
            import jax

            keys = jax.random.split(key, steps)
            probability, state = predict(initial, state)
            first = draw(probability, keys[0])

            def advance(carry, rng):
                last, state = carry
                probability, state = predict(features(last), state)
                token = draw(probability, rng)
                return (token, state), token

            (_, state), remaining = jax.lax.scan(advance, (first, state), keys[1:])
            return xp.concatenate((first[:, None], remaining.swapaxes(0, 1)), axis=1), state
        output = []
        for _ in range(steps):
            probability, state = predict(initial, state)
            token = draw(probability, None)
            output.append(token)
            initial = features(token)
        return (xp.stack(output, dim=1) if backend == "torch" else xp.stack(output, axis=1)), state

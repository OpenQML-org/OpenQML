"""Native, accelerator-resident objectives for quantum actor--critic training.

No environment, rollout runner or simulator is imported by this module. Tensor
arguments select NumPy, PyTorch or JAX; targets are explicitly stop-gradient.
Time-major traces distinguish termination from time-limit truncation.
"""

import math
from typing import NamedTuple

import numpy as np

from .errors import ValidationError
from .parameters import real


class AdvantageTargets(NamedTuple):
    advantages: object
    returns: object


class VTraceTargets(NamedTuple):
    values: object
    advantages: object


class CategoricalStatistics(NamedTuple):
    log_probability: object
    entropy: object
    valid: object


class PPOLoss(NamedTuple):
    loss: object
    policy_loss: object
    value_loss: object
    entropy: object
    approximate_kl: object
    clip_fraction: object


class SACLoss(NamedTuple):
    loss: object
    actor_loss: object
    critic_loss: object
    temperature_loss: object
    td_error: object


class ActorCriticOutput(NamedTuple):
    policy: object
    value: object
    log_std: object = None


class _Ops:
    def __init__(self, like):
        module = type(like).__module__
        self.like = like
        if module.startswith("torch"):
            import torch

            self.name, self.xp = "torch", torch
        elif module.startswith(("jax", "jaxlib")):
            import jax.numpy as jnp

            self.name, self.xp = "jax", jnp
        else:
            self.name, self.xp = "numpy", np

    def array(self, value, *, boolean=False):
        if self.name == "torch":
            return self.xp.as_tensor(
                value,
                device=self.like.device,
                dtype=self.xp.bool if boolean else self.like.dtype,
            )
        return self.xp.asarray(value, dtype=bool if boolean else self.like.dtype)

    def stop(self, value):
        if self.name == "torch":
            return value.detach()
        if self.name == "jax":
            import jax

            return jax.lax.stop_gradient(value)
        return value

    def cat(self, values, axis=0):
        if self.name == "torch":
            return self.xp.cat(values, dim=axis)
        return self.xp.concatenate(values, axis=axis)

    def maximum(self, left, right):
        return self.xp.maximum(left, self.array(right))


def _floating(value):
    module = type(value).__module__
    if not module.startswith(("torch", "jax", "jaxlib")) and not isinstance(value, np.ndarray):
        value = np.asarray(value, dtype=float)
    if str(value.dtype).removeprefix("torch.") not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValidationError("RL inputs must be real floating tensors.")
    # Accumulate losses and traces in at least float32 under mixed precision.
    if str(value.dtype).removeprefix("torch.") in {"float16", "bfloat16"}:
        return (
            value.float() if type(value).__module__.startswith("torch") else value.astype("float32")
        )
    return value


def _same(ops, reference, *values):
    arrays = tuple(ops.array(value) for value in values)
    if any(value.shape != reference.shape for value in arrays):
        raise ValidationError("RL transition tensors must have identical shapes.")
    return arrays


def _unit(value, name):
    value = real(value)
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be in [0, 1].")
    return value


def _positive(value, name, *, zero=False):
    value = real(value)
    if value < 0 or (value == 0 and not zero):
        raise ValidationError(f"{name} must be {'nonnegative' if zero else 'positive'}.")
    return value


def _trace(rewards, values, next_values, terminated, truncated):
    rewards = _floating(rewards)
    if rewards.ndim < 1 or rewards.shape[0] < 1:
        raise ValidationError("Traces need a nonempty time axis.")
    ops = _Ops(rewards)
    rewards = ops.stop(rewards)
    values, next_values = (ops.stop(x) for x in _same(ops, rewards, values, next_values))
    term = ops.array(terminated, boolean=True)
    trunc = ops.xp.zeros_like(term) if truncated is None else ops.array(truncated, boolean=True)
    if term.shape != rewards.shape or trunc.shape != rewards.shape:
        raise ValidationError("Episode masks must match the transition shape.")
    return ops, rewards, values, next_values, ~term, ~(term | trunc)


def _affine_suffix(offset, multiplier, ops):
    """Parallel affine suffix scan, including zero multipliers at episode ends.

    Hillis--Steele composition takes log2(T) launches without division, a dense
    triangular matrix, CPU copies, or a serial Python loop over transitions.
    """
    if ops.name == "jax":
        import jax

        result, _ = jax.lax.associative_scan(
            lambda later, earlier: (earlier[0] + earlier[1] * later[0], earlier[1] * later[1]),
            (offset[::-1], multiplier[::-1]),
        )
        return result[::-1]
    size, distance = offset.shape[0], 1
    while distance < size:
        offset = ops.cat(
            (offset[:-distance] + multiplier[:-distance] * offset[distance:], offset[-distance:])
        )
        multiplier = ops.cat(
            (multiplier[:-distance] * multiplier[distance:], multiplier[-distance:])
        )
        distance *= 2
    return offset


def generalized_advantages(
    rewards, values, next_values, terminated, *, truncated=None, discount=0.99, trace_decay=0.95
):
    """GAE with per-transition next values, including final observations.

    Truncations bootstrap next_values but stop the trace. Supply final-observation
    values at autoreset boundaries, never values of the next episode's reset.
    """
    discount, trace_decay = _unit(discount, "discount"), _unit(trace_decay, "trace_decay")
    ops, r, v, nv, bootstrap, continuation = _trace(
        rewards, values, next_values, terminated, truncated
    )
    delta = r + discount * bootstrap * nv - v
    advantage = _affine_suffix(delta, discount * trace_decay * continuation, ops)
    return AdvantageTargets(advantage, advantage + v)


def vtrace_targets(
    rewards,
    values,
    next_values,
    terminated,
    target_log_probs,
    behavior_log_probs,
    *,
    truncated=None,
    discount=0.99,
    rho_clip=1.0,
    c_clip=1.0,
    policy_rho_clip=1.0,
):
    """IMPALA V-trace for stale asynchronous/QPU actor trajectories.

    Importance ratios and targets are detached. This corrects policy lag; it
    does not correct quantum measurement bias or arbitrary replay sampling.
    """
    discount = _unit(discount, "discount")
    rho_clip = _positive(rho_clip, "rho_clip")
    c_clip = _positive(c_clip, "c_clip")
    pg_clip = _positive(policy_rho_clip, "policy_rho_clip")
    if c_clip > rho_clip:
        raise ValidationError("c_clip must not exceed rho_clip.")
    ops, r, v, nv, bootstrap, continuation = _trace(
        rewards, values, next_values, terminated, truncated
    )
    target, behavior = (ops.stop(x) for x in _same(ops, r, target_log_probs, behavior_log_probs))
    log_ratio = target - behavior
    # Clip in log space before exponentiation, including very stale actors.
    rho = ops.xp.exp(ops.xp.clip(log_ratio, None, math.log(rho_clip)))
    c = ops.xp.exp(ops.xp.clip(log_ratio, None, math.log(c_clip)))
    pg = ops.xp.exp(ops.xp.clip(log_ratio, None, math.log(pg_clip)))
    error = _affine_suffix(
        rho * (r + discount * bootstrap * nv - v), discount * c * continuation, ops
    )
    following = ops.cat((error[1:], ops.xp.zeros_like(error[:1])))
    advantage = pg * (r + discount * bootstrap * (nv + continuation * following) - v)
    return VTraceTargets(v + error, advantage)


def categorical_statistics(logits, actions, *, action_mask=None):
    """Stable masked categorical log probabilities and entropy, without sampling.

    Invalid action indices, masked actions, empty rows or nonfinite allowed
    logits yield valid=False and NaN log_probability. The native flag remains
    on device so compiled learners need no host synchronization.
    """
    logits = _floating(logits)
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValidationError("Categorical logits need a nonempty action dimension.")
    ops, width = _Ops(logits), logits.shape[-1]
    xp = ops.xp
    mask = (
        xp.ones_like(logits, dtype=xp.bool if ops.name == "torch" else bool)
        if action_mask is None
        else ops.array(action_mask, boolean=True)
    )
    if mask.shape != logits.shape:
        raise ValidationError("Action masks must match logits.")
    # A finite placeholder makes invalid/empty rows differentiably well-defined.
    allowed = mask & xp.isfinite(logits)
    finite = xp.where(allowed, logits, ops.array(-float("inf")))
    maximum = xp.amax(finite, axis=-1, keepdims=True)
    maximum = xp.where(xp.isfinite(maximum), maximum, xp.zeros_like(maximum))
    exponent = xp.where(allowed, xp.exp(finite - maximum), xp.zeros_like(logits))
    normalizer = ops.maximum(xp.sum(exponent, axis=-1, keepdims=True), 1e-30)
    log_probs = finite - maximum - xp.log(normalizer)
    entropy = -xp.sum(
        (exponent / normalizer) * xp.where(allowed, log_probs, xp.zeros_like(logits)), axis=-1
    )
    if ops.name == "torch":
        actions = xp.as_tensor(actions, device=logits.device)
    else:
        actions = xp.asarray(actions)
    if actions.shape != logits.shape[:-1] or "int" not in str(actions.dtype):
        raise ValidationError("Actions must be integers matching the logits' leading shape.")
    index = xp.clip(actions, 0, width - 1)
    if ops.name == "torch":
        selected = xp.gather(log_probs, -1, index.long().unsqueeze(-1))[..., 0]
        selected_mask = xp.gather(mask, -1, index.long().unsqueeze(-1))[..., 0]
    else:
        selected = xp.take_along_axis(log_probs, index[..., None], axis=-1)[..., 0]
        selected_mask = xp.take_along_axis(mask, index[..., None], axis=-1)[..., 0]
    valid = (
        (actions >= 0)
        & (actions < width)
        & selected_mask
        & xp.all(~mask | xp.isfinite(logits), axis=-1)
    )
    return CategoricalStatistics(xp.where(valid, selected, ops.array(float("nan"))), entropy, valid)


def ppo_loss(
    log_probs,
    old_log_probs,
    advantages,
    values,
    returns,
    *,
    entropy=None,
    old_values=None,
    clip_ratio=0.2,
    value_clip=None,
    value_coefficient=0.5,
    entropy_coefficient=0.0,
    normalize_advantages=True,
):
    """Clipped PPO objective with detached rollout targets and optional value clipping."""
    log_probs = _floating(log_probs)
    ops, xp = _Ops(log_probs), _Ops(log_probs).xp
    old, advantage, values, returns = _same(
        ops, log_probs, old_log_probs, advantages, values, returns
    )
    old, advantage, returns = ops.stop(old), ops.stop(advantage), ops.stop(returns)
    clip = _positive(clip_ratio, "clip_ratio")
    vc = _positive(value_coefficient, "value_coefficient", zero=True)
    ec = _positive(entropy_coefficient, "entropy_coefficient", zero=True)
    if type(normalize_advantages) is not bool:
        raise ValidationError("normalize_advantages must be a boolean.")
    size = log_probs.numel() if ops.name == "torch" else log_probs.size
    if size == 0:
        raise ValidationError("PPO needs a nonempty batch.")
    if normalize_advantages:
        centered = advantage - xp.mean(advantage)
        advantage = centered / xp.sqrt(xp.mean(centered * centered) + 1e-8)
    log_ratio = log_probs - old
    ratio = xp.exp(log_ratio)
    policy = -xp.mean(xp.minimum(ratio * advantage, xp.clip(ratio, 1 - clip, 1 + clip) * advantage))
    value_error = (values - returns) ** 2
    if value_clip is not None:
        bound = _positive(value_clip, "value_clip")
        if old_values is None:
            raise ValidationError("Value clipping requires old_values.")
        previous = ops.stop(_same(ops, log_probs, old_values)[0])
        clipped = previous + xp.clip(values - previous, -bound, bound)
        value_error = xp.maximum(value_error, (clipped - returns) ** 2)
    value = 0.5 * xp.mean(value_error)
    mean_entropy = ops.array(0.0) if entropy is None else xp.mean(_same(ops, log_probs, entropy)[0])
    kl = xp.mean((ratio - 1) - log_ratio)
    fraction = xp.mean(ops.array(xp.abs(ratio - 1) > clip))
    return PPOLoss(
        policy + vc * value - ec * mean_entropy, policy, value, mean_entropy, kl, fraction
    )


def squashed_gaussian(mean, log_std, noise):
    """Reparameterized tanh-Normal action and stable transformed log density.

    noise is caller-owned standard Gaussian noise; no hidden RNG or device
    transfer. Actions live in (-1,1); external affine bounds need their own
    log-Jacobian correction when densities are used outside this action space.
    """
    mean = _floating(mean)
    ops, xp = _Ops(mean), _Ops(mean).xp
    log_std, noise = _same(ops, mean, log_std, noise)
    if mean.ndim < 1 or not mean.shape[-1]:
        raise ValidationError("Gaussian actions need a nonempty event dimension.")
    latent = mean + xp.exp(log_std) * noise
    normal = -0.5 * (noise * noise + math.log(2 * math.pi)) - log_std
    # log(1-tanh(z)^2) = 2*(log(2)-z-softplus(-2z)).
    softplus = xp.logaddexp(xp.zeros_like(latent), -2 * latent)
    correction = 2 * (math.log(2) - latent - softplus)
    return xp.tanh(latent), xp.sum(normal - correction, axis=-1)


def sac_loss(
    log_probs,
    actor_q1,
    actor_q2,
    q1,
    q2,
    rewards,
    next_q1,
    next_q2,
    next_log_probs,
    terminated,
    log_temperature,
    *,
    discount=0.99,
    target_entropy=None,
):
    """Twin-critic maximum-entropy SAC losses, with separated gradient ownership.

    actor_q* must be computed with critic parameters frozen while preserving
    gradients to actions. Critic targets, actor temperature and temperature
    policy densities are detached here. Truncations intentionally bootstrap.
    """
    log_probs = _floating(log_probs)
    ops, xp = _Ops(log_probs), _Ops(log_probs).xp
    aq1, aq2, q1, q2, rewards, nq1, nq2, nlp = _same(
        ops, log_probs, actor_q1, actor_q2, q1, q2, rewards, next_q1, next_q2, next_log_probs
    )
    term = ops.array(terminated, boolean=True)
    if term.shape != log_probs.shape:
        raise ValidationError("Terminations must match the SAC transition shape.")
    log_alpha = ops.array(log_temperature)
    if log_alpha.ndim != 0:
        raise ValidationError("log_temperature must be a scalar tensor.")
    alpha = ops.stop(xp.exp(log_alpha))
    target = ops.stop(
        rewards + _unit(discount, "discount") * (~term) * (xp.minimum(nq1, nq2) - alpha * nlp)
    )
    actor = xp.mean(alpha * log_probs - xp.minimum(aq1, aq2))
    td = xp.stack((q1 - target, q2 - target))
    critic = 0.5 * xp.mean(xp.sum(td * td, axis=0))
    temperature = (
        ops.array(0.0)
        if target_entropy is None
        else -xp.mean(log_alpha * ops.stop(log_probs + real(target_entropy)))
    )
    return SACLoss(actor + critic + temperature, actor, critic, temperature, ops.stop(td))


def surrogate_fit_loss(predictions, measurements, *, binary_logits=False):
    """Fit a readout surrogate on detached control/measurement replay pairs.

    Default: MSE for Pauli expectations. binary_logits=True: stable BCE logits
    for aggregated binary features, as in arXiv:2503.09119 (v7, October 2026).
    """
    predictions = _floating(predictions)
    ops, xp = _Ops(predictions), _Ops(predictions).xp
    targets = ops.stop(_same(ops, predictions, measurements)[0])
    if type(binary_logits) is not bool:
        raise ValidationError("binary_logits must be a boolean.")
    if binary_logits:
        return xp.mean(
            xp.logaddexp(xp.zeros_like(predictions), predictions) - predictions * targets
        )
    return xp.mean((predictions - targets) ** 2)

"""JAX quantum PPO: batched environments, compiled learners and Optax state."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..errors import ValidationError
from ..rl import categorical_statistics, squashed_gaussian_log_probability
from ..rl_training import PPOTrainingLoop
from ._jax_training import make_train_step


class PPOTrainer(PPOTrainingLoop):
    """Functional-policy counterpart of the PyTorch training loop.

    Parameters and optimizer state stay JAX PyTrees. JIT/scan microbatches bound
    activation retention. Checkpoints use NPZ arrays and JSON tree structure,
    with no pickle; environment trajectories restart on loading.
    """

    def __init__(
        self,
        model,
        parameters,
        optimizer,
        environments,
        *,
        config=None,
        seed=0,
        optimizer_state=None,
    ):
        self.model, self.parameters, self.optimizer = model, parameters, optimizer
        self.optimizer_state = (
            optimizer.init(parameters) if optimizer_state is None else optimizer_state
        )
        self._initialize_loop(
            environments, model.observation_size, model.action_size, model.continuous, config, seed
        )
        self.key = jax.random.PRNGKey(seed)
        self.seed = seed
        self.scale = jnp.asarray((self.environments.high - self.environments.low) / 2)
        self.offset = jnp.asarray((self.environments.high + self.environments.low) / 2)
        self._apply = getattr(model, "apply", model)
        self._policy = jax.jit(self._apply)
        self._learn = make_train_step(
            optimizer,
            self._objective,
            microbatch_size=self.config.microbatch_size,
            max_grad_norm=self.config.max_grad_norm,
            has_aux=True,
        )

    def _sample(self, observations, masks, deterministic=False):
        output = self._policy(self.parameters, jnp.asarray(observations))
        self.key, key = jax.random.split(self.key) if not deterministic else (self.key, self.key)
        if self.continuous:
            noise = (
                jnp.zeros_like(output.policy)
                if deterministic
                else jax.random.normal(key, output.policy.shape)
            )
            raw = output.policy + jnp.exp(output.log_std) * noise
            actions = jnp.tanh(raw) * self.scale + self.offset
            log_probs = (
                squashed_gaussian_log_probability(output.policy, output.log_std, raw)
                - jnp.log(self.scale).sum()
            )
        else:
            masked = jnp.where(jnp.asarray(masks), output.policy, -jnp.inf)
            raw = jnp.argmax(masked, -1) if deterministic else jax.random.categorical(key, masked)
            actions = raw
            log_probs = categorical_statistics(
                output.policy, raw, action_mask=masks
            ).log_probability
        return tuple(np.asarray(x) for x in (raw, actions, log_probs, output.value))

    def _value(self, observations):
        return np.asarray(self._policy(self.parameters, jnp.asarray(observations)).value)

    def _objective(self, parameters, batch):
        config = self.config
        output = self._apply(parameters, batch["observations"])
        if self.continuous:
            log_probs = (
                squashed_gaussian_log_probability(output.policy, output.log_std, batch["actions"])
                - jnp.log(self.scale).sum()
            )
            latent = output.policy + jnp.exp(output.log_std) * batch["noise"]
            entropy = (
                -squashed_gaussian_log_probability(output.policy, output.log_std, latent)
                + jnp.log(self.scale).sum()
            )
        else:
            statistics = categorical_statistics(
                output.policy, batch["actions"], action_mask=batch["action_masks"]
            )
            log_probs, entropy = statistics.log_probability, statistics.entropy
        log_ratio = log_probs - batch["log_probs"]
        ratio = jnp.exp(log_ratio)
        policy = -jnp.minimum(
            ratio * batch["advantages"],
            jnp.clip(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch["advantages"],
        )
        value = (output.value - batch["returns"]) ** 2
        if config.value_clip is not None:
            clipped = batch["values"] + jnp.clip(
                output.value - batch["values"], -config.value_clip, config.value_clip
            )
            value = jnp.maximum(value, (clipped - batch["returns"]) ** 2)
        value = 0.5 * value
        loss = policy + config.value_coefficient * value - config.entropy_coefficient * entropy
        return loss, {
            "policy_loss": policy,
            "value_loss": value,
            "entropy": entropy,
            "approximate_kl": (ratio - 1) - log_ratio,
            "clip_fraction": (jnp.abs(ratio - 1) > config.clip_ratio).astype(jnp.float32),
        }

    def _update(self, batch):
        count, config = len(batch["observations"]), self.config
        totals, weight, stop = {}, 0, False
        for _ in range(config.epochs):
            self.key, key = jax.random.split(self.key)
            indices = np.asarray(jax.random.permutation(key, count))
            for start in range(0, count, config.minibatch_size):
                rows = indices[start : start + config.minibatch_size]
                mini = {name: jnp.asarray(value[rows]) for name, value in batch.items()}
                if self.continuous:
                    self.key, key = jax.random.split(self.key)
                    mini["noise"] = jax.random.normal(key, (len(rows), self.action_size))
                self.parameters, self.optimizer_state, metrics = self._learn(
                    self.parameters, self.optimizer_state, mini
                )
                metrics = {name: np.asarray(value).item() for name, value in metrics.items()}
                for name, value in metrics.items():
                    if name not in {"samples", "skipped"}:
                        totals[name] = totals.get(name, 0.0) + len(rows) * value
                weight += len(rows)
                self.updates += not metrics["skipped"]
                stop = metrics["skipped"] or (
                    config.target_kl is not None and metrics["approximate_kl"] > config.target_kl
                )
                if stop:
                    break
            if stop:
                break
        return {name: value / max(1, weight) for name, value in totals.items()} | {
            "early_stop": stop
        }

    def save(self, path):
        # A matching, initialized trainer supplies the trusted PyTree structure
        # on loading; only numbered array leaves are stored in the archive.
        leaves = jax.tree_util.tree_leaves((self.parameters, self.optimizer_state, self.key))
        metadata = {
            "schema": "openqml.jax_ppo",
            "version": 1,
            "config": asdict(self.config),
            "timesteps": self.timesteps,
            "updates": self.updates,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "continuous": self.continuous,
            "low": self.environments.low.tolist(),
            "high": self.environments.high.tolist(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(handle, "wb") as stream:
                np.savez_compressed(
                    stream,
                    metadata=np.asarray(json.dumps(metadata)),
                    **{f"leaf_{i}": np.asarray(value) for i, value in enumerate(leaves)},
                )
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def load(self, path):
        reference, structure = jax.tree_util.tree_flatten(
            (self.parameters, self.optimizer_state, self.key)
        )
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if (
                metadata.get("schema") != "openqml.jax_ppo"
                or metadata.get("version") != 1
                or metadata["config"] != asdict(self.config)
                or metadata["observation_size"] != self.observation_size
                or metadata["action_size"] != self.action_size
                or metadata["continuous"] != self.continuous
                or metadata["low"] != self.environments.low.tolist()
                or metadata["high"] != self.environments.high.tolist()
                or set(archive.files) != {"metadata"} | {f"leaf_{i}" for i in range(len(reference))}
            ):
                raise ValidationError("JAX PPO checkpoint configuration differs.")
            leaves = [archive[f"leaf_{i}"] for i in range(len(reference))]
            if any(
                value.shape != old.shape or value.dtype != old.dtype
                for value, old in zip(leaves, reference, strict=True)
            ):
                raise ValidationError("JAX PPO checkpoint array shapes/dtypes differ.")
            leaves = [
                jax.device_put(value, getattr(old, "sharding", None))
                for value, old in zip(leaves, reference, strict=True)
            ]
        self.parameters, self.optimizer_state, self.key = jax.tree_util.tree_unflatten(
            structure, leaves
        )
        self.timesteps, self.updates = metadata["timesteps"], metadata["updates"]
        self.environments.reset(seed=self.seed)
        return self

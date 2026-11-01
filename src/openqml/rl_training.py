"""Environment-independent PPO configuration and complete rollout orchestration.

Environments follow reset(seed=...) -> (observation, info) and step(action) ->
(observation, reward, terminated, truncated, info). No Gymnasium dependency or
bundled environment is required. Framework learners own all trainable state.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .rl import generalized_advantages


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 64
    microbatch_size: int = 32
    discount: float = 0.99
    trace_decay: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    max_rollout_mb: float = 64

    def __post_init__(self):
        for name in ("rollout_steps", "epochs", "minibatch_size", "microbatch_size"):
            integer(getattr(self, name), name, 1)
        for name in ("discount", "trace_decay"):
            if not 0 <= real(getattr(self, name)) <= 1:
                raise ValidationError(f"{name} must lie in [0, 1].")
        for name in ("clip_ratio", "max_grad_norm", "max_rollout_mb"):
            if real(getattr(self, name)) <= 0:
                raise ValidationError(f"{name} must be positive.")
        for name in ("value_clip", "target_kl"):
            value = getattr(self, name)
            if value is not None and real(value) <= 0:
                raise ValidationError(f"{name} must be positive or None.")
        for name in ("value_coefficient", "entropy_coefficient"):
            if real(getattr(self, name)) < 0:
                raise ValidationError(f"{name} must be nonnegative.")


class EnvironmentBatch:
    def __init__(self, environments, observation_size, action_size, continuous, *, seed=0):
        if callable(environments) or (
            hasattr(environments, "step") and hasattr(environments, "reset")
        ):
            environments = [environments]
        if isinstance(environments, (str, bytes)):
            raise ValidationError(
                "Provide an environment or nonempty sequence of factories/environments."
            )
        values = tuple(environments)
        if not values:
            raise ValidationError("At least one environment is required.")
        self.owned = [callable(value) for value in values]
        self.envs = [
            value() if owned else value for value, owned in zip(values, self.owned, strict=True)
        ]
        if len({id(env) for env in self.envs}) != len(self.envs):
            raise ValidationError("Each rollout stream needs an independent environment instance.")
        if any(
            not callable(getattr(e, "step", None)) or not callable(getattr(e, "reset", None))
            for e in self.envs
        ):
            raise ValidationError("Each environment needs reset and step methods.")
        self.observation_size, self.action_size = observation_size, action_size
        self.continuous = continuous
        self.low, self.high = np.full(action_size, -1.0), np.full(action_size, 1.0)
        if continuous:
            bounds = [
                (
                    np.asarray(getattr(getattr(e, "action_space", None), "low", self.low)),
                    np.asarray(getattr(getattr(e, "action_space", None), "high", self.high)),
                )
                for e in self.envs
            ]
            bounds = [
                (np.broadcast_to(a, (action_size,)), np.broadcast_to(b, (action_size,)))
                for a, b in bounds
            ]
            self.low, self.high = (np.broadcast_to(x, (action_size,)).copy() for x in bounds[0])
            if (
                not np.isfinite(self.low).all()
                or not np.isfinite(self.high).all()
                or np.any(self.high <= self.low)
            ):
                raise ValidationError("Continuous PPO requires finite, ordered action bounds.")
            if any(
                not np.array_equal(a, self.low) or not np.array_equal(b, self.high)
                for a, b in bounds
            ):
                raise ValidationError(
                    "Batched environments must share their continuous action bounds."
                )
        self.returns = np.zeros(len(self.envs))
        self.lengths = np.zeros(len(self.envs), dtype=np.int64)
        self.reset(seed=seed)

    def _observation(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.observation_size,) or not np.isfinite(value).all():
            raise ValidationError("Environment observations must be finite feature vectors.")
        return value.copy()

    def reset(self, *, seed=None):
        values = [
            env.reset(seed=None if seed is None else seed + i) for i, env in enumerate(self.envs)
        ]
        self.observations = np.stack([self._observation(value[0]) for value in values])
        self.infos = [dict(value[1]) for value in values]
        self.returns.fill(0)
        self.lengths.fill(0)

    @property
    def masks(self):
        if self.continuous:
            return None
        result = np.asarray(
            [info.get("action_mask", np.ones(self.action_size, dtype=bool)) for info in self.infos],
            dtype=bool,
        )
        if result.shape != (len(self.envs), self.action_size) or not result.any(axis=-1).all():
            raise ValidationError("Every discrete state needs a nonempty action mask.")
        return result

    def step(self, actions):
        actions = np.asarray(actions)
        shape = (len(self.envs), self.action_size) if self.continuous else (len(self.envs),)
        if (
            actions.shape != shape
            or actions.dtype.kind not in "iuf"
            or not np.isfinite(actions).all()
        ):
            raise ValidationError("Policy actions have an invalid shape or numeric value.")
        if self.continuous:
            epsilon = np.finfo(actions.dtype).eps if actions.dtype.kind == "f" else 0
            tolerance = 8 * epsilon * np.maximum(1, np.maximum(abs(self.low), abs(self.high)))
            if np.any(actions < self.low - tolerance) or np.any(actions > self.high + tolerance):
                raise ValidationError("Continuous policy actions exceed environment bounds.")
            actions = np.clip(actions, self.low, self.high)
        elif (
            np.any(actions < 0)
            or np.any(actions >= self.action_size)
            or np.any(actions != np.floor(actions))
            or not self.masks[np.arange(len(self.envs)), actions.astype(np.int64)].all()
        ):
            raise ValidationError("Discrete policy selected an invalid or masked action.")
        rewards, terminated, truncated = [], [], []
        final_observations, episodes = [], []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            observation, reward, term, trunc, info = env.step(
                action if self.continuous else int(action)
            )
            if not isinstance(term, (bool, np.bool_)) or not isinstance(trunc, (bool, np.bool_)):
                raise ValidationError("Environment termination flags must be booleans.")
            reward = real(reward)
            info = dict(info)
            self.returns[i] += reward
            self.lengths[i] += 1
            final = self._observation(observation)
            if term or trunc:
                final_key = next(
                    (
                        key
                        for key in ("final_observation", "final_obs", "terminal_observation")
                        if key in info
                    ),
                    None,
                )
                if final_key is not None:
                    final = self._observation(info[final_key])
                    reset_info = dict(info.get("reset_info", {}))
                else:
                    observation, reset_info = env.reset()
                episodes.append({"return": float(self.returns[i]), "length": int(self.lengths[i])})
                self.returns[i], self.lengths[i] = 0, 0
                info = reset_info
            final_observations.append(self._observation(final))
            self.observations[i] = self._observation(observation)
            self.infos[i] = info
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
        return (
            np.stack(final_observations),
            np.asarray(rewards),
            np.asarray(terminated),
            np.asarray(truncated),
            episodes,
        )

    def close(self):
        for env, owned in zip(self.envs, self.owned, strict=True):
            if owned and callable(getattr(env, "close", None)):
                env.close()


def prepare_ppo_batch(rollout):
    targets = generalized_advantages(
        rollout["rewards"],
        rollout["values"],
        rollout["next_values"],
        rollout["terminated"],
        truncated=rollout["truncated"],
        discount=rollout["discount"],
        trace_decay=rollout["trace_decay"],
    )
    advantages = targets.advantages
    advantages = (advantages - advantages.mean()) / np.sqrt(advantages.var() + 1e-8)
    batch = {
        key: value.reshape((-1,) + value.shape[2:])
        for key, value in rollout.items()
        if key in {"observations", "actions", "log_probs", "values", "action_masks"}
    }
    batch["advantages"], batch["returns"] = advantages.reshape(-1), targets.returns.reshape(-1)
    return batch


class PPOTrainingLoop:
    """Framework learners implement _sample, _value and _update on batched tensors."""

    def _initialize_loop(
        self, environments, observation_size, action_size, continuous, config, seed
    ):
        integer(seed, "seed")
        self.config = PPOConfig() if config is None else config
        if not isinstance(self.config, PPOConfig):
            raise ValidationError("config must be a PPOConfig.")
        self.observation_size, self.action_size, self.continuous = (
            observation_size,
            action_size,
            continuous,
        )
        self.environments = EnvironmentBatch(
            environments, observation_size, action_size, continuous, seed=seed
        )
        self.timesteps = self.updates = 0

    def collect(self, steps=None):
        steps = self.config.rollout_steps if steps is None else integer(steps, "steps", 1)
        n, a, o = len(self.environments.envs), self.action_size, self.observation_size
        shapes = {
            "observations": (steps, n, o),
            "actions": (steps, n, a) if self.continuous else (steps, n),
            **{
                key: (steps, n)
                for key in (
                    "log_probs",
                    "values",
                    "rewards",
                    "next_values",
                    "terminated",
                    "truncated",
                )
            },
        }
        if not self.continuous:
            shapes["action_masks"] = (steps, n, a)
        dtypes = {
            key: (bool if key in {"terminated", "truncated", "action_masks"} else np.float64)
            for key in shapes
        }
        dtypes["observations"] = np.float32
        if not self.continuous:
            dtypes["actions"] = np.int64
        needed = sum(
            math.prod(shape) * np.dtype(dtypes[key]).itemsize for key, shape in shapes.items()
        )
        # Retain room for flattened targets, device minibatches and host copies.
        if 3 * needed > self.config.max_rollout_mb * 2**20:
            raise ResourceError("PPO rollout/workspace exceeds max_rollout_mb.")
        rollout = {key: np.empty(shape, dtype=dtypes[key]) for key, shape in shapes.items()}
        episodes = []
        for t in range(steps):
            masks = self.environments.masks
            rollout["observations"][t] = self.environments.observations
            if masks is not None:
                rollout["action_masks"][t] = masks
            raw, actions, log_probs, values = self._sample(self.environments.observations, masks)
            if (
                np.shape(log_probs) != (n,)
                or np.shape(values) != (n,)
                or not np.isfinite(log_probs).all()
                or not np.isfinite(values).all()
                or not np.isfinite(actions).all()
            ):
                raise ValidationError("Policy produced invalid actions/log probabilities/values.")
            final, rewards, term, trunc, ended = self.environments.step(actions)
            for key, value in (
                ("actions", raw),
                ("log_probs", log_probs),
                ("values", values),
                ("rewards", rewards),
                ("terminated", term),
                ("truncated", trunc),
                ("next_values", self._value(final)),
            ):
                if np.shape(value) != rollout[key][t].shape or not np.isfinite(value).all():
                    raise ValidationError(f"Invalid PPO transition field {key}.")
                rollout[key][t] = value
            episodes.extend(ended)
        self.timesteps += steps * n
        rollout["discount"], rollout["trace_decay"] = self.config.discount, self.config.trace_decay
        return rollout, episodes

    def train(self, total_timesteps, *, callback=None):
        integer(total_timesteps, "total_timesteps", 1)
        if callback is not None and not callable(callback):
            raise ValidationError("callback must be callable or None.")
        target, history = self.timesteps + total_timesteps, deque(maxlen=100)
        while self.timesteps < target:
            steps = min(
                self.config.rollout_steps,
                math.ceil((target - self.timesteps) / len(self.environments.envs)),
            )
            rollout, episodes = self.collect(steps)
            metrics = self._update(prepare_ppo_batch(rollout))
            metrics |= {"timesteps": self.timesteps, "updates": self.updates, "episodes": episodes}
            history.append(metrics)
            if callback is not None and callback(dict(metrics)) is False:
                break
        return list(history)

    def evaluate(self, environment, *, episodes=5, max_episode_steps=10_000, seed=0):
        integer(episodes, "episodes", 1)
        integer(max_episode_steps, "max_episode_steps", 1)
        env = EnvironmentBatch(
            [environment], self.observation_size, self.action_size, self.continuous, seed=seed
        )
        if self.continuous and (
            not np.array_equal(env.low, self.environments.low)
            or not np.array_equal(env.high, self.environments.high)
        ):
            raise ValidationError("Evaluation action bounds differ from training.")
        scores, lengths = [], []
        try:
            for _ in range(episodes * max_episode_steps):
                _, actions, _, _ = self._sample(env.observations, env.masks, deterministic=True)
                _, _, _, _, ended = env.step(actions)
                if ended:
                    scores.append(ended[0]["return"])
                    lengths.append(ended[0]["length"])
                    if len(scores) == episodes:
                        return {
                            "mean_return": float(np.mean(scores)),
                            "std_return": float(np.std(scores)),
                            "returns": scores,
                            "lengths": lengths,
                        }
                if env.lengths[0] >= max_episode_steps:
                    raise ResourceError("Evaluation episode exceeds max_episode_steps.")
        finally:
            env.close()
        raise ResourceError("Evaluation did not finish the requested episodes.")

    def close(self):
        self.environments.close()

"""Complete discrete and bounded-continuous quantum PPO training."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..errors import ValidationError
from ..ir import integer
from ..rl import categorical_statistics, ppo_loss, squashed_gaussian_log_probability
from ..rl_training import PPOTrainingLoop
from ._torch_training import HybridTrainer


class PPOTrainer(PPOTrainingLoop):
    """Collect, bootstrap, compute GAE, optimize, evaluate and resume a policy.

    A model follows QuantumActorCritic's output/size contract. Environment state
    is restarted on checkpoint loading; model, optimizer and sampling RNG resume.
    DDP users must use equal rollout sizes/configurations and coordinated calls.
    """

    def __init__(self, model, optimizer, environments, *, config=None, seed=0, amp_dtype=None):
        integer(seed, "seed")
        self.model = model
        base = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            seed = (seed + 1_000_003 * torch.distributed.get_rank(model.process_group)) % (2**63)
        self._initialize_loop(
            environments, base.observation_size, base.action_size, base.continuous, config, seed
        )
        self.learner = HybridTrainer(
            model,
            optimizer,
            microbatch_size=self.config.microbatch_size,
            max_grad_norm=self.config.max_grad_norm,
            amp_dtype=amp_dtype,
            train_mode=False,
        )
        self.device = self.learner.device
        self.dtype = self.learner.parameters[0].dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.seed = seed
        self.scale = torch.as_tensor(
            (self.environments.high - self.environments.low) / 2,
            device=self.device,
            dtype=self.dtype,
        )
        self.offset = torch.as_tensor(
            (self.environments.high + self.environments.low) / 2,
            device=self.device,
            dtype=self.dtype,
        )
        if self.learner.distributed:
            sizes = torch.tensor(len(self.environments.envs), device=self.device)
            small, large = sizes.clone(), sizes.clone()
            torch.distributed.all_reduce(
                small, op=torch.distributed.ReduceOp.MIN, group=self.learner.group
            )
            torch.distributed.all_reduce(
                large, op=torch.distributed.ReduceOp.MAX, group=self.learner.group
            )
            if small.item() != large.item():
                raise ValidationError(
                    "Distributed PPO requires equal environment counts on every rank."
                )

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _sample(self, observations, masks, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self._tensor(observations))
            if self.continuous:
                noise = (
                    torch.zeros_like(output.policy)
                    if deterministic
                    else torch.randn(
                        output.policy.shape,
                        device=self.device,
                        dtype=self.dtype,
                        generator=self.generator,
                    )
                )
                raw = output.policy + output.log_std.exp() * noise
                actions = torch.tanh(raw) * self.scale + self.offset
                log_probs = (
                    squashed_gaussian_log_probability(output.policy, output.log_std, raw)
                    - self.scale.log().sum()
                )
            else:
                mask = torch.as_tensor(masks, device=self.device, dtype=torch.bool)
                logits = output.policy.masked_fill(~mask, -float("inf"))
                raw = (
                    logits.argmax(-1)
                    if deterministic
                    else torch.multinomial(logits.softmax(-1), 1, generator=self.generator)[:, 0]
                )
                actions = raw
                log_probs = categorical_statistics(
                    output.policy, raw, action_mask=mask
                ).log_probability
            return tuple(x.detach().cpu().numpy() for x in (raw, actions, log_probs, output.value))

    def _value(self, observations):
        self.model.eval()
        with torch.no_grad():
            return self.model(self._tensor(observations)).value.detach().cpu().numpy()

    def _update(self, batch):
        config = self.config
        count = len(batch["observations"])
        totals, weight, stop = {}, 0, False
        for _ in range(config.epochs):
            permutation = (
                torch.randperm(count, generator=self.generator, device=self.device).cpu().numpy()
            )
            for start in range(0, count, config.minibatch_size):
                indices = permutation[start : start + config.minibatch_size]
                mini = {key: torch.as_tensor(value[indices]) for key, value in batch.items()}
                records = []

                def objective(model, data, records=records):
                    output = model(data["observations"].to(self.dtype))
                    if self.continuous:
                        log_probs = (
                            squashed_gaussian_log_probability(
                                output.policy, output.log_std, data["actions"].to(self.dtype)
                            )
                            - self.scale.log().sum()
                        )
                        noise = torch.randn(
                            output.policy.shape,
                            device=self.device,
                            dtype=self.dtype,
                            generator=self.generator,
                        )
                        latent = output.policy + output.log_std.exp() * noise
                        entropy = (
                            -squashed_gaussian_log_probability(
                                output.policy, output.log_std, latent
                            )
                            + self.scale.log().sum()
                        )
                    else:
                        statistics = categorical_statistics(
                            output.policy, data["actions"], action_mask=data["action_masks"]
                        )
                        log_probs, entropy = statistics.log_probability, statistics.entropy
                    losses = ppo_loss(
                        log_probs,
                        data["log_probs"],
                        data["advantages"],
                        output.value,
                        data["returns"],
                        entropy=entropy,
                        old_values=data["values"],
                        clip_ratio=config.clip_ratio,
                        value_clip=config.value_clip,
                        value_coefficient=config.value_coefficient,
                        entropy_coefficient=config.entropy_coefficient,
                        normalize_advantages=False,
                    )
                    records.append(
                        (
                            len(data["observations"]),
                            {
                                name: value.detach()
                                for name, value in losses._asdict().items()
                                if name != "loss"
                            },
                        )
                    )
                    return losses.loss

                result = self.learner.step(mini, objective)
                metrics = {
                    name: sum(n * values[name] for n, values in records).item() / len(indices)
                    for name in records[0][1]
                }
                metrics |= {"loss": result["loss"], "grad_norm": result["grad_norm"]}
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + len(indices) * value
                weight += len(indices)
                self.updates += not result["skipped"]
                kl = torch.tensor(metrics["approximate_kl"], device=self.device)
                if self.learner.distributed:
                    torch.distributed.all_reduce(
                        kl, op=torch.distributed.ReduceOp.MAX, group=self.learner.group
                    )
                stop = result["skipped"] or (
                    config.target_kl is not None and kl.item() > config.target_kl
                )
                if stop:
                    break
            if stop:
                break
        return {name: value / max(1, weight) for name, value in totals.items()} | {
            "early_stop": stop
        }

    def save(self, path):
        # Reuse the learner's atomic file writer and safe tensor-only state
        # format; PPO-specific state is stored as ordinary dictionaries/tensors.
        from ._torch_training import _atomic_save

        learner_state = self.learner._checkpoint_state()
        generators = [self.generator.get_state()]
        if self.learner.distributed:
            generators = [None] * self.learner.world_size
            torch.distributed.all_gather_object(
                generators, self.generator.get_state(), group=self.learner.group
            )
        if self.learner.rank != 0:
            return None
        state = {
            "schema": "openqml.ppo_trainer",
            "version": 1,
            "learner": learner_state,
            "config": asdict(self.config),
            "generator": self.generator.get_state(),
            "generators_by_rank": generators,
            "timesteps": self.timesteps,
            "updates": self.updates,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "continuous": self.continuous,
            "low": torch.as_tensor(self.environments.low, dtype=torch.float64),
            "high": torch.as_tensor(self.environments.high, dtype=torch.float64),
        }
        return _atomic_save(state, Path(path))

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if (
            state.get("schema") != "openqml.ppo_trainer"
            or state.get("version") != 1
            or state["config"] != asdict(self.config)
            or state["observation_size"] != self.observation_size
            or state["action_size"] != self.action_size
            or state["continuous"] != self.continuous
            or not np.array_equal(state["low"].cpu().numpy(), self.environments.low)
            or not np.array_equal(state["high"].cpu().numpy(), self.environments.high)
        ):
            raise ValidationError("PPO checkpoint configuration differs.")
        self.learner.load_state_dict(state["learner"])
        self.generator.set_state(state["generators_by_rank"][self.learner.rank].cpu())
        self.timesteps, self.updates = state["timesteps"], state["updates"]
        self.environments.reset(seed=self.seed)
        return self

"""Microbatched hybrid optimization, DDP sample weighting and resumable state."""

import contextlib
import os
import tempfile
from pathlib import Path

import torch

from ...compute._norms import global_norm
from ...errors import ValidationError
from ...ir.core import integer
from ...ir.parameters import real


def _leaves(batch):
    if isinstance(batch, torch.Tensor):
        return [batch]
    if isinstance(batch, dict):
        return [leaf for value in batch.values() for leaf in _leaves(value)]
    if isinstance(batch, (tuple, list)):
        return [leaf for value in batch for leaf in _leaves(value)]
    raise ValidationError("Batches must be tensors or nested dictionaries/lists/tuples of tensors.")


def _map(batch, function):
    if isinstance(batch, torch.Tensor):
        return function(batch)
    if isinstance(batch, dict):
        return {key: _map(value, function) for key, value in batch.items()}
    values = [_map(value, function) for value in batch]
    return type(batch)(*values) if hasattr(batch, "_fields") else type(batch)(values)


def _atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(state, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


class HybridTrainer:
    """A scalar-mean loss(model, microbatch) or per-example loss vector.

    Backward runs before the next microbatch forward, so activation memory is
    bounded by a microbatch rather than the effective batch. With an existing
    DistributedDataParallel wrapper, one synchronized backward and global sample
    weighting handle unequal local batch sizes. All ranks must call each step.
    Batch-dependent layers (e.g. BatchNorm) retain microbatch semantics.
    """

    def __init__(
        self,
        model,
        optimizer,
        *,
        microbatch_size=32,
        max_grad_norm=1.0,
        amp_dtype=None,
        scheduler=None,
        train_mode=True,
    ):
        if not isinstance(model, torch.nn.Module) or not isinstance(
            optimizer, torch.optim.Optimizer
        ):
            raise ValidationError("Provide a PyTorch module and optimizer.")
        self.microbatch_size = integer(microbatch_size, "microbatch_size", 1)
        if max_grad_norm is not None and real(max_grad_norm) <= 0:
            raise ValidationError("max_grad_norm must be positive or None.")
        if type(train_mode) is not bool:
            raise ValidationError("train_mode must be a boolean.")
        if amp_dtype not in (None, torch.bfloat16, torch.float16):
            raise ValidationError("amp_dtype must be None, torch.bfloat16 or torch.float16.")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.max_grad_norm, self.amp_dtype, self.train_mode = max_grad_norm, amp_dtype, train_mode
        owned = {id(p) for p in model.parameters()}
        self.parameters = list(
            {id(p): p for group in optimizer.param_groups for p in group["params"]}.values()
        )
        if not self.parameters or any(id(p) not in owned for p in self.parameters):
            raise ValidationError("Optimizer parameters must belong to this model.")
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in self.parameters):
            raise ValidationError("Use one local device per trainer/DDP process.")
        if amp_dtype == torch.float16 and self.device.type != "cuda":
            raise ValidationError("float16 gradient scaling requires CUDA; use bfloat16 on CPU.")
        self.scaler = None
        if amp_dtype == torch.float16:
            scaler = getattr(torch.amp, "GradScaler", None)
            self.scaler = scaler("cuda") if scaler is not None else torch.cuda.amp.GradScaler()
        self.distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self.group = model.process_group if self.distributed else None
        self.world_size = torch.distributed.get_world_size(self.group) if self.distributed else 1
        self.rank = torch.distributed.get_rank(self.group) if self.distributed else 0
        self.updates = self.attempts = 0

    def step(self, batch, loss_function):
        leaves = _leaves(batch)
        if not leaves or any(x.ndim == 0 for x in leaves):
            raise ValidationError("Every batch tensor needs a shared nonempty sample dimension.")
        count = leaves[0].shape[0]
        if count < 1 or any(x.shape[0] != count for x in leaves):
            raise ValidationError("Every batch tensor needs the same nonempty sample dimension.")
        if not callable(loss_function):
            raise ValidationError("loss_function must be callable.")
        self.model.train(self.train_mode)
        global_count = torch.tensor(count, dtype=torch.int64, device=self.device)
        if self.distributed:
            torch.distributed.all_reduce(global_count, group=self.group)
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        self.optimizer.zero_grad(set_to_none=True)
        self.attempts += 1
        try:
            for start in range(0, count, self.microbatch_size):
                stop = min(count, start + self.microbatch_size)
                microbatch = _map(
                    batch,
                    lambda x, start=start, stop=stop: x[start:stop].to(
                        self.device, non_blocking=True
                    ),
                )
                sync = (
                    self.model.no_sync()
                    if self.distributed and stop < count
                    else contextlib.nullcontext()
                )
                with sync:
                    with torch.autocast(
                        self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
                    ):
                        loss = loss_function(self.model, microbatch)
                        if (
                            not isinstance(loss, torch.Tensor)
                            or loss.ndim > 1
                            or (loss.ndim == 1 and loss.shape != (stop - start,))
                        ):
                            raise ValidationError(
                                "Loss must be a scalar mean or a per-example vector."
                            )
                        loss = (
                            loss if loss.dtype in (torch.float32, torch.float64) else loss.float()
                        ).mean()
                    total_loss += loss.detach().to(torch.float64) * (stop - start)
                    weighted = loss * (
                        (stop - start) * self.world_size / global_count.to(loss.dtype)
                    )
                    (weighted if self.scaler is None else self.scaler.scale(weighted)).backward()
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            gradients = [p.grad for p in self.parameters if p.grad is not None]
            if not gradients or any(g.is_sparse for g in gradients):
                raise ValidationError(
                    "HybridTrainer requires at least one dense parameter gradient."
                )
            diagnostics = self.gradient_diagnostics()
            with torch.no_grad():
                norm = global_norm(gradients, torch, "torch")
                if self.max_grad_norm is not None:
                    factor = torch.clamp(self.max_grad_norm / torch.where(norm > 0, norm, 1), max=1)
                    groups = {}
                    for gradient in gradients:
                        groups.setdefault(gradient.dtype, []).append(gradient)
                    for group in groups.values():
                        torch._foreach_mul_(group, factor.to(dtype=group[0].real.dtype))
            finite = torch.isfinite(total_loss) & torch.isfinite(norm)
            accepted = finite.to(torch.int32)
            if self.distributed:
                torch.distributed.all_reduce(
                    accepted, op=torch.distributed.ReduceOp.MIN, group=self.group
                )
                torch.distributed.all_reduce(total_loss, group=self.group)
            skipped = not bool(accepted.item())
            if not skipped:
                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.updates += 1
            elif self.scaler is not None:
                self.scaler.update(new_scale=self.scaler.get_scale() * 0.5)
            return {
                "loss": float((total_loss / global_count).item()),
                "grad_norm": float(norm.item()),
                "samples": int(global_count.item()),
                "skipped": skipped,
                "updates": self.updates,
                **diagnostics,
            }
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def gradient_diagnostics(self):
        return {}

    def state_dict(self):
        model = self.model.module if self.distributed else self.model
        return {
            "schema": "openqml.hybrid_trainer",
            "version": 1,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "updates": self.updates,
            "attempts": self.attempts,
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
        }

    def load_state_dict(self, state, *, restore_rng=True):
        if state.get("schema") != "openqml.hybrid_trainer" or state.get("version") != 1:
            raise ValidationError("Unsupported hybrid training checkpoint.")
        if (state["scaler"] is None) != (self.scaler is None) or (
            (state["scheduler"] is None) != (self.scheduler is None)
        ):
            raise ValidationError("Checkpoint scaler/scheduler configuration differs.")
        rng = state
        if restore_rng and "rng_by_rank" in state:
            if self.distributed and len(state["rng_by_rank"]) != self.world_size:
                raise ValidationError(
                    "Checkpoint DDP world size differs; set restore_rng=False for a new topology."
                )
            rng = state["rng_by_rank"][self.rank]
        model = self.model.module if self.distributed else self.model
        model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.updates, self.attempts = state["updates"], state["attempts"]
        if restore_rng:
            torch.random.set_rng_state(rng["torch_rng"].cpu())
            if rng["cuda_rng"] is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda_rng"].cpu(), self.device)
        return self

    def _checkpoint_state(self):
        state = self.state_dict()
        if self.distributed:
            gathered = [None] * self.world_size
            torch.distributed.all_gather_object(
                gathered, {key: state[key] for key in ("torch_rng", "cuda_rng")}, group=self.group
            )
            state["rng_by_rank"] = gathered
        return state

    def save(self, path):
        state = self._checkpoint_state()
        if self.rank != 0:
            return None
        return _atomic_save(state, path)

    def load(self, path, *, restore_rng=True):
        return self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True), restore_rng=restore_rng
        )

"""Array-based supervised training over the existing native network trainers."""

from copy import deepcopy
from importlib import import_module

import numpy as np

from .architecture import HybridArchitecture, QuantumCellSpec
from .compute import compute_context, get_compute_kernel
from .errors import MissingDependencyError, ValidationError
from .ir import integer
from .parameters import real


def _require(name, extra):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"This operation needs {name}. Run: python -m pip install 'openqml[{extra}]'."
        ) from exc


def _numpy_chunks(value):
    """Bound validation temporaries for memmaps and noncontiguous array views."""
    kernel = get_compute_kernel()
    per_element = 2 * value.dtype.itemsize + 2
    budget = kernel.admit(1, 8, itemsize=per_element)
    size = max(1, int(budget * 2**20) // per_element)
    return np.nditer(
        value,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=size,
    )


def _data(value, name, *, check_finite=True):
    module = type(value).__module__.split(".")[0]
    if module == "torch":
        xp = _require("torch", "torch")
        valid = not value.is_complex() and (not check_finite or bool(xp.isfinite(value).all()))
    elif module in {"jax", "jaxlib"}:
        xp = _require("jax.numpy", "jax")
        valid = not np.issubdtype(value.dtype, np.complexfloating) and (
            not check_finite or bool(xp.isfinite(value).all())
        )
    else:
        value = np.asarray(value)
        valid = value.dtype.kind in "fibu" and (
            not check_finite or all(np.isfinite(chunk).all() for chunk in _numpy_chunks(value))
        )
    if not valid or value.ndim == 0 or 0 in value.shape:
        raise ValidationError(f"{name} must be a nonempty finite real array.")
    return value


class Model:
    """Fit/predict interface with one native parameter and optimizer owner.

    Accepts Python/NumPy arrays or arrays from the selected framework. Native
    modules and trainers remain accessible for custom objectives and stateful
    execution. Training frameworks load on fit, never on construction.
    """

    def __init__(
        self,
        input_features=None,
        output_features=None,
        *,
        task="regression",
        framework="torch",
        hidden_features=16,
        depth=2,
        num_qubits=2,
        quantum_depth=1,
        lr=1e-3,
        seed=0,
        device=None,
        backend="auto",
        architecture=None,
    ):
        if task not in {"regression", "classification", "forecast"}:
            raise ValidationError("Model task must be regression, classification or forecast.")
        if framework not in {"torch", "jax", "flax"}:
            raise ValidationError("Model framework must be torch, jax or flax.")
        if architecture is not None and not isinstance(architecture, HybridArchitecture):
            raise ValidationError("architecture must be a HybridArchitecture.")
        if architecture is not None and task == "forecast":
            raise ValidationError("Forecast models create their own quantile architecture.")
        self.input_features = (
            (None if architecture is None else architecture.input_features)
            if input_features is None
            else integer(input_features, "input_features", 1)
        )
        self.output_features = (
            (None if architecture is None else architecture.output_features)
            if output_features is None
            else integer(output_features, "output_features", 1)
        )
        self.hidden_features = integer(hidden_features, "hidden_features", 1)
        self.depth = integer(depth, "depth", 1)
        self.num_qubits = integer(num_qubits, "num_qubits", 1)
        self.quantum_depth = integer(quantum_depth, "quantum_depth", 1)
        if task == "forecast" and self.num_qubits < 2:
            raise ValidationError("Forecast models need a memory qubit and an environment qubit.")
        self.lr, self.seed = real(lr), integer(seed, "seed")
        if self.lr <= 0:
            raise ValidationError("lr must be positive.")
        self.task, self.framework, self.device, self.backend = (
            task,
            framework,
            None if device == "auto" else device,
            backend,
        )
        self.architecture = architecture
        self.module = self.trainer = self._forecast = None
        self.history = []
        self.best_epoch = self.stopped_epoch = None
        self._rng = np.random.default_rng(self.seed)

    def _build(self, features, outputs, microbatch_size):
        _require("torch" if self.framework == "torch" else "jax", self.framework)
        if self.framework != "torch":
            _require("optax", "jax")
        if self.framework == "flax":
            _require("flax", "flax")
        if self.architecture is None:
            if self.task == "forecast":
                from .finance.models import MarketForecastPlan

                self._forecast = MarketForecastPlan(
                    features,
                    outputs,
                    hidden_features=self.hidden_features,
                    depth=self.depth,
                    memory_qubits=self.num_qubits - 1,
                    environment_qubits=1,
                    quantum_depth=self.quantum_depth,
                )
                self.architecture = self._forecast.architecture()
            else:
                self.architecture = HybridArchitecture(
                    features,
                    outputs,
                    hidden_features=self.hidden_features,
                    depth=self.depth,
                    quantum=QuantumCellSpec(num_qubits=self.num_qubits, depth=self.quantum_depth),
                )
        elif (self.architecture.input_features, self.architecture.output_features) != (
            features,
            outputs,
        ):
            raise ValidationError("Data dimensions differ from the supplied architecture.")
        parameters = self.architecture.initial_parameters(self.seed, dtype="float32")
        self.input_features, self.output_features = features, outputs
        if self.framework == "torch":
            torch = _require("torch", "torch")
            device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            devices = (
                [torch.cuda.current_device() if device.index is None else device.index]
                if device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                self.module = self.architecture.build(
                    "torch", device=device, backend=self.backend, dtype=torch.float32
                )
            with torch.no_grad():
                for name, parameter in self.module.values.items():
                    parameter.copy_(
                        torch.as_tensor(parameters[name], dtype=parameter.dtype, device=device)
                    )
            self.trainer = self.module.trainer(lr=self.lr, microbatch_size=microbatch_size)
        else:
            jax, optax = _require("jax", self.framework), _require("optax", "jax")
            options = {"backend": self.backend}
            self.module = self.architecture.build(self.framework, **options)
            parameters = jax.tree_util.tree_map(lambda value: jax.numpy.asarray(value), parameters)
            if self.device is not None:
                platform = (
                    "gpu"
                    if str(self.device).split(":")[0] in {"gpu", "cuda"}
                    else str(self.device).split(":")[0]
                )
                index = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
                try:
                    chosen = jax.devices(platform)[index]
                except (IndexError, RuntimeError) as exc:
                    raise ValidationError(
                        f"Requested JAX device is unavailable: {self.device}."
                    ) from exc
                parameters = jax.device_put(parameters, chosen)

            def loss(network, weights, batch):
                return self._loss(network(weights, batch[0]), batch[1])

            arguments = (
                {"variables": {"params": parameters}}
                if self.framework == "flax"
                else {"parameters": parameters}
            )
            self.trainer = self.module.trainer(
                optax.adam(self.lr), loss, microbatch_size=microbatch_size, **arguments
            )

    def _convert(self, value, *, labels=False):
        if self.framework == "torch":
            torch = _require("torch", "torch")
            like = next(self.module.parameters())
            if type(value).__module__.split(".")[0] in {"jax", "jaxlib"}:
                raise ValidationError("Pass Python/NumPy or PyTorch data to a PyTorch model.")
            if isinstance(value, np.ndarray) and (
                not value.flags.writeable or any(s < 0 for s in value.strides)
            ):
                value = value.copy()
            return torch.as_tensor(
                value, dtype=torch.int64 if labels else like.dtype, device=like.device
            )
        jax = _require("jax", self.framework)
        if type(value).__module__.split(".")[0] == "torch":
            raise ValidationError("Pass Python/NumPy or JAX data to a JAX/Flax model.")
        value = jax.numpy.asarray(value, dtype=jax.numpy.int32 if labels else jax.numpy.float32)
        like = next(iter(self.trainer.parameters.values()))
        return jax.device_put(value, like.device)

    def _loss(self, prediction, target):
        if self.task == "forecast":
            return self._forecast.loss(prediction, target)
        if self.task == "classification":
            if target.shape != prediction.shape[:-1]:
                raise ValidationError(
                    "Classification targets must match the non-class output axes."
                )
            if self.framework == "torch":
                logp = prediction.log_softmax(dim=-1)
                value = -logp.gather(-1, target[..., None])[..., 0]
            else:
                jax = _require("jax", self.framework)
                logp = jax.nn.log_softmax(prediction, axis=-1)
                value = -jax.numpy.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
        else:
            if target.shape == prediction.shape[:-1] and prediction.shape[-1] == 1:
                target = target[..., None]
            if target.shape != prediction.shape:
                raise ValidationError("Regression targets must match the model output shape.")
            value = (prediction - target) ** 2
        axes = tuple(range(1, value.ndim))
        if not axes:
            return value
        return value.mean(dim=axes) if self.framework == "torch" else value.mean(axis=axes)

    def _check_targets(self, x, y, outputs, *, features=None):
        features = self.input_features if features is None else features
        if (
            x.ndim < 2
            or len(x) != len(y)
            or x.shape[-1] != features
            or 0 in x.shape
            or 0 in y.shape
        ):
            raise ValidationError("Inputs and targets must match the model's sample/feature axes.")
        prefix = (
            x.shape[:1]
            if self.architecture is not None and self.architecture.readout_pooling != "none"
            else x.shape[:-1]
        )
        if self.task == "classification":
            if y.shape != prefix:
                raise ValidationError("Classification labels must match the input sample axes.")
            module = type(y).__module__.split(".")[0]
            xp = (
                _require("torch", "torch")
                if module == "torch"
                else _require("jax.numpy", self.framework)
                if module in {"jax", "jaxlib"}
                else np
            )
            if module == "torch":
                integral = not xp.is_floating_point(y) or bool((y == y.floor()).all())
            elif xp is np:
                integral = y.dtype.kind in "ibu" or all(
                    (chunk == np.floor(chunk)).all() for chunk in _numpy_chunks(y)
                )
            else:
                integral = bool((y == xp.floor(y)).all())
            if not integral or int(y.min()) < 0 or int(y.max()) >= outputs:
                raise ValidationError("Classification labels must be integers in the class range.")
        elif y.shape != prefix + (outputs,) and not (
            self.task == "regression" and outputs == 1 and y.shape == prefix
        ):
            raise ValidationError("Targets must match the model's sample and output axes.")

    def _snapshot(self):
        trainer = self.trainer
        if self.framework == "torch":
            state = deepcopy(trainer.state_dict())
        else:
            jax = _require("jax", self.framework)
            state = (
                jax.tree_util.tree_map(lambda x: x.copy(), trainer.parameters),
                jax.tree_util.tree_map(lambda x: x.copy(), trainer.optimizer_state),
                trainer.updates,
                trainer.attempts,
                trainer.model,
            )
        return state, deepcopy(self._rng.bit_generator.state)

    def _restore(self, snapshot):
        state, rng = snapshot
        if self.framework == "torch":
            self.trainer.load_state_dict(state)
        else:
            (
                self.trainer.parameters,
                self.trainer.optimizer_state,
                self.trainer.updates,
                self.trainer.attempts,
                self.trainer.model,
            ) = state
            self.trainer._compiled = self.trainer._compiled_model = None
        self._rng.bit_generator.state = rng

    def evaluate(self, inputs, targets, *, batch_size=256):
        """Mean task loss in bounded batches, with native evaluation semantics."""
        if self.trainer is None:
            raise ValidationError("Call fit before evaluate.")
        x, y = _data(inputs, "inputs"), _data(targets, "targets")
        self._check_targets(x, y, self.output_features)
        size, total = integer(batch_size, "batch_size", 1), 0.0

        def accumulate():
            result = 0.0
            for start in range(0, len(x), size):
                batch = self._convert(x[start : start + size])
                prediction = (
                    self.module(batch)
                    if self.framework == "torch"
                    else self.trainer.model(self.trainer.parameters, batch)
                )
                target = self._convert(
                    y[start : start + size], labels=self.task == "classification"
                )
                result += float(self._loss(prediction, target).mean()) * len(batch)
            return result

        if self.framework == "torch":
            torch = _require("torch", "torch")
            training = self.module.training
            self.module.eval()
            try:
                with torch.no_grad():
                    total = accumulate()
            finally:
                self.module.train(training)
        else:
            total = accumulate()
        value = total / len(x)
        if not np.isfinite(value):
            raise ValidationError("Evaluation loss must be finite.")
        return value

    def fit(
        self,
        inputs,
        targets,
        *,
        epochs=1,
        batch_size=32,
        microbatch_size=None,
        shuffle=False,
        validation_data=None,
        patience=None,
        min_delta=0,
        restore_best=True,
    ):
        """Train with optional held-out loss monitoring and full-state early stopping.

        patience counts epochs without a min_delta improvement. With patience
        and restore_best=True, restore parameters, optimizer, counters and RNG
        to the best validation epoch. History retains every attempted epoch.
        """
        x, y = _data(inputs, "inputs"), _data(targets, "targets")
        epochs, size = integer(epochs, "epochs", 1), integer(batch_size, "batch_size", 1)
        micro = size if microbatch_size is None else integer(microbatch_size, "microbatch_size", 1)
        minimum = real(min_delta)
        if minimum < 0 or type(restore_best) is not bool:
            raise ValidationError("min_delta must be nonnegative and restore_best boolean.")
        if patience is not None:
            patience = integer(patience, "patience", 1)
            if validation_data is None:
                raise ValidationError("Early stopping requires validation_data=(inputs, targets).")
        validation = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
                raise ValidationError("validation_data must contain inputs and targets.")
            validation = tuple(_data(v, "validation data") for v in validation_data)
        if x.ndim < 2 or x.shape[0] != y.shape[0] or type(shuffle) is not bool:
            raise ValidationError("Inputs need [...,features] and matching target sample counts.")
        features = self.input_features or x.shape[-1]
        if x.shape[-1] != features:
            raise ValidationError("Input feature width differs from the model.")
        if self.task == "classification":
            outputs = self.output_features or max(2, int(y.max()) + 1)
        else:
            outputs = self.output_features or (1 if y.ndim == 1 else y.shape[-1])
        # Check all labels before creating an optimizer or applying an update.
        self._check_targets(x, y, outputs, features=features)
        if validation is not None:
            self._check_targets(*validation, outputs, features=features)
        if self.trainer is None:
            self._build(features, outputs, micro)
        elif self.framework == "torch":
            self.trainer.microbatch_size = micro
        elif self.trainer.options.get("microbatch_size") != micro:
            self.trainer.options["microbatch_size"] = micro
            self.trainer._compiled_model = None
        best, stale, snapshot = float("inf"), 0, None
        self.best_epoch = self.stopped_epoch = None
        for _ in range(epochs):
            order = self._rng.permutation(len(x)) if shuffle else None
            total, accepted = 0.0, 0
            for start in range(0, len(x), size):
                indices = (
                    slice(start, min(start + size, len(x)))
                    if order is None
                    else order[start : start + size]
                )
                batch = (
                    self._convert(x[indices]),
                    self._convert(y[indices], labels=self.task == "classification"),
                )
                metrics = (
                    self.trainer.step(batch, lambda network, b: self._loss(network(b[0]), b[1]))
                    if self.framework == "torch"
                    else self.trainer.step(batch)
                )
                count = len(batch[0])
                total += float(metrics["loss"]) * count
                accepted += int(not bool(metrics["skipped"]))
            row = {"loss": total / len(x), "updates": accepted}
            if validation is not None:
                row["val_loss"] = self.evaluate(*validation, batch_size=size)
            self.history.append(row)
            if validation is not None:
                if row["val_loss"] < best - minimum:
                    best, stale = row["val_loss"], 0
                    self.best_epoch = len(self.history)
                    if patience is not None and restore_best:
                        snapshot = self._snapshot()
                else:
                    stale += 1
                if patience is not None and stale >= patience:
                    self.stopped_epoch = len(self.history)
                    break
        if snapshot is not None:
            self._restore(snapshot)
        return self

    def _iter_predict_raw(self, inputs, batch_size, *, output=None):
        if self.trainer is None:
            raise ValidationError("Call fit before predict.")
        x = _data(inputs, "inputs", check_finite=False)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != self.input_features:
            raise ValidationError("Input feature width differs from the fitted model.")
        size = integer(batch_size, "batch_size", 1)
        kernel = get_compute_kernel()
        for start in range(0, len(x), size):
            # Never retain a context override or eval/no_grad mode across yield.
            with compute_context(kernel=kernel):
                batch = _data(x[start : start + size], "inputs")
                if self.framework == "torch":
                    torch = _require("torch", "torch")
                    training = self.module.training
                    self.module.eval()
                    try:
                        with torch.no_grad():
                            value = self.module(self._convert(batch))
                            result = value.detach().cpu().numpy()
                    finally:
                        self.module.train(training)
                else:
                    value = self.trainer.model(self.trainer.parameters, self._convert(batch))
                    result = np.asarray(value)
                kernel.finite(result)
                if output is not None:
                    result = self._decode_prediction(result, output)
            yield result

    def _predict_raw(self, inputs, batch_size):
        # Eager calls validate all inputs before producing a result. The iterator
        # instead validates each batch as it is consumed.
        x = _data(inputs, "inputs")
        single = x.ndim == 1
        result = np.concatenate(list(self._iter_predict_raw(x, batch_size)))
        return result[0] if single else result

    def _decode_prediction(self, raw, output):
        if output == "probabilities":
            p = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
            return p / p.sum(axis=-1, keepdims=True)
        if output == "quantiles":
            return self._forecast.decode(raw).quantiles
        if self.task == "classification":
            return np.argmax(raw, axis=-1)
        if self.task == "forecast":
            return self._forecast.decode(raw).median
        return raw[..., 0] if self.output_features == 1 else raw

    def iter_predict(self, inputs, *, batch_size=256, output="prediction"):
        """Yield NumPy prediction batches without retaining the complete output.

        output is 'prediction', 'probabilities' (classification), or 'quantiles'
        (forecast). A single input retains a length-one batch axis. Validation
        is lazy: a later invalid batch can fail after earlier batches were used.
        The caller must keep inputs and model weights stable while iterating.
        """
        if (
            output not in {"prediction", "probabilities", "quantiles"}
            or (output == "probabilities" and self.task != "classification")
            or (output == "quantiles" and self.task != "forecast")
        ):
            raise ValidationError("Select an output supported by the model task.")
        yield from self._iter_predict_raw(inputs, batch_size, output=output)

    def predict(self, inputs, *, batch_size=256):
        return self._decode_prediction(self._predict_raw(inputs, batch_size), "prediction")

    def predict_proba(self, inputs, *, batch_size=256):
        if self.task != "classification":
            raise ValidationError("predict_proba requires a classification model.")
        return self._decode_prediction(self._predict_raw(inputs, batch_size), "probabilities")

    def predict_quantiles(self, inputs, *, batch_size=256):
        if self.task != "forecast":
            raise ValidationError("predict_quantiles requires a forecast model.")
        return self._decode_prediction(self._predict_raw(inputs, batch_size), "quantiles")

    def save(self, path):
        """Atomically save configuration, training state, history and shuffle RNG."""
        from ._model_checkpoint import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path, *, device=None, backend="auto", max_bytes=256 * 2**20):
        """Restore a fitted model, optionally selecting a new device/backend."""
        from ._model_checkpoint import load_model

        return load_model(cls, path, device=device, backend=backend, max_bytes=max_bytes)

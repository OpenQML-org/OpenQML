"""Bounded recovery of submitted provider jobs and QPU circuit breaking."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from inspect import signature
from sys import modules
from threading import RLock

from ..compute.kernel import compute_context, get_compute_kernel
from ..errors import BackendError, ValidationError
from ..ir.core import integer
from ..ir.parameters import real

_RECOVERY_CONTEXT = ContextVar("openqml_job_recovery", default=None)


@contextmanager
def recovery_context(policy):
    token = _RECOVERY_CONTEXT.set(policy)
    try:
        yield
    finally:
        _RECOVERY_CONTEXT.reset(token)


@dataclass(frozen=True)
class RecoveryPolicy:
    max_attempts: int = 3
    initial_delay: float = 0.05
    max_delay: float = 1
    max_elapsed: float = 30
    failure_threshold: int = 3
    cooldown_seconds: float = 30
    retry_exceptions: tuple = (TimeoutError, ConnectionError)

    def __post_init__(self):
        for name in ("max_attempts", "failure_threshold"):
            integer(getattr(self, name), name, 1)
        for name in ("initial_delay", "max_delay", "max_elapsed", "cooldown_seconds"):
            value = real(getattr(self, name))
            if value < 0 or (name == "max_elapsed" and value == 0):
                raise ValidationError(f"{name} must be nonnegative; max_elapsed must be positive.")
        if not isinstance(self.retry_exceptions, tuple) or any(
            not isinstance(cls, type) or not issubclass(cls, Exception)
            for cls in self.retry_exceptions
        ):
            raise ValidationError("retry_exceptions must contain exception classes.")

    def transient(self, error):
        seen = set()
        for _ in range(8):
            if error is None or id(error) in seen:
                break
            seen.add(id(error))
            status = getattr(error, "status_code", None)
            if status is None:
                status = getattr(getattr(error, "response", None), "status_code", None)
            # Qiskit/IBM Runtime wait timeouts inherit JobTimeoutError, not
            # Python's TimeoutError. Read the already loaded optional SDK;
            # terminal JobError/RuntimeJobMaxTimeoutError remain non-transient.
            qiskit_timeout = getattr(
                modules.get("qiskit.providers.exceptions"), "JobTimeoutError", ()
            )
            if (
                isinstance(error, self.retry_exceptions)
                or isinstance(error, qiskit_timeout)
                or status in {429, 500, 502, 503, 504}
            ):
                return True
            error = error.__cause__ or error.__context__
        return False


class BackendJobError(BackendError):
    """A submitted job failed to return a usable result; its receipt is retained."""

    def __init__(self, message, *, pending=None, retryable=False, stage="result"):
        super().__init__(message)
        self.pending, self.retryable, self.stage = pending, retryable, stage
        self.job_id = None if pending is None else pending.job_id

    def resume(self, *, loader=None):
        if self.pending is None:
            raise ValidationError("This failure has no submitted job receipt.")
        return self.pending.result(loader=loader)


class BackendUnavailableError(BackendError):
    """A provider was rejected before submission by its recovery state."""

    submitted = False


class PendingJob:
    """Retain one job, its decoder and metadata; never re-submit quantum work."""

    def __init__(self, job, decoder, *, backend, metadata, policy=None):
        if not callable(getattr(job, "result", None)):
            raise BackendJobError("Provider returned no usable job handle.", stage="submission")
        self.job, self.decoder, self.backend, self.metadata = job, decoder, backend, metadata
        self.kernel = get_compute_kernel()
        self.policy = (_RECOVERY_CONTEXT.get() or RecoveryPolicy()) if policy is None else policy
        if not isinstance(self.policy, RecoveryPolicy):
            raise ValidationError("recovery must be RecoveryPolicy.")
        identifier = getattr(job, "job_id", None)
        try:
            self.job_id = identifier() if callable(identifier) else identifier
        except Exception:
            self.job_id = None
        if self.job_id is not None:
            self.job_id = str(self.job_id)
        self._lock, self._complete, self._value = RLock(), False, None
        try:
            parameters = signature(job.result).parameters
            self._timeout = "timeout" in parameters
        except (ValueError, TypeError):
            self._timeout = False

    def to_dict(self):
        return {
            "schema": "openqml.provider_job.v1",
            "job_id": self.job_id,
            "backend": self.backend,
            "metadata": self.metadata,
        }

    def result(self, *, loader=None):
        with self._lock, compute_context(kernel=self.kernel):
            if not self._complete:
                self._value = self._result(loader=loader)
                self._complete = True
            return self._value

    def _result(self, *, loader=None):
        if loader is not None:
            if self.job_id is None:
                raise ValidationError("Provider did not expose a resumable job ID.")
            restored = loader(self.job_id)
            identifier = getattr(restored, "job_id", None)
            identifier = identifier() if callable(identifier) else identifier
            if identifier is not None and str(identifier) != self.job_id:
                raise ValidationError("Retrieved job ID differs from the retained receipt.")
            self.job = restored
            try:
                self._timeout = "timeout" in signature(restored.result).parameters
            except (TypeError, ValueError):
                self._timeout = False
        started, delay = time.monotonic(), self.policy.initial_delay
        for attempt in range(self.policy.max_attempts):
            try:
                remaining = max(0, self.policy.max_elapsed - (time.monotonic() - started))
                result = self.job.result(timeout=remaining) if self._timeout else self.job.result()
            except Exception as exc:
                transient = self.policy.transient(exc)
                remaining = self.policy.max_elapsed - (time.monotonic() - started)
                wait = min(delay, self.policy.max_delay)
                if not transient or attempt + 1 == self.policy.max_attempts or wait >= remaining:
                    raise BackendJobError(
                        f"Provider {self.backend!r} job {self.job_id!r} result failed: {exc}",
                        pending=self,
                        retryable=transient,
                    ) from exc
                if wait:
                    time.sleep(wait)
                delay *= 2
                continue
            try:
                return self.decoder(result, self.metadata)
            except Exception as exc:
                raise BackendJobError(
                    f"Provider {self.backend!r} job {self.job_id!r} returned invalid data: {exc}",
                    pending=self,
                    stage="decode",
                ) from exc
        raise AssertionError("Unreachable result recovery state.")


class PendingBatch:
    """Retain completed chunks and resume only known jobs or unsubmitted chunks."""

    job_id = None

    def __init__(self, calls, combine):
        self.calls, self.combine = tuple(calls), combine
        self.values, self.failures = [None] * len(calls), {}
        self.lock = RLock()

    def take(self, index, call):
        try:
            value = call()
        except Exception as exc:
            with self.lock:
                self.failures[index] = exc
        else:
            with self.lock:
                self.values[index] = value
                self.failures.pop(index, None)

    def assemble(self):
        if self.failures:
            ids = [getattr(exc, "job_id", None) for exc in self.failures.values()]
            recoverable = all(
                (isinstance(exc, BackendJobError) and exc.pending is not None)
                or isinstance(exc, BackendUnavailableError)
                for exc in self.failures.values()
            )
            raise BackendJobError(
                f"Provider batch has {len(self.failures)} unresolved chunks; job IDs: {ids}.",
                pending=self,
                retryable=recoverable,
                stage="batch",
            )
        return self.combine(self.values)

    def result(self, *, loader=None):
        with self.lock:
            for index, error in tuple(self.failures.items()):
                if isinstance(error, BackendJobError) and error.pending is not None:
                    self.take(index, lambda error=error: error.resume(loader=loader))
                elif isinstance(error, BackendUnavailableError):
                    self.take(index, self.calls[index])
                # Unknown submission state is retained, never replayed.
            return self.assemble()


class CircuitBreaker:
    def __init__(self, policy):
        self.policy, self.lock = policy, RLock()
        self.failures, self.open_until, self.probing = 0, 0.0, False

    def available(self):
        with self.lock:
            return time.monotonic() >= self.open_until and not self.probing

    def acquire(self):
        with self.lock:
            if time.monotonic() < self.open_until or self.probing:
                raise BackendUnavailableError("QPU circuit breaker is open; no job was submitted.")
            if self.open_until:
                self.probing = True

    def success(self):
        with self.lock:
            self.failures, self.open_until, self.probing = 0, 0.0, False

    def failure(self):
        with self.lock:
            self.failures += 1
            if self.probing or self.failures >= self.policy.failure_threshold:
                self.open_until = time.monotonic() + self.policy.cooldown_seconds
            self.probing = False

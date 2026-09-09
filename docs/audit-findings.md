# Open audit findings

Issues found in the audit behind commits `153e688` and `6f7556a` that were
**not** fixed there — each is real and reproduced unless marked otherwise, but
either larger than that change or needing a decision first. Listed so they are
not rediscovered from scratch.

Twelve findings *were* fixed; see those two commits and the tests in
`tests/test_security.py`, `tests/test_metrics.py`.

## Optimiser

**`adam(return_best=True)` compares losses from different mini-batches.**
`models/base.py:124`. `best_loss` is updated from `gradient_fn(theta)`, which
under mini-batching sees a different random subset each step, so "best" can mean
"drew the easiest batch". `VariationalQuantumRegressor` defaults to
`batch_size=24`, so this is the default path. Fixing it means either evaluating
the full-batch loss for the comparison (costly) or returning the last iterate
and documenting it.

**`adam` never evaluates its final update.** Same file. Loss and gradient are
taken at `theta`, `theta` is updated, and the loop ends — so `maxiter=1` returns
the initial weights untouched. There is also no convergence exit at all: no
`tol`, no gradient-norm check. A NaN loss silently returns the initial weights,
because `if loss < best_loss` is never true for NaN.

## Preprocessing

**Whether to rescale is decided per fold.** `models/base.py:84`. The scaler is
correctly fit on the training fold only — there is no leakage, traced end to end
— but the *decision* whether a scaler exists at all depends on that fold's
range. A column in `[-1, 1]` except for one outlier row gets min-max squashed in
the four folds that train on the outlier and left alone in the fold that holds
it out, so one run fits models on two different representations and reports the
difference as model variance.

**Unscaled test values are not clipped.** `models/base.py:90`. The docstring
says unseen values are clipped rather than wrapped, but when no scaler was
fitted the value passes straight through: with `angle_embedding`'s `scale=π`,
a test point at 7.0 becomes 7π ≡ π, encoded identically to 1.0.

## Models

**`encode_states` ignores `feature_map` for complex input.** `models/kernel.py:48`
returns amplitude-encoded states whenever `np.iscomplexobj(X)`. Every
`StateClassificationTask` yields complex statevectors, so on task 3 every
`QuantumKernelClassifier` configuration collapses to the same kernel — and
`compare()` prints them as separate rows, distinctly labelled, carrying
identical scores. `n_qubits` is ignored on that branch too.

**`alpha * I` does not regularise a non-PSD kernel.** `models/kernel.py:99`. The
quantum fidelity Gram is PSD so the quantum arm is safe, but the classical
`poly` kernel with a negative `coef0` is indefinite: eigenvalues around −9.8
against `alpha=1e-3`. `np.linalg.solve` returns a solution without complaint,
and near a −`alpha` eigenvalue the dual coefficients explode.

**`utils.clone` uses deep params.** `utils.py:32` calls `get_params()`, which
defaults to `deep=True` in scikit-learn, then passes the result to `__init__`.
Composite estimators (`Pipeline`, `GridSearchCV`, `ColumnTransformer`) reject
the nested `step__param` keys, so the run aborts on fold 0 — contradicting the
claim that a scikit-learn estimator drops straight in. **Code reading only:**
scikit-learn is not installed here, so this was not executed. `get_params(deep=False)`
is the likely fix.

## Store and server

**Id allocation races and writes are not atomic.** `_store.py:61`. `_next_id` is
`max(existing) + 1` under a `threading.RLock`, which is in-process only, while
the store's stated contract is cross-process. Two processes publishing at once
compute the same id and the second silently overwrites the first. `write_text`
has no temp-file-plus-rename, so an interrupted write leaves a truncated JSON
that makes every later `get`/`list` raise.

**The REST client trusts server-controlled data.** `_store.py:118-171`. Not
reproduced — there is no server fixture and `RestStore` is entirely untested.
Reported: `entity` and `identifier` are interpolated into URLs without
`quote()`; `dataset.data_file` is an arbitrary local path chosen by the server
and handed to `np.load` (with `allow_pickle=False`, so not code execution, but
still a server-directed file open); and stdlib `urlopen` follows redirects while
preserving headers, so a malicious server could 302 the `Authorization` bearer
token to another host. Worth doing behind a fake-server fixture.

## Simulator

**`z_jacobian` leaves the device state destroyed.** `backends/statevector.py:733`
takes `psi = self._state` without copying and un-applies every gate in place, so
after a jacobian call `device.states` holds `|0…0⟩`. Not currently a wrong
answer — both callers read their outputs before the sweep — but the device is
shared and long-lived, and nothing says the state is consumed.

**An empty batch produces one phantom row.** `backends/__init__.py:110`.
`_batch_size` returns 0 for a `(0, k)` input and `run_batch` then does
`max(1, 0)`, so zero inputs yield one output row.

**Two "independent" shot evaluations at one seed are identical.**
`backends/__init__.py:121`. Each helper call builds a fresh simulator, restarting
the RNG, so `z_expectations(..., shots=256, seed=0)` twice returns byte-identical
noise. The models dodge this by caching one device per fit; the public helpers
do not.

## Tasks

**`estimation_procedure={"type": "none"}` trains and scores on the same rows.**
`tasks/split.py:70` hands back a fold whose train and test indices are identical,
with nothing marking the result in the run, its summary, or the leaderboard.
Nothing bundled trips it, but a user-created supervised task with this procedure
publishes a 1.00.

## Not reproduced — claims that did not survive checking

- A flow calling `openqml.reset_local_store`: blocked both before and after the
  fix. `rpartition` yields module `openqml`, which does not match the
  `"openqml."` prefix.
- `openqml_evil` slipping past the prefix check: the trailing dots already
  handle it.

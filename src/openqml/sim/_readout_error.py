"""Allocate Schmidt loss using the backward supports of the declared observables."""

import math
from collections.abc import Mapping, Set
from itertools import islice
from numbers import Real

import numpy as np

from ..errors import ResourceError, ValidationError
from ..ir.observables import Hamiltonian
from ..ir.parameters import real
from .pauli import _CLIFFORD, _ROTATIONS, _clifford, _clifford_table, _masks


def _rewind(word, name, wires, mask, table, generator):
    # A term factors into a known Pauli word and an arbitrary operator on
    # disjoint unknown sites. Only the support of the latter affects the bound.
    x, z, unknown = word
    if not ((x | z | unknown) & mask):
        return word
    if name == "swap":
        return tuple(m ^ mask if (m & mask).bit_count() == 1 else m for m in word)
    if not unknown & mask:
        if table is not None:
            (x, z), _ = _clifford(x, z, wires, table)
            return x, z, unknown
        if generator is not None:
            gx, gz = generator
            if ((gx & z).bit_count() + (gz & x).bit_count()) % 2 == 0:
                return word
    return x & ~mask, z & ~mask, unknown | mask


class ReadoutErrorBudget:
    def __init__(self, circuit, observables, absolute_error, splits, max_work, workspace):
        from ._mps import _routed

        count, width = len(observables), circuit.num_qubits
        if isinstance(absolute_error, np.ndarray) and absolute_error.ndim == 0:
            absolute_error = absolute_error.item()
        if isinstance(absolute_error, Real):
            targets = (real(absolute_error),) * count
        else:
            if isinstance(absolute_error, (str, bytes, Mapping, Set)):
                raise ValidationError("absolute_error requires ordered readout tolerances.")
            try:
                targets = tuple(real(v) for v in islice(absolute_error, count + 1))
            except TypeError as exc:
                raise ValidationError("absolute_error requires real readout tolerances.") from exc
        if len(targets) != count or any(t < 0 for t in targets):
            raise ValidationError("Supply one nonnegative absolute_error per observable.")
        terms = [o.terms if isinstance(o, Hamiltonian) else (o,) for o in observables]
        nterms = sum(map(len, terms))
        # Single-site gates also change the Pauli axes seen at earlier cuts.
        events = splits + sum(len(op.qubits) == 1 for op in circuit.operations) if splits else 0
        estimate = 192 * (1 + events + nterms + width) + 32 * count * (2 + splits)
        estimate += 3 * nterms * ((width + 7) // 8)
        if events * nterms + splits * count + nterms * width > max_work or estimate > workspace:
            raise ResourceError("Observable error planning exceeds the work or memory budget.")
        self.targets = np.asarray(targets)
        self.used = np.zeros(count)
        self.one_side = np.zeros((splits, count))
        self.crossing = np.zeros((splits, count))
        self.index = 0
        if not splits:
            return
        order = list(range(width))
        gates = [
            (gate.name if gate is not None else "swap", wires)
            for gate, wires in _routed(circuit, order)
        ]
        rows, coefficients, words = [], [], []
        for row, terms_in_row in enumerate(terms):
            for term in terms_in_row:
                if not term.coefficient or term.paulis.count("I") == width:
                    continue
                rows.append(row)
                coefficients.append(abs(term.coefficient))
                words.append((*_masks("".join(term.paulis[q] for q in order)), 0))
        norms = np.bincount(rows, weights=coefficients, minlength=count)
        if not np.isfinite(norms).all():
            raise ValidationError("Observable coefficient norms exceed finite precision.")
        # Sum each side directly: subtracting an almost equal crossing mass from
        # a total can erase small nonzero terms and understate an error bound.
        index = splits
        for name, wires in reversed(gates):
            if index == 0:
                break
            mask = sum(1 << q for q in wires)
            table = _clifford_table(name) if name in _CLIFFORD and name != "swap" else None
            axis = _ROTATIONS.get(name)
            generator = (mask if axis != "Z" else 0, mask if axis != "X" else 0) if axis else None
            split = len(wires) == 2
            if split:
                index -= 1
                left = (1 << (min(wires) + 1)) - 1
            for k, word in enumerate(words):
                if split:
                    support = word[0] | word[1] | word[2]
                    weights = self.crossing if support & left and support & ~left else self.one_side
                    weights[index, rows[k]] += coefficients[k]
                words[k] = _rewind(word, name, wires, mask, table, generator)

    def cutoff(self):
        remaining = len(self.one_side) - self.index
        limit = 1.0
        for a, b, target, used in zip(
            self.one_side[self.index],
            self.crossing[self.index],
            self.targets,
            self.used,
            strict=True,
        ):
            scale = a + b
            if not scale:
                continue
            # Solve a*w+b*sqrt(w) <= remaining allowance/2. Scaling avoids
            # squaring the observable norm or subtracting nearly equal roots.
            allowance = max(0.0, float(target - used)) / remaining * 0.5
            if allowance >= scale:
                continue
            if allowance == 0:
                return 0.0
            t, fraction = allowance / scale, b / scale
            root = (
                2 * t / (fraction + math.sqrt(fraction * fraction + 4 * (a / scale) * t))
                if t
                else 0.0
            )
            limit = min(limit, root * root)
        return limit

    def record(self, weight):
        self.used += 2 * (
            self.one_side[self.index] * weight + self.crossing[self.index] * math.sqrt(weight)
        )
        self.index += 1

"""Bound serialized IR work before constructing expressions, matrices or operations."""

import json

from ..errors import ValidationError


def load_json(text, max_bytes, *, label="IR"):
    from . import integer

    max_bytes = integer(max_bytes, "max_bytes", 1)
    if not isinstance(text, str):
        raise ValidationError(f"{label} JSON must be text.")
    try:
        if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
            raise ValidationError(f"{label} JSON exceeds max_bytes.")

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValidationError(f"Duplicate JSON field {key!r}.")
                result[key] = value
            return result

        def bounded_integer(value):
            # Older Python runtimes do not cap expensive decimal-to-integer
            # conversion. Apply a consistent source budget before int().
            if len(value.lstrip("-")) > 4300:
                raise ValidationError(f"{label} JSON integer literal exceeds its digit budget.")
            return int(value)

        return json.loads(text, object_pairs_hook=unique_object, parse_int=bounded_integer)
    except ValidationError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ValidationError(f"Invalid {label} JSON.") from exc


def check_expression_payload(value, location="expression"):
    pending, nodes = [(value, 0)], 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if depth > 64 or nodes > 4096:
            raise ValidationError(f"IR exceeds expression depth or node budget at {location}.")
        if not isinstance(node, dict):
            raise ValidationError(f"Invalid expression at {location}.")
        args = node.get("args", [])
        if not isinstance(args, list):
            raise ValidationError(f"Expression arguments must be a JSON array at {location}.")
        if len(args) > 2:
            raise ValidationError(f"Invalid expression arity at {location}.")
        terms = node.get("terms", {})
        if not isinstance(terms, dict):
            raise ValidationError(f"Expression terms must be a JSON object at {location}.")
        nodes += len(terms)
        if nodes > 4096:
            raise ValidationError(f"IR exceeds expression node budget at {location}.")
        pending.extend((arg, depth + 1) for arg in args)


def check_ir_payload(payload, *, max_operations, max_depth, max_matrix_qubits, max_matrix_entries):
    from . import integer

    max_operations = integer(max_operations, "max_operations")
    max_depth = integer(max_depth, "max_depth")
    max_matrix_qubits = integer(max_matrix_qubits, "max_matrix_qubits")
    max_matrix_entries = integer(max_matrix_entries, "max_matrix_entries")

    def fail(limit, location):
        raise ValidationError(f"IR exceeds {limit} at {location}.")

    pending, count, matrix_entries = [(payload, 0, "")], 0, 0
    while pending:
        body, depth, prefix = pending.pop()
        if depth > min(max_depth, 128):
            fail("max_depth", prefix or "circuit")
        if not isinstance(body, dict) or not isinstance(body.get("operations"), list):
            raise ValidationError("Operations must be a JSON array.")
        check_expression_payload(body.get("global_phase"), prefix + "global_phase")
        for index, op in enumerate(body["operations"]):
            location = f"{prefix}operations[{index}]"
            count += 1
            if count > max_operations:
                fail("max_operations", location)
            if not isinstance(op, dict):
                raise ValidationError(f"Invalid operation at {location}.")
            if any(not isinstance(op.get(key), list) for key in ("qubits", "clbits", "params")):
                raise ValidationError("Operation operands and parameters must be JSON arrays.")
            for i, value in enumerate(op["params"]):
                check_expression_payload(value, f"{location}.params[{i}]")
            matrices = []
            if op.get("matrix") is not None:
                matrices.append(op["matrix"])
            if op.get("kraus") is not None:
                if not isinstance(op["kraus"], list):
                    raise ValidationError(f"Kraus payload must be a JSON array at {location}.")
                matrices.extend(op["kraus"])
            if matrices:
                if len(op["qubits"]) > max_matrix_qubits:
                    fail("max_matrix_qubits", location)
                # Check the aggregate before walking entries or decoding complex numbers.
                remaining = max_matrix_entries - matrix_entries
                if remaining < 1 or 2 * len(op["qubits"]) >= remaining.bit_length():
                    fail("max_matrix_entries", location)
                dimension = 2 ** len(op["qubits"])
                matrix_entries += len(matrices) * dimension**2
                if matrix_entries > max_matrix_entries:
                    fail("max_matrix_entries", location)
                for matrix in matrices:
                    if (
                        not isinstance(matrix, list)
                        or len(matrix) != dimension
                        or any(not isinstance(row, list) or len(row) != dimension for row in matrix)
                    ):
                        raise ValidationError(f"Invalid matrix dimensions at {location}.")
            blocks = op.get("blocks", [])
            if not isinstance(blocks, list):
                raise ValidationError("Blocks must be a JSON array.")
            if len(blocks) > 2:
                raise ValidationError(f"Invalid branch count at {location}.")
            pending.extend(
                (block, depth + 1, f"{location}.blocks[{i}].") for i, block in enumerate(blocks)
            )


def bounded_circuit_json(circuit, maximum, *, indent=2, collect=True):
    """Admit expanded IR output before constructing nested dictionaries."""
    from . import integer
    from .parameters import _postorder, _tree_profile

    maximum = integer(maximum, "max_bytes", 1)
    pending, profile, checked_names, lower = [(circuit, 0)], {}, set(), 0
    while pending:
        body, depth = pending.pop()
        if depth > 128:
            raise ValidationError("Serialized IR exceeds its control-depth budget.")
        lower += 64
        roots = [body.global_phase]
        for operation in body.operations:
            lower += 64 + len(operation.qubits) + len(operation.clbits)
            if operation.condition is not None:
                lower += len(operation.condition.bits)
            if operation.matrix is not None:
                lower += 9 * len(operation.matrix) ** 2
            if operation.kraus is not None:
                lower += 9 * sum(len(matrix) ** 2 for matrix in operation.kraus)
            roots.extend(operation.params)
            pending.extend((block, depth + 1) for block in operation.blocks)
            if lower > maximum:
                raise ValidationError("IR JSON exceeds max_bytes.")
        for root in roots:
            lower += 6 * _tree_profile(root, profile)[0]
            if lower > maximum:
                raise ValidationError("IR JSON exceeds max_bytes.")
            for value in _postorder((root,), checked_names):
                checked_names.add(id(value))
                if any(len(name) > maximum for name, _ in value.terms):
                    raise ValidationError("IR JSON exceeds max_bytes.")
    pieces, size = [], 0
    try:
        encoder = json.JSONEncoder(indent=indent, allow_nan=False)
        for piece in encoder.iterencode(circuit.to_dict()):
            size += len(piece.encode("utf-8"))
            if size > maximum:
                raise ValidationError("IR JSON exceeds max_bytes.")
            if collect:
                pieces.append(piece)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError("IR JSON exceeds its serialization limits.") from exc
    return "".join(pieces) if collect else None

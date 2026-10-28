"""Bounded Python-source inspection, without execution or program translation."""

import ast
import io
import json
import sys
import tokenize
from dataclasses import asdict, dataclass

from .errors import ConversionError, Diagnostic, ValidationError
from .ir import integer


@dataclass(frozen=True)
class PythonSourceReport:
    filename: str
    source_bytes: int | None
    source_characters: int
    lines: int | None
    tokens: int
    nodes: int
    imports: tuple[str, ...]
    ast_parseable: bool | None
    static_scan_complete: bool
    diagnostics: tuple[Diagnostic, ...]
    diagnostics_truncated: bool = False

    @property
    def compatible(self):
        """Source inspection never certifies whole-program conversion."""
        return False

    def raise_for_errors(self):
        raise ConversionError(self.diagnostics)

    def to_dict(self):
        return {
            "schema": "openqml.python_inspection",
            "scope": "static_source_inspection_only",
            "code_executed": False,
            "whole_program_conversion_supported": False,
            "runtime_validity": "not_proven",
            **asdict(self),
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def inspect_python(
    source,
    *,
    filename="<python>",
    max_source_bytes=8_000_000,
    max_tokens=200_000,
    max_nodes=100_000,
    max_nesting=64,
    max_statement_tokens=2048,
    max_diagnostics=256,
):
    """Find syntactic conversion risks; never import, evaluate or call source code.

    A completed scan is not a semantic proof. Aliases, helper functions, import
    shadowing and runtime values can hide behavior. Token/statement/nesting
    preflight precedes AST allocation; max_nodes bounds subsequent inspection.
    This is a circuit-boundary aid, not a sandbox or an arbitrary Python compiler.
    Columns use the AST's one-based UTF-8 byte offsets.
    """
    if not isinstance(source, str) or not isinstance(filename, str) or not filename:
        raise ValidationError("Python inspection requires source text and a filename label.")
    source, filename = str.__str__(source), str.__str__(filename)
    limits = [max_source_bytes, max_tokens, max_nodes, max_nesting, max_statement_tokens]
    for value in limits:
        integer(value, "inspection limit", 1)
    integer(max_diagnostics, "max_diagnostics", 2)
    try:
        size = None if len(source) > max_source_bytes else len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValidationError("Python source must be valid UTF-8 text.") from exc
    diagnostics = [
        Diagnostic(
            "PYTHON_PROGRAM_UNSUPPORTED",
            "Arbitrary Python programs are not converted. Capture a concrete SDK circuit or "
            "build IR; review external execution context separately.",
            filename,
        )
    ]
    count = nodes = 0
    imports, aliases = set(), {}
    truncated = False

    def note(code, message, node=None, severity="warning"):
        nonlocal truncated
        location = filename if node is None else f"{filename}:{node.lineno}:{node.col_offset + 1}"
        if len(diagnostics) < max_diagnostics:
            diagnostics.append(Diagnostic(code, message, location, severity))
        else:
            truncated = True

    def report(valid, complete):
        if truncated:
            diagnostics[-1] = Diagnostic(
                "PYTHON_DIAGNOSTIC_LIMIT",
                "Additional diagnostics were omitted by the limit.",
                filename,
                "warning",
            )
        return PythonSourceReport(
            filename=filename,
            source_bytes=size,
            source_characters=len(source),
            lines=None
            if size is None
            else source.count("\n") + int(bool(source) and not source.endswith("\n")),
            tokens=count,
            nodes=nodes,
            imports=tuple(sorted(imports)),
            ast_parseable=valid,
            static_scan_complete=complete,
            diagnostics=tuple(diagnostics),
            diagnostics_truncated=truncated,
        )

    if size is None or size > max_source_bytes:
        note("PYTHON_SOURCE_LIMIT", "Source exceeds max_source_bytes.", severity="error")
        return report(None, False)
    nesting = statement = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            count += 1
            statement += 1
            if token.type == tokenize.NEWLINE:
                statement = 0
            if token.type == tokenize.INDENT or token.string in {"(", "[", "{"}:
                nesting += 1
            elif token.type == tokenize.DEDENT or token.string in {")", "]", "}"}:
                nesting -= 1
            if count > max_tokens or statement > max_statement_tokens or nesting > max_nesting:
                note(
                    "PYTHON_PARSE_LIMIT",
                    "Token, statement or nesting preflight limit exceeded.",
                    severity="error",
                )
                return report(None, False)
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, tokenize.TokenError, IndentationError, RecursionError) as exc:
        note("PYTHON_SYNTAX", str(exc), severity="error")
        return report(False, False)

    # Import names are clues, not bindings proven across Python scopes.
    syntax = []
    for node in ast.walk(tree):
        nodes += 1
        if nodes > max_nodes:
            note("PYTHON_AST_LIMIT", "AST inspection exceeds max_nodes.", severity="error")
            return report(True, False)
        syntax.append(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module = (
                    alias.name
                    if isinstance(node, ast.Import)
                    else "." * node.level + (node.module or "")
                )
                imports.add(module)
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = (
                    module if isinstance(node, ast.Import) else f"{module}.{alias.name}"
                )
                root = module.split(".")[0]
                if root not in sys.stdlib_module_names:
                    note(
                        "PYTHON_EXTERNAL_DEPENDENCY",
                        f"Import {module!r}: runtime behavior is not translated.",
                        node,
                    )

    control = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Match,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )
    noise = {
        "NoiseModel",
        "QuantumError",
        "depolarizing_error",
        "amplitude_damping_error",
        "thermal_relaxation_error",
        "depolarize",
        "mixed_unitary",
        "noise_model",
    }
    hardware = {
        "Schedule",
        "ScheduleBlock",
        "Play",
        "PulseGate",
        "QiskitRuntimeService",
        "IBMQ",
        "calibrations",
        "add_calibration",
        "pulse",
    }
    custom = {"to_gate", "to_instruction", "Gate", "Instruction", "Operation", "define_gate"}

    def resolved_name(node):
        parts = _name(node).split(".")
        return aliases.get(parts[0], parts[0]) + (
            "." + ".".join(parts[1:]) if len(parts) > 1 else ""
        )

    for node in syntax:
        if isinstance(node, control):
            # comprehension nodes have no own location; their parent carries it.
            if hasattr(node, "lineno"):
                note(
                    "PYTHON_DYNAMIC_CONTROL",
                    "Python control flow needs runtime capture or manual IR translation.",
                    node,
                )
        if isinstance(node, ast.Call):
            resolved = resolved_name(node.func)
            tail = resolved.split(".")[-1]
            keys = {keyword.arg for keyword in node.keywords}
            if tail in {"exec", "eval", "compile", "__import__", "import_module"}:
                note(
                    "PYTHON_DYNAMIC_EXECUTION",
                    "Dynamic execution/import cannot be resolved by static source inspection.",
                    node,
                )
            if tail in noise or "noise_model" in keys or "noise" in keys or ".noise." in resolved:
                note(
                    "PYTHON_NOISE_CONTEXT",
                    "Potential noise configuration: only explicit IR channels are preserved. "
                    "A circuit alone does not carry its simulator's noise model.",
                    node,
                )
            if (
                tail in hardware
                or ".pulse." in resolved
                or keys & {"backend", "calibrations"}
                or ("target" in keys and resolved.startswith(("qiskit.", "qiskit_aer.")))
            ):
                note(
                    "PYTHON_HARDWARE_CONTEXT",
                    "Backend and calibration behavior needs review outside circuit IR.",
                    node,
                )
            if tail in custom:
                note(
                    "PYTHON_CUSTOM_INSTRUCTION",
                    "Custom instructions need a supported definition, matrix or adapter.",
                    node,
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            note(
                "PYTHON_OPAQUE_DEFINITION",
                "Function/class behavior is not proven by this syntactic scan.",
                node,
            )
        if isinstance(node, ast.ClassDef) and any(
            resolved_name(base).split(".")[-1] in custom for base in node.bases
        ):
            note(
                "PYTHON_CUSTOM_INSTRUCTION",
                "Custom instruction classes need an explicit supported representation.",
                node,
            )
    return report(True, True)

"""Frontend-independent source limits and comment handling."""

import re

from .errors import conversion_error
from .ir import integer


def clean_source(text, max_source_bytes=2_000_000):
    integer(max_source_bytes, "max_source_bytes", 1)
    if (
        not isinstance(text, str)
        or len(text) > max_source_bytes
        or len(text.encode("utf-8")) > max_source_bytes
    ):
        raise conversion_error("SOURCE_LIMIT", "OpenQASM source exceeds the configured byte limit.")
    # Remove comments without changing strings, line numbers or character offsets.
    output, index = [], 0
    while index < len(text):
        if text[index] == '"':
            start = index
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise conversion_error("SYNTAX", "Unterminated source string.")
            output.append(text[start:index])
        elif text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            output.append(" " * (end - index))
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise conversion_error("SYNTAX", "Unterminated block comment.")
            end += 2
            output.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            index = end
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def validate_header(text, version=None):
    header = re.match(r"\s*OPENQASM\s+([23])(?:\.0)?\s*;", text)
    if not header:
        raise conversion_error("QASM_VERSION", "Expected an OpenQASM 2.0 or 3.0 header.")
    actual = int(header.group(1))
    if version is not None and (type(version) is not int or version != actual):
        raise conversion_error(
            "QASM_VERSION", "Source header differs from requested source format."
        )
    return actual


def check_literal_registers(text, max_register_bits):
    """Reject oversized literal declarations before an SDK constructs their storage.

    SDK-resolved constant/expression sizes also receive a post-parse width check.
    This preflight does not claim a process-level SDK parsing memory limit.
    """
    integer(max_register_bits, "max_register_bits", 1)
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    declarations = re.finditer(
        r"\b(?:qreg|creg)\s+[A-Za-z_]\w*\s*\[\s*(\d+)\s*\]"
        r"|\b(?:qubit|bit)\s*(?:\[\s*(\d+)\s*\])?\s+[A-Za-z_]\w*",
        code,
    )
    total = 0
    for declaration in declarations:
        total += int(declaration.group(1) or declaration.group(2) or "1")
        if total > max_register_bits:
            raise conversion_error("REGISTER_LIMIT", "Register size exceeds the configured limit.")


def expand_includes(text, include_paths, version, max_source_bytes, max_include_depth):
    """Inline custom files from explicit roots with cumulative read/expansion limits."""
    from pathlib import Path

    integer(max_include_depth, "max_include_depth")
    if isinstance(include_paths, (str, bytes)):
        raise conversion_error("INCLUDE_PATH", "include_paths must be a sequence of directories.")
    try:
        roots = tuple(Path(p).resolve(strict=True) for p in include_paths)
    except (TypeError, OSError, ValueError, RuntimeError) as exc:
        raise conversion_error(
            "INCLUDE_PATH", "Include roots must be existing directories."
        ) from exc
    if any(not p.is_dir() for p in roots):
        raise conversion_error("INCLUDE_PATH", "Include roots must be directories.")
    remaining = max_source_bytes - len(text.encode("utf-8"))
    directive = re.compile(r'"(?:\\.|[^"\\])*"|\binclude\b(?:\s*"([^"\n]+)"\s*;)?')
    standard = "qelib1.inc" if version == 2 else "stdgates.inc"

    def expand(code, parent, stack):
        nonlocal remaining
        pieces, end = [], 0
        for match in directive.finditer(code):
            filename = match.group(1)
            if filename is None:
                if not match.group().startswith('"'):
                    raise conversion_error("SYNTAX", "Malformed include directive.")
                continue
            if filename == standard:
                continue
            if not roots:
                raise conversion_error(
                    "INCLUDE_PATH", "Custom includes require explicit include_paths."
                )
            if len(stack) >= max_include_depth:
                raise conversion_error(
                    "INCLUDE_DEPTH_LIMIT", "Custom include nesting exceeds max_include_depth."
                )
            target = Path(filename)
            searches = ((parent,) if parent is not None else ()) + roots
            candidates = (target,) if target.is_absolute() else tuple(p / target for p in searches)
            resolved = None
            for candidate in candidates:
                try:
                    candidate = candidate.resolve()
                except (OSError, ValueError, RuntimeError):
                    continue
                if any(candidate.is_relative_to(root) for root in roots) and candidate.is_file():
                    resolved = candidate
                    break
            if resolved is None:
                raise conversion_error(
                    "INCLUDE_PATH", "Include file is missing or outside the explicit roots."
                )
            if resolved in stack:
                raise conversion_error("INCLUDE_CYCLE", "Custom include files contain a cycle.")
            try:
                with resolved.open("rb") as stream:
                    payload = stream.read(remaining + 1)
            except OSError as exc:
                raise conversion_error(
                    "INCLUDE_PATH", "Cannot read the requested include file."
                ) from exc
            if len(payload) > remaining:
                raise conversion_error(
                    "SOURCE_LIMIT", "Source and included files exceed max_source_bytes."
                )
            remaining -= len(payload)
            try:
                child = clean_source(payload.decode("utf-8"), max_source_bytes)
            except UnicodeError as exc:
                raise conversion_error("INCLUDE_ENCODING", "Include files must use UTF-8.") from exc
            pieces.extend(
                (
                    code[end : match.start()],
                    "\n",
                    expand(child, resolved.parent, stack + (resolved,)),
                    "\n",
                )
            )
            end = match.end()
        pieces.append(code[end:])
        return "".join(pieces)

    try:
        output = expand(text, None, ())
    except RecursionError as exc:
        raise conversion_error(
            "INCLUDE_DEPTH_LIMIT", "Include nesting exceeds the Python recursion limit."
        ) from exc
    if len(output.encode("utf-8")) > max_source_bytes:
        raise conversion_error("SOURCE_LIMIT", "Expanded source exceeds max_source_bytes.")
    return output

"""Minimal, dependency-free TOML emitter.

diffusion-pipe reads its configs with ``toml.load``; we only ever need to
*write* a small, well-known shape (top-level scalars/lists, single-level
``[table]`` sections, and ``[[array-of-table]]`` blocks for ``directory`` /
``prompts``). Rather than pull ``toml``/``tomli-w`` into the backend venv we
emit that subset by hand. Output is validated in tests by round-tripping
through the stdlib ``tomllib`` reader.
"""
from __future__ import annotations

import re

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _fmt_string(s: str) -> str:
    out = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{out}"'


def _fmt_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else _fmt_string(key)


def _fmt_float(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    # repr keeps a decimal point or exponent so TOML parses it as a float.
    return repr(value)


def _fmt_value(value) -> str:
    # bool must be checked before int (bool is a subclass of int).
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _fmt_float(value)
    if isinstance(value, str):
        return _fmt_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    raise TypeError(f"Cannot serialize {type(value).__name__} to TOML: {value!r}")


def _is_array_of_tables(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def _dump_table(table: dict, prefix: list[str], lines: list[str]) -> None:
    scalars: list[tuple[str, object]] = []
    subtables: list[tuple[str, dict]] = []
    array_tables: list[tuple[str, list]] = []

    for key, value in table.items():
        if value is None:
            continue
        if isinstance(value, dict):
            subtables.append((key, value))
        elif _is_array_of_tables(value):
            array_tables.append((key, value))
        else:
            scalars.append((key, value))

    for key, value in scalars:
        lines.append(f"{_fmt_key(key)} = {_fmt_value(value)}")

    for key, value in subtables:
        path = prefix + [key]
        header = ".".join(_fmt_key(p) for p in path)
        lines.append("")
        lines.append(f"[{header}]")
        _dump_table(value, path, lines)

    for key, items in array_tables:
        path = prefix + [key]
        header = ".".join(_fmt_key(p) for p in path)
        for item in items:
            lines.append("")
            lines.append(f"[[{header}]]")
            _dump_table(item, path, lines)


def dumps(table: dict) -> str:
    """Serialize a dict to a TOML string. ``None`` values are omitted."""
    lines: list[str] = []
    _dump_table(table, [], lines)
    return "\n".join(lines) + "\n"

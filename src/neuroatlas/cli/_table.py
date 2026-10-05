"""Print rows as an aligned table, CSV, Markdown or JSON -- one renderer for
every listing and report, so ``--format`` means the same thing everywhere.

One schema for every format:

* a missing number is ``None`` in a row: ``n/a`` in a table, empty in CSV,
  ``null`` in JSON (never the string "n/a", never NaN, which is not JSON);
* a field that does not apply is :data:`NOT_APPLICABLE`: ``-`` in a table,
  ``null`` in JSON;
* the lines a table prints under a row (``↳ ...``) are its ``note``, a column
  of CSV/Markdown and a field of JSON, so a machine reads the same reasons a
  person does;
* JSON may carry more fields than the table shows (``extra``), never fewer.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

FORMATS = ("table", "csv", "md", "json")


class _NotApplicable:
    """A cell for which the column has no meaning (shown ``-``, JSON null)."""

    def __repr__(self) -> str:
        return "NOT_APPLICABLE"

    def __bool__(self) -> bool:
        return False


NOT_APPLICABLE = _NotApplicable()


def _cell(value: Any, fmt: str = "table") -> str:
    if value is NOT_APPLICABLE:
        return "-"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "" if fmt == "csv" else "n/a"
    if isinstance(value, float):
        return f"{value:.3f}" if fmt != "csv" else repr(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def _json_value(value: Any) -> Any:
    if value is NOT_APPLICABLE:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def add_format_arg(parser) -> None:
    parser.add_argument("--format", choices=FORMATS, default="table",
                        help="Output format (default: table). csv, md and json carry the "
                             "table's ↳ lines as a `note` column; json uses null for "
                             "a missing value.")


def render(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
           fmt: str = "table", notes: Optional[Dict[int, List[str]]] = None,
           out=None, extra: Sequence[str] = (),
           labels: Optional[Mapping[str, str]] = None,
           formats: Optional[Mapping[str, str]] = None) -> None:
    """``notes`` maps a row index to lines printed under it in a table, and
    to its ``note`` field elsewhere. ``extra`` names fields only JSON carries.
    ``labels`` renames a column's header in the table (keys stay as they are
    in CSV and JSON); ``formats`` gives a column's number format in the table
    and Markdown (e.g. ``"{:.3g}"``) instead of three decimals."""
    out = out or sys.stdout
    # a caller that has notes for the table gets a note column elsewhere,
    # on every row, so CSV columns do not depend on the data
    with_note = fmt != "table" and notes is not None
    notes = notes or {}
    note_of = lambda i: "; ".join(notes.get(i, [])) or None        # noqa: E731
    if fmt == "json":
        payload = []
        wanted = [*columns, *(c for c in extra if c not in columns)]
        for i, r in enumerate(rows):
            # the row's own key order (identifying fields first), then the rest
            keys = [k for k in r if k in wanted] + [c for c in wanted if c not in r]
            item = {c: _json_value(r.get(c)) for c in keys if c != "note"}
            if with_note:
                item["note"] = note_of(i)
            payload.append(item)
        json.dump(payload, out, indent=2)
        out.write("\n")
        return
    keys = [*columns, "note"] if with_note else list(columns)
    def cell(row, column):
        value, spec = row.get(column), (formats or {}).get(column)
        if spec and fmt != "csv" and isinstance(value, (int, float)) \
                and not isinstance(value, bool) and value == value:
            return spec.format(value)
        return _cell(value, fmt)

    cells = [[cell(r, c) for c in columns] + ([note_of(i) or ""] if with_note else [])
             for i, r in enumerate(rows)]
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(keys)
        writer.writerows(cells)
        out.write(buf.getvalue())
        return
    if fmt == "md":
        out.write("| " + " | ".join(keys) + " |\n")
        out.write("|" + "|".join("---" for _ in keys) + "|\n")
        for row in cells:
            out.write("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |\n")
        return
    heads = [(labels or {}).get(c, c) for c in columns]
    widths = [max([len(h)] + [len(row[i]) for row in cells]) for i, h in enumerate(heads)]
    line = lambda values: "  ".join(v.ljust(w) for v, w in zip(values, widths)).rstrip()  # noqa: E731
    out.write(line(heads) + "\n")
    for i, row in enumerate(cells):
        out.write(line(row) + "\n")
        for note in notes.get(i, []):
            out.write(f"    ↳ {note}\n")

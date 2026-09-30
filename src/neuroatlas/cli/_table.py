"""Print rows as an aligned table, CSV, Markdown or JSON -- one renderer for
every listing and report, so ``--format`` means the same thing everywhere."""
from __future__ import annotations

import csv
import io
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

FORMATS = ("table", "csv", "md", "json")


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "n/a" if value != value else f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def add_format_arg(parser) -> None:
    parser.add_argument("--format", choices=FORMATS, default="table",
                        help="Output format (default: table).")


def render(rows: Sequence[Dict[str, Any]], columns: Sequence[str],
           fmt: str = "table", notes: Optional[Dict[int, List[str]]] = None,
           out=None) -> None:
    """``notes`` maps a row index to lines printed under it (table only)."""
    out = out or sys.stdout
    if fmt == "json":
        json.dump([{c: r.get(c) for c in columns} for r in rows], out, indent=2, default=str)
        out.write("\n")
        return
    cells = [[_cell(r.get(c)) for c in columns] for r in rows]
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(cells)
        out.write(buf.getvalue())
        return
    if fmt == "md":
        out.write("| " + " | ".join(columns) + " |\n")
        out.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in cells:
            out.write("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |\n")
        return
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    line = lambda values: "  ".join(v.ljust(w) for v, w in zip(values, widths)).rstrip()
    out.write(line(columns) + "\n")
    for i, row in enumerate(cells):
        out.write(line(row) + "\n")
        for note in (notes or {}).get(i, []):
            out.write(f"    ↳ {note}\n")

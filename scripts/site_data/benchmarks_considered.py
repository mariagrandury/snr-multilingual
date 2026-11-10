"""Data of the benchmarks-considered view (benchmarksConsidered): the rows of the
paper's table of every benchmark considered (documents/paper/tables/
app_benchmarks_considered.tex, written by make_benchmarks_considered_table.py),
one row per benchmark with its group (analysed or screened), languages, trained
languages, task, formats and low-resource language examples.

Reading the published table keeps the view identical to the paper.
"""
import re
from pathlib import Path

TABLE = Path(__file__).resolve().parents[2] / "documents/paper/tables/app_benchmarks_considered.tex"


def _clean(cell: str) -> str:
    cell = re.sub(r"\\cite\w*\{[^}]*\}|\\citeneeded\{\}", "", cell)
    return cell.replace("--", "–").replace("\\", "").strip()


def build() -> dict:
    rows, group = [], None
    for line in TABLE.read_text().splitlines():
        if "\\textit{Analysed}" in line:
            group = "analysed"
        elif "\\textit{Screened" in line:
            group = "screened"
        elif group and line.count("&") == 5:
            name, langs, trained, task, formats, low = (_clean(c) for c in line.rstrip("\\ ").split("&"))
            rows.append(dict(name=name, group=group, langs=int(langs), trained=int(trained), task=task,
                             formats=formats, low_resource=None if low == "–" else low))
    return {"benchmarks_considered": rows}

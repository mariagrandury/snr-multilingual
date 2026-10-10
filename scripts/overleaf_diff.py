#!/usr/bin/env python3
"""Compare an Overleaf export of the paper with documents/paper/ in this repo.

The Overleaf project and the repo do not share a layout (Overleaf keeps
main.tex at its root, may keep the generated pages flat in sections/ or in
sections/rqs/, the tables in tables/), so files are paired by NAME, and by
path only when a name occurs more than once. A file left unpaired is then
paired across an appendix renumbering (Overleaf app_06_snr.tex with the repo's
app_snr.tex). For each pair:

  text (.tex .bib .sty .bst .cls)  same / differs, and for a differing
      file which side moved: the repo's git history is searched for the
      Overleaf copy. "repo ahead" means the Overleaf copy IS an older repo
      version, so only the repo changed since (copy the repo file to
      Overleaf). "diverged" means it matches no repo version: the base is the
      repo version closest to it, and the counts say how many lines Overleaf
      and the repo each changed against that base (Overleaf lines to port,
      repo lines to push).
  figures and other binaries  same / differs, by checksum.

Files present on one side only are listed. Generator code and analysis CSVs
under documents/paper/ are not part of the Overleaf project and are ignored,
and a repo folder the export does not have at all (figures/ in a sources-only
export) is summarised in one line.

    python3 scripts/overleaf_diff.py                       # newest folder under documents/overleaf/
    python3 scripts/overleaf_diff.py documents/overleaf/paper-overleaf-10-08-20h
    python3 scripts/overleaf_diff.py EXPORT --diff 04_analysis   # word diff in the terminal
    python3 scripts/overleaf_diff.py EXPORT --code               # VS Code diff tab per differing file
    python3 scripts/overleaf_diff.py EXPORT --html               # one HTML report next to the export

The export may also be the .zip Overleaf downloads: it is unpacked next to
it, into a folder of the same name. In every diff, red is the Overleaf text
and green the repo text.
"""
import argparse
import difflib
import hashlib
import html
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "documents" / "paper"
EXPORTS = REPO / "documents" / "overleaf"
TEXT = {".tex", ".bib", ".sty", ".bst", ".cls"}
FIGURE = {".png", ".pdf", ".jpg", ".jpeg", ".svg", ".eps"}
# what of documents/paper/ belongs to the Overleaf project: the LaTeX sources and the figures it includes
SKIP_DIRS = {"__pycache__", "figures_svg", "figures_pdf"}
HISTORY = 200   # repo versions of a file searched for the Overleaf copy


def files(root, paper):
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or SKIP_DIRS & set(rel.parts) or p.name.startswith("."):
            continue
        if paper and p.suffix.lower() not in TEXT | FIGURE:
            continue
        out.append(rel)
    return out


def pair(left, right):
    """Pair two lists of relative paths by name; a name on several paths pairs by path."""
    by_l, by_r = defaultdict(list), defaultdict(list)
    for p in left:
        by_l[p.name].append(p)
    for p in right:
        by_r[p.name].append(p)
    pairs, only_l, only_r = [], [], []
    for name in sorted(set(by_l) | set(by_r)):
        ls, rs = by_l.get(name, []), by_r.get(name, [])
        if len(ls) == 1 and len(rs) == 1:
            pairs.append((ls[0], rs[0]))
            continue
        rest = list(rs)
        for l in ls:
            match = next((r for r in rest if r == l), None)
            if match is None:
                only_l.append(l)
            else:
                pairs.append((l, match))
                rest.remove(match)
        only_r += rest
    # an appendix renumbered on one side: app_06_snr.tex <-> app_snr.tex
    norm = lambda p: re.sub(r"^app_\d\d_", "app_", p.name)
    nl, nr = defaultdict(list), defaultdict(list)
    for p in only_l:
        nl[norm(p)].append(p)
    for p in only_r:
        nr[norm(p)].append(p)
    for key in set(nl) & set(nr):
        if len(nl[key]) == 1 and len(nr[key]) == 1:
            pairs.append((nl[key][0], nr[key][0]))
            only_l.remove(nl[key][0])
            only_r.remove(nr[key][0])
    return sorted(pairs), only_l, only_r


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def history(rel):
    """(commit, date, text) of the repo versions of documents/paper/<rel>, newest first."""
    path = f"documents/paper/{rel.as_posix()}"
    log = git("log", f"-{HISTORY}", "--follow", "--name-only", "--format=@%h %cs", "--", path).decode()
    out, head = [], None
    for line in log.splitlines():
        if line.startswith("@"):
            head = line[1:].split()
        elif line.strip() and head:
            blob = git("show", f"{head[0]}:{line.strip()}")
            out.append((head[0], head[1], blob.decode("utf-8", "replace")))
            head = None
    return out


def changed_lines(a, b):
    sm = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def provenance(rel_repo, ov_text, repo_text):
    """Which side moved: the Overleaf copy against the repo's history of the file."""
    versions = history(rel_repo)
    for commit, date, text in versions:
        if text == ov_text:
            return f"repo ahead (Overleaf = repo {commit} of {date})"
    if not versions:
        return "diverged (file not in repo history)"
    base = min(versions, key=lambda v: changed_lines(v[2], ov_text))
    return (f"diverged ({changed_lines(base[2], ov_text)} lines on Overleaf, "
            f"{changed_lines(base[2], repo_text)} in the repo, against repo {base[0]} of {base[1]})")


def word_diff(a, b, color=True):
    """git's word diff of two files, without its four header lines."""
    flag = "--color-words" if color else "--word-diff=porcelain"
    out = subprocess.run(["git", "--no-pager", "diff", "--no-index", "--no-prefix", flag, "--", str(a), str(b)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode("utf-8", "replace")
    plain = lambda l: re.sub(r"\x1b\[[0-9;]*m", "", l)
    return "\n".join(l for l in out.splitlines() if not plain(l).startswith(("diff --git", "index ", "--- ", "+++ ")))


def porcelain_html(text):
    """git's porcelain word diff as HTML: red the Overleaf text, green the repo text."""
    out = []
    for line in text.splitlines():
        if line.startswith(("diff ", "index ", "--- ", "+++ ")):
            continue
        if line.startswith("@@"):
            out.append(f'\n<span class="hunk">{html.escape(line)}</span>\n')
        elif line == "~":
            out.append("\n")
        elif line[:1] == "-":
            out.append(f"<del>{html.escape(line[1:])}</del>")
        elif line[:1] == "+":
            out.append(f"<ins>{html.escape(line[1:])}</ins>")
        else:
            out.append(html.escape(line[1:]))
    return "".join(out)


def export_dir(arg):
    if arg is None:
        found = sorted((p for p in EXPORTS.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if not found:
            sys.exit(f"no export under {EXPORTS}")
        return found[-1]
    p = Path(arg).resolve()
    if not p.exists():
        sys.exit(f"no such export: {arg}")
    if p.suffix == ".zip":
        dest = p.with_suffix("")
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        return dest
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", nargs="?", help="the Overleaf export folder or .zip (default: the newest under documents/overleaf/)")
    ap.add_argument("--diff", metavar="NAME", nargs="*", help="print the word diff of the differing text files (or those whose name contains NAME)")
    ap.add_argument("--code", metavar="NAME", nargs="*", help="open a VS Code diff per differing text file (or those whose name contains NAME)")
    ap.add_argument("--html", nargs="?", const="", metavar="OUT", help="write one HTML report (default: next to the export)")
    a = ap.parse_args()
    ov = export_dir(a.export)
    pairs, only_ov, only_repo = pair(files(ov, False), files(PAPER, True))

    rows, differing = [], []
    for o, r in pairs:
        po, pr = ov / o, PAPER / r
        if sha(po) == sha(pr):
            rows.append(("same", o, r, ""))
            continue
        if po.suffix.lower() in TEXT:
            note = provenance(r, po.read_text(errors="replace"), pr.read_text(errors="replace"))
            differing.append((o, r))
        else:
            note = "binary differs"
        rows.append(("DIFF", o, r, note))

    print(f"Overleaf export: {ov.relative_to(REPO) if REPO in ov.parents else ov}")
    print(f"repo:            {PAPER.relative_to(REPO)}")
    print(f"{len(pairs)} paired files, {sum(r[0] == 'same' for r in rows)} identical, {len(differing)} text files differ, "
          f"{len(only_ov)} only on Overleaf, {len(only_repo)} only in the repo\n")
    for state, o, r, note in rows:
        if state != "same":
            moved = "" if o == r else f"  (repo: {r})"
            print(f"  {state:4}  {o}{moved}  {note}")
    if only_ov:
        print("\nonly on Overleaf:")
        print("\n".join(f"  {p}" for p in only_ov))
    if only_repo:
        print("\nonly in the repo:")
        have = {q.parts[0] for q in files(ov, False) if len(q.parts) > 1}
        absent = defaultdict(int)
        for p in only_repo:
            if len(p.parts) > 1 and p.parts[0] not in have:
                absent[p.parts[0]] += 1
            else:
                print(f"  {p}")
        for d, n in sorted(absent.items()):
            print(f"  {d}/: {n} files (the export has no {d}/ folder)")

    pick = lambda names: [d for d in differing if not names or any(n in str(d[1]) for n in names)]
    if a.diff is not None:
        for o, r in pick(a.diff):
            print(f"\n=== {o}  vs  repo {r}\n" + word_diff(ov / o, PAPER / r))
    if a.code is not None:
        for o, r in pick(a.code):
            subprocess.run(["code", "--diff", str(ov / o), str(PAPER / r)])
    if a.html is not None:
        out = Path(a.html) if a.html else ov.parent / f"{ov.name}-diff.html"
        body = [f"<h1>Overleaf {html.escape(ov.name)} vs repo documents/paper</h1>",
                "<p><del>red</del>: only on Overleaf &nbsp; <ins>green</ins>: only in the repo</p><table>"]
        body += [f"<tr><td>{s}</td><td>{html.escape(str(o))}</td><td>{html.escape(str(r)) if o != r else ''}</td>"
                 f"<td>{html.escape(n)}</td></tr>" for s, o, r, n in rows if s != "same"]
        body += [f"<tr><td>overleaf only</td><td>{html.escape(str(p))}</td><td></td><td></td></tr>" for p in only_ov]
        body += [f"<tr><td>repo only</td><td></td><td>{html.escape(str(p))}</td><td></td></tr>" for p in only_repo]
        body.append("</table>")
        for o, r in differing:
            body.append(f'<h2 id="{html.escape(str(r))}">{html.escape(str(o))}</h2>'
                        f"<pre>{porcelain_html(word_diff(ov / o, PAPER / r, color=False))}</pre>")
        style = ("body{font-family:sans-serif;margin:16px;max-width:1200px} pre{white-space:pre-wrap;font-size:13px;"
                 "background:#fafafa;padding:8px} del{background:#fdd;color:#900} ins{background:#dfd;color:#060;"
                 "text-decoration:none} .hunk{color:#888} td{padding:2px 8px;font-size:13px;border-bottom:1px solid #eee}")
        out.write_text(f"<!doctype html><meta charset='utf-8'><title>Overleaf diff</title><style>{style}</style>"
                       + "\n".join(body))
        print(f"\nHTML report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

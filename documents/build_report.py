#!/usr/bin/env python3
"""Render documents/report/*.md to a PDF with PyMuPDF — no LaTeX needed.

    python documents/build_report.py            # documents/snr_predictivity_report.pdf
    python documents/build_report.py --md documents/report/other.md --out x.pdf

Markdown → HTML (python-markdown: tables, fenced code, footnotes) → PDF
(PyMuPDF Story). Relative image paths in the markdown resolve against the
markdown file's directory, so the report can point at the committed figures
under src/pretrain/ and src/signal-and-noise/ without copying them.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown
import pymupdf

HERE = Path(__file__).resolve().parent
CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.35; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; border-bottom: 0.5pt solid #999; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt 0; }
p { margin: 0 0 6pt 0; text-align: justify; }
li { margin: 0 0 2pt 0; }
table { border-collapse: collapse; font-size: 8pt; margin: 4pt 0 8pt 0; }
th, td { border: 0.5pt solid #888; padding: 2pt 4pt; vertical-align: top; }
th { background: #eee; }
code { font-family: monospace; font-size: 8.5pt; }
pre { font-family: monospace; font-size: 8pt; background: #f4f4f4; padding: 4pt; }
img { max-width: 100%; }
.caption { font-size: 8pt; color: #444; }
blockquote { margin: 4pt 12pt; color: #333; font-style: italic; }
"""


def render(md_path: Path, out_path: Path) -> Path:
    text = md_path.read_text()
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "footnotes",
                                               "attr_list", "sane_lists"])
    # Story resolves <img src> against an Archive; mount the filesystem root and
    # rewrite relative paths to absolute ones (root-relative, no leading slash).
    def _abs(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (md_path.parent / src).resolve()
        return f'src="{p.relative_to("/")}"'
    html = re.sub(r'src="([^"]+)"', _abs, html)

    story = pymupdf.Story(html=html, user_css=CSS, archive=pymupdf.Archive("/"))
    writer = pymupdf.DocumentWriter(str(out_path))
    page = pymupdf.paper_rect("a4")
    margin = 48
    where = page + (margin, margin, -margin, -margin)
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # DocumentWriter stores images as raw pixel streams (~15 MB for three
    # figures); rewriting with deflate brings the file to ~1 MB.
    doc = pymupdf.open(str(out_path))
    data = doc.tobytes(garbage=4, deflate=True, deflate_images=True)
    doc.close()
    out_path.write_bytes(data)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md", type=Path, default=HERE / "report" / "snr_predictivity_report.md")
    p.add_argument("--out", type=Path, default=HERE / "snr_predictivity_report.pdf")
    a = p.parse_args()
    out = render(a.md, a.out)
    print(f"wrote {out} ({pymupdf.open(str(out)).page_count} pages)")


if __name__ == "__main__":
    main()

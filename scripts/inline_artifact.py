"""Build the publishable copy of the research-question compendium.

The source keeps relative image paths so it opens from disk and its diffs stay
readable. An artifact can load images only from a data URI, so the published
copy inlines them. Output is gitignored build noise, not a second source.

    python3 scripts/inline_artifact.py
"""
import base64, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "documents" / "research-questions.html"
OUT = REPO / "documents" / "build" / "research-questions.html"

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml", ".gif": "image/gif", ".webp": "image/webp"}


def main():
    html = SRC.read_text()
    missing = []

    def inline(m):
        rel = m.group(1)
        f = (SRC.parent / rel).resolve()
        if not f.is_file():
            missing.append(rel)
            return m.group(0)
        mime = MIME.get(f.suffix.lower(), "application/octet-stream")
        return f'src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"'

    out, n = re.subn(r'src="((?!data:|https?:)[^"]+)"', inline, html)
    if missing:
        print("missing figures, nothing written:", *missing, sep="\n  ")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out)
    print(f"inlined {n} figures -> {OUT.relative_to(REPO)} ({len(out)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

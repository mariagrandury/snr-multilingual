import re
from pathlib import Path

from mkdocs.structure.files import File

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# READMEs included via --8<-- use image paths relative to their original
# location. Map each source dir to the URL prefix of the page that includes
# its README so those relative paths resolve in the rendered site.
INCLUDES = [
    ("src/signal-and-noise/analysis/rq00_gate_and_curves", "signal-noise"),
    ("src/signal-and-noise/analysis/rq01_scaling_predictability", "signal-noise"),
    ("src/signal-and-noise/analysis/rq02_decision_accuracy", "signal-noise"),
    ("src/signal-and-noise/analysis/rq03_noise_and_snr", "signal-noise"),
    ("src/signal-and-noise/analysis/rq04_surrogates", "signal-noise"),
    ("src/signal-and-noise/analysis/rq05_design_decisions", "signal-noise"),
    ("src/signal-and-noise/analysis/rq06_language_transfer", "signal-noise"),
    ("src/signal-and-noise/analysis/rq07_external_frameworks", "signal-noise"),
    ("src/signal-and-noise/analysis/rq08_subset_selection", "signal-noise"),
    ("src/signal-and-noise/analysis/rq09_benchmark_design", "signal-noise"),
]

_PLACEHOLDER = (
    "!!! note\n"
    "    This README is not present in this checkout, so it is unavailable in\n"
    "    this build.\n"
)


def on_pre_build(config):
    """With `snippets.check_paths: true`, a missing included README fails the
    build — write a placeholder where the file is absent (an RQ directory
    that a shallow checkout lacks). No-op wherever the READMEs exist."""
    for src_rel, _ in INCLUDES:
        readme = REPO_ROOT / src_rel / "README.md"
        if not readme.is_file():
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(_PLACEHOLDER)


ANALYSIS = REPO_ROOT / "src/signal-and-noise/analysis"
_HIGHLIGHT = re.compile(r"<!-- highlight: (\w+) -->")
_BLOCK = re.compile(r"<!-- BEGIN auto:highlight[^>]*-->\s*## Highlighted result\s*(.*?)<!-- END auto:highlight -->", re.S)


def on_page_markdown(markdown, page, config, files):
    """Showcase pages quote an RQ's regenerated "Highlighted result" with
    `<!-- highlight: rqNN_name -->`, so their headline numbers follow the
    pipeline instead of being copied by hand."""
    def quote(m):
        readme = ANALYSIS / m.group(1) / "README.md"
        found = _BLOCK.search(readme.read_text()) if readme.is_file() else None
        return found.group(1).strip() if found else "*No highlighted result in this checkout.*"
    return _HIGHLIGHT.sub(quote, markdown)


def on_files(files, config):
    docs_dir = config["docs_dir"]
    site_dir = config["site_dir"]
    use_directory_urls = config["use_directory_urls"]

    for src_rel, dest_prefix in INCLUDES:
        src_root = REPO_ROOT / src_rel
        if not src_root.is_dir():
            continue
        for img in src_root.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = img.relative_to(src_root).as_posix()
            url_path = f"{dest_prefix}/{rel}"
            f = File(
                path=url_path,
                src_dir=docs_dir,
                dest_dir=site_dir,
                use_directory_urls=use_directory_urls,
            )
            f.abs_src_path = str(img)
            files.append(f)
    return files

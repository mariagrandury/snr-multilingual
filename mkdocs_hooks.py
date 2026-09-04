from pathlib import Path

from mkdocs.structure.files import File

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# READMEs included via --8<-- use image paths relative to their original
# location. Map each source dir to the URL prefix of the page that includes
# its README so those relative paths resolve in the rendered site.
INCLUDES = [
    ("src/signal-and-noise/analysis/rq00_acc_vs_flops", "signal-noise"),
    ("src/signal-and-noise/analysis/rq01_decision_accuracy", "signal-noise"),
    ("src/signal-and-noise/analysis/rq02_snr_definition", "signal-noise"),
    ("src/signal-and-noise/analysis/rq03_allenai_comparison", "signal-noise"),
    ("src/signal-and-noise/analysis/rq04_smooth_subtasks", "signal-noise"),
    ("src/signal-and-noise/analysis/rq05_benchmark_creation", "signal-noise"),
    ("src/signal-and-noise/analysis/rq06_proxy_predictivity", "signal-noise"),
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

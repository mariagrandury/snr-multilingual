from pathlib import Path

from mkdocs.structure.files import File

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# READMEs included via --8<-- use image paths relative to their original
# location. Map each source dir to the URL prefix of the page that includes
# its README so those relative paths resolve in the rendered site.
INCLUDES = [
    ("src/signal-and-noise/results/snr_definition", "signal-noise"),
    ("src/signal-and-noise/results/allenai_comparison", "signal-noise"),
    ("src/signal-and-noise/results/benchmark_creation", "signal-noise"),
    ("src/signal-and-noise/results/smooth_subtasks", "signal-noise"),
]

_PLACEHOLDER = (
    "!!! note\n"
    "    These results are generated on the cluster and are not checked into\n"
    "    the repository, so they are unavailable in this build.\n"
)


def on_pre_build(config):
    """The included READMEs live under gitignored results/ dirs (generated on
    the cluster). With `snippets.check_paths: true`, a fresh clone (e.g. the
    Netlify deploy preview) would fail the build — write a placeholder where
    the real file is absent. No-op wherever the results actually exist."""
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

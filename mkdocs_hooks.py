import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
# Images referenced by an included README are published under this prefix at
# their repo-relative path, so same-named figures in different dirs never clash.
IMAGE_PREFIX = "repo"

# A docs/ stub is a page whose whole body is one `--8<-- "<README>"` line.
STUB = re.compile(r'\A\s*--8<--\s+"([^"]+)"\s*\Z')
LINK = re.compile(r'(!?\[[^\]]*\])\(([^)\s]+)((?:\s+"[^"]*")?)\)')
FENCE = re.compile(r"^\s*(```|~~~)")

_PLACEHOLDER = (
    "!!! note\n"
    "    This README is not present in this checkout, so it is unavailable in\n"
    "    this build.\n"
)

# Filled in on_files: stub page src_uri -> README repo path, and the reverse.
STUBS: dict[str, str] = {}
PAGES: dict[str, str] = {}


def _outside_fences(text):
    """Yield (line, in_fence) so links inside code blocks stay untouched."""
    in_fence = False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            in_fence = not in_fence
            yield line, True
        else:
            yield line, in_fence


def _resolve(readme, target):
    """Repo-relative path a README-relative link points to, or None when the
    link is external, an in-page anchor, or leaves the repo."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
        return None
    path = target.split("#", 1)[0]
    if not path:
        return None
    rel = posixpath.normpath(posixpath.join(posixpath.dirname(readme), path))
    return None if rel.startswith("..") else rel


def _rewrite(target, readme, page_uri, blob_url):
    rel = _resolve(readme, target)
    if rel is None:
        return target
    anchor = "#" + target.split("#", 1)[1] if "#" in target else ""
    page_dir = posixpath.dirname(page_uri)
    abs_path = REPO_ROOT / rel
    readme_in_dir = posixpath.join(rel, "README.md")
    if rel in PAGES or (abs_path.is_dir() and readme_in_dir in PAGES):
        return posixpath.relpath(PAGES.get(rel) or PAGES[readme_in_dir], page_dir) + anchor
    if abs_path.is_file() and abs_path.suffix.lower() in IMAGE_EXTS:
        return posixpath.relpath(f"{IMAGE_PREFIX}/{rel}", page_dir)
    if abs_path.exists() and blob_url:
        kind = "tree" if abs_path.is_dir() else "blob"
        return f"{blob_url.replace('/blob/', f'/{kind}/')}/{rel}{anchor}"
    # Target is missing from the repo: leave it so mkdocs warns about it.
    return target


def on_files(files, config):
    STUBS.clear()
    PAGES.clear()
    for f in files.documentation_pages():
        m = STUB.match(Path(f.abs_src_path).read_text())
        if m:
            STUBS[f.src_uri] = m.group(1)
            PAGES[m.group(1)] = f.src_uri

    published = set()
    for readme in STUBS.values():
        src = REPO_ROOT / readme
        if not src.is_file():
            continue
        for line, in_fence in _outside_fences(src.read_text()):
            if in_fence:
                continue
            for m in LINK.finditer(line):
                rel = _resolve(readme, m.group(2))
                if (
                    rel is None
                    or rel in published
                    or Path(rel).suffix.lower() not in IMAGE_EXTS
                    or not (REPO_ROOT / rel).is_file()
                ):
                    continue
                published.add(rel)
                f = File(
                    path=f"{IMAGE_PREFIX}/{rel}",
                    src_dir=config["docs_dir"],
                    dest_dir=config["site_dir"],
                    use_directory_urls=config["use_directory_urls"],
                )
                f.abs_src_path = str(REPO_ROOT / rel)
                files.append(f)
    return files


def on_page_markdown(markdown, page, config, files):
    """Inline a stub's README with its relative links resolved against the
    README's own directory: other included READMEs become site pages, images
    point at their published copy, and any other repo file links to GitHub."""
    readme = STUBS.get(page.file.src_uri)
    if readme is None:
        return markdown
    src = REPO_ROOT / readme
    if not src.is_file():
        return _PLACEHOLDER
    repo_url = (config.get("repo_url") or "").rstrip("/")
    blob_url = f"{repo_url}/blob/main" if repo_url else ""

    def sub(m):
        new = _rewrite(m.group(2), readme, page.file.src_uri, blob_url)
        return f"{m.group(1)}({new}{m.group(3)})"

    return "".join(
        line if in_fence else LINK.sub(sub, line)
        for line, in_fence in _outside_fences(src.read_text())
    )

import logging
import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks")

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


# Our own repository on GitHub: READMEs link its files by absolute URL.
OWN_REPO = re.compile(
    r"^https?://github\.com/(?:mariagrandury|swiss-ai)/snr-multilingual"
    r"/(?:blob|tree|raw)/[^/]+/([^#?]*)",
    re.I,
)


def _resolve(readme, target):
    """Repo-relative path a README link points to (relative, or an absolute
    URL into our own repo), or None when the link is external, an in-page
    anchor, or leaves the repo."""
    own = OWN_REPO.match(target)
    if own:
        return posixpath.normpath(own.group(1).rstrip("/") or ".")
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
        return None
    path = target.split("#", 1)[0]
    if not path:
        return None
    rel = posixpath.normpath(posixpath.join(posixpath.dirname(readme), path))
    return None if rel.startswith("..") else rel


def _rewrite(target, readme, page_uri):
    """Site URL for a README link, or None when the target is a repo file the
    site does not publish (the repo cannot be linked during review)."""
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
        if _withheld(rel):
            return None
        return posixpath.relpath(f"{IMAGE_PREFIX}/{rel}", page_dir)
    if abs_path.exists():
        return None
    # Target is missing from the repo: leave it so mkdocs warns about it.
    return None if OWN_REPO.match(target) else target


def _link_readme(text, readme, page_uri):
    def sub(m):
        new = _rewrite(m.group(2), readme, page_uri)
        if new is None:
            return _dropped(m.group(1))
        return f"{m.group(1)}({new}{m.group(3)})"

    return "".join(
        line if in_fence else LINK.sub(sub, line)
        for line, in_fence in _outside_fences(text)
    )


# --- Double-blind review -------------------------------------------------
# The site must not link or name our HF orgs, W&B, GitHub, personal pages,
# authors or cluster paths. A link to any of them becomes the
# `extra.anonymity_notice` sentence of mkdocs.yml (pages write
# `{{ anonymity_notice }}`, app.js reads <meta name="anonymity-notice">);
# a bare mention in text or code becomes a neutral placeholder.
NOTICE_TOKEN = "{{ anonymity_notice }}"
_NOTICE = {"text": ""}  # the wording lives only in mkdocs.yml extra.anonymity_notice
_PEOPLE = r"mariagrandury|aromanou|maria[_ ]?grandury|antonia[_ ]?romanou|grandury|romanou"
# Affiliation, cluster and personal hosts: any URL containing one becomes the notice.
_AFFIL = r"swiss[- _]?ai|cscs|clariden|epfl|claude\.ai/"
_ORGS = r"swiss-ai|msnr|msnr-data|multilingual-snr|snr-models"
_URL = re.compile(r"https?://[^\s<>\"'`)\]]+")
_BLOCKED_URL = re.compile(
    r"^https?://(?:[\w-]+\.)*(?:wandb\.ai|wandb\.me)/[\w-]"
    rf"|^https?://(?:www\.)?(?:github\.com|huggingface\.co|hf\.co)/(?:datasets/|spaces/)?(?:{_PEOPLE}|{_ORGS})(?:[/?#]|$)"
    rf"|{_PEOPLE}|{_AFFIL}",
    re.I,
)
_REDACT = [
    (re.compile(r"/(?:iopsstor|capstor)/(?:scratch|store|users)/cscs/[\w.-]+"), "/path/to/storage"),
    (re.compile(r"/(?:iopsstor|capstor)(?:/(?:scratch|store|users))?(?:/cscs)?"), "/path/to/storage"),
    (re.compile(rf"/users/(?:{_PEOPLE})"), "~"),
    (re.compile(r"[\w-]*epflnlp"), "anon-entity"),
    # A whole `<org>/<repo>` id becomes the notice: keeping the repo name
    # (e.g. Apertus-70B-2509, Megatron-LM) would still point at the org.
    (re.compile(rf"(?<![\w/.-])(?:{_ORGS})/[\w.-]+"), lambda m: _NOTICE["text"]),
    (re.compile(_PEOPLE, re.I), "anon"),
    # Affiliation and unreleased internal checkpoints in prose and code.
    (re.compile(r"custom_swissai_hf"), "custom_hf"),
    (re.compile(r"apertus3(?:[- ]a06)?(?:-[\w{},*.-]*)?|[\w-]*from8b[\w-]*|\ba06\b", re.I), "internal-checkpoint"),
    # Our own runs and containers are named after the Apertus architecture
    # (`apertus-350M-fwEdu60-...`, `apertus-eval`); the released Apertus
    # models (`Apertus-8B-2509`, capitalised) are third-party and stay.
    (re.compile(r"\bapertus-(?=[\d*{])"), "model-"),
    (re.compile(r"\bapertus-(eval|nemo)\b"), r"\1-container"),
    (re.compile(r"\bapertus-(?=data-mix)"), ""),
    # Cluster filesystems, Slurm accounts and machine names (the affiliation).
    (re.compile(r"iopsstor", re.I), "scratch"),
    (re.compile(r"capstor", re.I), "archive"),
    (re.compile(r"(?<!\w)(?:a139|infra01)(?!\w)"), "<account>"),
    (re.compile(r"\balps\d*\b", re.I), "cluster"),
    # Our Azure ML storage account and blob container.
    (re.compile(r"(?<![\w<>-])[a-z0-9]{3,24}\.blob\.core\.windows\.net(?=[/\s\"'`]|$)"), "<account>.blob.core.windows.net"),
    (re.compile(r"azureml-blobstore-[0-9a-f-]+"), "<container>"),
    (re.compile(r"snreswsstorage\w*"), "<account>"),
    (re.compile(r"\b(?:CSCS|Clariden|Alps)(?:\s+(?:cluster|supercomputer))?\b"), "our cluster"),
    (re.compile(r"cscs|clariden", re.I), "cluster"),
    (re.compile(r"swiss[- _]?ai|\bepfl\b", re.I), "anon-org"),
    (re.compile(r"[\w.+-]+@(?!example\.)[\w-]+(?:\.[\w-]+)+"), "anon@example.org"),
]
# What must never reach the built site; on_post_build warns on any hit.
# The affiliation, cluster and internal checkpoints are redacted above; a hit
# here means a page or data file bypassed _anonymize (static JSON, raw HTML).
_LEAK = re.compile(rf"{_PEOPLE}|epfl|swiss[- _]?ai|cscs|clariden|claude\.ai/|apertus3|from8b|github\.com/swiss-ai|huggingface\.co/(?:datasets/)?(?:{_ORGS})\b|wandb\.ai/\w|iopsstor|capstor|(?<!\w)(?:a139|infra01)(?!\w)|\balps\d*\b|anon-org/|\ba06\b|snreswsstorage|azureml-blobstore-[0-9a-f]", re.I)


# Figures whose pixels name internal lines (the a06 / distillation / Swiss-AI
# reference legends of the custom_swissai_hf and all/external scopes, and the
# rq00 external-floor figure): the text checks cannot read inside a PNG, so
# these are not published while under review and their reference keeps its
# alt text plus the notice.
_WITHHELD_IMAGE = re.compile(r"/custom_swissai_hf/|/all/external/|/above_random_external[^/]*$", re.I)


def _withheld(rel):
    return bool(_WITHHELD_IMAGE.search("/" + rel) or _LEAK.search(rel))


def _dropped(label):
    """A link that cannot be shown: keep its text, append the notice."""
    text = label[2:-1] if label.startswith("!") else label[1:-1]
    return f"{text} (*{_NOTICE['text']}*)"


def _anonymize(markdown):
    def link(m):
        return _dropped(m.group(1)) if _BLOCKED_URL.search(m.group(2)) else m.group(0)

    def bare(m):
        return f"*{_NOTICE['text']}*" if _BLOCKED_URL.search(m.group(0)) else m.group(0)

    out = []
    for line, in_fence in _outside_fences(markdown.replace(NOTICE_TOKEN, _NOTICE["text"])):
        if not in_fence:
            line = LINK.sub(link, line)
        line = _URL.sub(bare if not in_fence else (lambda m: _NOTICE["text"] if _BLOCKED_URL.search(m.group(0)) else m.group(0)), line)
        for pattern, repl in _REDACT:
            line = pattern.sub(repl, line)
        out.append(line)
    return "".join(out)


def on_config(config):
    _NOTICE["text"] = (config.get("extra") or {}).get("anonymity_notice") or ""
    if not _NOTICE["text"]:
        log.warning("anonymity: mkdocs.yml extra.anonymity_notice is missing; blocked links lose their notice")
    return config


ANALYSIS = REPO_ROOT / "src/signal-and-noise/analysis"
_HIGHLIGHT = re.compile(r"<!-- highlight: (\w+) -->")
_BLOCK = re.compile(r"<!-- BEGIN auto:highlight[^>]*-->\s*## Highlighted result\s*(.*?)<!-- END auto:highlight -->", re.S)


def on_files(files, config):
    STUBS.clear()
    PAGES.clear()
    for f in files.documentation_pages():
        m = STUB.match(Path(f.abs_src_path).read_text())
        if m:
            STUBS[f.src_uri] = m.group(1)
            PAGES[m.group(1)] = f.src_uri

    # READMEs included by a stub, and showcase pages (which may embed a repo
    # figure by its repo-relative path, e.g. ../../src/.../figure.png).
    sources = [(readme, REPO_ROOT / readme) for readme in STUBS.values()]
    sources += [(f"docs/{f.src_uri}", Path(f.abs_src_path)) for f in files.documentation_pages() if f.src_uri not in STUBS]
    published = set()
    for readme, src in sources:
        if not src.is_file():
            continue
        for line, in_fence in _outside_fences(src.read_text()):
            if in_fence:
                continue
            for m in LINK.finditer(line):
                rel = _resolve(readme, m.group(2))
                if (
                    rel is None
                    or rel.startswith("docs/")
                    or rel in published
                    or Path(rel).suffix.lower() not in IMAGE_EXTS
                    or not (REPO_ROOT / rel).is_file()
                    or _withheld(rel)
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

    # The benchmark catalogue lives in configs/; the Benchmarks tab reads it.
    f = File(
        path="interactive/data/benchmarks.csv",
        src_dir=config["docs_dir"],
        dest_dir=config["site_dir"],
        use_directory_urls=config["use_directory_urls"],
    )
    f.abs_src_path = str(REPO_ROOT / "configs/multilingual_benchmarks.csv")
    files.append(f)
    return files


def on_page_markdown(markdown, page, config, files):
    """Stub pages inline their README with its relative links resolved
    against the README's own directory: other included READMEs become site
    pages, images point at their published copy, and other repo files keep
    their text plus the anonymity notice. Showcase pages quote an RQ's
    regenerated "Highlighted result" with `<!-- highlight: rqNN_name -->`, so
    their headline numbers follow the pipeline instead of being copied by
    hand. Every page is then anonymized."""
    readme = STUBS.get(page.file.src_uri)
    if readme is not None:
        src = REPO_ROOT / readme
        markdown = _link_readme(src.read_text(), readme, page.file.src_uri) if src.is_file() else _PLACEHOLDER
    else:
        def quote(m):
            rq = f"src/signal-and-noise/analysis/{m.group(1)}/README.md"
            found = _BLOCK.search((REPO_ROOT / rq).read_text()) if (REPO_ROOT / rq).is_file() else None
            if not found:
                return "*No highlighted result in this checkout.*"
            return _link_readme(found.group(1).strip(), rq, page.file.src_uri)

        markdown = _HIGHLIGHT.sub(quote, markdown)
        markdown = _link_repo_images(markdown, f"docs/{page.file.src_uri}", page.file.src_uri)
    return _anonymize(markdown)


def _link_repo_images(text, page_path, page_uri):
    """A showcase page's images that live in the repo outside docs/ point at
    their published copy (on_files); every other link is left alone."""
    def sub(m):
        rel = _resolve(page_path, m.group(2))
        if not m.group(1).startswith("!") or rel is None or rel.startswith("docs/") or not (REPO_ROOT / rel).is_file():
            return m.group(0)
        if _withheld(rel):
            return _dropped(m.group(1))
        return f"{m.group(1)}({posixpath.relpath(f'{IMAGE_PREFIX}/{rel}', posixpath.dirname(page_uri))}{m.group(3)})"

    return "".join(line if in_fence else LINK.sub(sub, line) for line, in_fence in _outside_fences(text))


_HREF = re.compile(r'<a\b[^>]*\bhref="([^"]*)"[^>]*>(.*?)</a>', re.S)


def on_post_page(output, page, config):
    """Safety net on the rendered HTML (theme, raw HTML in pages), and the
    <meta> app.js reads the notice from."""
    output = _HREF.sub(
        lambda m: f'{m.group(2)} (<em>{_NOTICE["text"]}</em>)' if _BLOCKED_URL.search(m.group(1)) else m.group(0),
        output,
    )
    meta = f'<meta name="anonymity-notice" content="{_NOTICE["text"]}">'
    return output.replace("</head>", f"{meta}\n</head>", 1)


def on_post_build(config):
    """Warn (fails `mkdocs build --strict`) on any identity left in the site."""
    for f in Path(config["site_dir"]).rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS - {".svg"} and _LEAK.search(str(f.relative_to(config["site_dir"]))):
            log.warning("anonymity: published file name %s names the affiliation", f.relative_to(config["site_dir"]))
        if f.suffix.lower() not in {".html", ".js", ".json", ".csv", ".xml", ".txt", ".css", ".svg"}:
            continue
        rel = str(f.relative_to(config["site_dir"]))
        hits = set(m.group(0) for m in _LEAK.finditer(rel + "\n" + f.read_text(errors="ignore")))
        if hits:
            log.warning("anonymity: %s contains %s", f.relative_to(config["site_dir"]), sorted(hits))

"""Double-blind review: the one place the text that stands in for a removed link lives.

The website (mkdocs: `docs/`, `mkdocs.yml`, `mkdocs_hooks.py`, all at the repo
root next to this file, so a hook can `from anonymity import REMOVED_LINK`; not
wired yet), the docs and the paper must not link to any of our Hugging Face organisations
or profiles, W&B, GitHub, or a personal or professional site while the paper is
under review: a single such link is grounds for automatic rejection
(src/signal-and-noise/analysis/RULES.md, rule 19). Wherever a link to one of
them would go, write REMOVED_LINK instead; change the wording here only.
"""

REMOVED_LINK = "Link momentarily removed for double blind review"

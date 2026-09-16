"""The deck palette, re-exported from the analysis so the figure scripts under
documents/ and under src/signal-and-noise/analysis/ share one style
(``analysis/style.py`` is the definition)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "signal-and-noise"))
from analysis.style import *  # noqa: E402,F401,F403

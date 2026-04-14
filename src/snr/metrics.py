"""Thin wrapper around signal-and-noise metrics.

Re-exports the core functions from the reference implementation
(Heineman et al., 2025) at src/signal-and-noise/.
"""

import sys
from pathlib import Path

# Make the reference implementation importable
_SNR_ROOT = str(Path(__file__).resolve().parent.parent / "signal-and-noise")
if _SNR_ROOT not in sys.path:
    sys.path.insert(0, _SNR_ROOT)

from snr.metrics import decision_acc_fast as decision_accuracy  # noqa: E402
from snr.metrics import signal_to_noise_ratio  # noqa: E402

__all__ = ["signal_to_noise_ratio", "decision_accuracy"]

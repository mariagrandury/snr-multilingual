## 3. Rank the benchmarks of your own suite

Export your scores as a CSV with one row per (recipe, size, checkpoint,
task). The analysis runs in your browser; the file is never uploaded.

<div class="viz" data-viz="upload"></div>

| column | required | meaning |
|---|---|---|
| `recipe` | yes | the design variant you are choosing between (a data mix, an architecture, …) |
| `task` | yes | the benchmark (one row per task; per-language tasks are separate tasks) |
| `score` | yes | the metric; higher is better unless `lower_is_better` is set |
| `size` | no | model size label (`150M`, `1B`); with two sizes you get decision accuracy small → large |
| `step` | no | training step; several late checkpoints per recipe give the noise estimate and the early → final decision accuracy |
| `chance` | no | chance level of the task (e.g. `0.25` for four options) |
| `lower_is_better` | no | `1` for losses and bits per byte |

The computation mirrors
`snr/metrics.py` (*{{ anonymity_notice }}*):
signal is `(max − min) / mean` of the recipes' final scores, noise is the
standard deviation over each recipe's last checkpoints divided by their mean,
and decision accuracy is the share of recipe pairs ordered the same way by the
proxy and by the target (ties count as agreement only when both sides tie).
For the full 22 SNR definitions, use the [analysis pipeline](signal-noise/index.md).

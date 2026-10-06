# Decision accuracy against the tokens of the language seen

Does a language's BPB rank the design variants like the reference once the
proxy has seen enough of that language? Per proxy size and tenth of the run,
the mean over languages of DA-goal (against the 1.7B final), DA-ckpt (against
the proxy size's own final) and DA-size, against the training tokens of the
language the proxies had seen; every figure over the multi-axis pairs, with
mono-axis twins of the DA-goal and DA-ckpt figures. The figures and tables are
written by
[`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py)
(pool `predictivity_all`), from the ladder-report snapshot of 2026-10-06; they
were moved here from `rq01_scaling_predictability/` on 2026-10-07. The paper
copy is `da_goal_multi_axes_across_langs_bpb_paper.png`. The above-chance
counterpart is in
[`../rq00_chance_vs_train_tokens/`](../rq00_chance_vs_train_tokens/README.md).

**The 600M dip is the depth axis, not a loader fault.** DA-goal at the final
checkpoint is 0.92 at 350M, 0.74 at 600M and 0.92 at 1B over the same pair
set. At the 600M final the shallow cell has a lower BPB than its deep twin in
96 % of the matched (L, scheme, language) comparisons, against at most 28 % at
every other size (0.5 % at 1.7B), so every deep/shallow pair flips against the
reference: in an L50 language, two of the six pairs, hence the many languages
at 0.67. The 600M shallow rung trains on 15 % more compute than the deep one
(3 to 9 % at the other sizes). The dip is a property of the 600M rung's
design, and the figure needs that sentence wherever it is shown.

<!-- BEGIN auto:da-vs-train-tokens (tokens_seen.py --pool predictivity_all) -->
## Decision accuracy against the tokens of the language seen

`tokens_seen.py`, `predictivity_all` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_all`.

**DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages of the share of design-variant pairs the proxy's BPB orders like the 1.7B final (rq02's kernel, every pair at seed 1904 of every scheme on the variants that train the language, ≥ 3 pairs; rule 15's multi-axis set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of the language they had seen, and a point's x the geometric mean over languages (50 languages; `da_goal_multi_axes_across_langs_bpb_cells.csv` has the per-language cells with the min and max over variants). The 1.7B line is its own early checkpoints against its final. `da_ckpt_multi_axes_across_langs_bpb` ranks each checkpoint against the proxy size's OWN final (DA-ckpt) and `da_size_multi_axes_across_langs_bpb` the final of every size against the 1.7B final (DA-size, one point per size: the 100 % end of the DA-goal lines); the `_vs_frac` twins of the goal and ckpt figures put the same points against the share of the run the checkpoint sits at, where lines that are apart on the token axis falling together says the schedule, not the exposure, decides. The `_mono_axis` twins of the goal and ckpt figures read only the pairs that move one design axis (rule 15).

| proxy size | tokens of a language at 1C | DA at 1C | tokens at 5C | DA at 5C | languages |
|---|---|---|---|---|---|
| 90M | 0.02 B | 0.95 | 0.08 B | 0.96 | 50 |
| 175M | 0.03 B | 0.92 | 0.15 B | 0.96 | 50 |
| 350M | 0.06 B | 0.85 | 0.29 B | 0.92 | 50 |
| 600M | 0.10 B | 0.72 | 0.51 B | 0.74 | 50 |
| 1B | 0.16 B | 0.79 | 0.80 B | 0.92 | 50 |

![DA-goal of BPB vs tokens seen](pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 10%.

![DA-goal of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_all/da_goal_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.60 (600M) to 0.94 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 20%, 600M never, 1B 40%.
- The mono-axis pairs read lower than the multi-axis set at 50 of 50 (size, tenth) points, by 0.07 on average.

![DA-goal of BPB vs share of the run](pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 10%.

![DA-ckpt of BPB vs tokens seen](pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.99 (90M) to 1.00 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-ckpt of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_all/da_ckpt_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.98 (90M) to 0.99 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 30%.
- The mono-axis pairs read lower than the multi-axis set at 43 of 45 (size, tenth) points, by 0.03 on average.

![DA-ckpt of BPB vs share of the run](pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_ckpt_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.99 (90M) to 1.00 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-size of BPB vs tokens seen](pretraining/predictivity_all/da_size_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_size_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_all/da_size_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
<!-- END auto:da-vs-train-tokens -->

Follow-ups:
- A DA-goal figure with the 600M shallow rung left out, to show how much of the 600M dip the depth axis alone explains.
- The same figures per language resource tier, to see whether low-resource languages reach the threshold later in tokens or only later in the run.
- A seed-pair twin (the seed set of decision accuracy), to place the curves against the seed noise floor.

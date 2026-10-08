# Decision accuracy against the tokens of the language seen

Does a language's BPB rank the design variants like the reference once the
proxy has seen enough of that language? Each point is the mean over 50
languages (one BPB task each) of a decision accuracy at one proxy size and
tenth of the run, placed at the tokens of that language the proxies had seen.

Setup: pool `predictivity_seeds`, the pairs at seed 1904 across every data
build (scheme A/B/C × temperature) among the variants that train the language,
multi-axis set, at least 3 pairs per language (median 36, 6 to 325), proxies
90M–1B and the 1.7B line as its own early checkpoints against its final, no
task filter. Three accuracies: DA-goal (against the 1.7B final), DA-ckpt
(against the proxy size's own final) and DA-size (each size's final against
the 1.7B final), DA-goal and DA-ckpt each with a mono-axis twin.

Numbers are from the ladder report of 2026-10-08 12:06 and the outputs of the
2026-10-08 refresh (detrended checkpoint noise; prose re-read 2026-10-08),
written by
[`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py);
the figures moved here from `rq01_scaling_predictability/` on 2026-10-07. The
above-chance counterpart is in
[chance against tokens seen](../rq00_chance_vs_train_tokens/README.md), and
decision accuracy over every task in
[decision accuracy](../rq02_decision_accuracy/README.md).

## Opening figure: DA-goal of BPB against tokens seen

![DA-goal of BPB vs tokens seen, paper copy](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.csv)

DA-goal, no filter, multi-axis pairs, pool `predictivity_seeds`, 50
languages; the paper copy of the first figure in the generated block below.

Key findings:
- **The 600M rung is the outlier, not the smallest proxy.** At the final
  checkpoint DA-goal is 0.96 at 90M and 175M, 0.92 at 350M, 0.74 at 600M and
  0.92 at 1B (standard error over languages 0.006–0.010).
- **Tokens seen do not set the level.** The 90M proxies read 0.95 at their
  first tenth on 0.008 B tokens of a language, while the 600M proxies read
  0.74 at their final on 0.51 B; 1B at its first tenth (0.08 B) reads 0.76
  against 0.96 for 175M at its final (0.15 B).
- **More training does not close the 600M gap.** 600M stays at 0.70–0.74 over
  the whole run, while 350M climbs from 0.83 to 0.92 and 1B from 0.76 to 0.92.
- **The reference settles early.** The 1.7B line reads 0.91 against its own
  final at the first tenth, 0.96 at the second and 0.996 at the ninth.
- **The 600M dip is sharpest in the smallest pair sets.** All 16 languages
  with four variants (six pairs) read exactly 0.67 at the 600M final, two of
  six pairs flipped, against 0.97–0.99 at 90M–1B; they are 16 of the 18
  languages at 0.67.
- **The 600M proxies agree with themselves, not with the reference.** Their
  DA-ckpt is 0.91 at the first tenth and 0.994 at the ninth, the highest proxy
  there, so the 600M order settles early on a ranking the 1.7B final does not
  share.
- **Final training loss does not single out 600M.** The shallow cell ends
  below its deep twin in 7 of 10 matched (L, data build) pairs at 600M and
  also at 1B, where DA-goal is 0.92
  ([`scaling_fit.csv`](../rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_fit.csv)).
  The depth axis is the likely driver: the 600M depth pairs are not
  size-matched (shallow has 3.7 % more non-embedding parameters,
  [RULES.md, "Size matching"](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)),
  and BPB orders them like the 1.7B final at 0.06 against 0.69–0.97 at the
  other proxies (DA-size, rq02's
  [`scale_convergence_da_size_transformation_multi_axes.csv`](../rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.csv)).

Follow-ups:
- A per-pair breakdown of the 600M final (which design axis each flipped pair
  moves), to confirm per language that the depth axis drives the dip; this
  folder's CSVs give only per-language DA, rq02's per-axis lines only the pool.
- The same figures on the benchmark-BPB twins. They now exist at every
  checkpoint of every seed-1904 cell (`bench_bpb.csv`, rebuilt 2026-10-08 with
  the seed replicates),
  but `tokens_seen.py` reads the per-language BPB only.

<!-- BEGIN auto:da-vs-train-tokens (tokens_seen.py --pool predictivity_seeds) -->
## Decision accuracy against the tokens of the language seen

`tokens_seen.py`, `predictivity_seeds` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds`.

**DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages of the share of design-variant pairs the proxy's BPB orders like the 1.7B final (rq02's kernel, every pair at seed 1904 of every data build on the variants that train the language, ≥ 3 pairs; rule 15's multi-axis set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of the language they had seen, and a point's x the geometric mean over languages (50 languages; `da_goal_multi_axes_across_langs_bpb_cells.csv` has the per-language cells with the min and max over variants). The 1.7B line is its own early checkpoints against its final. `da_ckpt_multi_axes_across_langs_bpb` ranks each checkpoint against the proxy size's OWN final (DA-ckpt) and `da_size_multi_axes_across_langs_bpb` the final of every size against the 1.7B final (DA-size, one point per size: the 100 % end of the DA-goal lines); the `_vs_frac` twins of the goal and ckpt figures put the same points against the share of the run the checkpoint sits at, where lines that are apart on the token axis falling together says the schedule, not the exposure, decides. The `_mono_axis` twins of the goal and ckpt figures read only the pairs that move one design axis (rule 15).

| proxy size | tokens of a language at 1C | DA at 1C | tokens at 5C | DA at 5C | languages |
|---|---|---|---|---|---|
| 90M | 0.02 B | 0.95 | 0.08 B | 0.96 | 50 |
| 175M | 0.03 B | 0.92 | 0.15 B | 0.96 | 50 |
| 350M | 0.06 B | 0.85 | 0.29 B | 0.92 | 50 |
| 600M | 0.10 B | 0.72 | 0.51 B | 0.74 | 50 |
| 1B | 0.16 B | 0.79 | 0.80 B | 0.92 | 50 |

![DA-goal of BPB vs tokens seen](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 10%.

![DA-goal of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.60 (600M) to 0.94 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 20%, 600M never, 1B 40%.
- The mono-axis pairs read lower than the multi-axis set at 50 of 50 (size, tenth) points, by 0.07 on average.

![DA-goal of BPB vs share of the run](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 10%.

![DA-ckpt of BPB vs tokens seen](pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.99 (90M) to 0.99 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-ckpt of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.97 (90M) to 0.99 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 30%.
- The mono-axis pairs read lower than the multi-axis set at 39 of 45 (size, tenth) points, by 0.03 on average.

![DA-ckpt of BPB vs share of the run](pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.99 (90M) to 0.99 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-size of BPB vs tokens seen](pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.96 (175M).
<!-- END auto:da-vs-train-tokens -->

Key findings (generated figures):
- **The mono-axis pairs read lower for DA-goal at every point (50 of 50), and for DA-ckpt at 39 of 45.** DA-goal at the final is 0.60 at 600M and 0.88–0.94 at the other proxies over the mono-axis pairs (median 12 per language), against 0.74 and 0.92–0.96 over the multi-axis set.
- **DA-size has no size trend.** The finals read 0.96 (90M), 0.96 (175M), 0.92 (350M), 0.74 (600M) and 0.92 (1B), so a larger proxy does not rank this BPB population more like the 1.7B.

Follow-ups:
- A DA-goal figure with the 600M shallow rung left out, to show how much of the 600M dip the depth axis alone explains.
- The same figures per language resource tier, to see whether low-resource languages reach the threshold later in tokens or only later in the run.
- A seed-pair twin (the seed set of decision accuracy), to place the curves against the seed noise floor.

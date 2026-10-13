# Decision accuracy against the tokens of the language seen

Does a language's BPB rank the design variants like the 1.7B reference once the
proxy has seen enough of that language? Each point averages one decision
accuracy over 50 languages (one BPB task each), at one proxy size and one
tenth of the run. It sits at the number of tokens of that language the proxies
had seen.

**Setup.**
- **Measured.** We measure the DA-goal of per-language BPB against the tokens of that language the model has seen. DA-goal is the share of design pairs that a checkpoint orders like the final 1.7B checkpoint.
- **Models.** We use every data build at seed 1904: the proxies 90M–1B at each evaluated tenth of their run, and the early checkpoints of the 1.7B run.
- **Tasks and pairs.** Each of the 50 trained languages gives one task, its BPB. A language's pairs are the pairs of design variants that train it, on any number of design axes, with at least three pairs per language.
- **How.** A point is the mean over languages at one size and tenth, placed at the tokens of the language seen.

Pools and pairs: pool `predictivity_seeds`, multi-axis set, with the pairs at
seed 1904 across every data build (scheme A/B/C × temperature) among the
variants that train the language. A language needs at least 3 pairs (median 45,
6 to 406), and no task filter applies.

The proxies are 90M–1B, and the 1.7B line compares the reference's own early
checkpoints against its final. Three accuracies appear: DA-goal (against the
1.7B final), DA-ckpt (against the proxy size's own final) and DA-size (each
size's final against the 1.7B final), with a mono-axis twin for DA-goal and
DA-ckpt.

Numbers are from the ladder report of 2026-10-10 04:31 and the outputs of the
2026-10-10 refresh (detrended checkpoint noise, prose re-read 2026-10-10).
They are written by
[`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py),
and the figures moved here from `rq01_scaling_predictability/` on 2026-10-07.

The above-chance counterpart is in
[chance against tokens seen](../rq00_chance_vs_train_tokens/README.md), and
decision accuracy over every task is in
[decision accuracy](../rq02_decision_accuracy/README.md).

## Opening figure: DA-goal of BPB against tokens seen

![DA-goal of BPB vs tokens seen, paper copy](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.csv)

DA-goal, no filter, multi-axis pairs, pool `predictivity_seeds`, 50
languages. This is the paper copy of the first figure in the generated block below.

Key findings:
- **The 600M rung is the outlier, not the smallest proxy.** At the final
  checkpoint DA-goal is 0.95 at 90M and 175M, 0.91 at 350M, 0.73 at 600M and
  0.92 at 1B (standard error over languages 0.006–0.011).
- **Tokens seen do not set the level.** The 90M proxies read 0.92 at their
  first tenth on 0.008 B tokens of a language, while the 600M proxies read
  0.73 at their final on 0.50 B, and 1B at its first tenth (0.08 B) reads 0.75
  against 0.95 for 175M at its final (0.15 B).
- **More training does not close the 600M gap.** 600M stays at 0.69–0.73 over
  the whole run, while 350M climbs from 0.80 to 0.91 and 1B from 0.75 to 0.92.
- **The reference settles early.** The 1.7B line reads 0.90 against its own
  final at the first tenth, 0.96 at the second and 0.994 at the ninth.
- **The 600M dip is sharpest in the smallest pair sets.** All 16 languages
  with four variants (six pairs) read exactly 0.67 at the 600M final, which
  means two of the six pairs flipped. The same languages read 0.97–0.99 at the
  other proxies, and no other language reads 0.67 at 600M.
- **The 600M proxies agree with themselves, not with the reference.** Their
  DA-ckpt is already 0.91 at the first tenth, above 350M (0.83) and 1B (0.77),
  and 0.97 at the ninth. The 600M order thus settles early on a ranking that
  the 1.7B final does not share.
- **Final training loss does not single out 600M.** The shallow cell ends
  below its deep twin in 7 of 10 matched (L, data build) pairs at 600M and
  also at 1B, where DA-goal is 0.92
  ([`scaling_fit.csv`](../rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_fit.csv)).
  The depth axis is the likely driver. The 600M depth pairs are not
  size-matched (shallow has 3.7 % more non-embedding parameters,
  [RULES.md, "Size matching"](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)),
  and on them BPB matches the 1.7B order at a DA-size of 0.06, against
  0.69–0.97 at the other proxies (rq02's
  [`scale_convergence_da_size_transformation_multi_axes.csv`](../rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.csv)).

Follow-ups:
- **Which axis each 600M flip moves.** A per-pair breakdown of the 600M final
  would confirm per language that the depth axis drives the dip. This
  folder's CSVs give only per-language DA, and rq02's per-axis lines give only
  the pool.
- **The benchmark-BPB twins.** The same figures would read the twins, which
  now exist at every checkpoint of every seed-1904 cell (`bench_bpb.csv`,
  rebuilt 2026-10-08 with the seed replicates). `tokens_seen.py` reads only
  the per-language BPB.

<!-- BEGIN auto:da-vs-train-tokens (tokens_seen.py --pool predictivity_seeds) -->
## Decision accuracy against the tokens of the language seen

`tokens_seen.py`, `predictivity_seeds` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds`.

**DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages of the share of design-variant pairs the proxy's BPB orders like the 1.7B final (rq02's kernel, every pair at seed 1904 of every data build on the variants that train the language, ≥ 3 pairs; rule 15's multi-axis set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of the language they had seen, and a point's x the geometric mean over languages (50 languages; `da_goal_multi_axes_across_langs_bpb_cells.csv` has the per-language cells with the min and max over variants). The 1.7B line is its own early checkpoints against its final. `da_ckpt_multi_axes_across_langs_bpb` ranks each checkpoint against the proxy size's OWN final (DA-ckpt) and `da_size_multi_axes_across_langs_bpb` the final of every size against the 1.7B final (DA-size, one point per size: the 100 % end of the DA-goal lines); the `_vs_frac` twins of the goal and ckpt figures put the same points against the share of the run the checkpoint sits at, where lines that are apart on the token axis falling together says the schedule, not the exposure, decides. The `_mono_axis` twins of the goal and ckpt figures read only the pairs that move one design axis (rule 15).

| proxy size | tokens of a language at 1C | DA at 1C | tokens at 5C | DA at 5C | languages |
|---|---|---|---|---|---|
| 90M | 0.02 B | 0.90 | 0.08 B | 0.94 | 50 |
| 175M | 0.03 B | 0.87 | 0.14 B | 0.94 | 50 |
| 350M | 0.06 B | 0.82 | 0.28 B | 0.89 | 50 |
| 600M | 0.10 B | 0.70 | 0.50 B | 0.74 | 50 |
| 1B | 0.16 B | 0.76 | 0.78 B | 0.92 | 50 |

![DA-goal of BPB vs tokens seen](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.94 (90M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 20%.

![DA-goal of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.57 (600M) to 0.92 (90M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 50%, 600M never, 1B 40%.
- The mono-axis pairs read lower than the multi-axis set at 42 of 50 (size, tenth) points, by 0.05 on average.

![DA-goal of BPB vs share of the run](pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.94 (90M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M never, 1B 20%.

![DA-ckpt of BPB vs tokens seen](pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.94 (1B) to 0.97 (350M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-ckpt of BPB vs tokens seen, mono-axis pairs](pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_mono_axis_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.94 (90M) to 0.99 (600M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 40%.
- The mono-axis pairs read lower than the multi-axis set at 20 of 45 (size, tenth) points, by 0.00 on average.

![DA-ckpt of BPB vs share of the run](pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_ckpt_multi_axes_across_langs_bpb_vs_frac.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.94 (1B) to 0.97 (350M).
- First tenth of the run at DA ≥ 0.75: 90M 10%, 175M 10%, 350M 10%, 600M 10%, 1B 10%.

![DA-size of BPB vs tokens seen](pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_size_multi_axes_across_langs_bpb.csv)

Key findings:
- Over 50 languages, the proxies' last point ranges from 0.74 (600M) to 0.94 (90M).
<!-- END auto:da-vs-train-tokens -->

Key findings (generated figures):
- **The mono-axis pairs read lower for DA-goal at every point (50 of 50), but for DA-ckpt only at 23 of 45.** Over the mono-axis pairs (median 13 per language), DA-goal at the final is 0.57 at 600M and 0.88–0.93 at the other proxies, against 0.73 and 0.91–0.95 over the multi-axis set.
- **DA-size has no size trend.** The finals read 0.95 (90M), 0.95 (175M), 0.91 (350M), 0.73 (600M) and 0.92 (1B), so a larger proxy does not rank this BPB population more like the 1.7B.

Follow-ups:
- **Without the 600M shallow rung.** A DA-goal figure with that rung left out would show how much of the 600M dip the depth axis alone explains.
- **Per resource tier.** The same figures per language resource tier would show whether low-resource languages reach the threshold later in tokens or only later in the run.
- **A seed-pair twin.** The seed set of decision accuracy would place the curves against the seed noise floor.

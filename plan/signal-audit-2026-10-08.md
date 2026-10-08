# Signal audit: why so little signal, DA and SNR? (2026-10-08)

Question: "investigate in depth why we have so little signal, DA, SNR. I really
expected at least the L1 models to show similar trends to what the
english-only papers like allenai show. Do we have a bug somewhere?"

Scope: data build, training, conversion, evaluation, the above-chance gate,
decision accuracy (DA) and SNR. Snapshot: `ladder_report.csv` of 2026-10-08
04:36 (cached in `src/signal-and-noise/data/ladder-report/`), pool
`predictivity` (seed 1904) unless stated. No job was submitted, no code or data
was changed. Audit scripts live in the session scratchpad (not committed); the
commands that matter are quoted below.

## 1. Verdict

There is no bug that explains the low signal. The models train correctly and
score where models of their size should: the 1.7B L1 deep cell matches
OLMo-2-1B (4T tokens) on HellaSwag and ARC through the same pipeline. The
analysis code reproduces AllenAI's own DA on DataDecide data to within 0.03.
The low DA is mostly what the design predicts. A typical mono-axis pair differs
by about 0.8 noise SD, which caps DA near 0.56, and that is what we observe.

On top of that, three of our measurement choices depress the numbers:

- The checkpoint noise is read over the WSD decay, where scores still rise. This
  inflates noise 1.5–4x on the tasks with real signal.
- Letter-format tasks are scored from bf16 logits. 21–38 % of their items end
  in an exact tie, broken towards the first option.
- Cloze tasks are scored with raw `acc`.

We also found one design confound (600M shallow is not size-matched) and one
real pipeline deviation (no EOD between training documents). The deviation
measured as harmless on loss.

## 2. Findings, ranked by how much they explain

### F1. The design produces small effects, so low DA is expected (explains DA; design, not a bug)

Evidence (scripts `design/q1.py`, `q2.py`, `q4*.py`; final checkpoints, gated,
71 mono-axis pairs at 1B and 1.7B: L 42, scheme 12, arch 10, T 4, activation 3):

- **Gap sizes.** Median |Δ| between the two cells of a mono-axis pair at 1.7B:
  - English core accuracy (HellaSwag, ARC-E, ARC-C, rf_MMLU): 0.68 points.
  - Multilingual gated accuracy: 0.89 points.
  - English val BPB: 0.0025 bits/byte.
  - Train loss: 0.043 nats.
  - Without the L1 pairs, typical |Δ| is 0.2–1.5 points.
- **Noise yardsticks at 1B.**
  - Seed SD (3 seeds, RMS over 4 designs): HellaSwag 0.25, ARC-E 0.78, ARC-C 0.68, rf_MMLU 0.29 points.
  - Median over gated accuracy tasks: 0.79 points.
  - Binomial SD at 1.7B: HellaSwag 0.50, ARC-E 0.92, ARC-C 1.42 points.
- **Share of 1.7B pair gaps above 2√2 seed SD** (√2σ is the SD of a difference
  between two runs): English core 0.28, multilingual gated 0.16, all English gated 0.15.
- **Spread across models vs DataDecide.** Cross-model SD of final scores, in
  points (DataDecide numbers from rq07's `allenai_snr_variants_per_task.csv`):

  | task | DataDecide 1B (25 recipes) | ours 1.7B, 24 non-L1 families | ours 1.7B, 4 L1 families | seed SD 1B |
  |---|---:|---:|---:|---:|
  | ARC-E | 6.26 | 0.87 | 4.19 | 0.78 |
  | ARC-C | 4.64 | 0.87 | 3.84 | 0.68 |
  | HellaSwag | 3.19 | 0.57 | 0.40 | 0.25 |
  | MMLU (ours rf_mmlu) | 2.70 | 0.40 | 1.86 | 0.29 |

  DataDecide's recipe spread is 4–7x our non-L1 spread. Every L ≥ 2 cell trains
  on the same 50 % DCLM-Edu English half, so English tasks barely see the L,
  scheme and T axes.
- **Predicted vs observed DA.** Noise-only model:
  P(agree) = Φ(zp)Φ(zr) + (1−Φ(zp))(1−Φ(zr)), z = |δ|/(√2σ), σ = smoothed seed SD.
  Mono-axis DA-size, gated, no reliability filter:

  | population | observed | predicted from noise alone |
  |---|---:|---:|
  | all gated accuracy (≈450 tasks) | 0.50–0.535 (90M–1B) | 0.56–0.57 |
  | accuracy, n_items ≥ 1000 | 0.54–0.60 | 0.60–0.61 |
  | English core 4 | 0.62–0.66 | 0.63–0.66 |
  | bBPB twins | 0.58–0.63 | 0.60–0.63 (raw checkpoint σ) |

  These observed values reproduce rq02's `da_all_per_task_both_axes.csv`. The
  0.58→0.70 quoted for mono-axis accuracy is rq02's `above_66_either`
  population (observed 0.576 → 0.693 against 0.637 → 0.652 from the same
  noise model). That filter selects tasks on DA itself (CLAUDE.md bug #17), so
  it is not the population to diagnose with.
  - A second, independent estimate (a Gaussian reliability model with no rank
    change) caps unfiltered DA at 0.58–0.60 for accuracy and 0.68–0.72 for bBPB.
  - Making every true gap 3x larger (DataDecide-like) lifts that cap only to
    0.59–0.60 and 0.64–0.68, because 57 % of accuracy decisions have no
    detectable true gap at all.
- **Where the design does create gaps, DA is AllenAI-like.**
  - The three L1 English-corpus pairs (DCLM-Edu / DCLM / FineWeb) span ARC-E by
    10.8 points (13.8 seed SD) at 1.7B.
  - Their DA-size is 1.00 at every proxy from 90M to 1B on ARC-E, ARC-C,
    rf_MMLU, `bpb_dclm` and the ARC bBPB twins. HellaSwag reaches 0.67–1.0.
  - So the L1 cells do show the AllenAI trend. It is diluted when pooled with
    pairs that have nothing to resolve.
- **Not all of it is noise.**
  - Of the accuracy pairs whose 1B gap exceeds 2√2σ, 34 % flip sign at 1.7B,
    against about 1 % expected from noise.
  - Among decisions large (> 2σ of a difference) at both the proxy and 1.7B,
    17–25 % flip for accuracy and 14–21 % for bBPB at every proxy size. For
    per-language BPB the flip rate is 0–0.6 %.
  - By axis (1B → 1.7B, accuracy / bBPB): depth 44 % / 56 %, scheme 30 % / 27 %,
    L 20 % / 20 %.
  - Allowing for these flips, the noise model predicts 0.53–0.54 for accuracy,
    against the observed 0.50–0.53.
  - The English core flips only 3 of 78 large-gap decisions.
  - Most flips are on multilingual and small tasks, and on the arch axis (see F5).

Impact: **high**. This alone accounts for DA ≈ 0.5–0.6.

### F2. Checkpoint noise is read over the WSD decay and counts the trend as noise (explains much of the SNR gap; definition)

- **The code.**
  - `run_apertus_snr_variants.per_model_inputs` (`analysis/rq03_noise_and_snr/run_apertus_snr_variants.py:128`)
    takes `np.std(window)` on the raw 80/85/90/95/100 % points (`utils.noise_checkpoints`, `analysis/utils.py:675-696`).
  - That window is exactly the WSD decay (1.7B: decay 64,800→81,000 of 81,000), where scores still rise.
  - Upstream `snr_simple.py:63-67` uses the last 5 checkpoints of a DataDecide run.
  - A synthetic pure ramp with no iid noise gives step noise 0.0035–0.007, not 0 (both analysis agents' unit tests).
- **Measured.** I compared the sample SD over the 5 window points with the
  residual SD after a linear fit (ddof n−2). Under pure noise the ratio is about 1.
  Script `coord/noise_trend.py`, 29 cells per size:

  | metric | ratio raw/detrended, 1B | ratio, 1.7B |
  |---|---:|---:|
  | HellaSwag | 3.7 | 3.7 |
  | ARC-E | 1.8 | 1.7 |
  | ARC-C | 1.5 | 2.1 |
  | OBQA, MMLU (at chance) | 1.0 | 1.0–1.2 |
  | all English accuracy tasks, median (most at chance) | 1.02 | 1.02 |
  | gated accuracy (analysis agent) | 1.42 | 1.58 |
  | bBPB twins (`coord/bbpb_trend.py`) | 1.46 | 1.50 |
  | train loss | 2.0 | 2.4 |
  | macro BPB | 3.5 | 3.5 |
  | English val (`ppl__dclm`) | 4.0 | 4.3 |

- **Window noise vs seed noise.** For macro BPB the window SD (0.027) is 8x the
  1B seed SD (0.0035). For accuracy, seed SD (0.016 English median) exceeds the
  detrended window SD (0.011).
- **Effect on SNR** (range across 29 families ÷ mean window SD,
  `coord/accnorm_effect.py`). At 1.7B, raw → detrended:
  - HellaSwag 6.1 → 21.4
  - ARC-E 12.7 → 20.1
  - ARC-C 8.4 → 15.7
  - OBQA 4.3 → 4.5
- **Effect on the noise-SD statistic.** Over all model pairs of the bBPB twins
  at 1.7B, the share of gaps below one window SD drops from 0.35 (raw) to 0.27
  (detrended). The "59 % below one noise SD" figure uses the raw, trend-inflated SD.
- **rq07 decomposition.** The analysis agents split the 6–9x gap into about
  2.2–4.5x smaller signal (F1) and about 2–3x larger noise.
  - Of the noise factor, about 1.5–2x is the trend.
  - About 1.1–1.2x comes from DataDecide keeping only 3–4 window points under
    our rule-4 grid (ddof 0 biases fewer points low).
  - The AllenAI CSV was built in June with the old last-5 window and was never
    regenerated (commit 9682d4a0).
  - rq07 is therefore not like-for-like on noise, metric (F4) or signal population (F1).

Impact: **high on SNR levels and the rq07 comparison; none on DA.**

### F3. bf16 logits make letter-format tasks tie on 21–38 % of items (verified measurement artifact)

- **The cause.** `src/evals/scripts/evaluate.sbatch:279` hard-codes
  `dtype=bfloat16` for vLLM. In the samples, 97–99 % of the gaps between option
  log-likelihoods on single-token (letter) answers are exact multiples of 1/16
  nat. The logits are quantised. Ties at the maximum are broken by `argmax`
  towards the first option.
- **Measured** with `coord/ties.py` on the samples files, final checkpoint:

  | task (format) | 1.7B L1 deep: tie at max / tie incl. gold / gold strict max / acc | 1.7B L50 deep: tie at max / acc |
  |---|---|---|
  | mmlu (letter) | 0.21 / 0.12 / 0.285 / 0.338 | 0.38 / 0.258 |
  | commonsense_qa (letter) | 0.23 / 0.06 / 0.209 / 0.257 | 0.35 / 0.201 |
  | belebele_spa_Latn (letter) | 0.29 / 0.15 / 0.209 / 0.279 | 0.32 / 0.266 |
  | global_mmlu_full_es (letter) | 0.33 / 0.13 / 0.235 / 0.269 | 0.19 / 0.237 |
  | arc_challenge (cloze) | 0.002 | 0.003 |
  | rf_mmlu (cloze) | 0.003 | 0.003 |

- **Letter bias.** On 1.7B L50 `global_mmlu_full_es`, 72.5 % of predictions are "A".
- **Direct check (CPU, HF fp32 vs the vLLM bf16 samples).** On 40 MMLU items
  of the 1.7B L1 final, fp32 has 0 ties at the max against vLLM's 7, and the
  argmax agrees on only 35 of 40. About 12 % of item decisions change from
  logit precision alone. On ARC-E (90M, 67 items × 4 choices) fp32 and vLLM
  agree on 65 of 67, with max |Δ| 0.18 nats.
- **Effect on the mean is small.** For 1.7B MMLU the eval agent computed
  0.319 under a random tie-break against 0.314 first-index (same micro-average
  over `global_mmlu_full_en`). The tie mostly removes information: the model's
  preference between two letters 1/16 nat apart is lost.
- **Reach.** This hits every letter-format original (Global-MMLU, Belebele,
  INCLUDE, MMLU, CSQA, BBH-mcq). Those are the families that RULES.md says
  "barely survive the gate". The `rf_` twins and bBPB are unaffected, because
  multi-token continuations rarely tie.
- **Scope.** The scan of 3,660 cell-iteration dirs and 12,841 results files
  found every eval in bf16 on vLLM, so the artifact is uniform across cells.

Impact: **medium on letter-format tasks**: noise and letter bias near chance,
with little effect on the mean. It does not explain the English cloze tasks.
How much DA it costs is not measured; that needs the fp32 re-scoring in §4.

### F4. Cloze originals are scored with raw `acc`, 0-shot (choice; moderate)

- **The setting.** `configs/tasks.json` has no `metric` for hellaswag (line 2044),
  arc_easy (162), arc_challenge (142) or openbookqa (3452). So
  `ladder_report._primary` reads `acc`. Only the `rf_` twins pin `acc_norm`.
- **What we run.** Every result is 0-shot (`n-shot` 0 in the results JSON). The
  scan found no non-zero fewshot for these tasks.
- **What DataDecide runs.** RC with `acc_per_char` (HellaSwag, ARC-E, MMLU) or
  `acc_uncond` (ARC-C, OBQA, CSQA) (analysis agent, from the DataDecide parquet).
- **Same results file, two metrics** (1.7B L1 deep, final,
  `harness/eval_20260912_005713_3363386/results_*.json` and `eval_20260923_011617_3485249`):
  - HellaSwag: acc 0.509, acc_norm 0.675
  - OBQA: acc 0.284, acc_norm 0.412
  - ARC-C: acc 0.410, acc_norm 0.434
  - ARC-E: acc 0.743, acc_norm 0.718
- **OBQA is gated out by the metric alone.**
  - Raw acc is 0.17–0.27 across cells and sizes (0.284 for the best 1.7B cell),
    so it fails the gate at every size.
  - On 500 items, 0.284 has a Wilson 95 % lower bound of 0.246.
  - Its acc_norm, 0.36 (1B) to 0.41 (1.7B L1), would pass.
- **MMLU and CSQA** are the letter format, at chance, and gated. Their cloze
  twins (rf_mmlu acc_norm 0.379, rf_commonsense_qa acc 0.529) are what DataDecide's RC format measures.
- **Effect of switching to acc_norm** (extracted from the results JSONs for all
  cells; `coord/scan_accnorm.py`, `coord/accnorm_effect.py`):
  - Mono-axis DA-size at 1B (71 pairs): ARC-C 0.66 → 0.77, ARC-E 0.53 → 0.61,
    HellaSwag 0.88 → 0.87, OBQA 0.56 → 0.56.
  - SNR moves by 0–35 %.
  - The analysis agents agree: DA moves −0.03 to +0.18.

Impact: **medium.** It admits OBQA and helps ARC. It is not the main cause.

### F5. Deep vs shallow is not size-matched at 600M (verified confound; explains the "600M dip")

- **The configs.** In `src/pretrain/hyperparams/hyperparams_{deep,shallow}.json` (`configs.600M`):
  - Shallow 600M has 616.6M non-embedding parameters against deep's 594.5M (+3.7 %).
  - Its width is 2048 against 1536, so its tied embedding is 33 % larger.
  - It has 14 layers, the same as shallow 350M.
  - At 175M/350M/1B/1.7B the non-embedding counts match within −2.3 % to +0.8 %.
  - Tokens are 100 × N, so the 600M shallow cell gets about 15 % more compute (6·(N + d·V)·D).
- **The flip.**
  - Shallow beats deep on `bpb_dclm` in 9 of 10 settings at 600M, and loses at every other size.
  - Depth-pair DA-size on per-language BPB: 0.95 (175M), 0.72 (350M), **0.04 (600M)**, 0.89 (1B).
  - On `bpb_dclm`: 0.90 / 0.70 / 0.20 / 0.90.
  - This is the "unexplained 600M dip" (`plan/decision_accuracy.md` §8.7) and the
    "16 languages at exactly 0.67" (two flipped depth pairs out of six).

Impact: **medium on the arch axis and on any LM/BPB DA at 600M**; low elsewhere.

### F6. No EOD (or BOS) between training documents; evals prepend BOS (verified deviation; measured harmless)

- **The build.**
  - `src/pretrain/data/create_data_mixture.py:825-836` tokenizes with
    `add_special_tokens=False` and writes the ids with no EOD.
  - Megatron's GPTDataset adds no separator, and the runs set
    `eod_mask_loss`, `reset_attention_mask` and `reset_position_ids` to False.
  - Documents are glued back to back. Example from the A English build:
    `...Peaceful dreams!\n\n\nShop Now!` || `"Backscatter" is basically ...`.
- **Checked on disk.** In 20,000 random documents of each of the A, DCLMP, FWEB
  and L50 builds, 0 start with `<s>` (id 1) and 0 end with `</s>` (id 2).
  Tokens 0–3 never appear in 1.4M sampled tokens.
- **The eval side.** `auto_evals_cscs.py:534` sets `BOS=true`
  (`evaluate.sbatch:280-282`, `add_bos_token=True`). Every prompt therefore
  starts with a token the model never saw as input.
- **Measured effect.**
  - On the 90M L1 final checkpoint (HF, CPU), loss on 40 documents is 2.994 with
    BOS against 3.010 without. BOS acts as a harmless attention sink.
  - Missing document boundaries cost every cell alike (all builds share the
    builder), so they do not bias DA.
  - Rebuilding the data would retrain the grid, which CLAUDE.md forbids.

Impact: **low on DA/SNR.** Record it as a known deviation from standard practice.

### F7. Training anomalies that add noise but not a bias (suspicious; low–medium)

From the training audit (two independent agents; scripts in the scratchpad `train/`):

- **Warmup loss spikes.**
  - 90 % of batch-504 cells spike more than 1 nat during LR warmup (350M 22/31,
    600M 29/34, 1B 20/22, 1.7B 28/32, 3B 4/6).
  - Examples: 1.7B L1 deep 8.37 at iteration 993 (grad norm 92); 1.7B L50 deep 17.77 at 2,666.
  - All recover within 38–665 iterations, and final losses stay on trend.
  - `plan/90M-rung-anomaly.md:178-181` says 350M/600M had "no spike at all";
    that is wrong at per-iteration resolution.
  - `ladder_report.py:577` flags a cell as diverged only if its final loss is
    more than 0.25 above its best, so recovered spikes are invisible.
  - A spike that hits one cell of a pair can add noise to DA.
- **350M sits above the size trend in every family.**
  - Median residual +0.04 nats; +0.011 to +0.014 BPB on `bpb_dclm` in 5 of 6 families.
  - It is the shortest run (16,660 steps). AdEMAMix's β3 = 0.9999 gives a
    6,930-step slow-EMA half-life, 42 % of the run (9 % at 1.7B), longer than its 3,300-step decay.
  - The planned diagnostic `diag-350M-L2-deep-seed1904-beta3f0.2` was never run.
- **Clip 0.1 is active on 77–91 % of steps after 60 % of the 1.7B run.**
  It is about a 0.8x scale on the gradient, which Adam's normalizer mostly cancels.
- **The `loss` column is one iteration's loss.** It comes from
  `ladder_report.py:139`. Consecutive iterations at the 1.7B final differ by up
  to 0.02 nats, the size of a deep−shallow effect. Read `train_loss` DA and the
  loss-based residuals with that in mind.

### F8. Stale or mismatched tables (bookkeeping; low)

- `da_all_per_task_both_axes.csv` was written 10-07 20:49, before the 10-08
  04:36 report. HellaSwag now has 28 families at 1.7B against 26 in the CSV,
  and values move by ≤0.04. A `FORCE=1` refresh fixes it.
- rq07 aliases `global_mmlu_full_en` to `mmlu` although a real `mmlu` task now
  exists. No effect today: both are gated at 1B.
- `snr/snr_simple.compute_snr_small_scale` pools all mixes' last 5 checkpoints
  into one SD, so between-mix variance lands in the noise. It disagrees with the
  paper (§4.1, per-model noise). Our pipeline does not call it; do not use it
  for a reproduction claim.

## 3. Verified fine, and how

| area | check | result |
|---|---|---|
| Mixture per cell | printed `data_path` in the training `.out` of the 1.7B L1 A/DCLMP/FWEB, L50 and L8 cells | L1 reads `1.0 data/english_dclm`, `data/DCLMP/english_dclm` and `data/FWEB/english_dclm` respectively; L50 reads `0.5 english_dclm + 0.5 data-92B/fineweb_L50` |
| English builds distinct | md5 of the first MiB; document counts | md5 values abae30ad / b497cd2a / a8487961 all differ; 137.7M / 141.9M / 266.1M documents |
| Epochs | `.idx` token sums vs draws | Each English build is 184.0B tokens, so the 1.7B L1 cell reads 0.909 epoch and the 1.7B L50 cell reads 0.45 epoch of English and 0.91 of FineWeb-2; `--split 100,0,0`, validation rows excluded at build |
| Tokenizer | build, training (`HuggingFaceTokenizer swiss-ai/Apertus-70B-2509`) and eval | Identical everywhere; 0 ids ≥ 131072 |
| Decoded text | 20,000 random documents per build | Readable, correct language, no garbage or duplication |
| Training args | args block of jobs 3285404 (1.7B) and 3496841 (90M) | Match `megatron_args.sh`: LR 6.93e-4 / 1.43e-3, WSD 1-sqrt to 0, warmup 3,200 / 1,200, decay 16,200 / 5,400, wd 0.1, clip 0.1, init 0.007888 / 0.013662, AdEMAMix α 8, β3 0.9999, warmups = train_iters |
| AdEMAMix full-run warmup | `megatron/core/optimizer/ademamix.py` | Intended: α rises linearly to 8 over the run, as in Pagliardini et al. 2024. The side effect at 350M is F7 |
| LR schedule | per-iteration logs | Matches the analytic schedule to 5e-7 relative over all 81,000 / 27,000 iterations |
| Resumes | iteration counter, consumed samples, optimizer `step` in `common.pt` | Contiguous; samples = iter × GBS; no LR jump |
| NaN / skipped iterations | logs | 0 |
| Scaling | `bpb_dclm`, L1 deep | 0.985 → 0.938 → 0.905 → 0.854 → 0.819 → 0.782 (90M→1.7B); rungs are 10–40 seed SD apart |
| RoPE | `pretrain_gpt.py:121-136`, HF `config.json` | Factor 8 in training and eval; the printed `32.0` is inert, as the memory note says |
| Conversion | HF fp32 on CPU, 90M L1 final, 49k packed training tokens | 2.943 nats against Megatron 2.97–3.06; `score_bpb` held-out at 1.7B is 2.42 against training 2.36, a gap equal to the 90M's (+0.05) |
| HF config | `config.json` | xielu, rms eps 1e-5, θ 5e5, tied embeddings, qk_norm, GQA all match Megatron |
| vLLM vs HF | 90M, ARC-E, 67 items × 4 choices | Max abs Δ 0.18 nats, mean 0.04; argmax agrees on 65/67 (bf16 noise) |
| Harness | results JSON config | No chat template, `add_bos_token`, max length 4096, 0-shot, vLLM bf16, TP 1 |
| Per-item scores | HellaSwag / ARC / MMLU samples, 1.7B L1 final | Recomputed acc equals the results JSON and `ladder_report.csv` (0.508863, 0.743266, 0.410410, 0.337416) |
| Attribution | regex `(.+)-iter(\d+)$` (`ladder_report.py:459-466`); newest run wins (`:269-288`); scan of 3,660 dirs / 12,841 results files | 0 model-path vs (cell, iter) mismatches, 0 `--limit` runs |
| HellaSwag 0.313882 at 90M L1 deep and dclmP | per-item samples | Coincidence: both score 3,152/10,042, but their log-likelihoods differ on every item, they disagree on 562 items, and acc_norm differs (0.3587 vs 0.3553). Different model paths and job ids |
| Scores vs public models | same pipeline (`reference_hf` parquets, vLLM bf16, 0-shot), acc / acc_norm | Ours 1.7B L1: HellaSwag 0.509/0.675, ARC-E 0.743/0.718, ARC-C 0.410/0.434. OLMo-2-0425-1B (4T tokens): 0.508/0.683, 0.726/0.737, 0.385/0.423. gemma-3-1b-pt: 0.473/0.621, 0.721/0.722, 0.349/0.382. No depressed scores |
| Gate | `above_random.py:167-265` | Wilson one-sided 95 % matches the closed form; n_items from tasks.json; NA mask passes |
| DA kernel | independent recompute; cube vs per-call path | 0 mismatches over 9 tasks × 2 pair sets × 5 sizes |
| DA definitions | code | Proxy and reference are each family's final checkpoint (`require_final`); ties by sign; orientation-invariant; MIN_PAIRS 3 |
| DA on DataDecide | our `pair_agreement` vs upstream on the public parquet, 150M/300M/750M → 1B | ARC-E 0.930/0.953/0.967 vs 0.930/0.953/0.970; HellaSwag and MMLU identical; ARC-C within 0.007 |
| Synthetic tests | analysis agents' unit tests on our functions | Large gap → DA 1.0, SNR 37–137; pure noise → DA 0.505; ties order-invariant (upstream is not); orientation symmetric; Wilson exact; a pure ramp shows up as noise (F2) |

## 4. Proposed fixes and checks

CPU only, analysis side. Each needs your decision, because it moves populations:

1. **Noise.** Define checkpoint noise as the residual SD after a linear fit
   over the window (ddof n−2), or use the seed noise where replicates exist.
   Implement it in `utils`/`per_model_inputs` and add a RULES.md rule-4 note.
   Re-quote rq03, rq04, rq07, rq12 and the "share below one noise SD" figure.
2. **rq07.** Regenerate the AllenAI side with the same noise definition and an
   equal number of window points. Compare RC/normalised scores on both sides.
   State that the signal populations differ (25 corpora vs 29 design variants).
3. **Metric.** Pin `metric: acc_norm` for hellaswag, arc_easy, arc_challenge and
   openbookqa in `configs/tasks.json`, or add them as variants the way the
   `rf_` twins are. The values are already in every results JSON, so no
   re-evaluation is needed. OBQA then enters the gated population.
4. **Depth at 600M.** Flag the 600M depth pairs, or drop them from mono-axis DA,
   and say so in rq02 / `plan/decision_accuracy.md` §8.7.
5. **Headline DA.** Report it per axis and with the effect-size context of F1:
   the L1 corpus pairs at 1.00 against the L/scheme/T pairs at 0.5. Do not quote
   a pooled 0.58–0.70 without naming its pair set and filter.
6. **Loss column.** Store a 50–100-iteration mean in `ladder_report.py`.
7. **Refresh.** Run `FORCE=1` so the rq02 tables match the current report.

GPU jobs, for your approval (none submitted):

- **(a) fp32 re-scoring of letter-format tasks, to size F3.** Six final
  checkpoints: 1B and 1.7B of L1 deep, L8 deep and L50 deep. Tasks: mmlu,
  commonsense_qa, belebele (spa, rus, deu), global_mmlu_full (es, ru, de) and
  hellaswag as a control.
  - `evaluate.sbatch` hard-codes bf16 at line 279, so the job uses a scratch
    copy, writes to a separate log root and W&B project, and leaves the live
    tree untouched. Run it from `src/evals`, because the script takes `REPO_DIR=$PWD`:

    ```bash
    mkdir -p /iopsstor/scratch/cscs/mariagrandury/audit-fp32/logs
    sed 's/^COMMON_MODEL_ARGS="dtype=bfloat16"/COMMON_MODEL_ARGS="dtype=float32"/' \
      src/evals/scripts/evaluate.sbatch > /iopsstor/scratch/cscs/mariagrandury/audit-fp32/evaluate_fp32.sbatch
    cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals
    for c in lm-1.7B-L1-deep-seed1904/iter_0081000 lm-1.7B-L50-deep-seed1904/iter_0081000; do
      cell=${c%%/*}; it=${c##*_}; it=$((10#$it))
      LM_EVAL_BACKEND=vllm TOKENIZER=swiss-ai/Apertus-70B-2509 BOS=true APPLY_CHAT_TEMPLATE=false \
      EVAL_WORKERS=4 TP=1 PP=1 WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=msnr-audit-fp32 \
      LOGS_ROOT=/iopsstor/scratch/cscs/mariagrandury/audit-fp32/logs HARNESS_INCLUDE_PATH=$PWD/tasks \
      TASKS=mmlu,commonsense_qa,belebele_spa_Latn,belebele_rus_Cyrl,global_mmlu_full_es,global_mmlu_full_ru,hellaswag \
      sbatch --account=infra01 --job-name=audit-fp32-$cell-iter$it --time=02:00:00 --export=ALL \
        /iopsstor/scratch/cscs/mariagrandury/audit-fp32/evaluate_fp32.sbatch \
        /capstor/store/cscs/swissai/infra01/msnr/msnr-hf-models/$c $cell-iter$it
    done
    ```

    Extend the loop to the other four cells after the first two succeed.
  - Success measure: the tie rate drops to about 0; compare acc, gate status and
    the L1-vs-L50 ordering with the bf16 values.
  - About 1 node-hour per cell.
- **(b) Optional: BOS off vs on.** The same command with `BOS=false` on
  hellaswag and arc_easy for two cells. F6 already shows BOS is harmless on
  loss, so this is low priority.
- **(c) Optional: the planned 350M β3 diagnostic.** Run
  `python3.11 src/pretrain/launch_trainings.py cscs --size 350M --langs 2 --seed 1904 --ademamix-beta3-factor 0.2 --dry-run`
  first, then the real launch if you want it. About 39 node-hours. It tests
  whether the 350M rung's offset (F7) is the slow EMA.

## 5. What could not be checked

- **DA cost of the bf16 ties (F3).** It needs job (a).
- **Seed noise at 1.7B.** There are no 1.7B replicates, so the 1B seed SD was
  borrowed. bBPB has no seed replicates either, so its noise-only DA prediction
  rests on the checkpoint SD.
- **Whether replicate seeds change the data order** as well as the init. Not checked.
- **Published DataDecide numbers in their own harness format.** We compared
  through rq07's CSV and the public parquet only. External numbers quoted from
  model cards (OLMo-1B v1, TinyLlama, Pythia) are in a different harness and
  were used only as a range check.
- **Long-term effect of warmup spikes** on rare-token and low-resource-language
  embeddings (F7). Not measured.
- **The 1B L1 deep bBPB anomaly.** rq13 reports its `bbh_mcq` twins rising
  from 1.6 to 3.2–3.5 bits over the last four checkpoints, a collaborator-trained
  cell on the 45,740-iteration grid. Not investigated; a replicate seed would
  tell whether it is the seed.

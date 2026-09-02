# Multilingual small-to-large predictivity: training plan

## Goal

Find which model sizes and token budgets can serve as proxies for comparing design choices, and how that depends on the number of languages. Concretely: at a given number of languages, when does a small model rank a design choice the way a large model does? This study is not meant to predict Apertus 2's absolute performance; it is meant to tell us which proxy settings are reliable for making Apertus 2 design decisions when the dataset consists of a given number of languages.

The number of languages and the model size are the axes being analyzed, so neither is the design choice that's altered within a given model setting (i.e., they are not the factors being varied for the sake of model comparison/ranking). The design choice is varied as a separate intervention with at least three levels (see the intervention section). The outcome metric is per-language bits-per-byte (BPB) on a fixed held-out validation set.

## Parameter-count convention

All model sizes below are **non-embedding parameters**: the transformer width and depth only, excluding the token embedding and output-projection matrices.

This follows the Signal-and-Noise paper (Heineman et al., 2025), which uses the Bhagia et al. (2024) compute-efficient model ladder. That ladder defines its sizes (190M, 370M, 760M, 1.3B) "considering only non-embedding parameters," varying width and depth, and sets training tokens as multiples of Chinchilla-optimal (1×C = 20N). We use the same convention because the 131K Apertus vocabulary makes the embedding and output-projection matrices roughly 130–270M parameters on their own, which would otherwise account for most of the parameters in the small models and distort both the size axis and the scaling-law fit.

We should set the Megatron width and depth to hit each non-embedding target. The public OLMo-ladder configs (190M, 370M, 760M, 1.3B) are a close template for four of the sizes; aligning those four to the published values would let you reuse the configs and match the Signal-and-Noise setup directly.

Compute-planning note: with a 131K vocabulary, the output projection costs about 2·d_model·V floating-point operations per token, which roughly doubles the per-token cost of the smallest models relative to their non-embedding parameter count.

## Models

Sizes are non-embedding parameters. Cells marked ×3 get three seeds (different initialization and data-order seed); the rest get one seed.

| Languages | 90M | 175M | 350M | 600M | 1B  | 1.7B |
| --------- | --- | ---- | ---- | ---- | --- | ---- |
| 1         | ✓   | ×3   | ✓    | ✓    | ×3  | ✓    |
| 2         | ✓   | ×3   | ✓    | ✓    | ×3  | ✓    |
| 8         | ✓   | ✓    | ✓    | ✓    | ✓   | ✓    |
| 15        | ✓   | ✓    | ✓    | ✓    | ✓   | —    |
| 30        | ✓   | ×3   | ✓    | ✓    | ×3  | ✓    |
| 50        | ✓   | ✓    | ✓    | ✓    | ✓   | —    |
| 100       | ✓   | ×3   | ✓    | ✓    | ×3  | ✓    |

<!-- BEGIN generated: pretrain_progress.py --plot -->
| Axis | Values |
| ---- | ------ |
| Size (non-embedding) | 90M, 175M, 350M, 600M, 1B, 1.7B (1.7B at L ∈ {1, 2, 8, 30, 100}) |
| Language setting L | 1, 2, 8, 15, 30, 50, 100 (English + L−1 FineWeb-2 languages; L=1 is 100% English) |
| Seed | 1904; ×3 seeds (64, 313, 1904) on the 175M, 600M columns at L ∈ {1, 2, 50, 100} |
| Data scheme | A everywhere; B only where its language set differs — L ∈ {8, 15, 30} |
| Architecture | deep (baseline) and shallow (the model-depth intervention) |

**56 runs** at one intervention level (scheme A, deep — the plan grid).
Counting both architectures and scheme B where it differs: **146 runs**.

![Planned runs per grid cell](../src/pretrain/pretrain_progress_plan.png)

![Finished models per grid cell](../src/pretrain/pretrain_progress_simple.png)

![Eval work outstanding per grid cell](../src/pretrain/eval_progress.png)
<!-- END generated -->

(The 200-language setting was dropped on 2026-08-13 to fit the compute budget
and deadline. The ×3-seed rows were 1, 30 and 100; L=2 was added to them, and
the 1.7B row gained L=2, taking the grid from 52 to 56.)

## Intervention axis (the design choice under test)

Each (model size, number-of-languages) cell is trained under at least three different settings (an "intervention"). The analysis asks whether a proxy (smaller models) ranks those levels the way the reference (a larger model) does. The reference at each number of languages is the largest model trained there.

### Possible interventions

- **Tokenizer**, original (swiss-ai/Apertus-70B-2509) versus the V2 candidate vs. one of the others Clara has trained that's the same size. BPB is comparable across tokenizers because its denominator is bytes, so the metric needs no change.
- **Model depth**, comparing a deeper-narrower against a shallower-wider model at the same non-embedding size. This is the most controlled option (same data, same tokenizer, same size), but the effect of aspect ratio on loss is usually small near the optimum, so the signal may be too weak to estimate decision accuracy. Use it only if you expect a clear effect at these sizes.
- **Sampling temperature**, comparing T = 1 against about larger values. This is the most multilingual-specific choice and the one most likely to show a ranking that flips with scale, since the optimal temperature shifts with capacity and the number of languages. The two levels train on different distributions and the winner depends on how you aggregate across languages, so fix the decision criterion first; it is also only meaningful above one language.

## Token budget

The token budget is set per model size, and held constant across all language settings for a given size. Each size trains on 5× Chinchilla-optimal:

D(N) = 5 × 20 × N = 100 × N (non-embedding N).

| Size (non-emb) | 1×C = 20N | D(N) = 5×C |
| -------------- | --------- | ---------- |
| 90M            | 1.8B      | 9B         |
| 175M           | 3.5B      | 17.5B      |
| 350M           | 7B        | 35B        |
| 600M           | 12B       | 60B        |
| 1B             | 20B       | 100B       |
| 1.7B           | 34B       | 170B       |

Compute the exact token count for each model from its actual non-embedding parameter count (D = 100 × N).

Reasoning:

- A fixed multiple of Chinchilla per size is the standard scaling-ladder setup (Bhagia et al. train each rung at 1×–10×C; DataDecide uses 5×C for data decisions). Holding the same multiple across sizes keeps every rung in the same training regime, which is what the scaling-law fit over size needs.
- The 5× multiple puts the ~2B-scale model near 200B tokens, inside ATLAS's empirical pretrain-versus-finetune crossover band for ~2B multilingual models (about 144–283B), so the largest models are adequately trained for 200 languages.
- Holding D constant across language settings for a given size means that, within a size, only the language mix changes between settings. Differences then come from the number of languages and the per-language token reduction, not from a changing token count.

## Data mixtures

- **Sources:** FineWeb2 for the non-English languages. Ayush recommended using the hq variant at fineweb2-hq. English from `dclm-edu-filterrobots_fine` (there is no eng_Latn in FineWeb2). Default tokenizer: swiss-ai/Apertus-70B-2509 (the V1 tokenizer).
- **English share:** 50% in every multilingual setting; the other 50% is the FineWeb2 languages. The 1-language setting is 100% English. (See open question 2.)
- **Allocation within the FineWeb2 50%:** temperature sampling with T = 1 by default (proportional to estimated per-language tokens). (See open question 1.)
- **Language counts include English,** so the FineWeb2 language list for an L-setting has L − 1 entries. The lists are nested: the 1-language FineWeb2 list is a subset of the 7-language list, which is a subset of the 14-, 29-, 49-, and 99-language lists, all drawn from the 199 FineWeb2 languages used in the prior 200-language run. The validation build covers the 99-language list plus English (the largest trained setting). The lists are generated by `src/pretrain/data/generate_language_sets.py` from the FineWeb2 distribution and the benchmark availability in `configs/tasks.json`: the top-k subsets by train-split UTF-8 bytes, excluding `und_*` subsets and `hau_Latn` (absent from the swiss-ai filtered dataset dir); for the 99-language list, subsets with no benchmark in the lm-eval harness are replaced by the next subsets by bytes that have at least two benchmark families and are not a script variant of a kept language (2026-08-21: gmh, nrm, bew, tat, div, epo, hif, ltz → kin, jav, xho, hat, fao, zul, ibo, sot). Scheme B swaps the small settings for script/family-diverse picks. See `plan/benchmark_selection.md`.

To avoid tokenizing English once per setting, build the English data once and the FineWeb2 data once per setting, then blend them 50/50 at training time with the Megatron data loader's blend weights. So the artifacts are: one English dataset, one FineWeb2 dataset per multilingual setting, and one fixed validation set.

| Setting (L) | FineWeb2 languages | FineWeb2 build tokens | English share at training time |
| ----------- | ------------------ | --------------------- | ------------------------------ |
| 1           | 0 (English only)   | —                     | 100%                           |
| 2           | 1                  | 93.5B                 | 50%                            |
| 8           | 7                  | 93.5B                 | 50%                            |
| 15          | 14                 | 55B                   | 50%                            |
| 30          | 29                 | 93.5B                 | 50%                            |
| 50          | 49                 | 55B                   | 50%                            |
| 100         | 99                 | 93.5B                 | 50%                            |

Each FineWeb2 build is sized to half of the largest budget at that setting, with about 10% headroom: 93.5B where a 1.7B model trains (settings 2, 8, 30, 100; half of 170B plus headroom), 55B otherwise (half of 100B plus headroom). The English dataset is built once to 187B, which covers the 1-language setting's largest need (170B) and the English half of every other setting. The build script reports the realized per-language token counts and warns when a language runs out of data; record any shortfall.

## Validation set

Build one validation set, fixed and independent of temperature, token budget, and the language set, reused by every model. For each language, the validation set is the first 5M tokens of that language's first parquet file, capped at 30% of that file's rows so that single-file languages keep training data. The build records, per language, the token count, the UTF-8 byte count (the BPB denominator), and the number of leading rows assigned to validation (val_doc_count). Every training build is given this manifest and skips exactly those leading rows of the first file, so training and validation never overlap. This handles single-file languages, which exist in the tail.

Build it once over the 99-language list (the largest trained setting) plus English. A language whose first file is a single document gets no validation data and is flagged by the script.

## Commands

Run paths and `--output_prefix` values are placeholders; adjust to the cluster layout. The `$FW_Lx` placeholders are the FineWeb2 language lists described above (`$FW_L2` has 1 language, `$FW_L8` has 7, `$FW_L15` has 14, `$FW_L30` has 29, `$FW_L50` has 49, `$FW_L100` has 99), each a comma-separated list of `{lang}_{script}` codes with no spaces.

### Step 1: build the validation set (once)

```bash
python create_data_mixture.py \
  --build_validation \
  --languages $FW_L100 \
  --val_tokens_per_language 5000000 \
  --val_max_fraction 0.3 \
  --output_prefix outputs/validation
```

This writes `outputs/validation.fineweb_<lang>.bin`/`.idx` per language, `outputs/validation.dclm.bin`/`.idx` for English, and `outputs/validation.manifest.json` (per-language tokens, bytes, and val_doc_count). English is included automatically and does not need to be in `$FW_L100`.

### Step 2: build the English dataset (once)

```bash
python create_data_mixture.py \
  --target_tokens 187000000000 \
  --fineweb_pct 0 --dclm_pct 100 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/english_dclm
```

### Step 3: build the FineWeb2 dataset per setting

Each uses `--fineweb_pct 100 --dclm_pct 0`, `--temperature 1.0`, and the validation manifest. The 1-language setting has no FineWeb2 build; it trains on the English dataset alone.

**L = 2:**

```bash
python create_data_mixture.py \
  --target_tokens 93500000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L2 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L2
```

**L = 8:**

```bash
python create_data_mixture.py \
  --target_tokens 93500000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L8 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L8
```

**L = 15:**

```bash
python create_data_mixture.py \
  --target_tokens 55000000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L15 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L15
```

**L = 30:**

```bash
python create_data_mixture.py \
  --target_tokens 93500000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L30 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L30
```

**L = 50:**

```bash
python create_data_mixture.py \
  --target_tokens 55000000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L50 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L50
```

**L = 100:**

```bash
python create_data_mixture.py \
  --target_tokens 93500000000 \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages $FW_L100 \
  --temperature 1.0 \
  --validation_manifest outputs/validation.manifest.json \
  --output_prefix outputs/fineweb_L100
```

## Training

For each (size, setting), train for D(N) tokens (the per-size budget above). Compose the data at training time with the Megatron data loader:

- **1-language setting:** the English dataset alone (weight 1.0).
- **Multilingual settings:** the English dataset and that setting's FineWeb2 dataset, blended 50/50.

Set the trainer's total token count to D(N) for each size. The largest model at a setting trains close to one pass over the blended data; the smaller models draw a fraction, which the loader's shuffling makes a proportional sample. Seeds re-run with a different initialization and data-order seed, drawing a different sample. If exact per-language training token counts at every size matter more than tokenizer time, build a separate dataset per (size, setting) instead of subsampling one; that costs more tokenization.

Before the largest run at a setting, check the realized FineWeb2 build size that the script reports. At high language counts lower-resource languages can run out, so the realized size can fall below the build target; if it is below the largest model's FineWeb2 half (85B at the 1.7B settings), reduce that model's token count to avoid repeating data, and record it.

Log the final checkpoints (for example the last 30, spaced about 1000 steps), so that per-language BPB and the checkpoint-to-checkpoint noise estimate can be computed over the final window, matching the Signal-and-Noise noise definition.

**As implemented (2026-08-21).** Each run saves **20 checkpoints** evenly spaced — **40 at the 1B and 60 at the 1.7B**, the two reference rungs, whose intervals also stay near the ~2000-iter Azure-spot eviction window. The interval is per size, `train_iters / n`, and 40 and 60 are multiples of 20, so checkpoint *k* sits at *k*/*n* of training at **every** size and the grids stay index-aligned across the ladder, which is what lets SNR compare checkpoint *k* between sizes. Because D = 5 × Chinchilla, the 1×C operating point (`train_iters / 5`) is always checkpoint *n*/5 — 4, 8 or 12 — on-grid at every size. Evaluation covers every 2nd checkpoint, **the run's final one**, and **the checkpoint nearest each half-decade FLOPs milestone** (`auto_evals_*.py --every 2` + `configs.milestone_iters`, ~1 extra per run — see "The compute axis" below); the odd late checkpoints are converted to HF and kept, so the checkpoint-noise window can be densified later by lowering `--every` without retraining.

Note the deviation from "the last 30": with 20 checkpoints per run (40/60 at the reference rungs) the whole grid is smaller than that, and the dense tail is 5. Checkpoint noise is therefore estimated over 5 late checkpoints, not 30. Raising it means lowering the save interval — cheap in compute (checkpoints are written by training anyway) but it multiplies conversion and eval volume, which is the actual constraint (see `plan/compute-budget.md`).

## Checkpointing at defined token counts

Save checkpoints at several token counts within each run, not only at the end, so per-language BPB can be read at defined operating points and fit over training tokens as well as over model size. Two reference points are useful for each (size, language count):

1. **The single-language Chinchilla-optimal point,** 20 × N (non-embedding N). It depends only on model size: 90M → 1.8B, 175M → 3.5B, 350M → 7B, 600M → 12B, 1B → 20B, 1.7B → 34B. It is within the 5×C training budget for every size, so it is always an intermediate checkpoint. (Implemented 2026-08-21 as a per-size save interval of `train_iters / n` with n = 20 checkpoints per run, 40 at the 1B and 60 at the 1.7B — denser sampling of the reference models, on the same k/20 grid since 40 and 60 are multiples of 20: schedules are rounded to the grid, so checkpoint k sits at k/n of training, the final checkpoint is on-grid, and the 1×C point is exactly checkpoint n/5 — 4, 8 or 12 — everywhere.)
2. **The ATLAS compute-optimal point** for that model size and language count, N × r(K). ATLAS reports that adding languages without degrading per-language loss scales model size by 1.18 and total tokens by 1.66 per doubling of the language count (their worked case: one to four languages is ×1.4 model size and ×2.74 total tokens). The compute-optimal tokens-per-parameter ratio therefore grows by 1.66/1.18 ≈ 1.41 per doubling, anchored at 20 (Chinchilla) for one language: r(K) = 20 × 1.41^log2(K). This is approximate and should be double-checked.

r(K), tokens per parameter:

| K    | 1   | 2   | 8   | 15  | 30  | 50  | 100 |
| ---- | --- | --- | --- | --- | --- | --- | --- |
| r(K) | 20  | 28  | 56  | 76  | 107 | 137 | 193 |

ATLAS compute-optimal tokens, N × r(K), in billions:

| Size | 1    | 2    | 8    | 15   | 30   | 50   | 100  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 90M  | 1.8  | 2.5  | 5.0  | 6.8  | 9.6  | 12.4 | 17.4 |
| 175M | 3.5  | 4.9  | 9.7  | 13.3 | 18.7 | 24.0 | 33.8 |
| 350M | 7.0  | 9.8  | 19.5 | 26.6 | 37.4 | 48.1 | 67.6 |
| 600M | 12.0 | 16.9 | 33.4 | 45.5 | 64.1 | 82.4 | 116  |
| 1B   | 20.0 | 28.1 | 55.7 | 75.9 | 107  | 137  | 193  |
| 1.7B | 34.0 | 47.8 | 94.7 | 129  | 182  | 234  | 328  |

The K = 1 column is the single-language Chinchilla point. Cells at K ≥ 30 exceed the 5×C training budget (r(K) > 100 tokens per parameter), so training to 5×C does not reach the ATLAS compute-optimal point there: to capture that checkpoint at 30 languages and above, extend those runs to N × r(K), otherwise the final checkpoint is the 5×C budget. At 15 languages and below the ATLAS point is within the budget and is an intermediate checkpoint. Logging a few additional counts per run (for example 1×C and 2×C) gives several token points for the fit. Whether the ATLAS ratio actually holds for our mixtures is testable without new full runs — see "Is 5 × C the right budget at L ≥ 30?" below.

## The compute axis (2026-09-02)

Every checkpoint has a known (N, D), so the S&N-style "metric vs compute" plot
needs no dedicated save grid — each size's curve is plotted at its own x. Two
things had to be fixed for that axis to be trustworthy across models.

### One FLOPs convention, applied to external models too

    FLOPs = 6 × (N_non_emb + d_model × vocab_size) × D

The embedding *lookup* is free, the output projection is not: with a 131,072
vocab, including it nearly doubles the small rungs' compute, so omitting it
bends the ladder. Our cells tie embeddings, so the `params` already recorded in
`configs/models.json` is exactly that sum (90M: 193,560,576 = 92,897,280 +
768 × 131,072) — the numbers were right. What was missing was the ability to
tell that apart from an external model, which declares a nominal total
(`Qwen3-1.7B-Base: 1.7e9`) that is the same quantity only if its embeddings are
tied too.

Implemented as `src/evals/scripts/utils/configs.flops_params()` — one
definition replacing the formula that was duplicated across four call sites —
returning both N and a **basis**: `non_emb+dV` when the shape is recorded,
`declared_total` when it fell back to `params`. `sync_models_json.py` now writes
`n_non_emb`, `d_model` and `vocab_size` per cell, so our models are explicitly
on the convention; the basis is logged to the W&B run config and published as a
`flops_basis` column, so a point on a different footing is visible on the plot
rather than silently mixed in. Moving an external model onto the convention is
a data edit (add the three shape fields to its entry), not a code change.

### Milestone evals: measured IsoFLOP slices, ~1 extra eval per run

Curves need nothing extra; the *vertical* read does — "at 1e20 FLOPs, which
size/shape wins", and decision accuracy between two sizes at matched compute.
Those were interpolated between evaluated points. The eval due-rule now also
marks the saved checkpoint nearest each half-decade FLOPs milestone
(`configs.milestone_iters`, 1e18…1e21, skipping anything still inside LR
warmup, where a large model is not a decision-relevant comparison).

Against the real 20/40/60 save grid, **exactly one milestone per run is not
already due** (ck19 / ck7 / ck3 depending on size) and the worst
nearest-checkpoint error is 2.3 % of a run, usually under 1 %:

| Milestone | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| 1e19 | ck19 **new** | ck6 | ck2 | — | — | — |
| 3.2e19 | — | ck19 **new** | ck6 | ck2 | ck2 | — |
| 1e20 | — | — | ck19 **new** | ck7 **new** | ck6 | ck3 **new** |
| 3.2e20 | — | — | — | — | ck19 **new** | ck10 |

1e19, 3.2e19 and 1e20 each carry 3–4 sizes: those are the IsoFLOP slices, now
made of evaluated points. Cost is ~10 % more evaluated checkpoints, and
**nothing needs retraining** — the checkpoints already exist on the save grid;
the rule only marks one more of them as due, and both watchers are idempotent.

## Evaluation

Per-language BPB on the fixed validation set, using the same per-language text for every model. For each language, BPB = (sum of per-token negative log-likelihood in bits) / (validation bytes for that language), with the byte counts taken from the validation manifest. Evaluate each model on the languages it was trained on. Optionally, also evaluate each model on languages it was not trained on (the validation set covers the 99-language list plus English), which gives a zero-shot cross-lingual transfer read at no extra training cost.

## Analysis

The goal is to find which model sizes and token budgets rank a design choice the way the largest model does, and how that depends on the number of languages.

### Ranking stability (decision accuracy)

At each number of languages, both intervention levels are trained at every size. For each proxy size, compare its ordering of the levels against the largest model's ordering at the same number of languages. With a two-level intervention, the ordering is one decision per language, so the languages provide the population: decision accuracy at a (proxy size, language count) is the fraction of languages where the proxy and the reference agree on which level is better. Map this over proxy size and number of languages to find the smallest reliable proxy at each language count.

Caveats:

- It needs multiple levels (our "design choice intervention") and signal within these levels. If the two levels are close in per-language BPB, the decision is near a coin flip and decision accuracy is low for a reason unrelated to the proxy. Need to measure variability.
- It needs the large reference model trained on the same levels at that number of languages, so it can only be used where a reference size exists.
- Need to choose a criterion: per-language agreement, or agreement on a macro-average across languages. They can disagree...

### Prediction ability

Fit per-language BPB as a function of non-embedding size across scales, and measure how closely the proxy sizes predict the reference's absolute per-language BPB, and how that changes as the number of languages grows.

Caveats:

- It is sensitive to a constant offset. Small models often sit at a fixed distance from large ones in absolute loss even when preserving ordering; the fit then reports large error even when every decision the proxy implies is correct. Prediction error and decision accuracy can point in opposite directions.
- It assumes a functional form, which the multilingual bend and the small number of size points can break.
- Absolute per-language BPB spans a wide range across languages, so an aggregate is dominated by the high-BPB low-resource tail unless normalized.

## Sampling temperature: what T = 1 actually allocates (2026-09-02)

Allocation within the FineWeb-2 half is `p_i ∝ (bytes_i / Σ bytes)^(1/T)`
(`create_data_mixture.py --temperature`, default 1.0). The FineWeb-2 byte
distribution is extreme enough that this choice decides whether the tail
languages are trained at all.

At L100 the head/tail share ratio is **16,581 : 1** at T = 1 — Russian takes
28 % of the multilingual half, Sesotho 0.0017 %. Per-language tokens in the
FineWeb-2 half, by top rung (`min` = smallest language; `ep` = maximum epochs
any language's data is repeated, so >1 means duplication):

| Top rung | FW half | T = 1 | T = 2 | T = 3.33 (α = 0.3) |
|---|---:|---|---|---|
| 90M | 4.6 B | min 0.1 M · 66 langs <10 M · ep 0.0 | min 3.2 M · 27 <10 M · ep 0.0 | min 11 M · 0 <10 M · ep 0.1 |
| 350M | 17.2 B | min 0.3 M · 46 <10 M · ep 0.0 | min 12 M · 0 <10 M · ep 0.1 | min 41 M · ep 0.4 |
| **1B** | 47.2 B | **min 0.8 M · 32 <10 M · 66 <100 M** | **min 33 M · 0 <10 M · 27 <100 M · ep 0.3** | min 112 M · 0 <100 M · ep 1.2 |
| 1.7B | 83.6 B | min 1.4 M · 24 <10 M · 57 <100 M | min 58 M · 0 <10 M · 10 <100 M · ep 0.6 | min 198 M · ep 2.1 |

Read the 90M row: **at L100 with T = 1, 66 of 99 languages receive under 10 M
tokens and the smallest receives 80 K.** The small rungs are not training on
100 languages; they are training on ~35 plus noise. This lands directly on the
2026-08-21 L100 decision — `kin`, `jav`, `xho`, `hat`, `fao`, `zul`, `ibo`,
`sot` were swapped in *because they have benchmarks*, and at T = 1 they get
1.4–2.2 M tokens at the top rung and ~100 K at 90M. Benchmark coverage without
data share buys nothing: a per-language SNR of ~0 there is an artefact of the
mixture, not a property of the benchmark, which is the opposite of what this
study is trying to measure.

### Recommendation, given that 1.7B may not be trained

Dropping the 1.7B rung removes 40 % of the sweep's node-hours (15,086 of
37,860) and makes **1B the reference at every L**. The 1B row above is then the
best case, not the middle one — 66 of 99 languages under 100 M tokens in the
model every decision-accuracy comparison is anchored on.

**Adopt T = 2 sweep-wide.** It is the only setting that fixes the floor without
introducing a second problem:

- It lifts the smallest language from 0.8 M to **33 M tokens** at the 1B rung,
  a 40× change, and empties the "<10 M" column at every rung from 350M up.
- **It repeats nothing.** Max epochs 0.3 at 1B, 0.6 even at 1.7B — every
  language is still trained on unseen data. T = 3.33 crosses into duplication
  (1.2 epochs at 1B, 2.1 at 1.7B), which confounds the language-count axis
  with a data-repetition axis.
- One value for every cell. **T must not vary with L** — the intervention is
  the language *count* at fixed English share; a T that moves with L confounds
  the two and makes the L-ladder uninterpretable.

Cost: the FineWeb-2 builds are per-setting, so this is a rebuild of L8…L100.
The L100 build is already owed (the 8-language swap), so its share is free;
L8/L15/L30/L50 are the added cost. L1 is unaffected (100 % English) and L2 is
nearly so.

**If that rebuild cannot be afforded**, the fallback is to keep T = 1 and
report the constraint honestly: define a per-language token floor (100 M is the
natural line) and restrict per-language benchmark claims to languages above it,
treating the rest as BPB-only. That is a smaller result — it silently reduces
L100 from 99 languages to ~33 for the benchmark analysis — but it is at least
not a wrong one.

Either way, record the choice with the data: the mixture manifest already
stores per-language target tokens, so the analysis can filter on them.

## Is 5 × C the right budget at L ≥ 30? How to check it (2026-09-02)

Open question 3 (below) asks whether D = 100·N is enough once the language
count grows, since ATLAS-style multilingual scaling puts the compute-optimal
token/parameter ratio well above Chinchilla's ~20 for large K. The check does
**not** need new full runs.

1. **Intermediate checkpoints cannot answer it.** Under the WSD schedule the LR
   has not decayed at checkpoint k, so its loss is biased upward; treating it as
   "a run trained to D_k" inflates the fitted optimum. Chinchilla-style fits
   need *annealed* endpoints, and the sweep as designed has exactly one per
   (size, L) — six points, too few for a 5-parameter fit.
2. **WSD makes annealed endpoints cheap.** Branch a run at fraction f, decay
   over the standard 20 % window, and the result is a properly annealed model
   at D = f·100·N for ≈ 20 % of f·D extra compute. Two sizes (350M, 600M) ×
   L ∈ {1, 30, 100} × f ∈ {0.25, 0.5} = 12 branches ≈ 6 % of one level's
   node-hours — affordable even after dropping 1.7B.
3. **Fit and read off.** With 3 annealed points per (size, L), fit
   L(N, D) = E + A/N^α + B/D^β per language setting and compare the implied
   D_opt/N against 20 (Chinchilla) and 100 (ours). The question is whether that
   ratio *moves with L*; the absolute value matters less than the trend.
4. **Do the cheap falsification first.** The temperature table above suggests
   the L100 problem may not be a compute-budget problem at all: at T = 1 the
   tail languages are data-starved, not under-trained. Before spending anything
   on cooldown branches, check whether the languages driving the multilingual
   loss bend are the ones with negligible token allocations. That is a
   spreadsheet over the mixture manifest, not a sweep, and it may answer the
   question outright — and if T changes per the section above, this check must
   be redone against the new allocation.

## Open questions

1. Sampling temperature is set to T = 1 (proportional to estimated tokens) by default. Should the FineWeb2 proportion instead be tempered toward uniform (for example alpha = 0.3, i.e. T ≈ 3.3) to give lower-resource languages more weight? Similarly, the datasets are all fixed to 50% English. Is this sound or should it change as languages are added (for example, scaling down with the number of languages)? — **quantified above; recommendation: T = 2 sweep-wide, and never varying with L.**
2. Still need to make sure we can create Megatron configs for the specific parameter counts we had in mind (maybe Maria already has these, just not 100% sure).
3. 5 × C tokens does not give the ATLAS compute-optimal point for ≥ 30 languages. Should we instead base full dataset sizes on ATLAS numbers for K = 200? This would be a ridiculous number of tokens (462B for the 1.7B model). — **method to settle it above ("Is 5 × C the right budget"): 12 WSD cooldown branches, ≈ 6 % of a level.**
4. Which design choice to use as the intervention: tokenizer, model depth, or sampling temperature. If the tokenizer, all data and the validation sets are built once per tokenizer.

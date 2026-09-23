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
| 1         | ✓   | ×3   | ✓    | ×3   | ×3  | ✓    |
| 2         | ✓   | ×3   | ✓    | ×3   | ×3  | ✓    |
| 8         | ✓   | ✓    | ✓    | ✓    | ✓   | ✓    |
| 15        | ✓   | ✓    | ✓    | ✓    | ✓   | ✓    |
| 30        | ✓   | ✓    | ✓    | ✓    | ×3  | ✓    |
| 50        | ✓   | ×3   | ✓    | ×3   | ×3  | ✓    |
| 100       | ✓   | ✓    | ✓    | ✓    | ✓   | ✓    |

The ×3 columns use **different seed triples**, because two people fill them:
175M and 600M run (64, 313, 1904) at L ∈ {1, 2, 50}; the 1B column is
aromanou's and runs (28, 1797, 1904) at L ∈ {1, 2, 30}, plus L50 (added
2026-09-10 to match the other ×3 columns — those two cells are new). The grid
names the seeds that exist on disk — under the wrong triple the launcher
would submit two more runs per cell and the auto-eval watcher would never
evaluate the ones already trained.

Her 1B runs were saved under the earlier 20-checkpoint regime (every 2287
iters, to 45740), while the 1B rung now saves 40 (every 1143 iters, to
45720; the regime itself is unchanged — 20 per run, 40 at the 1B, 60 at the
1.7B). None of their checkpoints lands on that grid (15 cells: seed 1904 at
L1, L2, L15, L30 and schemeB L8/L15/L30, and seeds 28/1797 at L1, L2, L30 and
schemeB L30; their schedule also ends at 45,740, the new 1B L50 ×3 at 45,720), so the
watchers now read due checkpoints on the run's *own* grid
(`launch_trainings.due_iters`): every save of a 20-checkpoint run sits at the
same k/20 fraction as every 2nd save of a 40-checkpoint run, so the two are
evaluated at the same points and compared checkpoint for checkpoint.

Changes of 2026-09-10: the 1.7B row gained L15 and L50, so every size now
trains at every setting; L100 lost its ×3 cells; L100 is no longer a
scheme-A cell at all — it exists only as the flattened `AT3` mixture below;
the 1B column gained its ×3 at L50; and `AT3` runs the whole ladder at L50
as well as L100 (the flattened L50 build covers the 1.7B draw, see
"Sampling temperature").

<!-- BEGIN generated: pretrain_progress.py --plot -->
| Axis | Values |
| ---- | ------ |
| Size (non-embedding) | 90M, 175M, 350M, 600M, 1B, 1.7B, 3B, every size at every setting except 3B at L ∈ {8, 15} only |
| Language setting L | 1, 2, 8, 15, 30, 50 (English + L−1 FineWeb-2 languages; L=1 is 100% English) |
| Seed | 1904 everywhere; ×3 on the marked columns — 64, 313, 1904 at 175M, L ∈ {1, 2, 50} · 64, 313, 1904 at 600M, L ∈ {1, 2, 50} · 28, 1797, 1904 at 1B, L ∈ {1, 2, 30} |
| Data scheme | **A** (L ∈ {1, 2, 8, 15, 30, 50}) · **AT3** (L ∈ {15, 30, 50}; T=3; L15 stops at 1.7B, L30 stops at 1.7B; L15 is deep only; L30 is deep only) · **B** (L ∈ {8, 15, 30}) · **ZH** (L ∈ {2}; L2 stops at 1.7B; deep only) · **BT3** (L ∈ {30}; T=3; L30 stops at 1.7B; deep only) · **ES** (L ∈ {2}; L2 stops at 1.7B; deep only) · **DCLMP** (L ∈ {1}; deep only) · **FWEB** (L ∈ {1}; deep only) |
| Architecture | deep (baseline) and shallow (the model-depth intervention) |

**56 runs** at one intervention level (scheme A, deep — the plan grid).
Counting every scheme and the architectures each is trained in: **186 runs**.

![Planned runs per grid cell](../src/pretrain/pretrain_progress_plan.png)

![Finished models per grid cell](../src/pretrain/pretrain_progress_simple.png)

![Eval work outstanding per grid cell](../src/pretrain/eval_progress.png)
<!-- END generated -->

(The 200-language setting was dropped on 2026-08-13 to fit the compute budget
and deadline. The ×3-seed rows were 1, 30 and 100; L=2 was added to them, and
the 1.7B row gained L=2, taking the grid from 52 to 56. The 2026-09-10 changes
above kept the scheme-A deep grid at 56 cells — L100's ×3 rows came out, the
1.7B row gained L15 and L50, the 1B column gained ×3 at L50 — and took the
whole sweep, over every scheme and the architectures each is trained in, to
191 runs. It is 185 since 2026-09-22 — 173 after the untrained shallow replicates
and the 1B ×3 at L50 were dropped again.)

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
- **Allocation within the FineWeb2 50%:** temperature sampling, `p_i ∝ p_i^(1/T)`. T = 1 (proportional to estimated per-language tokens) for the baseline scheme A. The flattened scheme `AT3` uses the same language lists at a higher temperature and supplies two settings: L50, where it is the temperature *intervention* against A's T = 1, and L100, which exists **only** as the flattened build. See "Sampling temperature" below for the measured allocations — including why the value is under revision from T = 3 to T = 2.
- **Language counts include English,** so the FineWeb2 language list for an L-setting has L − 1 entries. The lists are nested: the 1-language FineWeb2 list is a subset of the 7-language list, which is a subset of the 14-, 29-, 49-, and 99-language lists, all drawn from the 199 FineWeb2 languages used in the prior 200-language run. The validation build covers the 99-language list plus English (the largest trained setting). The lists are generated by `src/pretrain/data/generate_language_sets.py` from the FineWeb2 distribution and the benchmark availability in `configs/tasks.json`: the top-k subsets by train-split UTF-8 bytes, excluding `und_*` subsets and `hau_Latn` (absent from the swiss-ai filtered dataset dir); for the 99-language list, subsets with no benchmark in the lm-eval harness are replaced by the next subsets by bytes that have at least two benchmark families and are not a script variant of a kept language (2026-08-21: gmh, nrm, bew, tat, div, epo, hif, ltz → kin, jav, xho, hat, fao, zul, ibo, sot). Scheme B swaps the small settings for script/family-diverse picks. See `plan/benchmark_selection.md`.

To avoid tokenizing English once per setting, build the English data once and the FineWeb2 data once per setting, then blend them 50/50 at training time with the Megatron data loader's blend weights. So the artifacts are: one English dataset, one FineWeb2 dataset per multilingual setting, and one fixed validation set.

| Setting (L) | FineWeb2 languages | FineWeb2 build tokens | English share at training time |
| ----------- | ------------------ | --------------------- | ------------------------------ |
| 1           | 0 (English only)   | —                     | 100%                           |
| 2           | 1                  | 92B                   | 50%                            |
| 8           | 7                  | 92B                   | 50%                            |
| 15          | 14                 | 92B                   | 50%                            |
| 30          | 29                 | 92B                   | 50%                            |
| 50          | 49                 | 92B (A and AT3)       | 50%                            |
| 100         | 99                 | 92B (AT3 only)        | 50%                            |

Each FineWeb2 build is sized to half of the largest budget at that setting, with about 10% headroom, and that number is **derived from the grid** rather than tabulated (`build_data_mixtures.largest_size` reads `launch_trainings.scheme_sizes`): 92B where a 1.7B model trains (half of 167B plus headroom), 52B where the largest rung is the 1B (half of 94B plus headroom). Since the 1.7B row gained L15 and L50 on 2026-09-10, every scheme-A and AT3 build is 92B; only ZH and ES at L2, capped at the 1B rung, are 52B. The English dataset is built once to 184B, which covers the 1-language setting's largest need and the English half of every other setting.

The build script reports the realized per-language token counts and warns when a language runs out of data; record any shortfall. **The builder never repeats data** — it prints the shortfall and moves on — so a target the source cannot reach yields a *smaller* build, not a flatter one. Two consequences already bite:

- The L15 and L50 builds on capstor (scheme A and scheme B's L15) are the 52B ones made when the 1B topped those settings. They are being rebuilt at 92B into a parallel root (`data/launch_builds.sh`, the `REBUILD` array) rather than overwritten, because already-trained cells read the 52B copies.

  **A bigger build is an *extension* of a smaller one, not a different dataset.** The per-language proportions come from the estimated token counts and the temperature alone — `target_tokens = total × prop` — and each language's parquet files are read in `sorted()` order from the same start, minus the same fixed validation skip, with no build-time shuffling. So the 92B build's token stream per language *begins with* the 52B build's and continues. Nothing is re-drawn — verified byte for byte on all three rebuilds on 2026-09-13 (every language section of each 52B build is the start of the 92B one: 14, 49 and 14 languages).

  That is why **no already-trained cell has to be re-run.** A model reads only its own budget out of the build, and at L15/L50 every rung through the 1B fits inside 52B — 4.7B at 90M (9 % of the pool) up to 47.2B at the 1B (91 %). Only the 1.7B exceeds it, at 83.6B (1.61 epochs), which is the sole reason the rebuild exists. The residual asymmetry is that the 1.7B reference reads a 92B pool while its proxies read the nested 52B one; since the proportions are identical and each rung already reads a different *fraction* of the pool, that difference is of the same kind and scale as a data-order seed change — which the ×3 seed columns already quantify as noise. Record it; do not re-run the column for it. One qualification, measured 2026-09-13: FineWeb-2's parquet files are grouped by CommonCrawl dump, so the extension is not a random superset of the smaller pool but a newer one — in scheme B's L15, crawls from 2021–24 are 4% of the 52B Russian and 14% of the 92B, 0% and 32% of the Chinese. The 1.7B reference therefore also reads a later crawl mix than its proxies.

  **Nothing is swapped into the training stage.** `launch_trainings.py cscs` reads the 92B copy (staged to `/iopsstor/scratch/cscs/mariagrandury/data-92B`) only for a cell the staged 52B build is too small for — the 1.7B cells at A-L15, A-L50 and B-L15 — and keeps every other rung, including cells not trained yet (the shallow ones, the new 1B ×3 seeds at L50), on the 52B pool their peers read. Replacing the stage files would change what those cells see, not just how much: Megatron shuffles over the whole file, so a different pool is a different sample order as well as a newer crawl mix. Without a big-enough rebuild the launcher still skips a cell drawing more than its staged build holds (`skip [data undersized]`), unless the build already realizes what the source allows at the current target.

  **For the paper:** the six 1.7B cells at A-L15, A-L50 and B-L15 (deep and shallow) train on a different build from their smaller rungs — the same languages in the same proportions, extended with newer crawls — and nothing in `configs/models.json` records which build a cell read. The record is this paragraph, the launcher's `(FineWeb-2 from /iopsstor/scratch/cscs/mariagrandury/data-92B…)` line when it submitted them, and the `data-92B/` paths in those cells' training logs. State it wherever results at those settings compare the 1.7B reference with its proxies.
- **No L2 language can feed a 1.7B.** A 1.7B draws 83.6B from the multilingual half, and the filtered subset holds about 71.8B of Russian by the builder's estimate (the L2 build realized 72.8B, so the estimates undercount slightly), 59.9B of Chinese and 23.4B of Spanish. The existing scheme-A L2 build is 72.8B, not 92B, for exactly this reason; Spanish is clean only through the 350M rung. All three train to 1.7B regardless (ES uncapped 2026-09-23), repeating 1.15x / 1.61x / 3.51x — under the ~4 epochs at which repeated tokens stop being worth close to fresh ones, and the price of having a second-language axis at all: without ES at the reference it holds one pair, below the three rule 5 needs.

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

**As implemented (2026-08-21).** Each run saves **20 checkpoints** evenly spaced — **40 at the 1B and 60 at the 1.7B**, the two reference rungs, whose intervals also stay near the ~2000-iter Azure-spot eviction window. The interval is per size, `train_iters / n`, and 40 and 60 are multiples of 20, so checkpoint *k* sits at *k*/*n* of training at **every** size and the grids stay index-aligned across the ladder, which is what lets SNR compare checkpoint *k* between sizes. Because D = 5 × Chinchilla, the 1×C operating point (`train_iters / 5`) is always checkpoint *n*/5 — 4, 8 or 12 — on-grid at every size. Evaluation covers **twelve checkpoints per run** — the ten tenths of training plus 85 % and 95 %, which is exactly the set `ladder._on_shared_grid` keeps — read on the grid the run actually saved at (`launch_trainings.due_iters`: a 20-save run yields every 2nd save, a 40-save one every 4th, a 60-save one every 6th, all landing on the same fractions). Until 2026-09-21 the evaluated set scaled with save density instead (20 at 1B, 32 at 1.7B and 3B) and the surplus was computed and then discarded by the loader; the third piece decided 09-02 — **the checkpoint nearest each half-decade FLOPs milestone** (~1 extra per run, see "The compute axis" below) — is **not implemented yet**: the `configs.milestone_iters` helper it needs does not exist, so the watchers today run only the twelve-checkpoint rule. Every saved checkpoint is still converted to HF and kept, so the checkpoint-noise window can be densified later without retraining.

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
Those were interpolated between evaluated points. The plan (09-02, not yet
implemented — the watchers still run only every-2nd+final) is for the eval
due-rule to also mark the saved checkpoint nearest each half-decade FLOPs
milestone (a `configs.milestone_iters` helper, 1e18…1e21, skipping anything
still inside LR warmup, where a large model is not a decision-relevant
comparison).

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

### Re-measured against the filtered subset (2026-09-10)

The table above is computed from the **raw** FineWeb-2 byte distribution
(`fineweb2-language-distribution.csv`). The builds read the much smaller
**swiss-ai filtered** dir, and its per-language totals — measured by
`create_data_mixture.py` sampling the real parquet files, and recorded as the
`estimated=` column of every build log — are what actually bounds an
allocation. Redone on those figures, with each language capped at what it
has (the builder never repeats data: it prints a shortfall and moves on):

| L100 · 92B target · 99 languages | T = 1 | T = 2 | T = 3 |
|---|---:|---:|---:|
| Build realizes | 92.0 B | 85.9 B | **75.4 B** |
| Feeds the 1.7B (draws 83.6 B) | yes (0.91 epochs) | yes (0.97 epochs) | **no, 8.2 B short — 1.11 epochs**, every language seen 1.11 times |
| Largest language's share | 18.8 % | 7.2 % | 4.9 % |
| Median language | 90 M | 373 M | 373 M |
| Smallest language | 3.5 M | 14.5 M | 14.5 M |
| Languages that exhaust their source | 0 | 56 | 60 |

| L50 · 52B target · 49 languages | T = 1 | T = 2 | T = 3 |
|---|---:|---:|---:|
| Build realizes | 52.0 B | 52.0 B | 51.4 B |
| Median language | 529 M | 946 M | 1,037 M |
| Smallest language | 46.7 M | 281 M | 336 M |
| Languages that exhaust their source | 0 | 0 | 8 |

| L50 · 92B target · 49 languages | T = 1 | T = 2 | T = 3 |
|---|---:|---:|---:|
| Build realizes | 92.0 B | 90.8 B | 87.1 B |
| Feeds the 1.7B (draws 83.6 B) | yes (0.91 epochs) | yes (0.92 epochs) | yes (0.96 epochs) |
| Median language | 935 M | 1,673 M | 1,834 M |
| Smallest language | 82.6 M | 335.6 M | 335.6 M |
| Languages that exhaust their source | 0 | 9 | 13 |

The 92B row is why the AT3 column runs the whole ladder at L50 (decided
2026-09-10): even at T = 3 the flattened L50 build covers the 1.7B draw
without repetition, so the temperature intervention gets the same 1.7B
reference as every other axis. (The same simulation, run on the builder's
own per-language estimates from the finished L50/L100 build plans,
reproduces every number in the 52B and L100 tables above.)

**This changes the recommendation's basis, not its answer: T = 2.** At L100,
T = 2 and T = 3 reach an *identical* tail — same median, same smallest
language — because 56 of the 99 have already run out of data at T = 2. Raising
the temperature further only takes tokens off the head, and the build then
falls short of what the 1.7B rung needs. **At L100 the tail is data-limited,
not allocation-limited**, which is the thing the raw-bytes table could not
show. T = 3 is therefore strictly dominated there. At L50 it is still viable
(51.4 B, and a slightly better floor than T = 2), but L50's flattened build is
what calibrates L100 against the T = 1 curve, so the two must share one
temperature.

The 2026-09-02 argument below still stands on its own terms, and its
conclusion is unchanged:

- It lifts the smallest language from 0.8 M to **33 M tokens** at the 1B rung,
  a 40× change, and empties the "<10 M" column at every rung from 350M up.
- **It repeats nothing.** Max epochs 0.3 at 1B, 0.6 even at 1.7B — every
  language is still trained on unseen data. (Re-measured, a higher T does not
  duplicate either: the builder cannot repeat, so it short-builds instead.
  Same conclusion, different failure.)
- One value for every cell. **T must not vary with L** — the intervention is
  the language *count* at fixed English share; a T that moves with L confounds
  the two and makes the L-ladder uninterpretable.

The last bullet is the one the 2026-09-10 grid does **not** honour, knowingly:
L2…L50 are T = 1 and L100 is flattened. The mitigation is that L50 is built
*both* ways, so the A-vs-flattened pair at L50 measures the temperature effect
directly and lets L100 be read against the T = 1 curve instead of being
compared to it naively. Any cross-L claim that skips that correction is
confounded, and the plan should say so wherever such a claim is made.

Cost: the FineWeb-2 builds are per-setting. Under the 2026-09-10 grid this is
two builds, not a sweep-wide rebuild: L50 and L100 in the flattened scheme's
own directory. Nothing already built is discarded, and no L100 model has
trained yet.

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

1. Sampling temperature is set to T = 1 (proportional to estimated tokens) by default. Should the FineWeb2 proportion instead be tempered toward uniform (for example alpha = 0.3, i.e. T ≈ 3.3) to give lower-resource languages more weight? Similarly, the datasets are all fixed to 50% English. Is this sound or should it change as languages are added (for example, scaling down with the number of languages)? — **quantified above; recommendation: T = 2 sweep-wide, and never varying with L.** The 2026-09-10 grid departs from this knowingly: L2…L50 stay at T = 1, L100 exists only flattened and is calibrated by the L50 pair, and the flattened scheme is coded at T = 3 (`AT3`), where the 1.7B at L100 repeats its data 1.11×.
2. Still need to make sure we can create Megatron configs for the specific parameter counts we had in mind (maybe Maria already has these, just not 100% sure).
3. 5 × C tokens does not give the ATLAS compute-optimal point for ≥ 30 languages. Should we instead base full dataset sizes on ATLAS numbers for K = 200? This would be a ridiculous number of tokens (462B for the 1.7B model). — **method to settle it above ("Is 5 × C the right budget"): 12 WSD cooldown branches, ≈ 6 % of a level.**
4. Which design choice to use as the intervention: tokenizer, model depth, or sampling temperature. If the tokenizer, all data and the validation sets are built once per tokenizer.

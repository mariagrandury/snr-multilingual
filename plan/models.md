# Predictivity sweep — model card sheet

All models trained in the small-to-large predictivity study
([plan](small-to-large-predictivity-training-plan.md) ·
[compute budget](compute-budget.md)). Source of truth: the two **reviewed**
hyperparams files — `src/pretrain/hyperparams/hyperparams_deep.json` (deep
baseline) and `src/pretrain/hyperparams/hyperparams_shallow.json` (shallow
depth-intervention variant) — including the D = 100·N training schedule,
stored per size in each config's `predictivity` block — plus the per-cell
knobs `launch_trainings.py` derives (`ADEMAMIX_WARMUP`, `INIT_STD`,
`SAVE_INTERVAL`). Regenerate this sheet if they change.
Updated 2026-08-28 (from the files as committed; earlier revisions of this
sheet carried the fixed-100B-token LRs and the 51-run grid).

## The grid

L ∈ {1, 2, 8, 15, 30, 50, 100} languages (English + L−1 FineWeb-2 languages,
lists per scheme in `src/pretrain/data/language_sets_scheme{A,B}.json`).
✓ = one seed (1904) · **×3** = seeds 28, 1797, 1904. The generated block in
[`src/pretrain/README.md`](../src/pretrain/README.md) is the live version of
this table.

| Languages | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| 1 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| 2 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 30 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| 50 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 100 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| **Runs/level** | 7 | 15 | 7 | 7 | 15 | 5 |

**56 runs per intervention level** (scheme A, deep). Counting both
architectures and scheme B where its language set differs (L ∈ {8, 15, 30}):
154 runs. Cell name = Slurm job name = checkpoint dir = W&B run name:
`lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>` (e.g.
`lm-1B-L30-deep-seed28`, `lm-1B-L8-schemeB-shallow-seed1904`). W&B project **`msnr`**
(entity `mariagrandury-epflnlp`).

## Transformations (the intervention axis)

Each cell trains at ≥2 levels of one design choice; the analysis asks whether
small models rank the levels the way the largest model at that L does (BPB per
language on the fixed validation set). Level 1 is always the baseline below:

| Axis | Levels | Status |
|---|---|---|
| Model depth | deep (width/depth ≈ 64) vs shallow (width/depth ≈ 128) at equal non-emb size | **wired**: `--arch deep\|shallow` in both launchers, same data |
| Data scheme | A (resource-ranked language sets) vs B (diversity-first) | **wired**: `--scheme B`, differs from A only at L ∈ {8, 15, 30} |
| Tokenizer | Apertus V1 (swiss-ai/Apertus-70B-2509) vs a candidate | open — BPB is byte-denominated so results stay comparable; data must be rebuilt per tokenizer |

## Architecture per size — deep baseline (`hyperparams_deep.json`)

Non-embedding parameter convention (Signal-and-Noise / OLMo ladder); tied
embeddings; head_dim 64; GQA ratio 4; FFN multiplier 4 (xIELU, non-gated);
width/depth ≈ 64 (the `find_hyperparams_deep.py` rule).

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Layers | 15 | 16 | 20 | 24 | 28 | 30 |
| d_model | 768 | 1024 | 1280 | 1536 | 1792 | 2304 |
| FFN size | 3072 | 4096 | 5120 | 6144 | 7168 | 9216 |
| Attention heads | 12 | 16 | 20 | 24 | 28 | 36 |
| KV groups | 3 | 4 | 5 | 6 | 7 | 9 |
| Non-emb params | 92.90M | 176.16M | 344.06M | 594.54M | 944.11M | 1,672.15M |
| Total params (tied) | 193.6M | 310.4M | 511.9M | 795.9M | 1,179.0M | 1,974.1M |

## Architecture per size — shallow variant (`hyperparams_shallow.json`)

The model-depth intervention level: the same six non-embedding sizes at
width/depth ≈ 128 (`find_hyperparams_shallow.py`: head_dim 64, FFN multiplier
4, GQA ratio 4 — both pinned to the deep ladder so the two differ in depth
only). Launched with `--arch shallow`.

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Layers | 8 | 10 | 14 | 14 | 17 | 20 |
| d_model | 1024 | 1280 | 1536 | 2048 | 2304 | 2816 |
| FFN size | 4096 | 5120 | 6144 | 8192 | 9216 | 11264 |
| Attention heads | 16 | 20 | 24 | 32 | 36 | 44 |
| KV groups | 4 | 5 | 6 | 8 | 9 | 11 |
| Non-emb params | 88.08M | 172.03M | 346.82M | 616.56M | 947.55M | 1,665.27M |
| vs deep target | −5.2% | −2.3% | +0.8% | +3.7% | +0.4% | −0.4% |
| Total params (tied) | 222.3M | 339.8M | 548.1M | 885.0M | 1,249.6M | 2,034.4M |

## Training schedule per size

D = 100 × N_non-emb exactly (5× Chinchilla), stored per size in each
config's `predictivity` block; one iteration = 504 × 4096 =
2,064,384 tokens. Iterations are rounded to the checkpoint grid (20 evenly
spaced checkpoints per run, 40 at 1.7B) so `save_interval = train_iters / 20`
divides exactly and the 1×C point (`train_iters / 5`) is checkpoint 4 (8 at
1.7B) at every size. Peak LR from the 6ND law at each run's own budget
(C = 6·N·100N); warmup ≈ 4% and WSD decay ≈ 20% of iterations (rounded to
100). Micro-batch and nodes are the cluster values (4× GH200 per node);
`launch_trainings.cscs_mbs` shrinks MBS where the layout needs it.

Deep baseline:

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Train tokens | 9.29B | 17.62B | 34.41B | 59.45B | 94.41B | 167.22B |
| Iterations | 4,500 | 8,540 | 16,660 | 28,800 | 45,740 | 81,000 |
| Peak LR | 1.428e-3 | 1.217e-3 | 1.029e-3 | 8.976e-4 | 7.996e-4 | 6.931e-4 |
| LR warmup iters | 200 | 300 | 700 | 1,200 | 1,800 | 3,200 |
| WSD decay iters | 900 | 1,700 | 3,300 | 5,800 | 9,100 | 16,200 |
| Micro-batch · nodes | 7 · 3 | 7 · 6 | 3 · 14 | 6 · 21 | 6 · 21 | 2 · 21 |
| Checkpoint interval (iters) | 225 | 427 | 833 | 1,440 | 2,287 | 2,025 (×40) |
| 1×C checkpoint (20N tokens) | iter 900 | 1,708 | 3,332 | 5,760 | 9,148 | 16,200 |

Shallow variant (its own N → slightly different schedules; no `nodes`
column in its file — the deep ladder's per-size node counts apply):

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Train tokens | 8.81B | 17.20B | 34.68B | 61.66B | 94.76B | 166.53B |
| Iterations | 4,260 | 8,340 | 16,800 | 29,860 | 45,900 | 80,680 |
| Peak LR | 1.447e-3 | 1.224e-3 | 1.027e-3 | 8.894e-4 | 7.988e-4 | 6.938e-4 |
| LR warmup iters | 200 | 300 | 700 | 1,200 | 1,800 | 3,200 |
| WSD decay iters | 900 | 1,700 | 3,400 | 6,000 | 9,200 | 16,100 |
| Micro-batch (config) | 7 | 14 | 8 | 4 | 3 | 2 |
| Checkpoint interval (iters) | 213 | 417 | 840 | 1,493 | 2,295 | 2,017 (×40) |
| 1×C checkpoint (20N tokens) | iter 852 | 1,668 | 3,360 | 5,972 | 9,180 | 16,136 |

## Shared training configuration (every run)

| | |
|---|---|
| Framework | swiss-ai/Megatron-LM fork (`pretrain_gpt.py`, commit `c92402e` + `src/pretrain/patches/`), bf16 + fp32 main grads, flash attention |
| Global batch · sequence | 504 × 4096 tokens (2,064,384 tokens/iter) |
| Tokenizer · vocab | swiss-ai/Apertus-70B-2509 (V1) · 131,072 (divisible-by-128 padding) |
| Positional | RoPE, base 500,000, rope-scaling factor 32 |
| Norm / activation | RMSNorm · xIELU · QK-layernorm (apex impl) · no biases |
| Dropout | 0.0 (attention and hidden) |
| Optimizer | AdEMAMix: β1 0.9, β2 0.999, β3 0.9999, α 8; β3/α warmup = the run's full schedule (`ADEMAMIX_WARMUP` = target iters, identical on every resume) |
| Regularization | weight decay 0.1 · grad clip 0.1 |
| LR schedule | WSD (warmup → constant → 1-sqrt decay), min-lr 0; `--use-checkpoint-opt_param-scheduler` so capped resumes stay on the saved curve |
| Init | normal, std = 0.008944 × √(1792 / d_model) (width-scaled, anchored at the 1B) |
| Parallelism | TP 1 · PP 1 · pure DP + distributed optimizer, overlap grad-reduce/param-gather |
| Checkpointing | torch_dist, async save, every `train_iters / 20` iters (/ 40 at 1.7B) — see the interval rows above |
| Seeds | 1904 baseline; 28 / 1797 / 1904 on ×3 cells (seed sets init AND data order) |

## Data per language setting

50/50 English (DCLM-edu) / FineWeb-2 blend at training time via Megatron blend
weights (`DATA_BLEND`); L = 1 is 100% English. Builds sized for the largest
run at each setting + 10% headroom (`build_data_mixtures.py`,
`LARGEST_SIZE_PER_SETTING`), validation rows excluded from training via the
manifest. Live per-language coverage: `src/pretrain/data/data_progress.py`.

| Setting | FW-2 languages | FW-2 build | Largest run |
|---|---|---|---|
| L1 | — (100% EN) | — | 1.7B · 167.2B tokens |
| L2 | 1 | 92.0B | 1.7B · 167.2B (source-limited: rus_Cyrl alone holds ~73B) |
| L8 | 7 | 92.0B | 1.7B · 167.2B |
| L15 | 14 | 52.0B | 1B · 94.4B |
| L30 | 29 | 92.0B | 1.7B |
| L50 | 49 | 52.0B | 1B |
| L100 | 99 | 92.0B | 1.7B |
| English (shared) | — | 184.0B | bounds the L1 1.7B run |
| Validation (fixed) | 99 + EN | 5M tokens/language | reused by every model |

Language lists: scheme A (resource-ranked) or scheme B (diversity-first),
nested across settings — `src/pretrain/data/language_sets_scheme{A,B}.json`.

## Evaluation

Auto-evals during training on **every 2nd checkpoint, each run's final
one, and the checkpoint nearest each half-decade FLOPs milestone** (`auto_evals_cscs.py` / `auto_evals_azure.py`): the `auto` benchmark
group of `configs/tasks.json`, expanded to one task per benchmark per
language the cell trains on (15 tasks at L1, 463 at L100), pushed to W&B
`msnr`. Per-language bits-per-byte on the fixed validation set (byte counts
from the validation manifest) on the languages each model trained on.
Reference at each L = the largest model trained there (1.7B or 1B).

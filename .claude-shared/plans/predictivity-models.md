# Predictivity sweep — model card sheet

All models trained in the small-to-large predictivity study
([plan](small-to-large-predictivity-training-plan.md) ·
[compute budget](predictivity-compute-budget.md)), generated from
`src/pretrain/hyperparams_predictivity.json` (source of truth — regenerate this
sheet if that file changes). A rendered version lives at the "Predictivity
Training Grid" artifact. Updated 2026-08-14.

## The grid

L ∈ {1, 2, 8, 15, 30, 50, 100} languages (English + L−1 FineWeb-2 languages,
lists per scheme in `src/pretrain/language_sets_scheme{A,B}.json`).
✓ = one seed (1904) · **×3** = seeds 28, 1797, 1904.

| Languages | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| 1 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| 2 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 30 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| 50 | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 100 | ✓ | **×3** | ✓ | ✓ | **×3** | ✓ |
| **Runs/level** | 7 | 13 | 7 | 7 | 13 | 4 |

**51 runs per intervention level × 3 levels ("transformations") = 153 runs.**
Run naming: `apertus-<size>-L<L>-seed<seed>` (e.g. `apertus-1B-L30-seed28`);
W&B project `predictivity`.

## Transformations (the intervention axis)

Each cell trains at ≥3 levels of one design choice; the analysis asks whether
small models rank the levels the way the largest model at that L does (BPB per
language on the fixed validation set). Level 1 is always the baseline below;
the choice of axis is plan open question 4:

| Candidate axis | Levels | Notes |
|---|---|---|
| Tokenizer | Apertus V1 (swiss-ai/Apertus-70B-2509) vs V2 candidate vs same-size alternative | BPB is byte-denominated → comparable across tokenizers; data rebuilt per tokenizer |
| Sampling temperature | T = 1 vs larger (e.g. T ≈ 3.3 / α = 0.3) | most multilingual-specific; meaningless at L = 1 |
| Model depth | deeper-narrower vs shallower-wider at equal non-emb size | most controlled; effect may be too small |

## Architecture per size

Non-embedding parameter convention (Signal-and-Noise / OLMo ladder); tied
embeddings; head_dim 64; GQA ratio 4; FFN multiplier 4 (xIELU, non-gated).

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Layers | 15 | 16 | 20 | 24 | 28 | 30 |
| d_model | 768 | 1024 | 1280 | 1536 | 1792 | 2304 |
| FFN size | 3072 | 4096 | 5120 | 6144 | 7168 | 9216 |
| Attention heads | 12 | 16 | 20 | 24 | 28 | 36 |
| KV groups | 3 | 4 | 5 | 6 | 7 | 9 |
| Non-emb params | 92.90M | 176.16M | 344.06M | 594.54M | 944.11M | 1,672.15M |
| Total params (tied) | 193.6M | 310.4M | 511.9M | 795.9M | 1,179.0M | 1,974.1M |

## Training schedule per size

D = 100 × N_non-emb exactly (5× Chinchilla); one iteration = 504 × 4096 =
2,064,384 tokens; LR from lr = 0.14015 · N^(−1/4); warmup ≈ 4% and WSD decay
≈ 20% of iterations (rounded to 100).

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| Train tokens | 9.29B | 17.62B | 34.41B | 59.45B | 94.41B | 167.22B |
| Iterations | 4,500 | 8,500 | 16,700 | 28,800 | 45,700 | 81,000 |
| Peak LR | 1.428e-3 | 1.217e-3 | 1.029e-3 | 8.976e-4 | 8.00e-4 | 6.931e-4 |
| LR warmup iters | 200 | 300 | 700 | 1,200 | 1,800 | 3,200 |
| WSD decay iters | 900 | 1,700 | 3,300 | 5,800 | 9,100 | 16,200 |
| Micro-batch (cluster) | 28 | 14 | 8 | 4 | 3 | 2 |
| MBS resolved · 2×H100 | 28 | 14 | 7 | 4 | 3 | 2 |
| MBS resolved · 8×H100 | 21 | 9 | 7 | 3 | 3 | 1 |
| Checkpoints (every 2,000 it) | 2 | 4 | 8 | 14 | 22 | 40 |
| 1×C checkpoint (20N tokens) | iter 900 | 1,700 | 3,340 | 5,760 | 9,140 | 16,200 |

MBS auto-resolves per node so 504 % (DP × MBS) == 0 (`azure/train.sh`); the
8-GPU 1.7B value of 1 can be hand-raised (`--set environment_variables.MBS=3`)
if H100-94GB memory allows. The cluster `nodes` column of
`hyperparams_predictivity.json` is currently inconsistent with GBS 504 on
4-GPU nodes (see budget sheet, Appendix B) — Azure is unaffected.

## Shared training configuration (every run)

| | |
|---|---|
| Framework | swiss-ai/Megatron-LM fork (`pretrain_gpt.py`), bf16 + fp32 main grads |
| Global batch · sequence | 504 × 4096 tokens (2,064,384 tokens/iter) |
| Tokenizer · vocab | swiss-ai/Apertus-70B-2509 (V1) · 131,072 (divisible-by-128 padding) |
| Positional | RoPE, base 500,000, rope-scaling factor 32 |
| Norm / activation | RMSNorm · xIELU · QK-layernorm (apex impl) · no biases |
| Dropout | 0.0 (attention and hidden) |
| Optimizer | AdEMAMix: β1 0.9, β2 0.999, β3 0.9999, α 8, β3/α-warmup 100,000 |
| Regularization | weight decay 0.1 · grad clip 0.1 |
| LR schedule | WSD (warmup → constant → 1-sqrt decay), min-lr 0 |
| Init | normal, std 0.008944 |
| Parallelism | TP 1 · PP 1 · pure DP + distributed optimizer, overlap grad-reduce/param-gather |
| Checkpointing | torch_dist every 2,000 iters, async save, `--use-checkpoint-opt_param-scheduler` |
| Seeds | 1904 baseline; 28 / 1797 / 1904 on ×3 cells (seed sets init AND data order) |

## Data per language setting

50/50 English (DCLM-edu) / FineWeb-2 blend at training time via Megatron blend
weights (`DATA_BLEND`); L = 1 is 100% English. Builds sized for the largest
run at each setting + 10% headroom (`build_data_mixtures.py`), validation
rows excluded from training via the manifest:

| Setting | FW-2 languages | FW-2 build | Largest run |
|---|---|---|---|
| L1 | — (100% EN) | — | 1.7B · 167.2B tokens |
| L2 | 1 | 52.0B | 1B · 94.4B |
| L8 | 7 | 92.5B | 1.7B · 167.2B |
| L15 | 14 | 52.0B | 1B |
| L30 | 29 | 92.5B | 1.7B |
| L50 | 49 | 52.0B | 1B |
| L100 | 99 | 92.5B | 1.7B |
| English (shared) | — | 184.5B | bounds the L1 1.7B run |
| Validation (fixed) | 99 + EN | 5M tokens/language | reused by every model |

Language lists: scheme A (resource-ranked) or scheme B (diversity-first),
nested across settings — `src/pretrain/language_sets_scheme{A,B}.json`.

## Evaluation

Per-language bits-per-byte on the fixed validation set (byte counts from the
validation manifest), on the languages each model trained on; auto-evals
during training every 5 checkpoints (`auto` task group) where applicable.
Reference at each L = the largest model trained there (1.7B or 1B).

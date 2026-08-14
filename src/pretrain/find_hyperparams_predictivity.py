#!/usr/bin/env python3
"""
Generate the model ladder for the small-to-large predictivity sweep.

Six sizes on one self-consistent architecture family — aspect ratio 64
(d_model = 64 x n_layers), GQA=4, FFN x4 — so the only thing that changes
across the ladder is scale. The four middle rungs reproduce the existing
`hyperparams_deep.json` sizes exactly; the 75M and 1.4B rungs extend the same
family down and up (see
.claude-shared/plans/small-to-large-predictivity-training-plan.md, "Parameter-
count convention").

Unlike `find_hyperparams_deep.py` (which trains every size to a single global
100B-token budget), this sweep trains each size to its OWN budget
D(N) = 5 x Chinchilla = 100 x N (non-embedding N). So the learning rate, batch
size, and iteration count are computed per size at that size's D(N), via the
Chinchilla C = 6 N D used by `set_lr_and_bs`.

Writes:
  hyperparams_predictivity.json           — per-size architecture + training config
  hyperparams_predictivity_explanation.txt — human-readable table
"""

import json
import sys
from pathlib import Path

from calculate_params_lr_bs import parameter_count
from find_hyperparams import _Tee, calculate_hyperparams

SCRIPT_DIR = Path(__file__).parent

VOCAB_SIZE = 131072
SEQ_LEN = 4096
GBS = 504
HEAD_DIM = 64
GQA_RATIO = 4
FFW_MULTIPLIER = 4
CHINCHILLA_MULTIPLE = 5  # train each size to 5 x Chinchilla-optimal (D = 100 x N)

# Tokens consumed per optimizer step at the fixed global batch size — this, not
# the theoretical optimal batch size, sets how many iters reach D(N) tokens.
TOKENS_PER_ITER = GBS * SEQ_LEN  # 504 x 4096 = 2,064,384

# WSD schedule shape, as fractions of train_iters. Mirrors the canonical
# 50000-iter run (2000 warmup, 10000 = last-20% decay) so the LR curve has the
# same shape at every size; both are scaled per size and snapped to 100 iters.
WARMUP_FRAC = 0.04
WSD_DECAY_FRAC = 0.20

# Ladder defined by depth only; width = 64 x n_layers gives aspect ratio 64.
# Label is the nominal non-embedding size (the realized count is written to the
# JSON and is what downstream token budgets use). `nodes` is the Slurm node
# count: the four shared rungs match hyperparams_deep.json; 75M and 1.4B are
# first estimates to tune against the per-size cost table once they run.
LADDER = [
    {"label": "75M",  "n_layers": 12, "nodes": 4},
    {"label": "175M", "n_layers": 16, "nodes": 6},
    {"label": "350M", "n_layers": 20, "nodes": 14},
    {"label": "600M", "n_layers": 24, "nodes": 21},
    {"label": "1B",   "n_layers": 28, "nodes": 21},
    {"label": "1.4B", "n_layers": 32, "nodes": 28},
]


def snap_100(x: float) -> int:
    """Round to the nearest 100 iters, with a floor of 100."""
    return max(100, round(x / 100) * 100)


def architecture(n_layers: int) -> dict:
    """Architecture knobs for a rung from its depth (aspect ratio 64, GQA=4,
    FFN x4)."""
    d_model = HEAD_DIM * n_layers
    num_heads = d_model // HEAD_DIM
    return {
        "n_layers": n_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_kv_heads": num_heads // GQA_RATIO,
        "ffw_size": d_model * FFW_MULTIPLIER,
    }


def print_ladder(json_configs: dict) -> None:
    """Print the realized ladder with each size's own 5xC token budget."""
    hdr = "  {:<6} {:<9} {:<8} {:<8} {:<10} {:<11} {:<5} {:<12} {:<8}"
    row = "  {:<6} {:<9} {:<8} {:<8} {:<10} {:<11.4f} {:<5} {:<12.6f} {:<8}"
    print(hdr.format("label", "n_layers", "d_model", "kv_heads",
                     "ffn", "n_non_emb", "mbs", "lr", "iters"))
    print("  " + "-" * 92)
    for label, c in json_configs.items():
        print(row.format(
            label, c["n_layers"], c["hidden_size"], c["num_query_groups"],
            c["ffn_hidden_size"], c["n_non_emb_params"] / 1e9,
            c["micro_batch_size"], c["lr"], c["train_iters"],
        ) + f"   D={c['train_tokens']/1e9:.1f}B")
    print()


def build_configs() -> dict:
    """Compute architecture + per-size training hyperparameters for every rung.

    Each rung's token budget is D = 100 x N (5 x Chinchilla on its own
    non-embedding parameter count), and lr / batch size / iters are set at that
    budget — not at a shared global token count.
    """
    json_configs = {}

    for rung in LADDER:
        arch = architecture(rung["n_layers"])
        _, n_non_emb = parameter_count(
            vocab_size=VOCAB_SIZE,
            n_layers=arch["n_layers"],
            d_model=arch["d_model"],
            num_heads=arch["num_heads"],
            num_kv_heads=arch["num_kv_heads"],
            ffw_size=arch["ffw_size"],
            verbose=False,
        )
        train_tokens = round(CHINCHILLA_MULTIPLE * 20 * n_non_emb)  # 100 x N

        # lr and mbs come from the Chinchilla C = 6 N D fit (batch-size
        # independent); the returned iter count is for the *theoretical*
        # optimal batch, so we discard it and derive iters from the fixed GBS.
        _, _, mbs, lr, _, _, _ = calculate_hyperparams(
            model_size=rung["label"],
            vocab_size=VOCAB_SIZE,
            seq_len=SEQ_LEN,
            desired_tokens=train_tokens,
            gbs=GBS,
            n_layers=arch["n_layers"],
            d_model=arch["d_model"],
            num_heads=arch["num_heads"],
            num_kv_heads=arch["num_kv_heads"],
            ffw_size=arch["ffw_size"],
            verbose=False,
        )
        train_iters = snap_100(train_tokens / TOKENS_PER_ITER)

        json_configs[rung["label"]] = {
            # Architecture
            "n_layers": arch["n_layers"],
            "hidden_size": arch["d_model"],
            "ffn_hidden_size": arch["ffw_size"],
            "num_attention_heads": arch["num_heads"],
            "num_query_groups": arch["num_kv_heads"],
            "n_non_emb_params": n_non_emb,
            "nodes": rung["nodes"],
            # Training — budget is per-size (5 x Chinchilla = 100 x N)
            "micro_batch_size": mbs,
            "global_batch_size": GBS,
            "seq_len": SEQ_LEN,
            "tokens_per_iter": TOKENS_PER_ITER,
            "train_tokens": train_tokens,
            "train_iters": train_iters,
            # WSD schedule, scaled per size so every run has the same LR shape
            "lr_warmup_iters": snap_100(WARMUP_FRAC * train_iters),
            "lr_wsd_decay_iters": snap_100(WSD_DECAY_FRAC * train_iters),
            # Learning rate, set at this size's own token budget
            "lr": round(lr, 8),
        }

    print("=" * 60)
    print("Predictivity ladder (5 x Chinchilla per size, D = 100 x N)")
    print("=" * 60)
    print_ladder(json_configs)
    return json_configs


if __name__ == "__main__":
    output_json = SCRIPT_DIR / "hyperparams_predictivity.json"
    output_txt = SCRIPT_DIR / "hyperparams_predictivity_explanation.txt"

    log_file = open(output_txt, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    json_configs = build_configs()

    sys.stdout = sys.__stdout__
    log_file.close()

    output = {
        "global": {
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "global_batch_size": GBS,
            "head_dim": HEAD_DIM,
            "chinchilla_multiple": CHINCHILLA_MULTIPLE,
        },
        "configs": json_configs,
    }
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Config saved to  {output_json}")
    print(f"Table saved to   {output_txt}")

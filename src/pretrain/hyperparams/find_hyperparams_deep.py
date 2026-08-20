#!/usr/bin/env python3
"""
Calculate params, lr, and batch size for the deep model configs in hyperparams_deep.json.

Outputs:
  hyperparams_deep.json          — updated with computed training hyperparams
  hyperparams_deep_explanation.txt — human-readable config table (mirrors find_hyperparams_shallow.py output)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from calculate_params_lr_bs import get_learning_rate
from find_hyperparams_shallow import calculate_hyperparams, print_configs_table

SCRIPT_DIR = Path(__file__).parent


class _Tee:
    """Write to multiple streams at once (e.g. stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def compute_configs(deep_data: dict) -> dict:
    g = deep_data["global"]
    vocab_size = g["vocab_size"]
    seq_len = g["seq_len"]
    desired_tokens = g["desired_tokens"]
    gbs = g["global_batch_size"]
    head_dim = g["head_dim"]

    print("=" * 60)
    print("Deep model configs")
    print("=" * 60)

    table_rows = []
    json_configs = {}

    for size_label, config in deep_data["configs"].items():
        nl = config["n_layers"]
        # Support both raw format (d_model, gqa_ratio, ffw_multiplier)
        # and already-computed format (hidden_size, num_attention_heads, ...)
        dm = config.get("d_model") or config["hidden_size"]
        nh = config.get("num_attention_heads") or (dm // head_dim)
        nkv = config.get("num_query_groups") or (nh // config["gqa_ratio"])
        ffw_size = config.get("ffn_hidden_size") or (dm * config["ffw_multiplier"])
        ffw_multiplier = ffw_size // dm

        _, n_non_emb, mbs, lr, _, train_iters, _ = calculate_hyperparams(
            model_size=size_label,
            vocab_size=vocab_size,
            seq_len=seq_len,
            desired_tokens=desired_tokens,
            gbs=gbs,
            n_layers=nl,
            d_model=dm,
            num_heads=nh,
            num_kv_heads=nkv,
            ffw_size=ffw_size,
            verbose=False,
        )

        table_rows.append(
            {
                "label": size_label,
                "n_layers": nl,
                "d_model": dm,
                "num_heads": nh,
                "num_kv_heads": nkv,
                "ffw_multiplier": ffw_multiplier,
                "n_non_emb": n_non_emb,
            }
        )

        # Predictivity-sweep schedule (D = 100 x N, ~4% warmup, ~20% WSD
        # decay, rounded to 100) mirrored per size so this file alone gives
        # the full training config; the predictivity launchers read it.
        p_iters = round(100 * n_non_emb / (gbs * seq_len) / 100) * 100
        json_configs[size_label] = {
            # Architecture
            "n_layers": nl,
            "hidden_size": dm,
            "ffn_hidden_size": ffw_size,
            "num_attention_heads": nh,
            "num_query_groups": nkv,
            "n_non_emb_params": n_non_emb,
            # Cluster layout and micro-batch are hand-tuned in the JSON
            # (audited nodes x MBS pairs, memory caps like the 90M MBS=7);
            # carry them over instead of resetting to the suggest_mbs value.
            **({"nodes": config["nodes"]} if "nodes" in config else {}),
            # Training
            "micro_batch_size": config.get("micro_batch_size", mbs),
            "global_batch_size": gbs,
            "seq_len": seq_len,
            # Top-level train_iters is the finished 36-sweep's record (the
            # predictivity launchers ignore it) — preserve, don't recompute.
            "train_iters": config.get("train_iters", train_iters),
            # Learning rate: 6ND law at the run's OWN predictivity budget
            # D = 100 x N, i.e. C = 600 x N^2 — not the legacy fixed 100B.
            "lr": round(get_learning_rate(6 * n_non_emb * (100 * n_non_emb)), 8),
            "predictivity": {
                "train_tokens": int(100 * n_non_emb),
                "train_iters": p_iters,
                "lr_warmup_iters": max(100, round(p_iters * 0.04 / 100) * 100),
                "lr_wsd_decay_iters": round(p_iters * 0.20 / 100) * 100,
            },
        }

    print_configs_table(
        "",
        table_rows,
        vocab_size=vocab_size,
        gbs=gbs,
        seq_len=seq_len,
        desired_tokens=desired_tokens,
        head_dim=head_dim,
    )

    return json_configs


if __name__ == "__main__":
    input_path = SCRIPT_DIR / "hyperparams_deep.json"
    output_txt = SCRIPT_DIR / "hyperparams_deep_explanation.txt"
    output_json = SCRIPT_DIR / "hyperparams_deep.json"

    with open(input_path) as f:
        deep_data = json.load(f)

    log_file = open(output_txt, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    json_configs = compute_configs(deep_data)

    sys.stdout = sys.__stdout__
    log_file.close()

    output = {
        "global": deep_data["global"],
        "configs": json_configs,
    }
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Config saved to  {output_json}")
    print(f"Table saved to   {output_txt}")

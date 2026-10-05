"""Derive hyperparams_swiglu.json — the deep family with SwiGLU in place of XIELU.

Unlike the deep and shallow families this is not a search: the activation is
the intervention, so every other shape parameter is COPIED from
hyperparams_deep.json and only `ffn_hidden_size` moves. Megatron's `--swiglu`
MLP holds three weight matrices per layer (gate, up, down) against XIELU's two,
so at an unchanged FFN width the family would carry ~33 % more non-embedding
parameters and the rung labels would stop meaning the same thing across the two
families. Solving 3 x ffn_swiglu = 2 x ffn_deep gives ffn = 8/3 x hidden; the
attention term cancels, so the match is exact wherever 8/3 x hidden is an
integer multiple of ROUND (90M, 600M, 1.7B) and within 0.6 % elsewhere — well
inside the -5.2 %..+3.7 % the shallow family already spans against deep.

Everything downstream follows the cell's OWN parameter count, as it does for
shallow: the token budget D = 100 x N, the LR from the 6ND law at that budget,
and the checkpoint grid. The `size` label stays nominal (analysis NON_EMB keys
on the label, not on the config).

    python find_hyperparams_swiglu.py          # rewrites hyperparams_swiglu.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_params_lr_bs import get_learning_rate, parameter_count  # noqa: E402

HERE = Path(__file__).resolve().parent
DEEP = HERE / "hyperparams_deep.json"
OUT = HERE / "hyperparams_swiglu.json"
# A multiple of 64 keeps the GEMMs tensor-core friendly. TP is 1 for every
# ladder cell (the KV-head counts force it), so nothing else constrains it.
ROUND = 64
# The deep rungs this family mirrors. 3B is deep only and out of scope here.
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]


def main() -> None:
    deep = json.loads(DEEP.read_text())
    g, C = deep["global"], deep["configs"]
    gbs, seq_len = g["global_batch_size"], g["seq_len"]
    configs = {}
    print("%-6s %-7s %-9s %-9s %-15s %-15s %s"
          % ("size", "hidden", "ffn deep", "ffn swi", "N deep", "N swiglu", "delta"))
    for size in SIZES:
        c = C[size]
        ffn = int(round(c["ffn_hidden_size"] * 2 / 3 / ROUND)) * ROUND
        _, n_non_emb = parameter_count(
            vocab_size=g["vocab_size"], n_layers=c["n_layers"], d_model=c["hidden_size"],
            num_heads=c["num_attention_heads"], num_kv_heads=c["num_query_groups"],
            ffw_size=ffn, n_experts=1, swiglu_or_geglu=True, tied_weights=True,
            verbose=False)
        raw_iters = 100 * n_non_emb / (gbs * seq_len)
        n_ckpts = 20 if raw_iters < 30000 else 40 if raw_iters < 60000 else 60
        p_iters = round(raw_iters / n_ckpts) * n_ckpts
        configs[size] = {
            "n_layers": c["n_layers"],
            "hidden_size": c["hidden_size"],
            "ffn_hidden_size": ffn,
            "num_attention_heads": c["num_attention_heads"],
            "num_query_groups": c["num_query_groups"],
            "n_non_emb_params": n_non_emb,
            # Deep's micro-batch is a hand-tuned memory cap, not a derived
            # value (suggest_mbs proposes 24 at 90M against the 7 that fits),
            # so it is copied rather than recomputed. The gated MLP's two
            # ffn-wide activations against XIELU's one make this family's
            # footprint ~4/3 of deep's per token at 2/3 the width, i.e. about
            # the same -- and deep's cap is the conservative side of that.
            "micro_batch_size": c["micro_batch_size"],
            "global_batch_size": gbs,
            "seq_len": seq_len,
            "train_iters": c["train_iters"],
            "lr": round(get_learning_rate(6 * n_non_emb * (100 * n_non_emb)), 8),
            "predictivity": {
                "train_tokens": int(100 * n_non_emb),
                "train_iters": p_iters,
                "lr_warmup_iters": max(100, round(p_iters * 0.04 / 100) * 100),
                "lr_wsd_decay_iters": round(p_iters * 0.20 / 100) * 100,
            },
        }
        print("%-6s %-7d %-9d %-9d %-15.0f %-15.0f %+.3f%%"
              % (size, c["hidden_size"], c["ffn_hidden_size"], ffn,
                 c["n_non_emb_params"], n_non_emb,
                 (n_non_emb - c["n_non_emb_params"]) / c["n_non_emb_params"] * 100))
    OUT.write_text(json.dumps({
        "global": {**g, "activation": "swiglu"},
        "configs": configs,
    }, indent=2) + "\n")
    print(f"\nConfig saved to {OUT}")


if __name__ == "__main__":
    main()

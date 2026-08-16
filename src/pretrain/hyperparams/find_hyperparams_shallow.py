import json
import math
import sys
from pathlib import Path

from calculate_params_lr_bs import parameter_count, set_lr_and_bs


def suggest_mbs(num_non_emb_parameters, gbs, verbose=True):
    """
    Suggest a micro-batch size (MBS) for a model given its non-embedding parameter count.

    Uses a power-law rule to scale MBS down as the model grows larger (bigger models have larger per-sample activation memory, so fewer samples fit per micro-batch). Then snaps the result to the nearest divisor of the global batch size (GBS) so that GBS % MBS == 0.
    """
    baseline_mbs = 3
    baseline_params = 1e9
    # The exponent 0.9 gives a slightly sub-linear decay
    suggested_mbs = baseline_mbs * (baseline_params / num_non_emb_parameters) ** 0.9
    # Round to nearest divisor of GBS so that GBS % MBS == 0
    divisors = [d for d in range(1, gbs + 1) if gbs % d == 0]
    rounded_mbs = min(divisors, key=lambda d: abs(d - suggested_mbs))
    if verbose:
        print(
            f"MBS {num_non_emb_parameters/1e9:.3f}B, suggested: {suggested_mbs}, rounded: {rounded_mbs} (nearest divisor of GBS {gbs})\n"
        )
    return rounded_mbs


def calculate_hyperparams(
    model_size,
    vocab_size,
    seq_len,
    desired_tokens,
    gbs,
    n_layers,
    d_model,
    num_heads,
    num_kv_heads,
    ffw_size,
    verbose=True,
):
    """Calculate hyperparameters for the model size."""
    if verbose:
        print("=" * 60)
        print(f"{model_size} config")
        print("=" * 60)
    num_parameters, num_non_emb_parameters = parameter_count(
        vocab_size=vocab_size,
        n_layers=n_layers,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        ffw_size=ffw_size,
        n_experts=1,
        swiglu_or_geglu=False,
        tied_weights=True,
        verbose=verbose,
    )
    mbs = suggest_mbs(num_non_emb_parameters, gbs=gbs, verbose=verbose)
    lr, bs, its, num_mbs = set_lr_and_bs(
        num_parameters=num_non_emb_parameters,
        desired_tokens=desired_tokens,
        n_layers=n_layers,
        d_model=d_model,
        seq_len=seq_len,
        micro_batch_size=mbs,
        verbose=verbose,
    )
    return num_parameters, num_non_emb_parameters, mbs, lr, bs, its, num_mbs


def print_configs_table(
    title, configs, vocab_size, gbs, seq_len, desired_tokens, head_dim
):
    """
    Print a formatted table of model configs with computed hyperparameters.
    """
    headers = [
        "label",
        "n_layers",
        "d_model",
        "wd_ratio",
        "head_dim",
        "n_heads",
        "gqa_ratio",
        "n_kv_heads",
        "ffw_mult",
        "n_non_emb",
        "mbs",
        "lr",
        "iters",
    ]
    hdr_fmt = "  {:<8} {:<9} {:<8} {:<9} {:<9} {:<8} {:<10} {:<11} {:<9} {:<11} {:<5} {:<12} {:<7}"
    row_fmt = "  {:<8} {:<9} {:<8} {:<9.0f} {:<9} {:<8} {:<10} {:<11} {:<9} {:<11.3f} {:<5} {:<12.6f} {:<7}"

    if title:
        print(f"  {title}")
    print(hdr_fmt.format(*headers))
    print("  " + "-" * 128)
    for cfg in configs:
        _, n_non_emb, mbs, lr, _, its, _ = calculate_hyperparams(
            model_size=cfg["label"],
            vocab_size=vocab_size,
            seq_len=seq_len,
            desired_tokens=desired_tokens,
            gbs=gbs,
            n_layers=cfg["n_layers"],
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            ffw_size=cfg["d_model"] * cfg["ffw_multiplier"],
            verbose=False,
        )
        print(
            row_fmt.format(
                cfg["label"],
                cfg["n_layers"],
                cfg["d_model"],
                cfg["d_model"] / cfg["n_layers"],
                head_dim,
                cfg["num_heads"],
                cfg["num_heads"] // cfg["num_kv_heads"],
                cfg["num_kv_heads"],
                cfg["ffw_multiplier"],
                n_non_emb / 1e9,
                mbs,
                lr,
                its,
            )
        )
    print()


def find_hyperparams_for_model_size(
    target_non_emb_params,
    vocab_size,
    seq_len,
    desired_tokens,
    gbs,
    head_dim,
    ffw_multipliers,
    gqa_ratios,
    width_depth_ratios,
):
    """
    For each target non-embedding parameter count, sweep all valid architectures to find the one that comes closest, then compute its hyperparameters.

    Constraints:
    - d_model is a multiple of head_dim
    - n_layers is a multiple of 2
    - width_depth_ratio = d_model / n_layers = [64, 128]
    - num_heads = d_model // head_dim
    - gqa_ratio = num_heads // num_kv_heads = [2, 3, 4]
    - ffw_multiplier = [3, 4, 6]
    """
    print("=" * 60)
    print("Architecture search for target model sizes")
    print("=" * 60)

    print("\n  Constraints:")
    print(f"   - d_model is a multiple of 512 (head_dim: {head_dim})")
    print(f"   - n_layers is a multiple of 2")
    print(f"   - width_depth_ratio = d_model / n_layers = {width_depth_ratios}")
    print(f"   - num_heads = d_model // head_dim")
    print(f"   - gqa_ratio = num_heads // num_kv_heads = {gqa_ratios}")
    print(f"   - ffw_multiplier = {ffw_multipliers}")
    print()

    d_model_options = range(512, 4097, 512)
    # d_model_options = [2**n for n in range(int(math.log2(head_dim)), 13)]  # up to 4096
    n_layers_options = range(2, 33, 2)

    # First pass: collect every structurally valid (n_layers, d_model, gqa, ffw) combination
    possible_configs = []
    for n_layers in n_layers_options:
        for d_model in d_model_options:
            if d_model / n_layers not in width_depth_ratios:
                continue
            num_heads = d_model // head_dim
            for gqa_ratio in gqa_ratios:
                if num_heads % gqa_ratio != 0:
                    continue
                num_kv_heads = num_heads // gqa_ratio
                for ffw_mult in ffw_multipliers:
                    ffw_size = d_model * ffw_mult
                    _, n_non_emb = parameter_count(
                        vocab_size=vocab_size,
                        n_layers=n_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        ffw_size=ffw_size,
                        verbose=False,
                    )
                    possible_configs.append(
                        {
                            "n_layers": n_layers,
                            "d_model": d_model,
                            "num_heads": num_heads,
                            "num_kv_heads": num_kv_heads,
                            "ffw_multiplier": ffw_mult,
                            "n_non_emb": n_non_emb,
                        }
                    )

    print(f"  {len(possible_configs)} valid configurations found\n")

    for cfg in possible_configs:
        cfg["width_depth_ratio"] = cfg["d_model"] / cfg["n_layers"]

    def best_per_target(pool):
        """For each target, pick the config in pool closest in parameter count."""
        return [
            {**min(pool, key=lambda c: abs(c["n_non_emb"] - t)), "target": t}
            for t in target_non_emb_params
        ]

    def best_ffw_for_ratio(ratio_pool):
        """Pick the ffw_multiplier that minimises total relative error across all targets."""
        unique_ffws = set(c["ffw_multiplier"] for c in ratio_pool)
        return min(
            unique_ffws,
            key=lambda ffw: sum(
                min(
                    abs(c["n_non_emb"] - t) / t
                    for c in ratio_pool
                    if c["ffw_multiplier"] == ffw
                )
                for t in target_non_emb_params
            ),
        )

    def labeled(configs):
        """Add a human-readable label derived from the target parameter count."""
        return [{**c, "label": f"{c['target'] / 1e6:.0f}M"} for c in configs]

    for wd_ratio in [128, 64]:
        ratio_pool = [c for c in possible_configs if c["width_depth_ratio"] == wd_ratio]

        print_configs_table(
            f"width_depth_ratio={wd_ratio:.0f}, ffw_multiplier=free",
            labeled(best_per_target(ratio_pool)),
            vocab_size=vocab_size,
            gbs=gbs,
            seq_len=seq_len,
            desired_tokens=desired_tokens,
            head_dim=head_dim,
        )

        best_ffw = best_ffw_for_ratio(ratio_pool)
        ffw_pool = [c for c in ratio_pool if c["ffw_multiplier"] == best_ffw]
        print_configs_table(
            f"width_depth_ratio={wd_ratio:.0f}, ffw_multiplier={best_ffw} (consistent across all sizes)",
            labeled(best_per_target(ffw_pool)),
            vocab_size=vocab_size,
            gbs=gbs,
            seq_len=seq_len,
            desired_tokens=desired_tokens,
            head_dim=head_dim,
        )

    return possible_configs


def calculate_hyperparams_for_model_configs(
    model_configs,
    vocab_size,
    seq_len,
    desired_tokens,
    gbs,
    head_dim,
):
    """
    Compute and display hyperparameters for a named dict of model configs.

    model_configs: dict mapping size label (e.g. "1B") to a config dict with keys:
        n_layers, d_model, gqa_ratio, ffw_multiplier
    """
    print("=" * 60)
    print("Model configs")
    print("=" * 60)

    configs = []
    json_configs = {}

    for size_label, config in model_configs.items():
        nl, dm = config["n_layers"], config["d_model"]
        nh = dm // head_dim
        nkv = nh // config["gqa_ratio"]
        ffw_size = dm * config["ffw_multiplier"]

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

        configs.append(
            {
                "label": size_label,
                "n_layers": nl,
                "d_model": dm,
                "num_heads": nh,
                "num_kv_heads": nkv,
                "ffw_multiplier": config["ffw_multiplier"],
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
            # Training
            "micro_batch_size": mbs,
            "global_batch_size": gbs,
            "seq_len": seq_len,
            "train_iters": train_iters,
            # Learning rate
            "lr": round(lr, 8),
            "predictivity": {
                "train_tokens": int(100 * n_non_emb),
                "train_iters": p_iters,
                "lr_warmup_iters": max(100, round(p_iters * 0.04 / 100) * 100),
                "lr_wsd_decay_iters": round(p_iters * 0.20 / 100) * 100,
            },
        }

    print_configs_table(
        "",
        configs,
        vocab_size=vocab_size,
        gbs=gbs,
        seq_len=seq_len,
        desired_tokens=desired_tokens,
        head_dim=head_dim,
    )

    json_path = Path(__file__).parent / "hyperparams_shallow.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "global": {
                    "vocab_size": vocab_size,
                    "seq_len": seq_len,
                    "desired_tokens": int(desired_tokens),
                    "global_batch_size": gbs,
                    "head_dim": head_dim,
                    "predictivity_schedule": (
                        "D = 100 x n_non_emb_params tokens (5x Chinchilla) at "
                        "global_batch_size x seq_len tokens/iter; "
                        "lr_warmup_iters ~4% and lr_wsd_decay_iters ~20% of "
                        "train_iters, all rounded to 100"
                    ),
                },
                "configs": json_configs,
            },
            f,
            indent=2,
        )
    print(f"Config saved to {json_path}")


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


if __name__ == "__main__":
    output_path = Path(__file__).parent / "hyperparams_shallow_explanation.txt"
    log_file = open(output_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    # Values for all model sizes

    vocab_size = 131072
    seq_len = 4096
    desired_tokens = 100e9  # 100B tokens
    gbs = 504  # global batch size
    head_dim = 64  # num_heads = d_model // head_dim
    ffw_multipliers = [3, 4, 6]  # feed-forward multiplier
    gqa_ratios = [2, 3, 4]  # grouped query attention ratio
    width_depth_ratios = [64, 128]  # d_model // n_layers

    # Validate calculations for 1B model config

    model_config_1b = {
        "n_layers": 16,
        "d_model": 2048,
        "gqa_ratio": 4,
        "ffw_multiplier": 6,
    }

    calculate_hyperparams(
        model_size="1B",
        vocab_size=vocab_size,
        seq_len=seq_len,
        desired_tokens=desired_tokens,
        gbs=gbs,
        n_layers=model_config_1b["n_layers"],
        d_model=model_config_1b["d_model"],
        num_heads=model_config_1b["d_model"] // head_dim,
        num_kv_heads=model_config_1b["d_model"]
        // head_dim
        // model_config_1b["gqa_ratio"],
        ffw_size=model_config_1b["d_model"] * model_config_1b["ffw_multiplier"],
    )

    # Find hyperparameters for other model sizes

    print("\n")
    print(
        "label    n_layers  d_model  wd_ratio  head_dim  n_heads  gqa_ratio  n_kv_heads  ffw_mult  n_non_emb   mbs   lr           iters  "
    )
    print(
        "--------------------------------------------------------------------------------------------------------------------------------"
    )
    print(
        "1B       16        2048     128       64        32       4          8           6         0.973       3     0.000791     56000 "
    )
    print("\n\n")

    find_hyperparams_for_model_size(
        target_non_emb_params=[100e6, 300e6, 500e6, 1e9],
        vocab_size=vocab_size,
        seq_len=seq_len,
        desired_tokens=desired_tokens,
        gbs=gbs,
        head_dim=head_dim,
        ffw_multipliers=ffw_multipliers,
        gqa_ratios=gqa_ratios,
        width_depth_ratios=width_depth_ratios,
    )

    print(
        "\n\nDECISION: We will use the configuration from the first table (width_depth_ratio=128, ffw_multiplier=free), keeping for the 1B model the original configuration.\n\n"
    )

    # Final model configs

    model_configs = {
        "100M": {
            "n_layers": 8,
            "d_model": 1024,
            "gqa_ratio": 2,
            "ffw_multiplier": 4,
        },
        "300M": {
            "n_layers": 12,
            "d_model": 1536,
            "gqa_ratio": 3,
            "ffw_multiplier": 4,
        },
        "500M": {
            "n_layers": 16,
            "d_model": 2048,
            "gqa_ratio": 4,
            "ffw_multiplier": 3,
        },
        "1B": {
            "n_layers": 16,
            "d_model": 2048,
            "gqa_ratio": 4,
            "ffw_multiplier": 6,
        },
    }

    calculate_hyperparams_for_model_configs(
        model_configs=model_configs,
        vocab_size=vocab_size,
        seq_len=seq_len,
        desired_tokens=desired_tokens,
        gbs=gbs,
        head_dim=head_dim,
    )

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Output written to {output_path}")

"""Fetch architecture hyperparameters from HuggingFace model configs."""

import csv
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

FIELDS = [
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
]

DERIVED_FIELDS = ["width_depth_ratio", "ffw_multiplier", "gqa_ratio"]


def derived_fields(row: dict) -> dict:
    def safe(num, den):
        try:
            return round(num / den, 3)
        except (TypeError, ZeroDivisionError):
            return "N/A"

    return {
        "width_depth_ratio": safe(row.get("hidden_size"), row.get("num_hidden_layers")),
        "ffw_multiplier": safe(row.get("intermediate_size"), row.get("hidden_size")),
        "gqa_ratio": safe(
            row.get("num_attention_heads"), row.get("num_key_value_heads")
        ),
    }


def fetch_hyperparams(
    models_file: str | Path = None,
    output_csv: str | Path = None,
) -> None:
    _dir = Path(__file__).parent
    models_file = Path(models_file or _dir / "hf_models.txt")
    output_csv = Path(output_csv or _dir / "hf_model_hyperparams.csv")
    models = Path(models_file).read_text().splitlines()
    models = [m.strip() for m in models if m.strip() and not m.startswith("#")]

    rows = []
    for model_id in models:
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            config = json.loads(Path(config_path).read_text())
            text_cfg = config.get("text_config", {})  # for multimodal models

            def get_field(key: str):
                return config.get(key) or text_cfg.get(key, "N/A")

            row = {"model": model_id.split("/")[-1]}
            row.update({f: get_field(f) for f in FIELDS})
            row.update(derived_fields(row))
            rows.append(row)
        except Exception as e:
            print(f"[WARN] {model_id}: {e}", file=sys.stderr)
            rows.append(
                {"model": model_id, **{f: "ERROR" for f in FIELDS + DERIVED_FIELDS}}
            )

    columns = ["model"] + FIELDS + DERIVED_FIELDS
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    col_widths = {
        c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = "  ".join(c.ljust(col_widths[c]) for c in columns)
    separator = "  ".join("-" * col_widths[c] for c in columns)
    print(header)
    print(separator)
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in columns))

    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    fetch_hyperparams()

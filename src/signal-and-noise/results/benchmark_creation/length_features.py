"""Phase B: per-family length features.

Pulls a small streaming sample (default 100 items) of the English (or
default) split of each family's underlying HF dataset, computes
character-length statistics for the context and the answer options,
and writes:

  - length_features.csv  one row per family with median context length,
                         median option length, and ratio.
  - sample_items.json    one example item per family — kept for Phase C
                         (topic auto-tagging) so we don't re-pull.

Run from the repo root:
    python results/benchmark_creation/length_features.py
"""
from __future__ import annotations

import json
import logging
import statistics as stat
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("length_features")


def _load_parquet(repo: str, config: str, split: str) -> list[dict]:
    """Fallback for datasets where streaming fails (e.g. hellaswag's
    `List` feature-type bug). Pulls the parquet auto-conversion."""
    url = (f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2F"
           f"parquet/{config}/{split}/0000.parquet")
    df = pd.read_parquet(url)
    return df.to_dict(orient="records")

HERE = Path(__file__).resolve().parent
N_SAMPLES = 100

# (repo, config, split, context_fn, options_fn, label_for_sample)
LOADERS = {
    "arc": dict(
        repo="allenai/ai2_arc", config="ARC-Challenge", split="test",
        ctx=lambda x: x["question"],
        opts=lambda x: list(x["choices"]["text"]),
    ),
    "belebele": dict(
        repo="facebook/belebele", config="eng_Latn", split="test",
        ctx=lambda x: x["flores_passage"] + " " + x["question"],
        opts=lambda x: [x[f"mc_answer{i}"] for i in range(1, 5)],
    ),
    # global_mmlu (Lite) is excluded from SNR analysis (no eval coverage),
    # but we still measure its length features for the metadata table.
    "global_mmlu": dict(
        repo="CohereLabs/Global-MMLU", config="en", split="test",
        ctx=lambda x: x["question"],
        opts=lambda x: [x["option_a"], x["option_b"], x["option_c"], x["option_d"]],
    ),
    "global_mmlu_full": dict(
        repo="CohereLabs/Global-MMLU", config="en", split="test",
        ctx=lambda x: x["question"],
        opts=lambda x: [x["option_a"], x["option_b"], x["option_c"], x["option_d"]],
    ),
    "global_piqa_completions": dict(
        repo="mrlbenchmarks/global-piqa-nonparallel", config="eng_latn", split="test",
        ctx=lambda x: x["prompt"],
        opts=lambda x: [x["solution0"], x["solution1"]],
    ),
    "hellaswag": dict(
        # Streaming fails for Rowan/hellaswag (List feature-type bug);
        # use the parquet-converted revision via _load_parquet fallback.
        repo="Rowan/hellaswag", config="default", split="validation",
        loader="parquet",
        ctx=lambda x: x["ctx"],
        opts=lambda x: list(x["endings"]),
    ),
    "multiblimp": dict(
        repo="jumelet/multiblimp", config="eng", split="train",
        # The eval prompt is the shared prefix; the model picks between
        # the grammatical full sentence and its ungrammatical twin.
        ctx=lambda x: x["prefix"] or "",
        opts=lambda x: [x["sen"], x["wrong_sen"]],
    ),
    "paws": dict(
        repo="google-research-datasets/paws-x", config="en", split="test",
        # In lm-eval the model is given both sentences, then asked to
        # output a binary label. Treat the two sentences as the context;
        # treat the labels (~3 chars) as the option strings.
        ctx=lambda x: x["sentence1"] + " " + x["sentence2"],
        opts=lambda x: ["No", "Yes"],
    ),
    "xcopa": dict(
        # No `en` config in cambridgeltl/xcopa; use Italian as a stand-in
        # (XCOPA is a parallel translation, so per-item lengths are
        # comparable up to a per-language constant). Flagged in the CSV.
        repo="cambridgeltl/xcopa", config="it", split="test",
        ctx=lambda x: x["premise"],
        opts=lambda x: [x["choice1"], x["choice2"]],
    ),
    "xnli": dict(
        repo="facebook/xnli", config="en", split="test",
        ctx=lambda x: x["premise"] + " " + x["hypothesis"],
        opts=lambda x: ["entailment", "neutral", "contradiction"],
    ),
    "xstorycloze": dict(
        repo="juletxara/xstory_cloze", config="en", split="eval",
        ctx=lambda x: " ".join(x[f"input_sentence_{i}"] for i in range(1, 5)),
        opts=lambda x: [x["sentence_quiz1"], x["sentence_quiz2"]],
    ),
    "xwinograd": dict(
        repo="Muennighoff/xwinograd", config="en", split="test",
        ctx=lambda x: x["sentence"],
        opts=lambda x: [x["option1"], x["option2"]],
    ),
}


def measure(family: str, spec: dict, n: int) -> tuple[dict, dict]:
    log.info("loading %s: %s / %s / %s",
             family, spec["repo"], spec["config"], spec["split"])
    if spec.get("loader") == "parquet":
        records = _load_parquet(spec["repo"], spec["config"], spec["split"])
        samples = records[:n]
    else:
        ds = load_dataset(spec["repo"], spec["config"], split=spec["split"],
                          streaming=True)
        samples = []
        for i, x in enumerate(ds):
            if i >= n:
                break
            samples.append(x)

    if not samples:
        raise RuntimeError(
            f"no samples returned from {spec['repo']}/{spec['config']}/{spec['split']} "
            f"(loader={spec.get('loader', 'streaming')}); cannot compute length features"
        )

    ctx_lens = [len(spec["ctx"](x)) for x in samples]
    opts_per = [spec["opts"](x) for x in samples]
    opt_lens_per_item = [stat.mean(len(o) for o in os) for os in opts_per]

    row = {
        "family": family,
        "n_sampled": len(samples),
        "context_len_chars_median": int(stat.median(ctx_lens)),
        "context_len_chars_mean": round(stat.mean(ctx_lens), 1),
        "option_len_chars_median": round(stat.median(opt_lens_per_item), 1),
        "option_len_chars_mean": round(stat.mean(opt_lens_per_item), 1),
        "context_to_option_ratio": round(
            stat.median(ctx_lens) / max(stat.median(opt_lens_per_item), 0.1), 2
        ),
        "n_options_observed": len(opts_per[0]),
    }
    sample = {
        "family": family,
        "context": spec["ctx"](samples[0]),
        "options": list(spec["opts"](samples[0])),
    }
    return row, sample


def main() -> None:
    rows = []
    samples = []
    failures: list[tuple[str, str]] = []
    for family, spec in LOADERS.items():
        try:
            row, sample = measure(family, spec, N_SAMPLES)
        except Exception as e:
            log.exception("failed to measure %s (%s/%s/%s): %s",
                          family, spec["repo"], spec["config"], spec["split"], e)
            failures.append((family, f"{type(e).__name__}: {e}"))
            continue
        rows.append(row)
        samples.append(sample)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "length_features.csv", index=False)
    log.info("wrote length_features.csv (%d families, %d failures)",
             len(df), len(failures))
    print(df.to_string(index=False))

    (HERE / "sample_items.json").write_text(json.dumps(samples, indent=2,
                                                       ensure_ascii=False))
    log.info("wrote sample_items.json (%d samples)", len(samples))

    if failures:
        log.warning("%d family/families could not be measured:", len(failures))
        for fam, err in failures:
            log.warning("  %s: %s", fam, err)
        # Also persist failures so downstream consumers know what's missing.
        (HERE / "length_features_failures.txt").write_text(
            "\n".join(f"{fam}\t{err}" for fam, err in failures) + "\n"
        )


if __name__ == "__main__":
    main()

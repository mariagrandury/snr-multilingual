"""Download + tokenize the FineWeb-Edu / FineWeb2-HQ mixture for one ratio.

The cluster consumed pre-tokenized mixtures frozen at CSCS
(mix_100B_<edu>_<fw2>, built upstream by the SwissAI data team); this script
recreates them on Azure: stream the raw corpora from the HF Hub, tokenize with
Megatron's tools/preprocess_data.py (alehc/swissai-tokenizer, --append-eod)
into .bin/.idx, and write a data_path.txt manifest of "<weight> <prefix>"
pairs that train.sh feeds to Megatron's --data-path. Weights are computed from
the actual .bin sizes, so the manifest reflects what is really on disk
(replaces create_data_mixture.py's symlink trick, which blob storage cannot
represent, and create_data_config.py's directory scan).

The exact language composition of the CSCS FineWeb2 corpus is not recoverable
from this repo; the documented approximation is the project's "main" language
group (configs/languages.json) minus English, weighted by each language's
corpus size on the Hub.

Runs inside the apertus-nemo container on gpu-nc80-lp (see jobs/prep.yml):
    python prepare_data.py --output <dir> --total-tokens-b 5.16 --edu-ratio 0.3
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

TOKENIZER = "alehc/swissai-tokenizer"
EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FW2_HQ_DATASET = "epfml/FineWeb2-HQ"     # top-10% quality filter, 20 languages
FW2_FULL_DATASET = "HuggingFaceFW/fineweb-2"  # all languages, no quality filter

# configs/languages.json iso2 -> FineWeb-2 config name (iso3_Script)
FW2_CONFIGS = {
    "ar": "arb_Arab", "de": "deu_Latn", "es": "spa_Latn", "eu": "eus_Latn",
    "fr": "fra_Latn", "hi": "hin_Deva", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "pt": "por_Latn", "ru": "rus_Cyrl", "sw": "swh_Latn", "te": "tel_Telu",
    "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl", "vi": "vie_Latn",
    "zh": "cmn_Hani",
}
BYTES_PER_TOKEN = 4.5  # rough utf-8 text bytes per swissai-tokenizer token

# configs/languages.json groups.main minus "en" (inlined: the AML job uploads
# only this directory, not the repo's configs/)
MAIN_LANGUAGES = ["es", "ru", "hi", "zh", "ja", "ar", "vi", "tr", "th", "sw", "eu"]


def hub_dataset_size(repo, config):
    """Total parquet bytes of one config, via the datasets-server size API."""
    import requests

    url = f"https://datasets-server.huggingface.co/size?dataset={repo.replace('/', '%2F')}&config={config}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()["size"]["config"]["num_bytes_original_files"]


def fw2_source(lang):
    """Prefer FineWeb2-HQ, fall back to full FineWeb-2 for languages it lacks."""
    from datasets import get_dataset_config_names

    config = FW2_CONFIGS[lang]
    if config in get_dataset_config_names(FW2_HQ_DATASET):
        return FW2_HQ_DATASET, config
    print(f"[{lang}] {config} not in FineWeb2-HQ, falling back to {FW2_FULL_DATASET}")
    return FW2_FULL_DATASET, config


def stream_jsonl(repo, config, byte_budget, jsonl_path):
    """Stream one corpus into a JSONL file until ~byte_budget of text."""
    from datasets import load_dataset

    written = 0
    with open(jsonl_path, "w") as f:
        for doc in load_dataset(repo, name=config, split="train", streaming=True):
            written += f.write(json.dumps({"text": doc["text"]}) + "\n")
            if written >= byte_budget:
                break
    print(f"{jsonl_path.name}: {written / 1e9:.2f} GB text")


def tokenize(jsonl_path, prefix, megatron_dir, workers):
    subprocess.run(
        ["python", str(Path(megatron_dir) / "tools" / "preprocess_data.py"),
         "--input", str(jsonl_path), "--json-keys", "text",
         "--tokenizer-type", "HuggingFaceTokenizer", "--tokenizer-model", TOKENIZER,
         "--append-eod", "--workers", str(workers), "--output-prefix", str(prefix)],
        check=True)
    jsonl_path.unlink()  # free NVMe before the next language


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-tokens-b", type=float, required=True,
                   help="mixture size in billions of tokens (full run: 103.2)")
    p.add_argument("--edu-ratio", type=float, default=0.3,
                   help="FineWeb-Edu fraction (0.3/0.6/0.9 for the 3 mixtures)")
    p.add_argument("--languages", nargs="*", default=MAIN_LANGUAGES)
    p.add_argument("--edu-config", default="sample-350BT",
                   help="fineweb-edu subset (sample-100BT is enough for pilots)")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    import os
    megatron_dir = os.environ["MEGATRON_LM_DIR"]
    args.output.mkdir(parents=True, exist_ok=True)
    total_bytes = args.total_tokens_b * 1e9 * BYTES_PER_TOKEN
    tmp = Path(tempfile.mkdtemp(prefix="jsonl_"))  # node-local NVMe

    # FineWeb-Edu (English) share
    plan = [(EDU_DATASET, args.edu_config, "fineweb_edu", total_bytes * args.edu_ratio)]

    # FineWeb2 share, split across languages proportionally to Hub corpus size
    sources = {lang: fw2_source(lang) for lang in args.languages}
    sizes = {lang: hub_dataset_size(*sources[lang]) for lang in args.languages}
    fw2_bytes = total_bytes * (1 - args.edu_ratio)
    for lang in args.languages:
        share = sizes[lang] / sum(sizes.values())
        plan.append((*sources[lang], f"fw2_{lang}", fw2_bytes * share))

    for repo, config, name, byte_budget in plan:
        print(f"=== {name}: {byte_budget / 1e9:.2f} GB from {repo}/{config} ===")
        jsonl = tmp / f"{name}.jsonl"
        stream_jsonl(repo, config, byte_budget, jsonl)
        tokenize(jsonl, args.output / name, megatron_dir, args.workers)

    # Manifest from actual token counts (uint32 tokens -> 4 bytes each);
    # prefixes are relative (train.sh prepends the mount dir) and keep the
    # _text_document suffix preprocess_data.py bakes into the file names.
    bins = sorted(args.output.glob("*.bin"))
    tokens = {b: b.stat().st_size / 4 for b in bins}
    with open(args.output / "data_path.txt", "w") as f:
        f.writelines(
            f"{tokens[b] / sum(tokens.values()):.6f} {b.stem}\n" for b in bins)
    print(f"Done: {sum(tokens.values()) / 1e9:.2f}B tokens in {args.output}")


if __name__ == "__main__":
    main()

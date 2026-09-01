#!/usr/bin/env python3
"""Per-language bits-per-byte and perplexity for a converted checkpoint.

BPB on the fixed held-out validation set is THE outcome metric of the
predictivity study (plan/small-to-large-predictivity-training-plan.md, line 1
of the design): the language count and model size are the axes, and per-
language BPB is what they are measured against.

It was not being computed. `megatron_args.sh` trains with `--split 100,0,0`,
so Megatron's validation and test splits are empty and no validation loss is
ever produced. Turning that on would give ONE aggregate loss over the 50/50
blend, which is not per-language and not BPB — and it would change the
training config mid-sweep, making old and new cells incomparable. Scoring
offline from the saved checkpoints avoids both problems and works uniformly
on every cell already trained.

Everything needed already exists: the validation build wrote one .bin per
language plus `validation.manifest.json`, which records each language's token
count AND its UTF-8 byte count — recorded specifically as the BPB denominator
— and training skipped exactly those rows, so the sets are disjoint.

    BPB = (sum of -log2 p(token)) / (UTF-8 bytes of that language)
    PPL = exp((sum of -ln p(token)) / (number of scored tokens))

One forward pass yields both. Cost is ~2% of a 90M training run and ~0.1% of
a 1.7B one — scoring a fixed 500M tokens scales with N while training scales
with N x D(N), so it is roughly constant per checkpoint across the ladder.

    python3.11 score_bpb.py --model <hf_dir> --out results.json
    # Quick correctness check. The small --batch-size/--seq-len are required
    # off-GPU: scoring materialises batch x seq x 131072 float32 logits, which
    # is 17 GB at the defaults and OOMs a login node instantly.
    python3.11 score_bpb.py --model <hf_dir> --languages dclm,fineweb_rus_Cyrl \
        --limit-docs 5 --max-tokens 4096 --seq-len 512 --batch-size 1 \
        --device cpu --out /tmp/bpb.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Mirrors data/create_data_mixture.py's MegatronDatasetWriter. That file
# hand-rolls the writer "avoids importing megatron, which has conflicts"; the
# reader follows the same rule, so the two stay a matched pair with no
# dependency on the Megatron checkout being present or importable.
IDX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODE = 4          # Megatron's DType enum value for int32
OUTPUT_DTYPE = np.int32


def read_indexed_dataset(prefix: str, limit_docs: int | None = None,
                         max_tokens: int | None = None) -> np.ndarray:
    """The whole dataset as one flat token array, documents in file order.

    Documents are CONCATENATED WITHOUT an EOD separator, which is what makes
    the numerator and denominator match: the .bin holds exactly the tokens the
    manifest counted, and the manifest's byte count is the UTF-8 length of the
    same text. Inserting EOD would add negative log-likelihood that no byte in
    the denominator pays for, inflating BPB.

    `max_tokens` truncates to a whole number of leading documents. It is a
    DETERMINISTIC prefix, so every model is scored on byte-identical text and
    the comparison between them stays exact — the sample is smaller, not
    different.
    """
    idx_path, bin_path = Path(f"{prefix}.idx"), Path(f"{prefix}.bin")
    with open(idx_path, "rb") as f:
        if f.read(len(IDX_HEADER)) != IDX_HEADER:
            raise ValueError(f"{idx_path} is not a Megatron .idx file")
        (version,) = struct.unpack("<Q", f.read(8))
        (code,) = struct.unpack("<B", f.read(1))
        if version != 1 or code != DTYPE_CODE:
            raise ValueError(f"{idx_path}: version {version}, dtype code {code} "
                             f"(expected 1 / {DTYPE_CODE}=int32)")
        (n_seq,) = struct.unpack("<Q", f.read(8))
        struct.unpack("<Q", f.read(8))                      # document count
        lengths = np.frombuffer(f.read(n_seq * 4), dtype=np.int32)

    if limit_docs:
        lengths = lengths[:limit_docs]
    if max_tokens:
        # Cut at a document boundary, never mid-document: a truncated document
        # would make the token count and the byte fraction disagree.
        keep = int(np.searchsorted(np.cumsum(lengths.astype(np.int64)),
                                   max_tokens, side="left")) + 1
        lengths = lengths[:keep]
    n = int(lengths.sum())
    # Sequences are stored contiguously in file order, so the first n tokens
    # of the .bin are exactly the leading documents.
    return np.fromfile(bin_path, dtype=OUTPUT_DTYPE, count=n)


def score_stream(model, tokens: np.ndarray, seq_len: int, batch_size: int,
                 device: str) -> tuple[float, int]:
    """(total -ln p in nats, number of tokens scored) over one token stream.

    Blocks overlap by one token so that every token except the very first has
    a predecessor to condition on: block b covers stream[b*L : b*L+L+1], the
    inputs are [:-1] and the targets [1:]. Without the overlap the first token
    of each block would go unscored, silently dropping a slice of the corpus
    from the numerator while the denominator kept all of its bytes.
    """
    import torch

    starts = list(range(0, max(len(tokens) - 1, 0), seq_len))
    total_nll, total_n = 0.0, 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        # Blocks are ragged only at the tail; group them by length so one
        # short final block does not force padding logic for the whole run.
        for length in sorted({min(seq_len + 1, len(tokens) - s) for s in chunk}):
            rows = [tokens[s:s + length] for s in chunk
                    if min(seq_len + 1, len(tokens) - s) == length]
            batch = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
            with torch.no_grad():
                logits = model(batch[:, :-1]).logits.float()
                nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1),
                    reduction="sum")
            total_nll += float(nll)
            total_n += batch[:, 1:].numel()
    return total_nll, total_n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF checkpoint directory")
    p.add_argument("--manifest", default="/capstor/store/cscs/swissai/infra01/"
                   "multilingual_data_mixtures/predictivity-data/"
                   "validation.manifest.json")
    p.add_argument("--data-dir", default="/iopsstor/scratch/cscs/mariagrandury/data",
                   help="where the validation .bin/.idx live (the manifest "
                        "records absolute paths from build time, which move)")
    p.add_argument("--out", required=True)
    p.add_argument("--languages", help="comma-separated subset (default: all)")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-docs", type=int,
                   help="score only the first N documents per language (testing)")
    p.add_argument("--max-tokens", type=int, default=1_000_000,
                   help="tokens per language to score, as a deterministic "
                        "leading-document prefix (0 = the full ~5M). The "
                        "default trades a BPB standard error of ~0.003 bits "
                        "for a 5x cheaper pass; the byte denominator is measured "
                        "by decoding, so it stays exact either way.")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads(Path(args.manifest).read_text())
    wanted = args.languages.split(",") if args.languages else list(manifest)
    missing = [l for l in wanted if l not in manifest]
    if missing:
        sys.exit(f"not in manifest: {missing}")

    print(f"loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()
    # The BPB denominator is measured, not inferred. Scaling the manifest's
    # byte count by the token fraction would be wrong for a prefix, because
    # bytes-per-token varies between documents — a bias identical across
    # models, so comparisons would survive, but the absolute BPB would not.
    # Decoding is exact: verified to reproduce the manifest byte count to the
    # byte (ratio 1.00000) on dclm, rus_Cyrl and cmn_Hani, i.e. the tokenizer
    # round-trips losslessly on Latin, Cyrillic and CJK alike.
    tok = AutoTokenizer.from_pretrained(args.model)

    out: dict[str, dict] = {}
    for lang in wanted:
        entry = manifest[lang]
        # The manifest's paths are from build time; re-root them so a staged
        # or mirrored copy works without rewriting the manifest.
        prefix = str(Path(args.data_dir) / Path(entry["bin"]).name[:-len(".bin")])
        tokens = read_indexed_dataset(prefix, args.limit_docs,
                                      args.max_tokens or None)
        if len(tokens) < 2:
            print(f"  {lang}: too few tokens, skipping", flush=True)
            continue
        nll, n = score_stream(model, tokens, args.seq_len, args.batch_size,
                              args.device)
        # Every token but the corpus's first must be scored exactly once. A
        # silent gap here would shrink the numerator while the denominator
        # kept all its bytes, i.e. flatter BPB for the same model.
        if n != len(tokens) - 1:
            sys.exit(f"{lang}: scored {n} targets, expected {len(tokens) - 1}")
        nbytes = len(tok.decode(tokens.tolist(),
                                skip_special_tokens=False).encode("utf-8"))
        if len(tokens) == entry["tokens"] and nbytes != entry["bytes"]:
            # Only meaningful on a full language, where the manifest is the
            # reference. A mismatch means the .bin and the manifest describe
            # different text — worth knowing before trusting any of this.
            print(f"    WARNING: {lang} decoded {nbytes} bytes, manifest says "
                  f"{entry['bytes']}", flush=True)
        out[lang] = {
            "tokens_scored": n,
            "bytes": nbytes,
            "nll_nats": nll,
            "bpb": nll / math.log(2) / nbytes,
            "ppl": math.exp(nll / n),
        }
        print(f"  {lang:24} bpb={out[lang]['bpb']:.4f}  ppl={out[lang]['ppl']:.2f}",
              flush=True)

    bpbs = [v["bpb"] for v in out.values()]
    result = {
        "model": args.model,
        "seq_len": args.seq_len,
        "max_tokens_per_language": args.max_tokens,
        "n_languages": len(out),
        # Macro average: every language counts once, regardless of how many
        # bytes it contributed. A byte-weighted mean would be dominated by the
        # few high-resource languages and would move with the language set.
        "macro_bpb": sum(bpbs) / len(bpbs) if bpbs else None,
        "languages": out,
    }
    # Write through a temp name: score_bpb.sbatch and launch_bpb.sh both treat
    # a NON-EMPTY bpb.json as "already scored", and these jobs are deliberately
    # drained through debug's hard 1:30 wall (debug_drain.sh), so a kill
    # mid-write would leave a short file that is skipped forever — the
    # checkpoint would drop out of the BPB analysis silently. Same rule as the
    # .bin staging in pretrain/data/stage_to_iopsstor.sh.
    tmp = Path(f"{args.out}.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(f"\nmacro BPB over {len(out)} languages: {result['macro_bpb']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

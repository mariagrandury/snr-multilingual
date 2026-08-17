#!/usr/bin/env python3
"""
Creates a blended Megatron-LM dataset from parquet sources.

Tokenizes raw text from FineWeb-2 (multilingual) and DCLM (English) parquet
files using a HuggingFace tokenizer, then writes the result in Megatron
IndexedDataset format (.bin/.idx).

Supports:
  - Selecting languages explicitly or by top-K (ranked by estimated tokens)
  - Temperature sampling for per-language token allocation within FineWeb-2
    (proportions proportional to p^(1/T); T=1 is proportional to tokens)
  - A fixed validation-set build mode (--build_validation) that is independent
    of temperature, target_tokens, and the language mixture
  - Excluding the validation rows from training via a validation manifest
    (--validation_manifest), so training and validation never overlap
  - Resumption after preemption via JSON checkpoint files

Validation strategy:
  The validation set for a language is the first --val_tokens_per_language
  tokens of that language's FIRST parquet file (in sorted filename order),
  capped at --val_max_fraction of that file's rows so that single-file
  languages keep training data. The build records, per language, the number of
  leading rows assigned to validation (val_doc_count). Training reads the
  manifest and skips exactly those leading rows of the first file, so the two
  sets are disjoint. This works for languages with only one parquet file.

Usage examples:
  # Build the fixed validation set ONCE (all FineWeb-2 languages + DCLM English).
  python create_data_mixture.py \
      --build_validation \
      --languages rus_Cyrl,deu_Latn,fra_Latn \
      --val_tokens_per_language 5_000_000 \
      --val_max_fraction 0.3 \
      --output_prefix outputs/validation

  # Training mixture: 50% English (DCLM) + 50% FineWeb-2, 110B tokens total,
  # proportional language allocation (T=1), validation rows excluded.
  python create_data_mixture.py \
      --target_tokens 110_000_000_000 \
      --fineweb_pct 50 --dclm_pct 50 \
      --languages rus_Cyrl,deu_Latn,fra_Latn \
      --temperature 1.0 \
      --validation_manifest outputs/validation.manifest.json \
      --output_prefix outputs/mix_3lang

  # English-only baseline: 100% DCLM, no FineWeb-2 languages.
  python create_data_mixture.py \
      --target_tokens 110_000_000_000 \
      --fineweb_pct 0 --dclm_pct 100 \
      --validation_manifest outputs/validation.manifest.json \
      --output_prefix outputs/mix_L1

Assumptions and behaviors:
  - One parquet row = one document = one sequence. No chunking or concatenation
    of rows; Megatron's data loader handles sequence packing at training time.
  - Token dtype is int32 (Apertus vocab size 131K fits comfortably).
  - Token counts per language are estimated by sampling a few files, computing
    a tokens-per-on-disk-byte ratio, and multiplying by total file size on
    disk. The ratio is measured per UTF-8 text byte and converted to on-disk
    bytes using the text/file size ratio from the same files' parquet footers,
    because parquet compression varies ~7x across languages.
  - Language ranking (--top_k_languages) and proportional allocation within
    FineWeb-2 are based on these estimated token counts.
  - Per-language allocation within FineWeb-2 uses temperature sampling:
    proportions are proportional to p^(1/T), where p is the estimated-token
    proportion. T=1 reproduces proportional-to-tokens; larger T flattens
    toward uniform (T = 1/alpha, so the common alpha=0.3 is T~3.3).
  - DCLM is treated as a single monolingual (English) source with a flat
    parquet directory.
  - FineWeb-2 language folders follow {lang}_{script} naming (e.g. deu_Latn).
  - When --validation_manifest is given, training skips the leading
    val_doc_count rows of each source's first file (the validation rows). With
    no manifest, training uses all rows and may overlap the validation set.
  - Sources are written sequentially (all of language A, then B, then DCLM).
    No cross-source shuffling; Megatron's data loader shuffles at training time.
  - Parquet files within each source are processed in sorted filename order
    for deterministic, resumable iteration.
  - Each source stops after the document that meets or first exceeds its token
    target (overshoot is at most one document, typically a few thousand tokens).
  - Empty or null text rows are filtered out before tokenization and are not
    written to the output.
  - If a source is exhausted before reaching its token target, a warning is
    printed and processing continues with the remaining sources.
  - Checkpoints are saved only after every completed parquet file (a progress
    line is still printed every N documents). On resume, the .bin file is
    truncated back to the last completed file to discard any partial writes,
    and the in-progress file is re-tokenized from its start (at most one
    ~1 GB parquet file re-done, never duplicated).
"""

import argparse
import json
import os
import struct
import time
from itertools import accumulate
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Apertus V1 tokenizer (131K vocab). Consider switching to the Apertus V2
# candidate to match Apertus 2 pretraining; if changed, rebuild the validation
# manifest and all data, since the token counts depend on the tokenizer.
TOKENIZER_NAME = "swiss-ai/Apertus-70B-2509"

FINEWEB_DIR = (
    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/"
    "fineweb-2_0_1-quality_10-filterrobots/data/output"
)
DCLM_DIR = (
    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/"
    "dclm-edu-filterrobots_fine/data/output"
)

OUTPUT_DTYPE = np.int32
DTYPE_CODE = 4  # Megatron DType enum value for int32
IDX_HEADER = b"MMIDIDX\x00\x00"

DEFAULT_BATCH_SIZE = 1024
DEFAULT_SAMPLE_FILES = 5
DEFAULT_SAMPLE_ROWS = 2000
DEFAULT_CHECKPOINT_INTERVAL = 50_000  # documents between checkpoint writes
DEFAULT_VAL_TOKENS_PER_LANGUAGE = 5_000_000  # validation budget per language
DEFAULT_VAL_MAX_FRACTION = 0.3  # cap validation at this fraction of first-file rows


# ---------------------------------------------------------------------------
# Megatron .bin/.idx writer (avoids importing megatron, which has conflicts)
# ---------------------------------------------------------------------------
class MegatronDatasetWriter:
    """Writes Megatron IndexedDataset .bin/.idx files.

    Supports resume by accepting pre-existing state.

    One parquet row = one document = one sequence, so the .idx document index
    is always exactly arange(len(sequence_lengths) + 1). It is generated in
    _write_idx rather than accumulated, so only sequence_lengths is tracked.
    """

    def __init__(
        self,
        output_prefix: str,
        *,
        resume: bool = False,
        sequence_lengths: Optional[List[int]] = None,
    ):
        self.output_prefix = output_prefix
        self.bin_path = output_prefix + ".bin"
        self.idx_path = output_prefix + ".idx"

        if resume and sequence_lengths is not None:
            self.sequence_lengths = sequence_lengths
            # Truncate .bin to the expected size in case the previous run
            # crashed mid-write, leaving trailing garbage bytes.
            expected_bytes = sum(sequence_lengths) * np.dtype(OUTPUT_DTYPE).itemsize
            actual_bytes = os.path.getsize(self.bin_path)
            if actual_bytes < expected_bytes:
                # truncate() would pad with NULs here, silently turning the gap
                # into token id 0 and corrupting the dataset at the seam. The
                # .bin is only ever flushed before its checkpoint is written, so
                # this means the two are out of sync (lost writes, wrong file).
                raise RuntimeError(
                    f"{self.bin_path} is SHORTER than its checkpoint expects "
                    f"({actual_bytes} < {expected_bytes} bytes). Refusing to "
                    f"zero-pad. The checkpoint and .bin disagree; inspect both "
                    f"before rerunning."
                )
            if actual_bytes > expected_bytes:
                print(
                    f"  Truncating .bin from {actual_bytes} to "
                    f"{expected_bytes} bytes (removing partial write)"
                )
                with open(self.bin_path, "ab") as f:
                    f.truncate(expected_bytes)
            self.bin_file = open(self.bin_path, "ab")
        else:
            self.sequence_lengths = []
            self.bin_file = open(self.bin_path, "wb")

    def add_document(self, token_ids: np.ndarray) -> int:
        """Write one document (a single sequence of token IDs).

        Returns the number of tokens written.
        """
        arr = token_ids.astype(OUTPUT_DTYPE)
        self.bin_file.write(arr.tobytes(order="C"))
        self.sequence_lengths.append(len(arr))
        return len(arr)

    def flush(self):
        self.bin_file.flush()

    def finalize(self):
        """Close .bin and write the .idx file."""
        self.bin_file.close()
        self._write_idx()

    def _write_idx(self):
        seq_lengths = np.array(self.sequence_lengths, dtype=np.int32)
        # One document per sequence: doc_indices == [0, 1, ..., n].
        doc_indices = np.arange(len(seq_lengths) + 1, dtype=np.int64)

        # Build sequence pointers (byte offsets into .bin)
        itemsize = np.dtype(OUTPUT_DTYPE).itemsize
        seq_pointers = np.zeros(len(seq_lengths), dtype=np.int64)
        if len(seq_lengths) > 0:
            seq_pointers[1:] = np.cumsum(seq_lengths[:-1].astype(np.int64)) * itemsize

        with open(self.idx_path, "wb") as f:
            f.write(IDX_HEADER)
            f.write(struct.pack("<Q", 1))  # version
            f.write(struct.pack("<B", DTYPE_CODE))
            f.write(struct.pack("<Q", len(seq_lengths)))  # sequence count
            f.write(struct.pack("<Q", len(doc_indices)))  # document count
            f.write(seq_lengths.tobytes(order="C"))
            f.write(seq_pointers.tobytes(order="C"))
            f.write(doc_indices.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def checkpoint_path(output_prefix: str) -> str:
    return output_prefix + ".checkpoint.json"


def _atomic_save_npy(path: str, arr: np.ndarray):
    """np.save to a tmp file, then rename into place.

    os.replace is atomic within a filesystem, so a reader either sees the
    previous checkpoint or the new one, never a half-written array.
    """
    tmp = path + ".tmp.npy"  # np.save would append .npy anyway; be explicit
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_checkpoint(
    output_prefix: str,
    source_progress: Dict,
    sequence_lengths: List[int],
    total_docs: int,
    total_toks: int,
    plan: List[dict],
):
    """Atomically write checkpoint to disk."""
    ckpt = {
        "source_progress": source_progress,
        "total_docs": total_docs,
        "total_toks": total_toks,
        "plan": plan,
        "num_sequences": len(sequence_lengths),
    }
    ckpt_file = checkpoint_path(output_prefix)
    # Every part of the checkpoint is written atomically (tmp file + rename).
    # These jobs are *designed* to be killed at the 12h wall, and a kill landing
    # inside a multi-hundred-MB np.save would leave a truncated .npy that the
    # JSON still points at: np.load then raises, every singleton successor dies
    # the same way, and a multi-day build becomes unresumable.
    _atomic_save_npy(ckpt_file + ".seq_lengths.npy",
                     np.array(sequence_lengths, dtype=np.int32))
    tmp = ckpt_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, ckpt_file)


def load_checkpoint(output_prefix: str) -> Optional[dict]:
    """Load checkpoint if it exists.

    Checkpoints written before doc_indices was dropped still carry a
    .doc_indices.npy next to this file; it is ignored (it only ever held
    arange) so in-flight builds resume across the change without a rebuild.
    """
    ckpt_file = checkpoint_path(output_prefix)
    if not os.path.isfile(ckpt_file):
        return None
    with open(ckpt_file, "r") as f:
        ckpt = json.load(f)
    ckpt["sequence_lengths"] = np.load(
        ckpt_file + ".seq_lengths.npy"
    ).tolist()
    return ckpt


def remove_checkpoint(output_prefix: str):
    """Remove checkpoint files after successful completion."""
    for suffix in ["", ".seq_lengths.npy", ".doc_indices.npy",
                   ".seq_lengths.npy.tmp.npy", ".doc_indices.npy.tmp.npy"]:
        path = checkpoint_path(output_prefix) + suffix
        if os.path.isfile(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
def text_bytes_per_file_byte(parquet_file: str) -> float:
    """Uncompressed `text` bytes per on-disk byte, from the parquet footer.

    Reads metadata only (no column data). Bridges the two byte units below:
    tokens are sampled per *uncompressed text* byte, but a source's volume is
    measured as *compressed on-disk* size across all columns.
    """
    meta = pq.ParquetFile(parquet_file).metadata
    text_uncompressed = sum(
        meta.row_group(r).column(i).total_uncompressed_size
        for r in range(meta.num_row_groups)
        for i in range(meta.row_group(r).num_columns)
        if meta.row_group(r).column(i).path_in_schema == "text"
    )
    return text_uncompressed / os.path.getsize(parquet_file)


def estimate_tokens_per_file_byte(
    tokenizer,
    parquet_files: List[str],
    sample_files: int = DEFAULT_SAMPLE_FILES,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> float:
    """Estimate tokens per on-disk parquet byte, by sampling rows.

    Returns a ratio in the same unit as get_total_file_bytes, so the product of
    the two is a token estimate. Tokens are counted per UTF-8 text byte (stable
    across documents), then converted to on-disk bytes with the text/file ratio
    measured on the same sampled files.

    That conversion is the whole point: parquet is compressed and carries
    columns besides `text`, and the resulting factor ranges from 0.51
    (deu_Latn) to 3.74 (hin_Deva) — a 7.3x spread. Multiplying a
    tokens-per-text-byte ratio straight by on-disk size therefore skewed the
    per-language proportions by that much, and those proportions ARE the
    mixture (props ~ estimated tokens at T=1). Within a language the factor is
    stable across files (<=5% between the sampled files and the whole set), so
    sampling it costs no extra I/O.

    Only the leading `sample_rows` rows of each sampled file are read. Reading
    the file whole to keep 2000 rows costs a full decompress and ~2 GB of RSS
    per file — 100 languages x 5 files is ~150 GB of I/O before a single token
    is written, on every fresh build.
    """
    files_to_sample = parquet_files[:sample_files]
    total_bytes = 0
    total_tokens = 0
    sampled_text_bytes = 0.0
    sampled_file_bytes = 0
    for f in files_to_sample:
        file_bytes = os.path.getsize(f)
        sampled_file_bytes += file_bytes
        sampled_text_bytes += text_bytes_per_file_byte(f) * file_bytes
        texts: List[Optional[str]] = []
        for batch in pq.ParquetFile(f).iter_batches(
            batch_size=min(sample_rows, DEFAULT_BATCH_SIZE), columns=["text"]
        ):
            texts.extend(batch.column("text").to_pylist())
            if len(texts) >= sample_rows:
                break
        texts = texts[:sample_rows]
        if not texts:
            continue
        for text in texts:
            if text is not None:
                total_bytes += len(text.encode("utf-8"))
        encoded = tokenizer(
            [t for t in texts if t is not None],
            add_special_tokens=False,
        )
        total_tokens += sum(len(ids) for ids in encoded["input_ids"])
    if total_bytes == 0 or sampled_file_bytes == 0:
        return 0.0
    tokens_per_text_byte = total_tokens / total_bytes
    return tokens_per_text_byte * (sampled_text_bytes / sampled_file_bytes)


def get_total_file_bytes(parquet_files: List[str]) -> int:
    """Sum of on-disk (compressed) parquet sizes, a proxy for content volume.

    Multiplied by the sampled tokens-per-*on-disk*-byte ratio to *estimate* a
    source's total tokens. This is still only an estimate: it drives language
    ranking (--top_k_languages) and the relative per-language split of the
    FineWeb-2 target, never the totals — each source stops at its precisely
    counted --target_tokens during the write.
    """
    return sum(os.path.getsize(f) for f in parquet_files)


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------
def discover_fineweb_languages(fineweb_dir: str) -> Dict[str, List[str]]:
    """Map language code -> sorted list of parquet file paths."""
    languages = {}
    for entry in sorted(os.listdir(fineweb_dir)):
        lang_dir = os.path.join(fineweb_dir, entry)
        if not os.path.isdir(lang_dir):
            continue
        parquets = sorted(
            os.path.join(lang_dir, f)
            for f in os.listdir(lang_dir)
            if f.endswith(".parquet")
        )
        if parquets:
            languages[entry] = parquets
    return languages


def discover_dclm_files(dclm_dir: str) -> List[str]:
    """Return sorted list of DCLM parquet file paths."""
    return sorted(
        os.path.join(dclm_dir, f)
        for f in os.listdir(dclm_dir)
        if f.endswith(".parquet")
    )


# ---------------------------------------------------------------------------
# Source planning (training-data build)
# ---------------------------------------------------------------------------
def build_plan(
    tokenizer,
    args,
    languages: Dict[str, List[str]],
    dclm_files: List[str],
) -> List[dict]:
    """Build the list of sources with per-source token targets.

    Each source entry: {
        "name": str,
        "files": [str, ...],
        "target_tokens": int,
        "estimated_tokens": int,
    }
    """
    fineweb_target = int(args.target_tokens * args.fineweb_pct / 100)
    dclm_target = int(args.target_tokens * args.dclm_pct / 100)

    sources: List[dict] = []

    # --- FineWeb-2 languages ---
    if fineweb_target > 0:
        if not (args.languages or args.top_k_languages):
            raise ValueError(
                "fineweb_pct > 0 requires --languages or --top_k_languages"
            )

        lang_est_tokens: Dict[str, int] = {}

        if args.languages:
            selected_langs = args.languages.split(",")
            missing = [l for l in selected_langs if l not in languages]
            if missing:
                raise ValueError(
                    f"Languages not found in {FINEWEB_DIR}: {missing}\n"
                    f"Available (sample): {list(languages.keys())[:20]}"
                )
            print("\nEstimating tokens for selected languages...")
            for lang in selected_langs:
                files = languages[lang]
                tpb = estimate_tokens_per_file_byte(
                    tokenizer, files,
                    sample_files=args.sample_files,
                    sample_rows=DEFAULT_SAMPLE_ROWS,
                )
                lang_est_tokens[lang] = int(tpb * get_total_file_bytes(files))
        else:
            # Estimate tokens for all languages, pick top K
            print(f"Estimating tokens for {len(languages)} languages...")
            all_lang_estimates = {}
            for i, (lang, files) in enumerate(languages.items()):
                tpb = estimate_tokens_per_file_byte(
                    tokenizer, files,
                    sample_files=args.sample_files,
                    sample_rows=DEFAULT_SAMPLE_ROWS,
                )
                all_lang_estimates[lang] = int(tpb * get_total_file_bytes(files))
                if (i + 1) % 100 == 0:
                    print(f"  Estimated {i + 1}/{len(languages)} languages...")

            ranked = sorted(all_lang_estimates.items(), key=lambda x: -x[1])
            selected_langs = [lang for lang, _ in ranked[: args.top_k_languages]]
            lang_est_tokens = {lang: all_lang_estimates[lang] for lang in selected_langs}
            print(f"\nTop {args.top_k_languages} languages by estimated tokens:")
            for lang in selected_langs:
                print(f"  {lang}: ~{lang_est_tokens[lang] / 1e9:.2f}B tokens")

        total_est = sum(lang_est_tokens.values())
        if total_est == 0:
            raise ValueError("No tokens estimated for selected languages")

        # Temperature-adjusted proportions: q_i proportional to p_i^(1/T).
        # T=1 -> proportional to estimated tokens; larger T -> flatter (more
        # uniform), upweighting lower-resource languages. T = 1/alpha.
        raw = {l: lang_est_tokens[l] / total_est for l in selected_langs}
        if args.temperature and args.temperature != 1.0:
            powed = {l: p ** (1.0 / args.temperature) for l, p in raw.items()}
            z = sum(powed.values())
            props = {l: powed[l] / z for l in selected_langs}
        else:
            props = raw

        for lang in selected_langs:
            sources.append({
                "name": f"fineweb_{lang}",
                "files": languages[lang],
                "target_tokens": int(fineweb_target * props[lang]),
                "estimated_tokens": lang_est_tokens[lang],
            })

    # --- DCLM (English) source ---
    if dclm_target > 0 and dclm_files:
        tpb = estimate_tokens_per_file_byte(
            tokenizer, dclm_files,
            sample_files=args.sample_files,
            sample_rows=DEFAULT_SAMPLE_ROWS,
        )
        dclm_est = int(tpb * get_total_file_bytes(dclm_files))
        sources.append({
            "name": "dclm",
            "files": dclm_files,
            "target_tokens": dclm_target,
            "estimated_tokens": dclm_est,
        })

    if not sources:
        raise ValueError(
            "No sources to process; check --fineweb_pct/--dclm_pct and "
            "language selection"
        )

    return sources


def print_plan(sources: List[dict], target_tokens: int):
    print("\n--- Plan ---")
    print(f"Total target: {target_tokens / 1e9:.2f}B tokens")
    print(f"Sources: {len(sources)}")
    for s in sources:
        print(
            f"  {s['name']:30s}  "
            f"target={s['target_tokens'] / 1e9:.4f}B  "
            f"estimated={s['estimated_tokens'] / 1e9:.4f}B  "
            f"files={len(s['files'])}"
        )
    print()


# ---------------------------------------------------------------------------
# Validation-set construction (fixed, mixture-independent)
# ---------------------------------------------------------------------------
def build_validation_set(
    tokenizer,
    args,
    languages: Dict[str, List[str]],
    dclm_files: List[str],
):
    """Build a fixed per-language validation set.

    For each language, the validation set is the first --val_tokens_per_language
    tokens of its FIRST parquet file (sorted), capped at --val_max_fraction of
    that file's rows so single-file languages keep training data. The build is
    INDEPENDENT of temperature, target_tokens, and the training mixture.

    For each language it writes a Megatron .bin/.idx and records, in a manifest:
      tokens, bytes (UTF-8; the BPB denominator), n_docs, val_doc_count (the
      number of leading rows of the first file assigned to validation), and the
      first file path. Training reads this manifest and skips the first
      val_doc_count rows of that file, so the two sets are disjoint.

    Run this ONCE over the full language list; reuse the output for every model.
    """
    if args.languages:
        selected = args.languages.split(",")
        missing = [l for l in selected if l not in languages]
        if missing:
            raise ValueError(f"Languages not found in {FINEWEB_DIR}: {missing}")
    else:
        selected = sorted(languages.keys())

    val_sources: List[Tuple[str, List[str]]] = [
        (f"fineweb_{lang}", languages[lang]) for lang in selected
    ]
    if dclm_files:
        val_sources.append(("dclm", dclm_files))

    manifest: Dict[str, dict] = {}
    print(f"\n--- Building validation set ({len(val_sources)} sources) ---")
    print(f"Budget per language: {args.val_tokens_per_language:,} tokens; "
          f"capped at {args.val_max_fraction:.0%} of the first file's rows")

    for name, files in val_sources:
        if not files:
            print(f"  WARNING: {name} has no files; skipping")
            continue

        first_file = files[0]
        # n_rows comes from the parquet footer (O(1)); we then stream the file
        # in batches and stop at the token budget. Validation consumes only the
        # first few thousand rows, so we never materialize the whole (multi-GB)
        # file — a full to_pylist() spikes ~2.5 GB per source and gets a login
        # node's long-running job reaped.
        pf = pq.ParquetFile(first_file)
        n_rows = pf.metadata.num_rows
        row_cap = int(args.val_max_fraction * n_rows)

        prefix = f"{args.output_prefix}.{name}"
        writer = MegatronDatasetWriter(prefix)
        toks = 0
        n_bytes = 0
        n_docs = 0
        val_doc_count = 0  # number of leading rows assigned to validation
        row_idx = 0
        stop = False

        for batch in pf.iter_batches(batch_size=args.batch_size, columns=["text"]):
            if stop:
                break
            batch_rows = batch.column("text").to_pylist()
            # Tokenize the whole batch, mapping None/empty to "" to keep row
            # alignment (empty text tokenizes to an empty id list).
            safe = [t if t is not None else "" for t in batch_rows]
            encoded = tokenizer(safe, add_special_tokens=False)["input_ids"]
            for text, ids in zip(batch_rows, encoded):
                if toks >= args.val_tokens_per_language or row_idx >= row_cap:
                    stop = True
                    break
                # rows[0:row_idx+1] are reserved for validation (boundary by row)
                val_doc_count = row_idx + 1
                row_idx += 1
                if text is None or len(text) == 0 or len(ids) == 0:
                    continue
                writer.add_document(np.array(ids, dtype=OUTPUT_DTYPE))
                toks += len(ids)
                n_bytes += len(text.encode("utf-8"))
                n_docs += 1

        writer.finalize()
        manifest[name] = {
            "bin": prefix + ".bin",
            "idx": prefix + ".idx",
            "tokens": toks,
            "bytes": n_bytes,
            "n_docs": n_docs,
            "val_doc_count": val_doc_count,
            "first_file": first_file,
        }
        flag = "" if n_docs > 0 else "  [WARNING: no validation docs]"
        print(f"  {name:30s}  {n_docs:,} docs, {toks:,} tokens, "
              f"{n_bytes:,} bytes, val_doc_count={val_doc_count}/{n_rows}{flag}")

    manifest_path = f"{args.output_prefix}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nValidation manifest: {manifest_path}")
    print(f"Sources with validation data: "
          f"{sum(1 for v in manifest.values() if v['n_docs'] > 0)}/{len(manifest)}")


# ---------------------------------------------------------------------------
# Main processing loop (training-data build)
# ---------------------------------------------------------------------------
def process_sources(
    tokenizer,
    writer: MegatronDatasetWriter,
    sources: List[dict],
    source_progress: Dict,
    total_docs: int,
    total_toks: int,
    args,
    val_skip_by_source: Optional[Dict[str, int]] = None,
) -> Tuple[int, int]:
    """Stream through sources, tokenize, and write to the dataset.

    If val_skip_by_source is given, the first val_skip_by_source[name] rows of
    each source's first file are skipped (they belong to the validation set).

    Returns updated (total_docs, total_toks).
    """
    val_skip_by_source = val_skip_by_source or {}
    plan_serializable = [
        {k: v for k, v in s.items() if k != "files"}
        for s in sources
    ]

    for src_idx, source in enumerate(sources):
        name = source["name"]
        target = source["target_tokens"]
        files = source["files"]
        val_skip = val_skip_by_source.get(name, 0)

        # Check if this source is already complete
        prog = source_progress.get(name, {"file_idx": 0, "source_toks": 0})
        start_file_idx = prog["file_idx"]
        source_toks = prog["source_toks"]

        if source_toks >= target:
            print(f"[{src_idx+1}/{len(sources)}] {name}: already complete, skipping")
            continue

        print(
            f"[{src_idx+1}/{len(sources)}] {name}: "
            f"target={target / 1e9:.4f}B tokens, "
            f"{len(files)} files"
            + (f", skipping {val_skip} validation rows in first file"
               if val_skip else "")
        )
        if start_file_idx > 0:
            print(f"  Resuming from file {start_file_idx}, {source_toks/1e9:.4f}B tokens already written")

        source_docs = 0

        for file_idx in range(start_file_idx, len(files)):
            if source_toks >= target:
                break

            parquet_path = files[file_idx]
            # Stream the file batch-by-batch. A whole-file read_table().to_pylist()
            # peaks at ~5 GB for a 1 GB DCLM parquet (Arrow table + Python str
            # objects), and is wasted entirely when a source hits its target in
            # the file's first batches. Batching bounds peak RSS independent of
            # file size, leaving the doc-count-driven sequence_lengths list as
            # the only term that grows with the build.
            skip = val_skip if file_idx == 0 else 0  # leading validation rows
            row_idx = 0

            for batch in pq.ParquetFile(parquet_path).iter_batches(
                batch_size=args.batch_size, columns=["text"]
            ):
                if source_toks >= target:
                    break

                n_rows = batch.num_rows
                texts = batch.column("text").to_pylist()
                # Exclude the validation rows from the first file of this source.
                # The boundary is by row index, so this is consistent on resume.
                if row_idx < skip:
                    texts = texts[skip - row_idx:]
                row_idx += n_rows

                batch_texts = [t for t in texts if t is not None and len(t) > 0]
                if not batch_texts:
                    continue
                encoded = tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                )

                for ids in encoded["input_ids"]:
                    if source_toks >= target:
                        break
                    if len(ids) == 0:
                        continue
                    token_arr = np.array(ids, dtype=OUTPUT_DTYPE)
                    doc_toks = writer.add_document(token_arr)
                    source_toks += doc_toks
                    total_toks += doc_toks
                    total_docs += 1
                    source_docs += 1

                    # Progress heartbeat only — do NOT save a checkpoint here.
                    # A mid-file checkpoint would record file_idx of a
                    # partially-consumed file while sequence_lengths already
                    # include that file's partial docs; on resume the .bin is
                    # truncated to those docs AND the file is re-tokenized from
                    # row 0, duplicating its head. Checkpoints are taken only at
                    # file boundaries (below), so resume redoes at most one file
                    # cleanly.
                    if total_docs % args.checkpoint_interval == 0:
                        writer.flush()
                        print(
                            f"    {total_docs:,} docs, "
                            f"{total_toks / 1e9:.4f}B tokens"
                        )

            # Completed this file — update progress to next file and
            # checkpoint immediately. Checkpoints happen only at file
            # boundaries, so on resume the .bin truncates back to a completed
            # file and re-tokenizing the in-progress file from row 0 never
            # duplicates data.
            source_progress[name] = {
                "file_idx": file_idx + 1,
                "source_toks": source_toks,
            }
            writer.flush()
            save_checkpoint(
                args.output_prefix,
                source_progress,
                writer.sequence_lengths,
                total_docs,
                total_toks,
                plan_serializable,
            )

            if source_docs > 0 and file_idx % 50 == 0:
                print(
                    f"    file {file_idx+1}/{len(files)}: "
                    f"{source_docs:,} docs, {source_toks / 1e9:.4f}B tokens"
                )

        if source_toks < target:
            print(
                f"  WARNING: {name} exhausted before target! "
                f"Shortfall: {(target - source_toks) / 1e9:.4f}B tokens"
            )
        print(f"  Done: {source_docs:,} docs, {source_toks / 1e9:.4f}B tokens")

    return total_docs, total_toks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create a blended Megatron-LM dataset from parquet sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target_tokens", type=int, default=None,
        help="Total target token count for the output dataset "
             "(required unless --build_validation)",
    )
    parser.add_argument(
        "--fineweb_pct", type=float, default=None,
        help="Percentage of tokens from FineWeb-2 (0-100); "
             "required unless --build_validation",
    )
    parser.add_argument(
        "--dclm_pct", type=float, default=None,
        help="Percentage of tokens from DCLM (0-100); "
             "required unless --build_validation",
    )
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated language codes to include from FineWeb-2 "
             "(e.g., rus_Cyrl,deu_Latn,fra_Latn). In --build_validation mode, "
             "the validation set is built for these languages (plus DCLM "
             "English); if omitted, all discovered languages are used.",
    )
    parser.add_argument(
        "--top_k_languages", type=int, default=None,
        help="Select the top K languages by estimated token count",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for per-language allocation within "
             "FineWeb-2; proportions are proportional to p^(1/T). T=1 (default) "
             "is proportional to estimated tokens; larger T flattens toward "
             "uniform (T = 1/alpha, so alpha=0.3 is T~3.3).",
    )
    parser.add_argument(
        "--output_prefix", type=str, required=True,
        help="Prefix for output .bin/.idx files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Tokenization batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--sample_files", type=int, default=DEFAULT_SAMPLE_FILES,
        help=f"Files to sample per source for token estimation (default: {DEFAULT_SAMPLE_FILES})",
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Documents between progress heartbeats (default: {DEFAULT_CHECKPOINT_INTERVAL:,}); "
             "checkpoints themselves are written only at parquet-file boundaries",
    )
    parser.add_argument(
        "--val_tokens_per_language", type=int,
        default=DEFAULT_VAL_TOKENS_PER_LANGUAGE,
        help="Validation tokens per language in --build_validation mode "
             f"(default: {DEFAULT_VAL_TOKENS_PER_LANGUAGE:,}). Taken from the "
             "start of each language's first file.",
    )
    parser.add_argument(
        "--val_max_fraction", type=float, default=DEFAULT_VAL_MAX_FRACTION,
        help="Cap the validation set at this fraction of the first file's rows "
             f"(default: {DEFAULT_VAL_MAX_FRACTION}), so single-file languages "
             "keep training data.",
    )
    parser.add_argument(
        "--build_validation", action="store_true",
        help="Build the fixed per-language validation set and exit. Ignores "
             "--target_tokens, --fineweb_pct, --dclm_pct, and --temperature.",
    )
    parser.add_argument(
        "--validation_manifest", type=str, default=None,
        help="Path to the validation manifest produced by --build_validation. "
             "When given, training skips the leading validation rows of each "
             "source's first file so training and validation do not overlap.",
    )
    args = parser.parse_args()

    # Load tokenizer (needed for both modes)
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # --- Validation-set build mode ---
    if args.build_validation:
        print("Discovering data sources (validation build)...")
        languages = discover_fineweb_languages(FINEWEB_DIR)
        dclm_files = discover_dclm_files(DCLM_DIR)
        print(f"  FineWeb-2: {len(languages)} languages")
        print(f"  DCLM: {len(dclm_files)} files")
        out_dir = os.path.dirname(os.path.abspath(args.output_prefix))
        os.makedirs(out_dir, exist_ok=True)
        build_validation_set(tokenizer, args, languages, dclm_files)
        return

    # --- Training-data build mode: validate arguments ---
    if args.target_tokens is None:
        parser.error("--target_tokens is required (unless --build_validation)")
    if args.fineweb_pct is None or args.dclm_pct is None:
        parser.error("--fineweb_pct and --dclm_pct are required (unless --build_validation)")
    if abs(args.fineweb_pct + args.dclm_pct - 100.0) > 0.01:
        parser.error("--fineweb_pct and --dclm_pct must sum to 100")
    if args.fineweb_pct > 0 and not args.languages and not args.top_k_languages:
        parser.error("Must specify --languages or --top_k_languages when --fineweb_pct > 0")

    # Load the validation manifest (to exclude validation rows from training)
    val_skip_by_source: Dict[str, int] = {}
    if args.validation_manifest:
        with open(args.validation_manifest, "r") as f:
            vm = json.load(f)
        for name, entry in vm.items():
            val_skip_by_source[name] = int(entry.get("val_doc_count", 0))
        print(
            f"Loaded validation manifest ({args.validation_manifest}): will skip "
            f"leading validation rows for {len(val_skip_by_source)} sources"
        )
    else:
        print(
            "WARNING: no --validation_manifest given; training data may overlap "
            "the validation set."
        )

    # Check for existing checkpoint
    ckpt = load_checkpoint(args.output_prefix)
    if ckpt is not None:
        print("Resuming from checkpoint!")
        print(f"  Documents: {ckpt['total_docs']:,}")
        print(f"  Tokens:    {ckpt['total_toks'] / 1e9:.4f}B")

    # Discover data
    print("Discovering data sources...")
    languages = discover_fineweb_languages(FINEWEB_DIR)
    dclm_files = discover_dclm_files(DCLM_DIR)
    print(f"  FineWeb-2: {len(languages)} languages")
    print(f"  DCLM: {len(dclm_files)} files")

    # Build or reload plan
    if ckpt is not None:
        # Rebuild source list with file paths, using saved plan for targets.
        saved_plan = ckpt["plan"]
        sources = []
        for sp in saved_plan:
            if sp["name"].startswith("fineweb_"):
                lang = sp["name"][len("fineweb_"):]
                files = languages.get(lang, [])
            elif sp["name"] == "dclm":
                files = dclm_files
            else:
                files = []
            sources.append({**sp, "files": files})
        source_progress = ckpt["source_progress"]
        total_docs = ckpt["total_docs"]
        total_toks = ckpt["total_toks"]
    else:
        sources = build_plan(tokenizer, args, languages, dclm_files)
        source_progress = {}
        total_docs = 0
        total_toks = 0

    print_plan(sources, args.target_tokens)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_prefix))
    os.makedirs(out_dir, exist_ok=True)

    # Initialize writer
    if ckpt is not None:
        writer = MegatronDatasetWriter(
            args.output_prefix,
            resume=True,
            sequence_lengths=ckpt["sequence_lengths"],
        )
    else:
        writer = MegatronDatasetWriter(args.output_prefix)

    t0 = time.time()

    total_docs, total_toks = process_sources(
        tokenizer, writer, sources, source_progress,
        total_docs, total_toks, args, val_skip_by_source,
    )

    writer.finalize()
    remove_checkpoint(args.output_prefix)

    elapsed = (time.time() - t0) / 60
    print(f"\n--- Done ---")
    print(f"Documents: {total_docs:,}")
    print(f"Tokens:    {total_toks / 1e9:.4f}B")
    print(f"Time:      {elapsed:.1f} min")
    print(f"Output:    {args.output_prefix}.bin / .idx")


if __name__ == "__main__":
    main()

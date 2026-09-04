"""
Merge results from multiple split evaluation directories into a single results file.
Used by aggregate_splits.sbatch to combine parallelized evaluation outputs and
by _run_per_task.sh to fold a job's per_task/<task>/ dirs into its eval dir.
"""
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser

# Per-task dicts that union cleanly across splits; everything else (config,
# environment, tokenizer info, timing) is taken from the first split.
MERGE_KEYS = ("results", "groups", "group_subtasks", "configs", "n-shot",
              "versions", "higher_is_better", "n-samples", "task_hashes")


def merge_split_results(split_dirs: list[Path], output_dir: Path,
                        move_samples: bool = False):
    """Merge results_*.json and samples_*.jsonl from multiple split dirs.

    Samples are copied up by default; `move_samples` moves them instead, so a
    job's per_task/ tree keeps only the small per-task results files rather
    than a second copy of every samples file."""
    merged_results = None

    for split_dir in split_dirs:
        # Shallowest first. A split job's eval dir now holds BOTH its merged
        # results_*.json at the top and one per task under per_task/<task>/
        # <model>/, and `**` matches every depth — an arbitrary [0] could pick
        # a single-task file and silently drop the rest of that split.
        result_files = sorted(split_dir.glob("**/results_*.json"),
                              key=lambda p: (len(p.relative_to(split_dir).parts), p.name))
        if not result_files:
            print(f"WARNING: No results file found in {split_dir}")
            continue

        result_file = result_files[0]
        with open(result_file) as f:
            split_results = json.load(f)

        if merged_results is None:
            # Use the first split as the base
            merged_results = split_results
        else:
            # Merge results from this split into the base
            for key in MERGE_KEYS:
                if key in split_results:
                    merged_results.setdefault(key, {}).update(split_results[key])

        for sample_file in split_dir.glob("**/samples_*.jsonl"):
            dest = output_dir / sample_file.name
            if dest.exists():
                continue
            if move_samples:
                shutil.move(str(sample_file), str(dest))
            else:
                shutil.copy2(sample_file, dest)

    if merged_results is None:
        raise RuntimeError("No results files found in any split directory")

    # Write merged results with a consistent timestamp
    # Use the timestamp from the base results file name
    base_result_files = list(split_dirs[0].glob("**/results_*.json"))
    timestamp = base_result_files[0].stem.replace("results_", "") if base_result_files else "merged"

    output_file = output_dir / f"results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)

    task_count = len(merged_results.get("results", {}))
    print(f"Merged {len(split_dirs)} splits -> {task_count} tasks in {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge split evaluation results")
    parser.add_argument("--split_dirs", nargs="+", type=Path, required=True,
                        help="Directories containing split results")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for merged results")
    parser.add_argument("--move-samples", action="store_true",
                        help="Move samples_*.jsonl into output_dir instead of copying")
    args = parser.parse_args()

    merge_split_results(args.split_dirs, args.output_dir, args.move_samples)

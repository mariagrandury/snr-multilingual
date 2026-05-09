"""
Merge results from multiple split evaluation directories into a single results file.
Used by aggregate_splits.sbatch to combine parallelized evaluation outputs.
"""
import json
import shutil
from pathlib import Path
from argparse import ArgumentParser


def merge_split_results(split_dirs: list[Path], output_dir: Path):
    """Merge results_*.json and samples_*.jsonl from multiple split dirs."""
    merged_results = None

    for split_dir in split_dirs:
        result_files = list(split_dir.glob("**/results_*.json"))
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
            if "results" in split_results:
                merged_results["results"].update(split_results["results"])
            if "configs" in split_results:
                merged_results["configs"].update(split_results["configs"])
            if "n-shot" in split_results:
                merged_results["n-shot"].update(split_results["n-shot"])
            if "versions" in split_results:
                merged_results["versions"].update(split_results["versions"])
            if "higher_is_better" in split_results:
                merged_results["higher_is_better"].update(split_results["higher_is_better"])
            if "n-samples" in split_results:
                merged_results["n-samples"].update(split_results["n-samples"])

        # Copy sample files
        for sample_file in split_dir.glob("**/samples_*.jsonl"):
            dest = output_dir / sample_file.name
            if not dest.exists():
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
    args = parser.parse_args()

    merge_split_results(args.split_dirs, args.output_dir)

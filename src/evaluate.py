import json
import time
from datetime import datetime
from pathlib import Path

import lm_eval
from lm_eval.loggers import WandbLogger

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
WANDB_ENTITY = "mariagrandury-epflnlp"
WANDB_PROJECT = "snr-experiments"


def run_evaluation(
    model_id: str,
    revision: str,
    tasks: list[str],
    *,
    checkpoint_index: int = 0,
    device: str = "cpu",
    batch_size: str | int = "auto",
    limit: int | None = None,
    log_wandb: bool = True,
) -> dict:
    """Run lm_eval on a single model checkpoint and save results.

    checkpoint_index: Position in the sorted checkpoint list, used as the x-axis in W&B charts.
    """
    model_short = model_id.split("/")[-1]
    run_name = f"{model_short}_{revision}"

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id} @ {revision} (index {checkpoint_index})")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}, Limit: {limit}")
    print(f"{'='*60}\n")

    model_args = f"pretrained={model_id},revision={revision},trust_remote_code=True"

    start_time = time.time()
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
    )

    elapsed = time.time() - start_time
    results["total_evaluation_time_seconds"] = round(elapsed, 2)
    print(f"Evaluation completed in {elapsed:.1f}s")

    # Save results locally
    output_dir = RESULTS_DIR / model_short / revision
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"

    # Save full results (except per-sample data)
    results_to_save = {k: v for k, v in results.items() if k != "samples"}
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    # Save samples as JSONL (one file per task)
    if "samples" in results:
        for task_name, samples in results["samples"].items():
            samples_file = output_dir / f"samples_{task_name}_{timestamp}.jsonl"
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"Samples saved to {samples_file}")

    # Log to W&B
    if log_wandb:
        import wandb

        wandb_logger = WandbLogger(
            init_args={
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "name": run_name,
                "group": model_id,
                "job_type": "eval",
                "tags": [model_short, revision],
                "config": {
                    "model_id": model_id,
                    "revision": revision,
                    "checkpoint_index": checkpoint_index,
                    "tasks": tasks,
                    "limit": limit,
                    "device": device,
                },
            },
        )
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        if "samples" in results:
            wandb_logger.log_eval_samples(results["samples"])

        # Set summary metrics for charting (avoids W&B auto-incrementing _step)
        #   x-axis = checkpoint_index, y-axis = score, grouped by model
        wandb.summary["checkpoint_index"] = checkpoint_index
        wandb.summary["revision"] = revision
        wandb.summary["total_evaluation_time_seconds"] = elapsed
        for task_name, task_results in results["results"].items():
            for metric_key, value in task_results.items():
                if metric_key == "alias" or "stderr" in metric_key:
                    continue
                wandb.summary[f"{task_name}/{metric_key}"] = value

        wandb.finish()
        print(f"Results logged to W&B: {WANDB_ENTITY}/{WANDB_PROJECT}/{run_name}")

    return results


def run_all(
    models: list[dict],
    checkpoints_per_model: dict[str, list[str]],
    tasks: list[str],
    *,
    device: str = "cpu",
    batch_size: str | int = "auto",
    limit: int | None = None,
    log_wandb: bool = True,
):
    """Run evaluation for all models and their resolved checkpoints."""
    for model_entry in models:
        model_id = model_entry["id"]
        revisions = checkpoints_per_model[model_id]
        for idx, revision in enumerate(revisions):
            run_evaluation(
                model_id,
                revision,
                tasks,
                checkpoint_index=idx,
                device=device,
                batch_size=batch_size,
                limit=limit,
                log_wandb=log_wandb,
            )

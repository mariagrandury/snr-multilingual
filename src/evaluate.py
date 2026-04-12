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
    device: str = "cpu",
    batch_size: str | int = "auto",
    limit: int | None = None,
    log_wandb: bool = True,
) -> dict:
    """Run lm_eval on a single model checkpoint and save results."""
    model_short = model_id.split("/")[-1]
    run_name = f"{model_short}_{revision}"
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id} @ {revision}")
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

    # Save results locally
    output_dir = RESULTS_DIR / model_short / revision
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"

    elapsed = time.time() - start_time
    results["total_evaluation_time_seconds"] = round(elapsed, 2)
    print(f"Evaluation completed in {elapsed:.1f}s")

    # Save full results (everything except per-sample data, which goes in JSONL)
    results_to_save = {k: v for k, v in results.items() if k != "samples"}
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    # Save samples
    if "samples" in results:
        for task_name, samples in results["samples"].items():
            samples_file = output_dir / f"samples_{task_name}_{timestamp}.jsonl"
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"Samples saved to {samples_file}")

    # Log to W&B
    if log_wandb:
        wandb_logger = WandbLogger(
            init_args={
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
                "name": run_name,
                "job_type": "eval",
            },
        )
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        if "samples" in results:
            wandb_logger.log_eval_samples(results["samples"])
        import wandb
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
        for revision in revisions:
            run_evaluation(
                model_id,
                revision,
                tasks,
                device=device,
                batch_size=batch_size,
                limit=limit,
                log_wandb=log_wandb,
            )

#!/usr/bin/env python3
"""One eval worker: load the model ONCE, then run tasks one at a time and write
each task's artifacts the moment it finishes.

This replaces the `lm_eval` CLI in evaluate.sbatch's inner loop
(scripts/_run_per_task.sh starts one of these per GPU). Both ways of calling
the CLI were wrong for a sweep: one call per task reloaded vLLM every time
(30-90 s, longer than most tasks), one call for every task loaded once but
wrote NOTHING until the very end, so a walltime kill or one bad dataset threw
the whole job away (../CLAUDE.md bug 13).

Here the model object is created once and handed to `lm_eval.simple_evaluate`
per task, so a task costs only its own dataset load and inference, and its
results_*.json + samples_*.jsonl are on disk before the next one starts. They
are written by lm_eval's own EvaluationTracker, so the files are exactly what
the CLI writes: full config, environment and tokenizer info, task hashes,
per-task timing.

Layout under --output_path (the job's eval_<ts>_<jobid> dir):

    inflight/<task>.claim       claim marker: exactly one worker owns the task
    inflight/<task>/            in progress (partial output after a kill)
    per_task/<task>/<model>/    results_<ts>.json + samples_<task>_<ts>.jsonl
    failed_tasks.log            "<task>\\t<Error>: <message>", one line per failure

A worker claims a task by creating `inflight/<task>.claim` with O_EXCL
(atomic, so N workers on the same job dir never run the same task twice; the
marker outlives the task, or a worker arriving later would claim it again)
and publishes it by renaming `inflight/<task>` to `per_task/<task>` once the
results file exists. The rename is atomic, so `per_task/<task>/` exists only
when complete — which is the "done" test in _eval_status.completed_tasks,
i.e. what makes the next job skip it. Whatever directory is left in inflight/
after a kill is the partial output of a task that was running.

Failure isolation: an exception inside one task is logged and the worker
moves on. Under torchrun/accelerate (WORLD_SIZE > 1, the hf and megatron_lm
backends) it is re-raised instead — the ranks must stay in lockstep through
lm_eval's collectives, and a rank that skipped a task would deadlock the rest;
what finished before the crash is on disk regardless.

The arguments mirror the lm_eval CLI's so evaluate.sbatch composes the command
the way it always did; --worker/--num_workers are the only additions.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_limit(s: str | None):
    if s is None:
        return None
    return int(s) if float(s) >= 1 and float(s).is_integer() else float(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="lm_eval backend: vllm, hf, megatron_lm, ...")
    p.add_argument("--model_args", default="", help="comma-separated key=value, as for lm_eval")
    p.add_argument("--tasks", required=True, help="comma-separated task names (the shared queue)")
    p.add_argument("--output_path", required=True, help="the job's eval dir")
    p.add_argument("--worker", type=int, default=0, help="this worker's index (logging only)")
    p.add_argument("--num_workers", type=int, default=1,
                   help="workers sharing --output_path; >1 turns on task claiming")
    p.add_argument("--batch_size", default=1)
    p.add_argument("--max_batch_size", type=int)
    p.add_argument("--device")
    p.add_argument("--limit", type=parse_limit)
    p.add_argument("--num_fewshot", type=int)
    p.add_argument("--gen_kwargs")
    p.add_argument("--system_instruction")
    p.add_argument("--apply_chat_template", nargs="?", const=True, default=False)
    p.add_argument("--fewshot_as_multiturn", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--confirm_run_unsafe_code", action="store_true")
    p.add_argument("--log_samples", action="store_true")
    p.add_argument("--write_out", action="store_true")
    p.add_argument("--include_path")
    p.add_argument("--metadata", help="JSON dict, as for lm_eval --metadata")
    return p.parse_args()


def memoise_env_info() -> None:
    """`pretty_env_info` in every results file shells out to nvidia-smi and
    `pip list` — seconds per call, identical for every task of this process.
    Compute it once. Patched on torch (the harness imports it from there at
    call time) and, for harness versions that bind it at import, on the
    harness module too."""
    try:
        import torch.utils.collect_env as ce
    except ImportError:               # no torch: the harness records "N/A"
        return
    cached = functools.lru_cache(maxsize=None)(ce.get_pretty_env_info)
    ce.get_pretty_env_info = cached
    try:
        import lm_eval.loggers.utils as lu
        if hasattr(lu, "get_pretty_env_info"):
            lu.get_pretty_env_info = cached
    except ImportError:
        pass


def claim(marker: Path) -> bool:
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main() -> int:
    args = parse_args()
    out = Path(args.output_path)
    inflight, per_task = out / "inflight", out / "per_task"
    failed_log = out / "failed_tasks.log"
    tasks = [t for t in args.tasks.split(",") if t]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1 and args.num_workers > 1:
        sys.exit("--num_workers > 1 is one process per worker; under "
                 "torchrun/accelerate use --num_workers 1")
    if rank == 0:
        inflight.mkdir(parents=True, exist_ok=True)
        per_task.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[eval_worker {args.worker}] {msg}", flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    memoise_env_info()
    import lm_eval
    from lm_eval.api.registry import get_model
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table, simple_parse_args_string

    model_args = simple_parse_args_string(args.model_args)
    if args.trust_remote_code:            # what the CLI flag does, both sides
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args["trust_remote_code"] = True
    metadata = json.loads(args.metadata) if args.metadata else {}
    gen_kwargs = simple_parse_args_string(args.gen_kwargs) if args.gen_kwargs else None

    t0 = time.time()
    log(f"loading {args.model} once: {model_args}")
    lm = get_model(args.model).create_from_arg_obj(
        model_args, {"batch_size": args.batch_size,
                     "max_batch_size": args.max_batch_size, "device": args.device})
    log(f"model ready in {time.time() - t0:.0f}s; {len(tasks)} task(s) in the queue")
    # One TaskManager for the whole run: building it indexes every task YAML
    # in the harness, which simple_evaluate would otherwise redo per call.
    task_manager = TaskManager(include_path=args.include_path,
                               metadata={**model_args, **metadata})

    done = failed = attempted = 0
    for task in tasks:
        if args.num_workers > 1 and not claim(inflight / f"{task}.claim"):
            continue                      # another worker has it
        if rank == 0:
            (inflight / task).mkdir(exist_ok=True)
        attempted += 1
        tracker = EvaluationTracker(output_path=str(inflight / task))
        t0 = time.time()
        try:
            results = lm_eval.simple_evaluate(
                model=lm, model_args=model_args, tasks=[task],
                num_fewshot=args.num_fewshot, batch_size=args.batch_size,
                max_batch_size=args.max_batch_size, device=args.device,
                limit=args.limit, write_out=args.write_out,
                log_samples=args.log_samples, evaluation_tracker=tracker,
                system_instruction=args.system_instruction,
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=args.fewshot_as_multiturn,
                gen_kwargs=gen_kwargs, task_manager=task_manager,
                confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                metadata=metadata)
            if results is None:           # a non-zero rank: rank 0 writes
                continue
            samples = results.pop("samples", None)
            # A pre-built LM is recorded as source "CUSTOM"/its class name;
            # restore the backend name the CLI writes so readers of
            # model_source/config.model see the same values as before.
            results["config"]["model"] = args.model
            tracker.general_config_tracker.model_source = args.model
            tracker.save_results_aggregated(results=results, samples=samples)
            if samples:
                for name in results["configs"]:
                    tracker.save_results_samples(task_name=name, samples=samples[name])
            # The tracker only warns when a write fails; check before publishing.
            if not any((inflight / task).glob("*/results_*.json")):
                raise RuntimeError("no results file written")
            os.rename(inflight / task, per_task / task)
        except Exception as e:
            if world_size > 1:
                raise
            traceback.print_exc()
            reason = f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"
            with open(failed_log, "a") as f:
                f.write(f"{task}\t{reason}\n")
            failed += 1
            log(f"FAILED {task} after {time.time() - t0:.0f}s — {reason}")
            continue
        done += 1
        print(make_table(results), flush=True)
        if "groups" in results:
            print(make_table(results, "groups"), flush=True)
        log(f"done {task} in {time.time() - t0:.0f}s ({done} done, {failed} failed)")

    log(f"finished: {done} done, {failed} failed, {attempted} attempted")
    # Non-zero only when this process actually failed tasks and saved none.
    # NOT "attempted but nothing done": under torchrun/accelerate every rank
    # but 0 gets None back from simple_evaluate for every task (rank 0 is the
    # only writer), so that test would fail the launcher — and with it the
    # whole job — on a run where every result landed correctly.
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())

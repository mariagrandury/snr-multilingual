"""Automatic evaluations
Usage (in a tmux session that never ends):
```
while true; do
    python scripts/automate.py
    sleep $(( 60*60 ))
done
```
Make sure to export your WANDB_API_KEY and LOGS_ROOT.
"""
from __future__ import annotations

import collections
import re
import os
import json
import subprocess
import shutil
from pathlib import Path


def unify(completed: list[str]) -> list[str]:
    completed_set = set(completed)
    unified = []
    for groupname, tasks in filter(lambda t: t[0] != ROOT_EVAL, TASKS["groups"].items()):
        if set(tasks) <= completed_set:
            unified.append(groupname)
    return unified


def get_running(as_jobname: bool = False) -> dict[str, dict[int, list[str]] | list[str]]:
    proc = subprocess.run(["squeue", "--me", '--format="%j"', "--noheader"],
                          capture_output=True, text=True)
    assert proc.returncode == 0

    jobnames = proc.stdout.strip().split("\n")
    if jobnames == [""]:
        return [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    jobnames = [re.match('^"(.*)"$', jobname).group(1) for jobname in jobnames]

    running = [] if as_jobname else collections.defaultdict(lambda: collections.defaultdict(list))
    for jobname in jobnames:
        rmatch = re.match(r"^eval_(.*)_([a-zA-Z]+)_([0-9]+)$", jobname)
        if rmatch is not None:
            if as_jobname:
                running.append(jobname)
            else:
                name, group, it = rmatch.groups()
                if group == ROOT_EVAL:
                    running[name][int(it)] += ALL_EVALS
                else:
                    running[name][int(it)].append(group)
    return running


def get_evaluated(model: str) -> dict[int, list[str]]:
    status = collections.defaultdict(list)
    for path in Path(CFG["logs_root"]).glob(f"{model}/iter_*/harness/eval_*/*/results*.json"):
        it = int(re.match("^iter_([0-9]+)$", path.parent.parent.parent.parent.name).group(1))
        with open(path) as f:
            info = json.load(f)
        for task in info["results"]:
            status[it].append(task)
    return {it: unify(tasks) for it, tasks in status.items()}


def get_available(model_dirs: list[Path]) -> list[int]:
    available = []
    for model_dir in model_dirs:
        for path in filter(lambda path: path.suffix == "", Path(model_dir).iterdir()):
            available.append(int(re.match("^iter_([0-9]+)$", path.name).group(1)))
    return available


def submit(name: str, model: dict, it: int, tasks: list[str]):
    task_alias = ROOT_EVAL if tasks == ALL_EVALS else " ".join(tasks)
    tasks = " ".join(tasks)
    path, = (model_dir for model_dir in model["model_dirs"]
             if Path(f"{model_dir}/iter_{it:07d}").exists())
    cmd = ["sbatch",
           f"--job-name=eval_{name}_{task_alias}_{it}",
           "scripts/evaluate.sbatch",
           str(path),
           str(it),
           model["tokens_per_iter"],
           name]
    env = {**os.environ,
           "LOGS_ROOT": CFG["logs_root"],
           "TOKENIZER": "alehc/swissai-tokenizer",
           "BOS": "true",
           "SIZE": str(model["size"]),
           "HF_TEMP_DIR": CFG["hf_temp_dir"],
           "TASKS": tasks}
    env.update(CFG["extra_env"])
    print("Launching", name, it, tasks, path)
    subprocess.run(cmd, env=env, stdout=subprocess.PIPE)


def submit_needed():
    running = get_running()
    for name, model in CFG["models"].items():
        status = get_evaluated(name)
        for it, tasks in running[name].items():
            if it in status:
                status[it] += tasks
            else:
                status[it] = tasks

        available = get_available(model["model_dirs"])
        for it in available:
            if (it - model["start_eval_from"]) % model["frequency"] == 0 and it >= model["start_eval_from"]:
                missing = sorted(set(ALL_EVALS) - set(status.get(it, [])))
                if len(missing) > 0:
                    if model["size"] < 70:
                        submit(name, model, it, missing)
                    else:
                        for task in missing:
                            submit(name, model, it, [task])


def update_hf_checkpoints():
    jobnames = get_running(as_jobname=True)
    for path in Path(CFG["hf_temp_dir"]).iterdir():
        if path.name in jobnames:  # Don't touch hf checkpoints of unfinished runs.
            continue
        name, it = re.match("^eval_(.*)_.*_([0-9]+)$", path.name).groups()
        it = int(it)
        dest = Path(CFG["hf_storage_dir"])/f"{name}_it{it}"
        if dest.exists():  # Checkpoint is already stored, probably from a job with different tasks that finished earlier.
            print("Removing", path)
            shutil.rmtree(path)
        else:
            print("Moving", path, "to", dest)
            shutil.move(path, dest)


def cleanup_hf_checkpoints():
    # Get model=>[(it, path)] mapping.
    stored = collections.defaultdict(list)
    for path in Path(CFG["hf_storage_dir"]).iterdir():
        rmatch = re.match("^(.*)_it([0-9]+)$", path.name)
        if rmatch is not None and rmatch.group(1) in CFG["models"]:
            name, it = rmatch.groups()
            stored[name].append((int(it), path))

    # Remove old checkpoints.
    keep = CFG["num_hf_checkpoints_to_keep"]
    for saved in stored.values():
        remove = sorted(saved, key=lambda t: t[0])[:-keep]
        for _, path in remove:
            print("Removing", path)
            shutil.rmtree(path)


def sync_wandb():
    print("Syncing wandb...")
    env = {**os.environ,
           "WANDB_SILENT": "true",
           "WANDB_RESUME": "allow",
           "WANDB_ENTITY": CFG["wandb_entity"],
           "WANDB_PROJECT": CFG["wandb_project"]}
    cmd = ["python3", "scripts/update_wandb.py", str(CFG["logs_root"])]
    for name in CFG["models"]:
        subprocess.run(cmd + [f"--name={name}"], env=env)


def main():
    submit_needed()
    update_hf_checkpoints()
    cleanup_hf_checkpoints()
    sync_wandb()


if __name__ == "__main__":
    with open("configs/automation.json") as f:
        CFG = json.load(f)
    with open("configs/tasks.json") as f:
        TASKS = json.load(f)
    ROOT_EVAL = TASKS["root"]
    ALL_EVALS = sorted([task for task in TASKS["groups"] if task != ROOT_EVAL])
    main()

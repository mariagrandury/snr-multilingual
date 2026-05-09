import pandas as pd
import numpy as np
from rich.table import Table
from rich.console import Console
from rich import box
from tqdm import tqdm

from snr.download.hf import pull_predictions_from_hf
from snr.dataloader import get_slice
from snr.metrics import decision_acc_fast
from snr.metrics import signal_to_noise_ratio
from snr.constants.signal import SNR_MODELS


DEFAULT_TASKS = [
    "minerva", "mmlu", "agi_eval", "arc_challenge", "arc_easy", "boolq",
    "csqa", "hellaswag", "openbookqa", "piqa", "socialiqa", "winogrande",
    "gsm8k", "mbpp", "mbppplus", "codex_humaneval", "codex_humanevalplus",
    "autobencher", "gsm_plus", "gsm_symbolic_main", "gsm_symbolic_p1",
    "gsm_symbolic_p2", "medmcqa", "minerva_math_500",
]


def compute_decision_accuracy(df, task, small_size, target_size="1B", target_step=69369):
    scores_small = get_slice(df, size=small_size, task=task)
    scores_target = get_slice(df, size=target_size, task=task, step=target_step)

    # Get the score at the highest step for each mix
    scores_small = scores_small.loc[scores_small.groupby("mix")["step"].idxmax()]
    if target_step is None:
        # No fixed target step → take the latest available step per mix.
        scores_target = scores_target.loc[scores_target.groupby("mix")["step"].idxmax()]

    decision_acc = decision_acc_fast(
        scores_small=scores_small.sort_values("model")["primary_score"],
        scores_target=scores_target.sort_values("model")["primary_score"],
    )

    return decision_acc


def compute_scaling_law_error(df, task, large_size):
    # Lazy-import: the ladder wrapper pulls in olmo-ladder ('scaling', 'fitting'),
    # which isn't always installed (e.g., the multilingual Apertus path doesn't need it).
    from snr.ladder_wrapper import run_ladder
    from snr.constants.ladder import LADDER_MODEL_NAMES

    if large_size == "7B":
        target_model = "peteish7"
    elif large_size == "13B":
        target_model = "peteish13-highlr"
    else:
        raise ValueError(large_size)

    _, _, error = run_ladder(df, task, train_models=LADDER_MODEL_NAMES, eval_models=[target_model])

    return error


def compute_snr_small_scale(df, task, small_size):
    scores_df = get_slice(df, size=small_size, task=task).sort_values("step")

    # Last 5 scores per mix. Half-trained models can have <5 ckpts on some mixes,
    # so we keep them as a list of (possibly unequal-length) 1-D arrays rather than
    # forcing a 2-D ndarray (which would raise on jagged input).
    scores_arrays = [
        np.array(lst[-5:])
        for lst in scores_df.groupby("mix")["primary_score"].apply(list)
    ]

    signal = [arr.mean() for arr in scores_arrays]
    noise = np.concatenate(scores_arrays)

    snr = signal_to_noise_ratio(signal, noise)

    return snr


def compute_snr_large_scale(df, task, large_size):
    if large_size == "1B":
        signal_models = "olmo2_1b"
        noise_model = "peteish1"
    elif large_size == "7B":
        signal_models = "olmo2_7b"
        noise_model = "peteish7"
    elif large_size == "13B":
        signal_models = "olmo2_13b"
        noise_model = "peteish13-highlr"
    elif large_size == "32B":
        signal_models = "olmo2_32b"
        noise_model = "peteish32"
    else:
        raise ValueError(large_size)

    signal_models = SNR_MODELS[signal_models]["models"]
    noise_df = get_slice(df, model=noise_model, task=task)

    signal_df = df[df["model_path"].isin(signal_models) & (df["task"] == task)]

    signal = list(signal_df["primary_score"])
    noise = list(noise_df.sort_values("step")["primary_score"])[-30:]

    snr = signal_to_noise_ratio(signal, noise)

    return snr


def _safe(fn, *args, **kwargs):
    """Run fn → NaN on failure, so missing-data cells don't kill the loop."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return float("nan")


def calculate_results(df, tasks, small_sizes, large_sizes_scaling, large_sizes_snr,
                     target_size="1B", target_step=69369):
    results = []
    for task in tqdm(tasks, desc="Running analysis"):
        row = {"Task": task,
               "Decision Accuracy": {},
               "Scaling Law Error": {},
               "SNR": {}}

        for small_size in small_sizes:
            row["Decision Accuracy"][small_size] = _safe(
                compute_decision_accuracy, df, task, small_size,
                target_size=target_size, target_step=target_step,
            )

        for large_size in large_sizes_scaling:
            row["Scaling Law Error"][large_size] = _safe(
                compute_scaling_law_error, df, task, large_size,
            )

        for size in small_sizes:
            row["SNR"][size] = _safe(compute_snr_small_scale, df, task, size)
        for large_size in large_sizes_snr:
            row["SNR"][large_size] = _safe(compute_snr_large_scale, df, task, large_size)

        results.append(row)
    return results


def render_table(results, small_sizes, large_sizes_scaling, large_sizes_snr):
    table = Table(title="Signal-and-Noise Analysis by Task", box=box.ASCII)

    # Add header
    decision_acc_headers = [f"{size}" for size in small_sizes]
    scaling_law_headers = [f"{size}" for size in large_sizes_scaling]
    snr_headers = [f"{size}" for size in list(small_sizes) + list(large_sizes_snr)]
    table.add_column("Task", justify="left")
    for size in decision_acc_headers:
        table.add_column(f"Decision\nAcc\n{size}", justify="left")
    for size in scaling_law_headers:
        table.add_column(f"Scaling\nLaw Err\n{size}", justify="left")
    for size in snr_headers:
        table.add_column(f"SNR\n{size}", justify="left")

    # Sort results alphabetically by task name
    sorted_results = sorted(results, key=lambda row: str(row["Task"]).lower())

    # Add rows
    for row in sorted_results:
        row_values = [str(row["Task"])]

        for size in decision_acc_headers:
            val = row["Decision Accuracy"].get(size, "")
            if isinstance(val, float) and np.isfinite(val):
                row_values.append(f"{int(round(val * 100))}%")
            else:
                row_values.append("-" if isinstance(val, float) else str(val))

        for size in scaling_law_headers:
            val = row["Scaling Law Error"].get(size, "")
            if isinstance(val, float) and np.isfinite(val):
                row_values.append(f"{val * 100:.1f}%")
            else:
                row_values.append("-" if isinstance(val, float) else str(val))

        for size in snr_headers:
            val = row["SNR"].get(size, "")
            if isinstance(val, float) and np.isfinite(val):
                row_values.append(f"{val:.1f}")
            else:
                row_values.append("-" if isinstance(val, float) else str(val))
        table.add_row(*row_values)

    console = Console()
    console.print(table)


def main(
    df=None,
    tasks=None,
    small_sizes=("150M", "300M", "750M"),
    large_sizes_scaling=("7B", "13B"),
    large_sizes_snr=("1B", "7B", "13B", "32B"),
    target_size="1B",
    target_step=69369,
):
    """Drive the analysis. Defaults reproduce the DataDecide pipeline; pass a
    pre-loaded df + custom sizes to run on other model families (e.g. Apertus)."""
    if df is None:
        local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name="core")
        df = pd.read_parquet(local_path)
    if tasks is None:
        tasks = DEFAULT_TASKS

    small_sizes = list(small_sizes)
    large_sizes_scaling = list(large_sizes_scaling)
    large_sizes_snr = list(large_sizes_snr)

    results = calculate_results(
        df, tasks, small_sizes, large_sizes_scaling, large_sizes_snr,
        target_size=target_size, target_step=target_step,
    )
    render_table(results, small_sizes, large_sizes_scaling, large_sizes_snr)
    return results


if __name__ == "__main__":
    main()

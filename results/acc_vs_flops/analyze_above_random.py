"""Above-random analysis for the per-(model, ckpt, task) eval scores
that feed the `acc_vs_flops/` curves.

For each task we know the number of answer options *n* (taken from the
family metadata in ``results/benchmark_creation/analyze.py``); the random
baseline is ``1 / n``. This script computes, for every task in the 12
Apertus pretraining grids:

  - random_baseline                (= 1 / n_options)
  - best_score                     max primary_score across all 12 models
                                   × all available checkpoints
  - mean_final_score               mean primary_score across the 12 models
                                   at each model's last available
                                   checkpoint
  - n_final_above_random           how many of those 12 final-step scores
                                   exceed the random baseline (out of 12)
  - frac_ckpts_above_random        fraction of *all* (model, ckpt) score
                                   points above the random baseline
  - lift_final                     mean_final_score − random_baseline
  - significant                    one-sample t-test of the 12 final-step
                                   scores against the random baseline,
                                   p < 0.05 (two-sided)

Outputs (next to the ``acc_vs_flops`` curves these analyse):

  - per_task_above_random.csv      one row per task
  - per_family_above_random.csv    one row per family (mean / fraction)
  - above_random.png               per-family strip plot of mean-final
                                   task scores with the family random
                                   baseline marked
  - above_random_bars.png          per-task lift bar chart, family-grouped
  - above_random_heatmap.png       task × size heatmap of mean final
                                   score (cell text = score; cell color
                                   = score − random_baseline, diverging
                                   colormap centered at 0 so each row's
                                   white corresponds to its own baseline)

Random baseline for ``truthfulqa`` (mc1 split) is the standard reported
value 0.232; the per-item option count is variable (4–13) but the
weighted average is ~4.3 across the canonical English set, and the
multilingual variants follow the same schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multilingual.analyze_snr_variants import assign_language, benchmark_family  # noqa: E402
from multilingual.smooth_subtasks import _is_language_aggregate  # noqa: E402
from snr.download.apertus import load_apertus_eval_results  # noqa: E402

HERE = Path(__file__).resolve().parent

# n_options per family. Most come from ``results/benchmark_creation/analyze.py``;
# truthfulqa is the only family in this set that has a non-integer effective
# n_options (mc1 is variable per item — the canonical English random baseline
# of 0.232 corresponds to ~4.31 options on average, which we use uniformly
# for the multilingual variants since they share the schema).
FAMILY_N_OPTIONS: dict[str, float] = {
    "arc": 4,  # variable (3–5, dataset-dependent)
    "belebele": 4,
    "global_mmlu": 4,
    "global_mmlu_full": 4,
    "global_piqa_completions": 2,
    "hellaswag": 4,
    "multiblimp": 2,
    "paws": 2,
    "truthfulqa": 4,  # variable (mc1 and mc2 differ per question)
    "truthfulqa_mc1": 4,
    "xcopa": 2,
    "xnli": 3,
    "xstorycloze": 2,
    "xwinograd": 2,
}


def _family_of(task: str) -> str:
    return benchmark_family(task)


def _baseline(family: str) -> float:
    return 1.0 / FAMILY_N_OPTIONS[family]


def _per_task_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate (model, ckpt) score points to one row per task. Filter
    to per-language aggregate tasks (e.g., ``global_mmlu_full_ar`` but
    not ``global_mmlu_full_ar_anatomy``) so the rows line up with the
    panels of the ``acc_vs_flops/per_benchmark`` grids."""
    df = df.copy()
    df["family"] = df["task"].map(_family_of)
    df = df[df["family"].isin(FAMILY_N_OPTIONS)].copy()
    df = df[[_is_language_aggregate(t, f) for t, f in zip(df["task"], df["family"])]].copy()
    df["language"] = df["task"].map(assign_language)
    df["random_baseline"] = df["family"].map(_baseline)
    df["above_random_ckpt"] = df["primary_score"] > df["random_baseline"]

    # Last checkpoint per (model, task).
    last_idx = df.groupby(["model", "task"])["step"].idxmax()
    last = df.loc[last_idx, ["model", "task", "primary_score"]].rename(
        columns={"primary_score": "final_score"}
    )

    rows: list[dict] = []
    for task, sub in df.groupby("task"):
        family = sub["family"].iloc[0]
        rb = sub["random_baseline"].iloc[0]
        finals = last[last["task"] == task]["final_score"].to_numpy()
        finals = finals[~np.isnan(finals)]
        # One-sample t-test: are the 12 final-step scores above random?
        if len(finals) >= 2 and np.std(finals) > 0:
            t, p = stats.ttest_1samp(finals, rb, alternative="greater")
            t = float(t)
            p = float(p)
        else:
            t = float("nan")
            p = float("nan")
        rows.append(
            {
                "task": task,
                "family": family,
                "language": sub["language"].iloc[0],
                "n_options": FAMILY_N_OPTIONS[family],
                "random_baseline": rb,
                "best_score": float(sub["primary_score"].max()),
                "mean_final_score": float(np.mean(finals)) if len(finals) else float("nan"),
                "std_final_score": (
                    float(np.std(finals, ddof=1)) if len(finals) >= 2 else float("nan")
                ),
                "n_models_final": int(len(finals)),
                "n_final_above_random": int(np.sum(finals > rb)),
                "frac_ckpts_above_random": float(sub["above_random_ckpt"].mean()),
                "lift_final": float(np.mean(finals) - rb) if len(finals) else float("nan"),
                "t_stat": t,
                "p_value_one_sided_greater": p,
                "significant_above_random": bool(p < 0.05) if not np.isnan(p) else False,
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "language"]).reset_index(drop=True)


def _per_family(per_task: pd.DataFrame) -> pd.DataFrame:
    pt = per_task.copy()
    pt["above_at_final"] = pt["mean_final_score"] > pt["random_baseline"]
    g = pt.groupby("family")
    out = pd.DataFrame(
        {
            "n_tasks": g.size(),
            "n_options": g["n_options"].first(),
            "random_baseline": g["random_baseline"].first(),
            "mean_final_median": g["mean_final_score"].median(),
            "mean_final_mean": g["mean_final_score"].mean(),
            "best_score_max": g["best_score"].max(),
            "lift_final_median": g["lift_final"].median(),
            "n_tasks_significant_above_random": g["significant_above_random"].sum(),
            "frac_tasks_above_random_at_final": g["above_at_final"].mean(),
        }
    ).reset_index()
    out = out.sort_values(["random_baseline", "mean_final_median"], ascending=[True, False])
    return out


# --- plotting ---------------------------------------------------------------


def _strip_plot(per_task: pd.DataFrame, per_family: pd.DataFrame, out_path: Path) -> None:
    """Per-family strip plot of mean-final-score across tasks; vertical
    family-specific random baseline; dots colored by above/below."""
    family_order = list(per_family["family"])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(family_order) + 1.5))
    rng = np.random.default_rng(0)

    for i, fam in enumerate(family_order):
        sub = per_task[per_task["family"] == fam]
        rb = float(per_family.loc[per_family["family"] == fam, "random_baseline"].iloc[0])
        scores = sub["mean_final_score"].to_numpy()
        sigs = sub["significant_above_random"].to_numpy()
        if len(scores) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(scores))
        # Color: green if significantly above random, gray if above but n.s.,
        # red if at/below random.
        colors: list[str] = []
        for s, sig in zip(scores, sigs):
            if s <= rb:
                colors.append("#d62728")  # red
            elif sig:
                colors.append("#2ca02c")  # green
            else:
                colors.append("#7f7f7f")  # gray (above but n.s.)
        ax.scatter(
            scores,
            np.full_like(scores, i, dtype=float) + jitter,
            c=colors,
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        # Random baseline tick at this family's row.
        ax.plot([rb, rb], [i - 0.40, i + 0.40], color="black", lw=2.0, alpha=0.85)
        # Family median, dashed.
        med = float(np.nanmedian(scores))
        ax.plot([med, med], [i - 0.30, i + 0.30], color="#1f77b4", lw=1.5, ls="--")

    ax.set_yticks(range(len(family_order)))
    ylabels = []
    for fam in family_order:
        row = per_family[per_family["family"] == fam].iloc[0]
        ylabels.append(
            f"{fam}\nn={int(row['n_tasks'])}, "
            f"baseline={row['random_baseline']:.3f}\n"
            f"sig>chance: {int(row['n_tasks_significant_above_random'])}/{int(row['n_tasks'])}"
        )
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean primary_score across the 12 final-step Apertus models")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        "Above-random check: per-task mean score at each model's last "
        "checkpoint, vs the family's 1/n_options baseline\n"
        "● green = significantly above random (one-sample t, p<0.05); "
        "● gray = above but n.s.; ● red = at/below random; "
        "vertical bar = random baseline; dashed = family median",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _lift_bars(per_task: pd.DataFrame, out_path: Path) -> None:
    """Lift = mean_final_score - random_baseline, arranged as a tasks × 1
    bar chart, sorted within each family by lift."""
    df = per_task.sort_values(["family", "lift_final"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.22 * len(df) + 1.5))
    y = np.arange(len(df))
    colors = [
        "#2ca02c" if (lift > 0 and sig) else "#7f7f7f" if lift > 0 else "#d62728"
        for lift, sig in zip(df["lift_final"], df["significant_above_random"])
    ]
    ax.barh(y, df["lift_final"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["task"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("mean_final_score − random_baseline (lift over chance)")
    ax.set_title(
        "Per-task lift over the 1/n_options random baseline at each "
        "model's final checkpoint (mean across the 12 Apertus models)",
        fontsize=10,
    )
    # Family separators.
    fam_changes = np.where(df["family"].values[1:] != df["family"].values[:-1])[0] + 1
    for fc in fam_changes:
        ax.axhline(fc - 0.5, color="black", lw=0.4, alpha=0.4)
    # Family labels on the right margin.
    fam_centers = []
    for fam, sub in df.groupby("family", sort=False):
        idx = sub.index.to_numpy()
        fam_centers.append((fam, float(np.mean(idx))))
    xmax = max(df["lift_final"].max() * 1.05, 0.05)
    for fam, c in fam_centers:
        ax.text(xmax, c, fam, fontsize=8, va="center", ha="left", style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_SIZE_ORDER = ["175M", "350M", "600M", "1B"]


def _per_size_final_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Mean primary_score across mixes at each model's final checkpoint,
    one row per (task, size). Used to fill the heatmap matrix."""
    df = df.copy()
    df["family"] = df["task"].map(_family_of)
    df = df[df["family"].isin(FAMILY_N_OPTIONS)].copy()
    df = df[[_is_language_aggregate(t, f) for t, f in zip(df["task"], df["family"])]].copy()
    # Last ckpt per (model, task), then average across the 3 mixes per (size, task).
    last_idx = df.groupby(["model", "task"])["step"].idxmax()
    last = df.loc[last_idx, ["task", "size", "primary_score"]]
    return last.groupby(["task", "size"])["primary_score"].mean().reset_index()


def _size_heatmap(df: pd.DataFrame, per_task: pd.DataFrame, out_path: Path) -> None:
    """Task × size heatmap. Cell text = mean final score across the 3
    mixes for that size. Cell color = score − random_baseline (one
    diverging colormap; white = 0 lift, i.e. at-baseline). The AVG
    column is the mean across the 4 sizes."""
    long = _per_size_final_mean(df)
    wide = long.pivot(index="task", columns="size", values="primary_score")
    wide = wide.reindex(columns=_SIZE_ORDER)

    # Order rows by family then language, like the bar plot.
    task_order = per_task.sort_values(["family", "language", "task"])["task"].tolist()
    wide = wide.reindex(task_order)

    # AVG = mean across the 4 sizes (skip NaNs to be safe; all 4 should be filled).
    wide["AVG"] = wide[_SIZE_ORDER].mean(axis=1, skipna=True)
    cols = _SIZE_ORDER + ["AVG"]

    rb = per_task.set_index("task")["random_baseline"].reindex(task_order).to_numpy()
    fams = per_task.set_index("task")["family"].reindex(task_order).to_numpy()

    score = wide[cols].to_numpy()
    lift = score - rb[:, None]  # broadcast: each row's baseline subtracted

    # Symmetric color limits → white at 0 lift = at-baseline for that row.
    # Clip vmax to the 90th percentile of |lift| so a few extreme rows
    # (multiblimp at +0.4) don't squash the gradient for the rest.
    vmax_full = float(np.nanmax(np.abs(lift)))
    vmax = float(np.nanpercentile(np.abs(lift), 90))
    vmax = max(vmax, 0.05)  # avoid a near-zero scale if everything is small
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(7, 0.22 * len(task_order) + 2.0))
    im = ax.imshow(
        lift, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order, fontsize=6)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel("")

    # Cell annotations: actual mean score (not the lift).
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            v = score[i, j]
            if np.isnan(v):
                continue
            # Pick black/white text based on |lift| relative to vmax for legibility.
            text_color = "black" if abs(lift[i, j]) < 0.55 * vmax else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6, color=text_color)

    # Family separators on the y-axis.
    fam_changes = np.where(fams[1:] != fams[:-1])[0] + 1
    for fc in fam_changes:
        ax.axhline(fc - 0.5, color="black", lw=0.5, alpha=0.7)
    # Bracket between 1B and AVG for visual separation.
    ax.axvline(len(_SIZE_ORDER) - 0.5, color="black", lw=0.8, alpha=0.7)

    # Family labels on the right margin (centered on each family's rows).
    for fam in pd.unique(fams):
        idx = np.where(fams == fam)[0]
        c = float(np.mean(idx))
        ax.text(
            len(cols) - 0.4,
            c,
            f"  {fam}\n  (b={1.0 / FAMILY_N_OPTIONS[fam]:.3f})",
            fontsize=7,
            va="center",
            ha="left",
            style="italic",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.18, extend="both")
    cbar.set_label(
        f"mean final score − random baseline "
        f"(white = at chance; saturated at ±{vmax:.2f}; max |lift| = {vmax_full:.2f})",
        fontsize=8,
    )

    ax.set_title(
        "Task × size heatmap of mean final-checkpoint score "
        "(mean across the 3 mixes)\n"
        "Cell text = score; color = score − random_baseline "
        "(diverging, 0 white = at chance for that row's family)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_apertus_eval_results()
    print(
        f"Loaded {len(df):,} rows | {df['model'].nunique()} models | "
        f"{df['task'].nunique()} tasks total"
    )

    per_task = _per_task_stats(df)
    print(
        f"  → {len(per_task)} tasks across {per_task['family'].nunique()} "
        "benchmark families with a defined random baseline"
    )
    skipped = sorted(set(df["task"].map(_family_of)) - set(FAMILY_N_OPTIONS))
    if skipped:
        print(
            f"  Skipped {len(skipped)} families without a defined n_options "
            f"(e.g. {skipped[:5]}…)"
        )

    per_family = _per_family(per_task)

    csv_task = HERE / "per_task_above_random.csv"
    csv_fam = HERE / "per_family_above_random.csv"
    per_task.to_csv(csv_task, index=False)
    per_family.to_csv(csv_fam, index=False)
    print(f"Wrote {csv_task.name}")
    print(f"Wrote {csv_fam.name}")

    _strip_plot(per_task, per_family, HERE / "above_random.png")
    print("Wrote above_random.png")

    _lift_bars(per_task, HERE / "above_random_bars.png")
    print("Wrote above_random_bars.png")

    _size_heatmap(df, per_task, HERE / "above_random_heatmap.png")
    print("Wrote above_random_heatmap.png")

    print("\nPer-family summary:")
    cols = [
        "family",
        "n_tasks",
        "n_options",
        "random_baseline",
        "mean_final_median",
        "lift_final_median",
        "n_tasks_significant_above_random",
        "frac_tasks_above_random_at_final",
    ]
    with pd.option_context("display.width", 160, "display.max_colwidth", 36):
        print(per_family[cols].to_string(index=False))

    n_tasks = len(per_task)
    n_sig = int(per_task["significant_above_random"].sum())
    n_above = int((per_task["mean_final_score"] > per_task["random_baseline"]).sum())
    print(
        f"\n{n_above}/{n_tasks} tasks have mean-final-score above the "
        f"family random baseline; {n_sig}/{n_tasks} are significantly "
        "above random (one-sample t-test, p<0.05)."
    )


if __name__ == "__main__":
    main()

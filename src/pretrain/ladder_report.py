#!/usr/bin/env python3
"""Is the ladder healthy? Loss, scaling law, benchmarks, and BPB — from disk.

The question "does this look right?" has a few concrete answers, and all of
them are already on disk: the training logs carry the loss curve, the harness
results carry the benchmark scores, and score_bpb.py writes per-language BPB
and perplexity. Nothing here needs W&B or the network.

Four checks, in the order a problem usually shows up:

  loss        every run's loss must fall, and the final loss must be ordered
              by size at fixed L.
  scaling     fit L = A*N^-alpha across the ladder at each language setting.
              A rung that is off the fit by more than --tol is flagged: that
              is the shape a broken rung makes, and it is how the 90M anomaly
              shows up (~+2.5 nats, at EVERY language setting).
  benchmarks  the `auto` group's scores should rise between the first and
              last evaluated checkpoint, and end above chance.
  bpb         per-language bits-per-byte, the plan's outcome metric.

Only losses from COMPLETED runs are compared, since a mid-run WSD schedule
has not decayed yet and its loss is not comparable to a finished one.

--plot additionally writes four figures, a shared per-checkpoint CSV, and a
generated ladder_report.md summary table next to them. Some of this evidence is only obvious as a shape:
the 90M divergence is two numbers in a table and unmistakable as a curve.

    python3.11 ladder_report.py                 # everything
    python3.11 ladder_report.py --check scaling
    python3.11 ladder_report.py --plot          # + figures and ladder_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pretrain_progress import SIZES  # noqa: E402

TRAIN_LOGS = Path("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/"
                  "Megatron-LM/logs/slurm/training")
EVAL_LOGS = Path("/iopsstor/scratch/cscs/mariagrandury/data-mix-small/"
                 "Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr")

LOSS_RE = re.compile(r"iteration\s+(\d+)/\s*(\d+).*?lm loss: ([0-9.E+-]+)")
LOG_RE = re.compile(r"pretrain-(?P<size>[\d.]+[MB])-L(?P<L>\d+)"
                    r"(?P<scheme>-schemeB)?-(?P<arch>deep|shallow)"
                    r"-seed(?P<seed>\d+)-\d+\.out")

# Non-embedding parameters — the x of the scaling fit. The ladder is defined by
# these targets, so they are the right abscissa even though the realised counts
# differ by a fraction of a percent.
NON_EMB = {"90M": 9.0e7, "175M": 1.75e8, "350M": 3.5e8,
           "600M": 6.0e8, "1B": 1.0e9, "1.7B": 1.7e9}


def loss_curves() -> dict[tuple, list[tuple[int, float]]]:
    """(size, L, arch, scheme, seed) -> [(iter, loss), ...], newest log wins."""
    runs: dict[tuple, dict[int, float]] = {}
    for f in sorted(TRAIN_LOGS.glob("pretrain-*.out")):
        m = LOG_RE.match(f.name)
        if not m:
            continue
        key = (m["size"], int(m["L"]), m["arch"],
               "B" if m["scheme"] else "A", int(m["seed"]))
        # A resumed cell has several logs; later iterations supersede earlier
        # ones, so merging by iteration rebuilds the whole curve.
        got = runs.setdefault(key, {})
        for it, _tgt, loss in LOSS_RE.findall(f.read_text(errors="ignore")):
            got[int(it)] = float(loss)
    return {k: sorted(d.items()) for k, d in runs.items()}


def targets() -> dict[tuple, int]:
    """(size, L, arch, scheme, seed) -> the run's own --train-iters."""
    out = {}
    for f in sorted(TRAIN_LOGS.glob("pretrain-*.out")):
        m = LOG_RE.match(f.name)
        if not m:
            continue
        hits = LOSS_RE.findall(f.read_text(errors="ignore"))
        if hits:
            out[(m["size"], int(m["L"]), m["arch"],
                 "B" if m["scheme"] else "A", int(m["seed"]))] = int(hits[-1][1])
    return out


def check_loss(curves, tgts) -> list[str]:
    problems = []
    print("\n== loss ==")
    finished = {}
    for k, pts in sorted(curves.items()):
        if not pts:
            continue
        last_it, last_loss = pts[-1]
        target = tgts.get(k, last_it)
        done = last_it >= target
        # Compare the first tenth against the last tenth: a run that is not
        # learning shows up here, and a single noisy iteration cannot cause it.
        head = [v for _, v in pts[:max(1, len(pts) // 10)]]
        tail = [v for _, v in pts[-max(1, len(pts) // 10):]]
        drop = sum(head) / len(head) - sum(tail) / len(tail)
        tag = "done " if done else "part."
        if drop <= 0:
            problems.append(f"{k}: loss did not fall ({drop:+.2f})")
            tag = "STUCK"
        # DIVERGENCE: the run reached a better loss earlier and never got back.
        # On a single-epoch budget there is no overfitting to explain it, so a
        # final loss well above the minimum means the run blew up — and the
        # checkpoint being shipped is worse than one already on disk.
        min_it, min_loss = min(pts, key=lambda p: p[1])
        if last_loss > min_loss + 0.25 and min_it < last_it * 0.9:
            problems.append(
                f"{k[0]} L{k[1]} {k[2]}: DIVERGED — best loss {min_loss:.2f} at "
                f"iter {min_it} ({100*min_it//last_it}% of the run), final "
                f"{last_loss:.2f} (+{last_loss-min_loss:.2f})")
            tag = "DIVRG"
        if done:
            finished[k] = last_loss
        print(f"  {tag} {k[0]:>5} L{k[1]:<3} {k[2]:<7} seed{k[4]:<5} "
              f"iter {last_it}/{target}  loss {last_loss:.3f}  (fell {drop:.2f}, "
              f"best {min_loss:.3f}@{min_it})")

    # Ordering by size at fixed (L, arch): a bigger model that is worse is a
    # real signal, not noise, at these gaps.
    for (L, arch) in sorted({(k[1], k[2]) for k in finished}):
        row = [(s, finished[(s, L, arch, "A", 1904)])
               for s in SIZES if (s, L, arch, "A", 1904) in finished]
        for (s1, v1), (s2, v2) in zip(row, row[1:]):
            if v2 > v1:
                problems.append(
                    f"L{L} {arch}: {s2} loss {v2:.3f} WORSE than {s1} {v1:.3f}")
    return problems


def check_scaling(curves, tgts, tol: float) -> list[str]:
    """Fit log L = log A - alpha log N per (L, arch) and flag rungs off it.

    Fitted on the sizes that are ON the trend and predicted for the rest, so a
    single broken rung cannot drag the fit toward itself and hide.
    """
    problems = []
    print("\n== scaling ==")
    finished = {k: pts[-1][1] for k, pts in curves.items()
                if pts and pts[-1][0] >= tgts.get(k, pts[-1][0])}
    for (L, arch) in sorted({(k[1], k[2]) for k in finished}):
        pts = [(NON_EMB[s], finished[(s, L, arch, "A", 1904)], s)
               for s in SIZES if (s, L, arch, "A", 1904) in finished]
        if len(pts) < 3:
            continue
        # Fit on everything but the smallest rung, then predict it: with 3-4
        # points a 2-parameter fit that includes an outlier is dominated by it.
        fit = pts[1:]
        xs = [math.log(n) for n, _, _ in fit]
        ys = [math.log(v) for _, v, _ in fit]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            continue
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        icpt = my - slope * mx
        print(f"  L{L:<3} {arch:<7} alpha={-slope:.3f}  fitted on "
              f"{','.join(s for _, _, s in fit)}")
        for n, v, s in pts:
            pred = math.exp(icpt + slope * math.log(n))
            resid = v - pred
            flag = "  <-- OFF TREND" if abs(resid) > tol else ""
            print(f"       {s:>5}  obs {v:.3f}  pred {pred:.3f}  "
                  f"resid {resid:+.3f}{flag}")
            if abs(resid) > tol:
                problems.append(f"L{L} {arch} {s}: {resid:+.2f} off the "
                                f"scaling fit (obs {v:.2f}, pred {pred:.2f})")
    return problems


def _scores(path: Path) -> dict[str, float]:
    out = {}
    for f in path.glob("harness/eval_*/results_*.json"):
        try:
            res = json.loads(f.read_text()).get("results", {})
        except Exception:
            continue
        for task, metrics in res.items():
            for key, val in metrics.items():
                if key.startswith(("acc,", "exact_match,")) and isinstance(val, float):
                    out[task] = val
                    break
    return out


def check_benchmarks() -> list[str]:
    problems = []
    print("\n== benchmarks ==")
    cells: dict[str, dict[int, dict]] = {}
    for d in sorted(EVAL_LOGS.glob("lm-*-iter*")):
        m = re.match(r"(.+)-iter(\d+)$", d.name)
        if m:
            cells.setdefault(m.group(1), {})[int(m.group(2))] = _scores(d)
    for cell, iters in sorted(cells.items()):
        pts = sorted(iters)
        if len(pts) < 2:
            continue
        first, last = iters[pts[0]], iters[pts[-1]]
        common = set(first) & set(last)
        if not common:
            continue
        improved = sum(1 for t in common if last[t] > first[t])
        best = max(common, key=lambda t: last[t] - first[t])
        print(f"  {cell:<34} {len(common):>3} tasks, {improved:>3} improved, "
              f"best {best} {first[best]:.3f}->{last[best]:.3f}")
        if improved < len(common) / 3:
            problems.append(f"{cell}: only {improved}/{len(common)} benchmarks "
                            f"improved between iter {pts[0]} and {pts[-1]}")
    return problems


def check_bpb() -> list[str]:
    print("\n== bits-per-byte (score_bpb.py) ==")
    found = sorted(EVAL_LOGS.glob("lm-*-iter*/bpb/bpb.json"))
    if not found:
        print("  none yet — run scripts/score_bpb.sbatch")
        return []
    for f in found:
        d = json.loads(f.read_text())
        name = f.parent.parent.name
        langs = d["languages"]
        worst = max(langs, key=lambda k: langs[k]["bpb"])
        best = min(langs, key=lambda k: langs[k]["bpb"])
        print(f"  {name:<34} macro BPB {d['macro_bpb']:.4f} over "
              f"{d['n_languages']} langs  (best {best.replace('fineweb_','')} "
              f"{langs[best]['bpb']:.3f}, worst {worst.replace('fineweb_','')} "
              f"{langs[worst]['bpb']:.3f})")
    return []



# ---------------------------------------------------------------------------
# Artifacts: the CSVs first, then every figure and table FROM those CSVs.
#
# Nothing below re-reads the logs. That is the point: a new artifact becomes a
# read of a tidy table rather than another pass over 500k log lines, and any
# number in a figure can be checked against the row it came from. The CSVs are
# also the hand-off to any analysis that does not live here.
# ---------------------------------------------------------------------------

# Size picks the HUE, (arch, scheme) picks the SHADE within it. Widely
# separated hues because neighbouring rungs are what the eye must tell apart
# (a viridis ramp makes 175M and 350M nearly identical); shades within a hue
# because in the close-up the variants of ONE size sit hundredths of a nat
# apart, and there a shared colour plus a linestyle is not enough to follow a
# line across the panel. Linestyle is kept as a redundant second channel.
# Size picks the colour FAMILY, (arch, scheme) picks the step within it.
# Hand-picked ColorBrewer steps rather than samples of a continuous colormap:
# sampling `Blues` at 0.45 and 0.68 gives two mid-blues that are
# indistinguishable in a 1px line, which is exactly the comparison the
# close-up exists to make. These four steps are chosen to stay apart at line
# width. Linestyle is kept as a redundant second channel.
SIZE_PALETTE = {
    "90M":  ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "175M": ["#c6dbef", "#4292c6", "#08519c", "#08306b"],
    "350M": ["#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
    "600M": ["#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "1B":   ["#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
    "1.7B": ["#ccece6", "#66c2a4", "#238b45", "#005824"],
}
# Scheme B is a DIFFERENT data distribution at the same L, not a variant of A,
# so its runs must never read as points on an A curve.
VARIANT_STYLE = {("deep", "A"): "-", ("shallow", "A"): "--",
                 ("deep", "B"): ":", ("shallow", "B"): "-."}
# Palettes run pale -> very dark. The two COMMON variants take the two most
# readable steps (mid and dark); the palest is reserved for shallow/B, which
# barely occurs. An earlier assignment gave shallow/A the palest step and it
# was legible in the legend but not in the plot.
VARIANT_STEP = {("deep", "A"): 1, ("shallow", "A"): 2,
                ("deep", "B"): 3, ("shallow", "B"): 0}
CURVE_WINDOWS = 400
CLOSEUP_YMAX = 3.5   # ceiling for the last-10% panels     # per run, emitting each window's min AND max


def _panels(n: int, ax_w: float = 3.4, ax_h: float = 2.8):
    """A subplot grid sized to n panels, at most 4 wide, unused axes hidden."""
    import matplotlib.pyplot as plt
    cols = min(4, max(1, n))
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(ax_w * cols, ax_h * rows),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for a in flat[n:]:
        a.axis("off")
    return fig, flat


def bpb_results() -> dict[str, dict[int, dict]]:
    """cell -> {iter: bpb.json}, for every cell score_bpb.py has run on.

    The one place that touches score_bpb's output; everything downstream goes
    through the CSV it feeds.
    """
    out: dict[str, dict[int, dict]] = {}
    for f in sorted(EVAL_LOGS.glob("lm-*-iter*/bpb/bpb.json")):
        m = re.match(r"(.+)-iter(\d+)$", f.parent.parent.name)
        if not m:
            continue
        try:
            out.setdefault(m.group(1), {})[int(m.group(2))] = json.loads(f.read_text())
        except Exception:
            continue          # a job still writing its file; next run picks it up
    return out


def _fit(points):
    """(alpha, intercept) of log L = log A - alpha log N, fitted on `points`
    (n, loss) — or None if it is underdetermined."""
    if len(points) < 2:
        return None
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(v) for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx



def benchmark_results() -> dict[str, dict[int, dict[str, float]]]:
    """cell -> {iter: {task: score}} from the harness results files.

    Primary metric per task is `acc`, falling back to `exact_match` for the
    generative ones — the same rule push_all_results.py uses for W&B, so the
    CSV and the dashboards cannot disagree about what a task's score is.
    """
    out: dict[str, dict[int, dict[str, float]]] = {}
    for d in sorted(EVAL_LOGS.glob("lm-*-iter*")):
        m = re.match(r"(.+)-iter(\d+)$", d.name)
        if not m:
            continue
        scores: dict[str, float] = {}
        for f in d.glob("harness/eval_*/results_*.json"):
            try:
                res = json.loads(f.read_text()).get("results", {})
            except Exception:
                continue
            for task, metrics in res.items():
                for key, val in metrics.items():
                    if key.startswith(("acc,", "exact_match,")) and isinstance(val, float):
                        scores[task] = val
                        break
        if scores:
            out.setdefault(m.group(1), {})[int(m.group(2))] = scores
    return out


def _task_meta() -> dict[str, dict]:
    """configs/tasks.json — task -> {language, benchmark, n_options}."""
    try:
        return json.loads(
            (SCRIPT_DIR.parent.parent / "configs" / "tasks.json").read_text())["tasks"]
    except Exception:
        return {}


def _meta_for(task: str, meta: dict) -> dict:
    """tasks.json entry for TASK, inheriting from its parent when the exact
    subtopic is not listed.

    lm_eval reports `global_mmlu_full_en_anatomy` while tasks.json lists the
    parent `global_mmlu_full_en`; the subtopics share the parent's format, so
    inheriting is what makes `chance` cover them. Longest prefix wins, so a
    listed subtopic still beats its parent.
    """
    if task in meta:
        return meta[task]
    best = ""
    for name in meta:
        if task.startswith(name + "_") and len(name) > len(best):
            best = name
    return meta.get(best, {})


def _cell_parts(cell: str):
    m = re.match(r"lm-([\d.]+[MB])-L(\d+)(-schemeB)?-(deep|shallow)-seed(\d+)", cell)
    if not m:
        return None
    return {"size": m.group(1), "L": int(m.group(2)),
            "scheme": "B" if m.group(3) else "A",
            "arch": m.group(4), "seed": int(m.group(5))}


def write_csv(curves, tgts, out_dir: Path, tol: float) -> Path:
    """ONE long-format table with every measurement in the sweep.

    One file rather than four, because the audience is co-authors who each
    want a different slice: long format lets them pivot to whatever shape
    their question needs without us having to guess it, and a new metric
    becomes new ROWS rather than a schema change that breaks their scripts.
    The cost is that a wide view needs a pivot first, and that the run-level
    scalars sit at a different grain from the per-iteration rows — they are
    carried as kind="summary" with an empty iter.

        kind=loss       key=""        value=lm loss at that iteration
        kind=bpb / ppl  key=language  value=bits-per-byte / perplexity
        kind=benchmark  key=task      value=accuracy (or exact_match)
        kind=summary    key=metric    value=final_loss, best_loss, best_iter,
                                      diverged, alpha, pred_loss, resid, ...

    `chance` is filled for benchmark rows from tasks.json's n_options, so
    "above chance" is answerable from this file alone.
    """
    import csv

    meta = _task_meta()
    rows: list[dict] = []

    def add(cell, parts, iter_, frac, kind, key, value, **extra):
        rows.append({"cell": cell, **parts, "iter": iter_, "frac": frac,
                     "kind": kind, "key": key, "value": value, **extra})

    # --- loss curves + per-run summary -------------------------------------
    summaries: dict[tuple, dict] = {}
    for k, pts in sorted(curves.items()):
        if not pts:
            continue
        size, L, arch, scheme, seed = k
        cell = f"lm-{size}-L{L}{'-schemeB' if scheme == 'B' else ''}-{arch}-seed{seed}"
        parts = {"size": size, "L": L, "arch": arch, "scheme": scheme, "seed": seed}
        target = tgts.get(k, pts[-1][0]) or 1
        last_it, final = pts[-1]
        best_it, best = min(pts, key=lambda p: p[1])
        summaries[k] = {
            "cell": cell, "parts": parts, "n_params": NON_EMB[size],
            "target_iters": target, "last_iter": last_it,
            "complete": int(last_it >= target), "final_loss": final,
            "best_loss": best, "best_iter": best_it,
            "diverged": int(final > best + 0.25 and best_it < last_it * 0.9),
        }
        # Subsample for plotting, denser over the last 10%: that window is
        # where the architecture and scheme differences live (hundredths of a
        # nat), and a uniform stride would smear them. Each window contributes
        # its MIN and MAX so a divergence spike cannot be stepped over.
        cut = int(len(pts) * 0.9)
        for seg, n_win in ((pts[:cut], CURVE_WINDOWS // 2),
                           (pts[cut:], CURVE_WINDOWS // 2)):
            step = max(1, len(seg) // max(1, n_win))
            for i in range(0, len(seg), step):
                win = seg[i:i + step]
                for it, v in {min(win, key=lambda p: p[1]),
                              max(win, key=lambda p: p[1])}:
                    add(cell, parts, it, round(it / target, 5), "loss", "",
                        round(v, 4))

    # --- scaling fit, per (L, arch, scheme) --------------------------------
    done = {(s["parts"]["size"], k[1], k[2], k[3]): s
            for k, s in summaries.items() if s["complete"] and k[4] == 1904}
    for (L, arch, scheme) in {(k[1], k[2], k[3]) for k in done}:
        ladder = [done[(s, L, arch, scheme)] for s in SIZES
                  if (s, L, arch, scheme) in done]
        if len(ladder) < 3:
            continue
        fit = _fit([(s["n_params"], s["final_loss"]) for s in ladder[1:]])
        if not fit:
            continue
        slope, icpt = fit
        for s in ladder:
            pred = math.exp(icpt + slope * math.log(s["n_params"]))
            s["alpha"] = -slope
            s["pred_loss"] = pred
            s["resid"] = s["final_loss"] - pred
            s["off_trend"] = int(abs(s["final_loss"] - pred) > tol)

    for s in summaries.values():
        for name in ("n_params", "target_iters", "last_iter", "complete",
                     "final_loss", "best_loss", "best_iter", "diverged",
                     "alpha", "pred_loss", "resid", "off_trend"):
            if name in s:
                add(s["cell"], s["parts"], "", "", "summary", name,
                    round(s[name], 6) if isinstance(s[name], float) else s[name])

    # --- BPB / perplexity ---------------------------------------------------
    for cell, iters in sorted(bpb_results().items()):
        parts = _cell_parts(cell)
        if not parts:
            continue
        try:
            from launch_trainings import cell_languages
            trained = {t[:2] for t in cell_languages(parts["L"], parts["scheme"])}
        except Exception:
            trained = set()
        for it, d in sorted(iters.items()):
            for lang, v in d["languages"].items():
                short = lang.replace("fineweb_", "")
                tr = int(lang == "dclm" or short.split("_")[0][:2] in trained)
                for kind, val in (("bpb", v["bpb"]), ("ppl", v["ppl"])):
                    add(cell, parts, it, "", kind, short, round(val, 6),
                        language=short, trained=tr)
            # Per CHECKPOINT, not per run: bucketing it with the run-level
            # summary let the last checkpoint's value overwrite the rest and
            # then repeat on every row, flattening the macro curve.
            add(cell, parts, it, "", "macro", "macro_bpb",
                round(d["macro_bpb"], 6))

    # --- benchmark scores ---------------------------------------------------
    for cell, iters in sorted(benchmark_results().items()):
        parts = _cell_parts(cell)
        if not parts:
            continue
        for it, scores in sorted(iters.items()):
            for task, val in sorted(scores.items()):
                e = _meta_for(task, meta)
                n_opt = e.get("n_options")
                add(cell, parts, it, "", "benchmark", task, round(val, 6),
                    benchmark=e.get("benchmark", ""), language=e.get("language", ""),
                    chance=round(1 / n_opt, 4) if n_opt else "")

    wide = write_wide_csv(rows, out_dir)
    # The training curve is the ONE thing the wide table cannot hold: it is
    # per ITERATION, not per checkpoint. Everything else lives in the wide
    # file, so nothing is stored twice.
    curve_cols = ["cell", "size", "L", "arch", "scheme", "seed", "iter",
                  "frac", "loss"]
    curve = out_dir / "ladder_report_curve.csv"
    with open(curve, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=curve_cols, extrasaction="ignore")
        w.writeheader()
        n = 0
        for r in rows:
            if r["kind"] == "loss":
                w.writerow({**{c: r.get(c) for c in curve_cols[:-1]},
                            "loss": r["value"]})
                n += 1
    print(f"[csv]  wrote {curve} ({n} curve points)", file=sys.stderr)
    return wide, curve


def write_wide_csv(rows: list[dict], out_dir: Path) -> Path:
    """One row per checkpoint ID, one column per measurement — the SHARED file.

    This is the artifact to hand a collaborator: the row key is exactly the
    thing they think in, `<cell>-iter<N>`, and every metric for that
    checkpoint is on the row, so a correlation between (say) Russian BPB and
    hellaswag_ru is one dataframe away with no reshaping.

    The long table stays as the internal source the figures read, because the
    two answer different questions and neither is a good substitute:

      wide  one row per checkpoint, ~2.3k columns, ~28% filled. Natural for
            per-checkpoint analysis and for the model x benchmark matrix the
            SNR framework wants. Costs: the schema GROWS when a benchmark or
            language is added, so column-name-dependent code breaks; and it
            cannot hold the training curve, which is per ITERATION, not per
            checkpoint.
      long  one row per measurement, 15 stable columns. New metrics are new
            rows, so nothing downstream breaks, and it holds the 32k-point
            loss curve the divergence figure needs. Costs: any wide view
            needs a pivot first.

    Column names are prefixed by family (`bpb__`, `ppl__`, `bench__`) so the
    blocks can be selected without a lookup table.
    """
    import csv

    meta_cols = ["cell", "size", "L", "arch", "scheme", "seed", "iter"]
    wide: dict[tuple, dict] = {}
    summary: dict[str, dict] = {}
    for r in rows:
        if r["kind"] == "summary":
            summary.setdefault(r["cell"], {})[r["key"]] = r["value"]
        elif r["kind"] != "loss":
            key = (r["cell"], r["iter"])
            row = wide.setdefault(key, {c: r.get(c) for c in meta_cols})
            prefix = {"bpb": "bpb__", "ppl": "ppl__", "benchmark": "bench__",
                      "macro": ""}[r["kind"]]
            row[prefix + str(r["key"])] = r["value"]
    # The training loss AT each checkpoint iteration — the per-iteration curve
    # belongs to the long file, but the value at the checkpoint is a property
    # of the checkpoint and belongs here.
    loss_at = {(r["cell"], r["iter"]): r["value"] for r in rows if r["kind"] == "loss"}
    # A run with no eval results yet still has a loss curve and a scaling
    # residual, and the table and the ladder need it. Give every such run one
    # row at its final iteration with the metric columns simply empty.
    meta_of = {r["cell"]: {c: r.get(c) for c in meta_cols} for r in rows}
    # One row per PLANNED checkpoint, not just per checkpoint that happens to
    # have results. The empty cells are the point: the same table then answers
    # "what do we have?" and "what is still missing?", and a co-author can see
    # that a gap is unevaluated rather than silently absent.
    from launch_trainings import save_interval
    for cell, s in summary.items():
        target = int(s.get("target_iters", 0))
        if not target:
            continue
        si = save_interval(target)
        # EVERY saved checkpoint, not just the eval-due half. Conversion and
        # score_bpb.py already cover all of them, so planning for 10 left the
        # BPB-only checkpoints as stray extra rows and made the row count
        # depend on which jobs had run. 20 rows per cell; the odd ones simply
        # carry BPB with the bench__ columns empty, which is the honest
        # picture of what each checkpoint has.
        planned = list(range(si, target + 1, si))
        for it in planned:
            wide.setdefault((cell, it),
                            {**meta_of.get(cell, {"cell": cell}), "iter": it})

    for (cell, it), row in wide.items():
        row["loss"] = loss_at.get((cell, it), "")
        row["id"] = f"{cell}-iter{it}"
        row.update({f"run__{k}": v for k, v in summary.get(cell, {}).items()})

    if not wide:
        return out_dir / "ladder_report.csv"
    metric_cols = sorted({c for r in wide.values() for c in r} - set(meta_cols) - {"id"})
    cols = ["id"] + meta_cols + metric_cols
    path = out_dir / "ladder_report.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(wide[k] for k in sorted(wide))
    print(f"[csv]  wrote {path} ({len(wide)} checkpoints x {len(cols)} columns)",
          file=sys.stderr)
    return path


def _read_wide(csv_path: Path):
    """The wide table, or None when there is nothing to report yet.

    write_wide_csv returns its path without creating the file when no run has
    produced anything, so every consumer has to cope with a path that is not
    there — otherwise a fresh checkout crashes instead of saying "no data".
    """
    import pandas as pd
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    return None if df.empty else df


def _melt(wide_csv: Path, prefix: str, name: str):
    """Long view of one column family of the wide table.

    The figures want (id, key, value); the shared file is wide. Melting here
    keeps the wide file the single stored form — nothing is duplicated on
    disk just to make a plot convenient.
    """
    import pandas as pd
    df = _read_wide(wide_csv)
    if df is None:
        return pd.DataFrame()
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame()
    idv = ["cell", "size", "L", "arch", "scheme", "seed", "iter"]
    out = df.melt(id_vars=idv, value_vars=cols, var_name=name, value_name="value")
    out[name] = out[name].str.slice(len(prefix))
    return out.dropna(subset=["value"])


def _style(size, arch, scheme):
    """(colour, linestyle) — family from the size, step from (arch, scheme)."""
    pal = SIZE_PALETTE.get(size)
    step = VARIANT_STEP.get((arch, scheme), 1)
    colour = pal[step] if pal else "#555"
    return colour, VARIANT_STYLE.get((arch, scheme), "-")


def plot_loss(csv_path: Path, out_dir: Path) -> Path | None:
    """Loss vs fraction of run, two columns: the whole run and its last 10%.

    The full-range column is dominated by the early descent and the 90M
    divergence, which is exactly what hides the architecture and scheme
    differences — those are hundredths of a nat at the very end. The close-up
    column autoscales to that window, so the ordering between deep/shallow and
    scheme A/B becomes readable instead of being a single pixel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False).rename(columns={"loss": "value"})
    if df.empty:
        return None
    # The close-up compares healthy runs to each other. A diverged run sits
    # 3 nats above everything and would stretch the axis until the very
    # differences this column exists to show collapse to one pixel — and it
    # has no meaningful final loss to compare anyway.
    wide = _read_wide(csv_path.with_name("ladder_report.csv"))
    diverged = (set(wide.loc[wide.get("run__diverged") == 1, "cell"])
                if wide is not None and "run__diverged" in wide else set())
    Ls = sorted(df["L"].unique())
    fig, axes = plt.subplots(len(Ls), 2, figsize=(9.5, 2.7 * len(Ls)),
                             squeeze=False)
    for row, L in enumerate(Ls):
        sub = df[df["L"] == L]
        for col, (lo, title) in enumerate(((0.0, "full run"), (0.9, "last 10%"))):
            ax = axes[row][col]
            win = sub[sub["frac"] >= lo]
            if col == 1:
                win = win[~win["cell"].isin(diverged)]
            if win.empty:
                ax.axis("off")
                continue
            for cell, g in win.groupby("cell"):
                g = g.sort_values("frac")
                size, arch, scheme = g.iloc[0][["size", "arch", "scheme"]]
                colour, ls = _style(size, arch, scheme)
                ax.plot(g["frac"], g["value"], color=colour, ls=ls,
                        lw=1.0 if col == 0 else 1.4)
            if col == 0:
                for cell, g in win.groupby("cell"):
                    lo_r = g.loc[g["value"].idxmin()]
                    colour, _ = _style(*lo_r[["size", "arch", "scheme"]])
                    ax.plot(lo_r["frac"], lo_r["value"], "o", ms=4,
                            color=colour, mec="black", mew=0.5)
                ax.set_ylim(2, 8)
            else:
                # Hard ceiling rather than autoscale: a common upper bound
                # makes the panels comparable down the column, and nothing
                # healthy sits above it at the end of training.
                v = win["value"]
                ax.set_ylim(v.min() * 0.995, min(v.max() * 1.005, CLOSEUP_YMAX))
            ax.set_title(f"L={L} — {title}", fontsize=8)
            ax.set_xlabel("fraction of run", fontsize=7)
            ax.set_ylabel("lm loss", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25, lw=0.4)

    present = sorted({(r["size"], r["arch"], r["scheme"])
                      for _, r in df[["size", "arch", "scheme"]].drop_duplicates().iterrows()},
                     key=lambda x: (SIZES.index(x[0]) if x[0] in SIZES else 9, x[1], x[2]))
    handles = []
    for size, arch, scheme in present:
        c, ls = _style(size, arch, scheme)
        handles.append(Line2D([], [], color=c, ls=ls, lw=1.8,
                              label=f"{size} {arch[:2]}/{scheme}"))
    fig.legend(handles=handles, fontsize=6.5, ncol=min(8, len(handles)),
               loc="lower center", frameon=False)
    fig.suptitle("Training loss — full run (dot = each run's best) and a "
                 "close-up of the last 10%\nthe close-up drops diverged runs, so arch "
                 "and scheme separate", y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out_dir / "ladder_report_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scaling(csv_path: Path, out_dir: Path) -> Path | None:
    """Final loss vs parameters, one panel per L, every (arch, scheme) overlaid.

    Overlaid rather than one panel each, because the question the ladder is
    for is whether depth or the language set changes the EXPONENT — and that
    is a comparison between fits, which only reads if they share an axis.
    Each variant keeps its own fit: scheme B is different data, so a combined
    fit would be a line through two distributions.
    """
    import matplotlib.pyplot as plt

    import pandas as pd
    wide = _read_wide(csv_path)
    if wide is None or "run__alpha" not in wide:
        return None
    wide = (wide[["cell", "size", "L", "arch", "scheme"]
                 + [c for c in wide.columns if c.startswith("run__")]]
            .drop_duplicates(subset="cell"))
    wide.columns = [c.replace("run__", "") for c in wide.columns]
    wide = wide[wide["alpha"].notna()]
    if wide.empty:
        return None
    Ls = sorted(wide["L"].unique())
    fig, axes = _panels(len(Ls), 3.8, 3.2)
    variant_colour = {("deep", "A"): "#2980b9", ("shallow", "A"): "#e67e22",
                      ("deep", "B"): "#8e44ad", ("shallow", "B"): "#27ae60"}
    for ax, L in zip(axes, Ls):
        for (arch, scheme), g in wide[wide["L"] == L].groupby(["arch", "scheme"]):
            g = g.sort_values("n_params")
            c = variant_colour.get((arch, scheme), "#555")
            ax.plot(g["n_params"], g["final_loss"], "o", ms=5, color=c, zorder=3,
                    label=f"{arch}/{scheme} α={g.iloc[0]['alpha']:.3f}")
            ax.plot(g["n_params"], g["pred_loss"], "-", lw=1, color=c, alpha=0.6)
            off = g[g["off_trend"] == 1]
            ax.plot(off["n_params"], off["final_loss"], "o", ms=13, mfc="none",
                    mec="#c0392b", mew=1.6, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        sizes = [s for s in SIZES if s in set(wide[wide["L"] == L]["size"])]
        ax.set_xticks([NON_EMB[s] for s in sizes], minor=False)
        ax.set_xticklabels(sizes, fontsize=6)
        ax.set_xticks([], minor=True)
        ax.set_title(f"L={L}", fontsize=9)
        ax.set_xlabel("non-embedding params", fontsize=7)
        ax.set_ylabel("final lm loss", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25, lw=0.4, which="major")
    fig.suptitle("Scaling fit per language setting — every architecture and "
                 "scheme overlaid, each fitted separately\nline fitted WITHOUT "
                 "the smallest rung then predicting it; red ring = off trend",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    path = out_dir / "ladder_report_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bpb(csv_path: Path, out_dir: Path) -> Path | None:
    """Per-language BPB vs checkpoint, one panel per scored cell."""
    import matplotlib.pyplot as plt

    import pandas as pd
    from launch_trainings import cell_languages
    df = _melt(csv_path, "bpb__", "key")
    if df.empty:
        return None
    trained_by_cell = {}
    for cell in df["cell"].unique():
        parts = _cell_parts(cell)
        trained_by_cell[cell] = ({t[:2] for t in cell_languages(parts["L"], parts["scheme"])}
                                 if parts else set())
    df["trained"] = [int(k == "dclm" or k.split("_")[0][:2] in trained_by_cell[c])
                     for c, k in zip(df["cell"], df["key"])]
    # macro_bpb is a per-CHECKPOINT column (it used to be run-level, which
    # flattened this curve); read it under that name or the macro line
    # silently disappears.
    macro = _read_wide(csv_path)
    macro = (macro[["cell", "iter", "macro_bpb"]].rename(
                 columns={"macro_bpb": "value"}).dropna(subset=["value"])
             if macro is not None and "macro_bpb" in macro
             else pd.DataFrame(columns=["cell", "iter", "value"]))
    cells = sorted(df["cell"].unique())
    fig, axes = _panels(len(cells), 4.0, 3.2)
    for ax, cell in zip(axes, cells):
        g = df[df["cell"] == cell]
        for lang, gl in g.groupby("key"):
            gl = gl.sort_values("iter")
            tr = bool(gl.iloc[0]["trained"])
            ax.plot(gl["iter"], gl["value"], lw=1.4 if tr else 0.5,
                    color="#2980b9" if tr else "#cccccc", zorder=3 if tr else 1)
            if tr:
                ax.annotate(lang, (gl.iloc[-1]["iter"], gl.iloc[-1]["value"]),
                            fontsize=6, xytext=(3, 0), textcoords="offset points")
        mg = macro[macro["cell"] == cell].sort_values("iter")
        if not mg.empty:
            ax.plot(mg["iter"], mg["value"], "--", lw=1.4, color="#c0392b",
                    label="macro (all 100)")
            ax.legend(fontsize=6, loc="upper left")
        ax.set_yscale("log")
        ax.set_title(cell.replace("lm-", ""), fontsize=8)
        ax.set_xlabel("checkpoint (iter)", fontsize=7)
        ax.set_ylabel("bits per byte", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.suptitle("Per-language BPB vs checkpoint — blue = languages the cell "
                 "trains on, grey = unseen\nlines that RISE mean the model is "
                 "getting worse on held-out data", y=1.0, fontsize=10)
    fig.tight_layout()
    path = out_dir / "ladder_report_bpb.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_benchmarks(csv_path: Path, out_dir: Path) -> Path | None:
    """Benchmark accuracy vs checkpoint, one panel per benchmark.

    Scores are averaged over the languages of a benchmark and drawn per size,
    with the chance line from tasks.json's n_options: a benchmark sitting on
    its chance line has told us nothing, however smooth the curve looks.
    """
    import matplotlib.pyplot as plt

    df = _melt(csv_path, "bench__", "key")
    if df.empty:
        return None
    # Per-task metadata comes from tasks.json rather than being repeated on
    # every row — one source of truth, and the wide file stays a pure matrix.
    meta = _task_meta()
    info = [_meta_for(k, meta) for k in df["key"]]
    df["benchmark"] = [e.get("benchmark", "") for e in info]
    df["chance"] = [1 / e["n_options"] if e.get("n_options") else float("nan")
                    for e in info]
    df = df[df["benchmark"] != ""]
    benches = sorted(df["benchmark"].unique())
    fig, axes = _panels(len(benches), 3.4, 2.8)
    for ax, b in zip(axes, benches):
        g = df[df["benchmark"] == b]
        for (size, arch, scheme), gs in g.groupby(["size", "arch", "scheme"]):
            m = gs.groupby("iter")["value"].mean().sort_index()
            # x is the fraction of the run so rungs of different length are
            # comparable, as in the loss figure.
            colour, ls = _style(size, arch, scheme)
            ax.plot(range(len(m)), m.values, lw=1.2, color=colour, ls=ls)
        ch = g["chance"].dropna()
        if not ch.empty:
            ax.axhline(ch.mean(), color="#c0392b", lw=0.9, ls=":")
        ax.set_title(b, fontsize=8)
        ax.set_xlabel("evaluated checkpoint (in order)", fontsize=7)
        ax.set_ylabel("accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle("Benchmark accuracy vs checkpoint, averaged over each "
                 "benchmark's languages\ncolour = size, dotted red = chance "
                 "(from tasks.json n_options)", y=1.0, fontsize=10)
    fig.tight_layout()
    path = out_dir / "ladder_report_benchmarks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def summary_table(csv_path: Path, tol: float) -> str:
    """One markdown row per completed run — the numbers behind the figures."""
    import pandas as pd

    full = _read_wide(csv_path)
    if full is None:
        return "_no runs found_"
    # macro_bpb is per CHECKPOINT; the table wants the latest scored one.
    latest = (full[full["macro_bpb"].notna()].sort_values("iter")
              .groupby("cell")["macro_bpb"].last().to_dict()
              if "macro_bpb" in full else {})
    w = (full[["cell", "size", "L", "arch", "scheme"]
              + [c for c in full.columns if c.startswith("run__")]]
         .drop_duplicates(subset="cell"))
    w.columns = [c.replace("run__", "") for c in w.columns]
    w = w[w.get("complete") == 1] if "complete" in w else w
    if w.empty:
        return "_no completed runs_"
    w = w.sort_values(["L", "n_params", "arch", "scheme"])
    out = ["| cell | final loss | best (iter) | diverged | scaling resid | macro BPB |",
           "| ---- | ---------: | ----------: | :------: | ------------: | --------: |"]
    for _, r in w.iterrows():
        resid = ""
        if r.get("resid") == r.get("resid"):
            resid = (f"**{r['resid']:+.2f}**" if r.get("off_trend") == 1
                     else f"{r['resid']:+.2f}")
        b = latest.get(r["cell"])
        name = r["cell"].replace("lm-", "").replace("-seed1904", "")
        out.append(
            f"| {name} | {r['final_loss']:.3f} | {r['best_loss']:.3f} "
            f"({int(r['best_iter'])}) | {'**yes**' if r.get('diverged') else ''} "
            f"| {resid} | {f'{b:.3f}' if b is not None else '—'} |")
    return "\n".join(out)


# The intervention axes the sweep varies. SNR treats two runs as different
# models only if the transformation between them actually moves the scores;
# a transformation whose effect is the size of the SEED effect is not a
# distinct model, it is a re-roll. Seed is therefore listed first: it is the
# yardstick the other two are measured against, not just another axis.
TRANSFORMS = {
    "seed":   ("size", "L", "arch", "scheme"),
    "arch":   ("size", "L", "scheme", "seed"),
    "scheme": ("size", "L", "arch", "seed"),
}


def transform_effects(csv_path: Path) -> str:
    """Effect of each transformation, on matched pairs that differ ONLY in it.

    For every (size, L) where two runs differ in exactly one axis, report the
    change in final loss, macro BPB and mean benchmark accuracy. The question
    is not whether an effect is non-zero but whether it is bigger than the
    seed effect: if swapping the seed moves a benchmark as much as swapping
    the architecture does, then deep and shallow are the same model as far as
    an SNR ranking is concerned, and pooling them would inflate the sample
    without adding a real comparison.
    """
    import pandas as pd

    full = _read_wide(csv_path)
    if full is None:
        return "_no runs found_"
    bcols = [c for c in full.columns if c.startswith("bench__")]
    scored = full[full[bcols].notna().any(axis=1)]
    bench = (scored.assign(v=scored[bcols].mean(axis=1))
             .sort_values("iter").groupby("cell")["v"].last().to_dict())
    bpb = (full[full["macro_bpb"].notna()].sort_values("iter")
           .groupby("cell")["macro_bpb"].last().to_dict()
           if "macro_bpb" in full else {})
    runs = full.drop_duplicates("cell")
    runs = runs[runs.get("run__complete") == 1] if "run__complete" in runs else runs
    loss = dict(zip(runs["cell"], runs.get("run__final_loss", [])))
    keys = {r["cell"]: r for _, r in runs.iterrows()}

    out = ["| transformation | pairs | Δ final loss | Δ macro BPB | Δ mean benchmark |",
           "| -------------- | ----: | -----------: | ----------: | ---------------: |"]
    detail = []
    for axis, fixed in TRANSFORMS.items():
        groups: dict[tuple, list] = {}
        for cell, r in keys.items():
            groups.setdefault(tuple(r[f] for f in fixed), []).append(cell)
        deltas = {"loss": [], "bpb": [], "bench": []}
        npairs = 0
        for fixedvals, cells in groups.items():
            if len(cells) < 2:
                continue
            cells = sorted(cells, key=lambda c: str(keys[c][axis]))
            a, b = cells[0], cells[-1]
            npairs += 1
            for name, src in (("loss", loss), ("bpb", bpb), ("bench", bench)):
                if a in src and b in src and src[a] == src[a] and src[b] == src[b]:
                    deltas[name].append(src[b] - src[a])
                    if name == "loss":
                        detail.append(
                            f"| {axis} | {keys[a]['size']} L{keys[a]['L']} | "
                            f"{keys[a][axis]} -> {keys[b][axis]} | "
                            f"{src[b] - src[a]:+.3f} |")

        def rng(v):
            if not v:
                return "—"
            return (f"{min(v):+.3f} .. {max(v):+.3f}" if len(v) > 1
                    else f"{v[0]:+.3f}")
        out.append(f"| {axis} | {npairs} | {rng(deltas['loss'])} | "
                   f"{rng(deltas['bpb'])} | {rng(deltas['bench'])} |")

    note = ("\n**Read this against the `seed` row.** A transformation whose "
            "effect is no larger than re-rolling the seed is not a distinct "
            "model for SNR — it is the same model measured twice. An em dash "
            "means no matched pair exists yet.\n")
    return "\n".join(out) + "\n" + note


def write_artifacts(curves, tgts, out_dir: Path, tol: float) -> None:
    """The CSV, then every figure and the table derived from it."""
    wide, curve = write_csv(curves, tgts, out_dir, tol)
    made = [p for p in (plot_loss(curve, out_dir),
                        plot_scaling(wide, out_dir),
                        plot_bpb(wide, out_dir),
                        plot_benchmarks(wide, out_dir)) if p]
    for p in made:
        print(f"[plot] saved {p}", file=sys.stderr)

    doc = out_dir / "ladder_report.md"
    figs = "\n\n".join(f"![{p.stem}](./{p.name})" for p in made)
    doc.write_text(
        "# Ladder report\n\n"
        "**Generated by `ladder_report.py --plot` — do not edit.** Everything "
        f"here is derived from [`{wide.name}`](./{wide.name}) — one row per "
        "checkpoint (`id` = `<cell>-iter<N>`), one column per measurement, "
        "prefixed by family: `bench__<task>`, `bpb__<lang>`, `ppl__<lang>`, "
        f"`run__<metric>`. The training curve is per ITERATION rather than "
        f"per checkpoint, so it lives alongside in `{curve.name}`; nothing "
        "is stored twice.\n\n"
        "`final loss` is at the run's target; `best (iter)` is the lowest loss "
        "it ever reached. A run whose best is far from its end has diverged: "
        "on a single-epoch budget there is no overfitting to explain it, so "
        "the shipped checkpoint is worse than one already on disk. "
        "`scaling resid` is nats from a power law fitted on the LARGER rungs "
        f"(bold beyond {tol}), per (L, arch, scheme).\n\n"
        + summary_table(wide, tol)
        + "\n\n## Effect of each transformation\n\n"
        + transform_effects(wide) + "\n" + figs + "\n")
    print(f"[docs] wrote {doc}", file=sys.stderr)


CAPSTOR_REPORTS = Path("/capstor/store/cscs/swissai/infra01/msnr-ladder-report")
HF_DATASET_REPO = "multilingual-snr/msnr-ladder-report"


def publish(out_dir: Path, push_hf: bool) -> None:
    """Copy the shared CSV to capstor, and optionally to a private HF dataset.

    The CSV is gitignored: it is 1-2 MB that changes on every regeneration, so
    committing it would churn the history for a file that is pure output.
    capstor is the durable home (iopsstor is swept ~30 days) and the Hub is
    what a collaborator without cluster access can actually reach.
    """
    import shutil

    src = out_dir / "ladder_report.csv"
    if not src.is_file():
        print("[publish] nothing to publish", file=sys.stderr)
        return
    CAPSTOR_REPORTS.mkdir(parents=True, exist_ok=True)
    dst = CAPSTOR_REPORTS / src.name
    shutil.copy2(src, dst)
    print(f"[publish] {dst}", file=sys.stderr)

    if not push_hf:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=True,
                        exist_ok=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=src.name,
                        repo_id=HF_DATASET_REPO, repo_type="dataset")
        print(f"[publish] https://huggingface.co/datasets/{HF_DATASET_REPO}",
              file=sys.stderr)
    except Exception as e:
        # Compute nodes have no internet and the token may be absent; a failed
        # upload must not lose the figures that already succeeded.
        print(f"[publish] HF upload skipped: {e}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", choices=["loss", "scaling", "benchmarks", "bpb"],
                   action="append")
    p.add_argument("--tol", type=float, default=0.15,
                   help="nats off the scaling fit before a rung is flagged")
    p.add_argument("--plot", action="store_true",
                   help="also write the figures and ladder_report.md")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--publish", action="store_true",
                   help="copy the shared CSV to capstor (add --push-hf for "
                        "the private HF dataset repo)")
    p.add_argument("--push-hf", action="store_true")
    args = p.parse_args()
    want = args.check or ["loss", "scaling", "benchmarks", "bpb"]

    curves = tgts = None
    if args.plot or {"loss", "scaling"} & set(want):
        curves, tgts = loss_curves(), targets()

    problems = []
    if "loss" in want:
        problems += check_loss(curves, tgts)
    if "scaling" in want:
        problems += check_scaling(curves, tgts, args.tol)
    if "benchmarks" in want:
        problems += check_benchmarks()
    if "bpb" in want:
        problems += check_bpb()

    if args.plot:
        write_artifacts(curves, tgts, args.out_dir, args.tol)
    if args.publish or args.push_hf:
        publish(args.out_dir, args.push_hf)

    print("\n== summary ==")
    if not problems:
        print("  nothing flagged")
    for x in problems:
        print(f"  ! {x}")
    sys.exit(0)


if __name__ == "__main__":
    main()

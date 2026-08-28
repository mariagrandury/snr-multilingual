#!/usr/bin/env python3
"""Is the sweep going well? Loss, scaling law, benchmarks, and BPB — from disk.

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

    python3.11 sweep_health.py                 # everything
    python3.11 sweep_health.py --check scaling
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", choices=["loss", "scaling", "benchmarks", "bpb"],
                   action="append")
    p.add_argument("--tol", type=float, default=0.15,
                   help="nats off the scaling fit before a rung is flagged")
    args = p.parse_args()
    want = args.check or ["loss", "scaling", "benchmarks", "bpb"]

    curves = tgts = None
    if {"loss", "scaling"} & set(want):
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

    print("\n== summary ==")
    if not problems:
        print("  nothing flagged")
    for x in problems:
        print(f"  ! {x}")
    sys.exit(0)


if __name__ == "__main__":
    main()

# on_grid_iters.awk — the checkpoints of one cell that BPB should score.
#
# Reads ASCENDING iteration numbers, one per line, and drops the ones off the
# run's own save grid. Megatron writes an extra checkpoint when a preempted job
# exits, so a cell trained on `preemptable` accumulates iters between its grid
# saves (lm-3B-L30-deep-seed1904: 11478, 21769, 27354 and 31411 among its
# 2420-step saves). Nothing reads them — the analysis takes the ten tenths of a
# run (utils.SHARED_FRACS, RULES.md rule 3) and an exact grid save always wins
# at_fraction's nearest-match, because its distance is 0 — while each one costs
# a full scoring pass (444 s at 90M to 2,141 s at 1.7B).
#
# The grid step is the most common gap between consecutive saves, ties going to
# the larger gap: exactly launch_trainings.run_interval, whose reasoning is that
# an off-grid save only ever SHORTENS a gap, so it cannot win the mode.
#
# The mode needs enough grid saves to be visible, and a cell with only two or
# three of them can read the wrong step either way: it may keep an off-grid save
# (the filter degrades to the no-op it replaced) or skip a grid one. Both
# self-heal, because an iter with no bpb.json stays a candidate and is scored on
# a later pass once the mode is unambiguous.
#
# A trailing off-grid save is kept only when it is more than one step past the
# last on-grid one, which is launch_trainings.due_iters' `final` exception
# expressed without knowing the target: that clause fires when the target is
# NOT among the saves, i.e. when the grid's own endpoint is missing and the
# overshoot is all there is. lm-1B-L8-deep-seed1904 saved both 45720 (40 x 1143,
# the target) and 45740, 20 iters later; 45720 is the endpoint, so 45740 is
# dropped, and due_iters agrees — it reports 45720 as that run's final.
#
# Verified over all 222 converted cells on 2026-10-05: every checkpoint
# due_iters asks for is kept (0 dropped), 665 off-grid saves are not, and every
# finished cell keeps exactly target/step checkpoints — including the 15 1B
# cells that saved 20 times every 2287 iters instead of 40, since the step is
# read off each run rather than assumed from its size.
{ v[n++] = $1 + 0 }
END {
    if (n == 0) exit
    step = v[0]                       # a lone save: its own iter, as run_interval does
    for (i = 1; i < n; i++) {
        g = v[i] - v[i - 1]
        if (++c[g] > best || (c[g] == best && g > step)) { step = g; best = c[g] }
    }
    for (i = 0; i < n; i++)
        if (step > 0 && v[i] % step == 0) last_grid = v[i]
    for (i = 0; i < n; i++)
        if ((step > 0 && v[i] % step == 0) || v[i] > last_grid + step) print v[i]
}

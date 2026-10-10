/* Additional interactive views of the noise-snr page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, css, fmt, pct, note, tiles, plot, mount, load, uniq } = S;

  const KINDS = [{ value: "acc", label: "accuracy tasks" }, { value: "bbpb", label: "bBPB twins" }, { value: "bpb", label: "per-language BPB" }];
  const DATA_LABEL = { A: "data A", AT3: "data A at T = 3", B: "data B", ZH: "data B (Chinese at L2)", ES: "data C (Spanish at L2)",
    DCLMP: "data B (unfiltered DCLM)", FWEB: "data C (FineWeb English)" };
  // label a log axis at 1, 2 and 5 times a power of ten only, as plain decimals
  const logTick = (v) => ([1, 2, 5].includes(Math.round(v / 10 ** Math.floor(Math.log10(v) + 1e-9))) ? d3.format("~g")(v) : "");
  const variantLabel = (m) => `L${m.L} · ${m.ladder} · ${DATA_LABEL[m.data] ?? m.data}`;
  const matches = (p, q, names) => {
    if (!q) return false;
    const t = q.trim().toLowerCase();
    return t.length > 0 && [p.task, p.fam, names[p.fam] ?? "", p.lang, S.langName(p.lang)].some((s) => String(s).toLowerCase().includes(t));
  };

  // ---------- signal vs noise map, with each task's curves across the design variants ----------
  S.VIEWS.noiseSnrMap = async (el) => {
    const d = await S.languages().then(() => load("noise_snr_map"));
    const rows = S.expand(d).map((r) => ({ ...r, task: d.tasks[r.i].t, fam: d.tasks[r.i].f, lang: d.tasks[r.i].l, kind: d.tasks[r.i].k }));
    const name = (p) => `${d.names[p.fam] ?? p.fam} · ${S.langName(p.lang)}`;
    let pinned = null;

    const curvesFor = (p, panel) => {
      if (p.kind !== "acc") {
        panel.append(note("Per-checkpoint curves are published for the accuracy tasks only; the bBPB twins and the per-language BPB have their signal and noise in the table above, not as curves on this site."));
        return;
      }
      const key = `${p.task}@${p.size}`;
      panel.dataset.key = key;
      const box = h("div", { class: "viz-empty" }, "Loading curves…");
      panel.append(box);
      load(`noise_snr_curves_${p.size.replace(".", "_")}`).then((c) => {
        if (panel.dataset.key !== key) return;
        const lines = c.curves[p.task];
        if (!lines) { box.replaceWith(note("No curves for this task at this size.")); return; }
        const pts = lines.flatMap(([mi, ...ys]) => ys.map((y, j) => (y == null ? null : {
          m: variantLabel(c.models[mi]), L: `L${c.models[mi].L}`, x: 100 * c.fracs[j], y: y / 1000 }))).filter(Boolean);
        const finals = pts.filter((q) => q.x === 100);
        const lo = d3.min(finals, (q) => q.y), hi = d3.max(finals, (q) => q.y);
        const ls = S.LS.map((L) => `L${L}`).filter((L) => pts.some((q) => q.L === L));
        box.replaceWith(plot({
          height: 300, marginRight: 120, width: 720,
          x: { label: "share of the run (%)", domain: [10, 100] },
          y: { label: "accuracy", grid: true },
          color: { domain: ls, range: S.series(S.LS.length).filter((_, i) => ls.includes(`L${S.LS[i]}`)), legend: true, label: "training languages" },
          marks: [
            Plot.rectX([0], { x1: 80, x2: 100, fill: css("--viz-s4"), fillOpacity: 0.12 }),
            Plot.text(["noise window:\nthe wobble of\neach line here"], { x: 80, frameAnchor: "top-left", dx: 4, dy: 4, fill: css("--viz-muted"), fontSize: 10, textAnchor: "start" }),
            Plot.line(pts, { x: "x", y: "y", z: "m", stroke: "L", strokeWidth: 1.4, strokeOpacity: 0.8 }),
            Plot.dot(finals, { x: "x", y: "y", fill: "L", r: 3 }),
            Plot.ruleX([0], { x: 100, y1: lo, y2: hi, dx: 10, stroke: css("--md-default-fg-color"), strokeWidth: 2 }),
            Plot.text([0], { x: 100, y: (lo + hi) / 2, dx: 16, textAnchor: "start", text: () => "signal: spread of\nthe final scores", fontSize: 10, fill: css("--viz-muted") }),
            Plot.tip(pts, Plot.pointer({ x: "x", y: "y", title: (q) => `${q.m}\n${q.x} % of the run: ${fmt(q.y, 3)}` })),
          ],
        }));
      });
    };

    const detail = (p, panel, rawMode) => {
      panel.replaceChildren(
        h("h4", {}, `${name(p)} `, h("span", { class: "viz-muted" }, `(${p.task}, ${p.size})`)),
        tiles([[fmt(p.signal, 3), "signal: relative sd of the finals across the size's design variants"],
          [fmt(p.x, 4), `noise: relative ${rawMode ? "raw" : "detrended"} checkpoint sd`],
          [`${fmt(p.snr, 1)}×`, "SNR = signal ÷ noise"],
          [p.da == null ? "–" : pct(p.da), p.size === "1.7B" ? "DA-size (1.7B is the reference)" : "DA-size: pairs ranked like 1.7B"]]));
      curvesFor(p, panel);
    };

    mount(el, [
      { key: "size", label: "Model size", type: "seg", options: S.LADDER, value: "1B" },
      { key: "kind", label: "Tasks", type: "seg", options: KINDS },
      { key: "noise", label: "Checkpoint noise", type: "seg", options: [{ value: "detr", label: "detrended (default)" }, { value: "raw", label: "raw sd" }] },
      { key: "q", label: "Find a benchmark or language", type: "text" },
    ], (s) => {
      const raw = s.noise === "raw";
      const sel = rows.filter((r) => r.size === s.size && r.kind === s.kind && r.signal > 0 && r.noise > 0);
      if (!sel.length) return h("div", { class: "viz-empty" }, "No task has an SNR here.");
      const both = sel.flatMap((r) => [r.noise, r.noise * (r.raw_over_detr ?? 1)]);
      const xd = [d3.min(both) / 1.2, d3.max(both) * 1.2], yd = [d3.min(sel, (r) => r.signal) / 1.2, d3.max(sel, (r) => r.signal) * 1.2];
      const pts = sel.map((r) => {
        const x = raw ? r.noise * (r.raw_over_detr ?? NaN) : r.noise;
        return { ...r, x, y: r.signal, snr: r.signal / x, hit: matches(r, s.q, d.names) };
      }).filter((p) => p.x > 0);
      const iso = [0.3, 1, 3, 10].map((k) => {
        const xEnd = Math.min(xd[1], yd[1] / k), xStart = Math.max(xd[0], yd[0] / k);
        return xEnd > xStart ? { k, seg: [{ x: xStart, y: k * xStart }, { x: xEnd, y: k * xEnd }] } : null;
      }).filter(Boolean);
      const hits = pts.filter((p) => p.hit);
      const hasDA = s.size !== "1.7B";
      const withDA = hasDA ? pts.filter((p) => p.da != null) : [];
      const noDA = hasDA ? pts.filter((p) => p.da == null) : pts;
      const med = d3.median(pts, (p) => Math.log10(p.snr));
      const below = pts.filter((p) => p.snr < 1).length;
      const panel = h("div", { class: "ladder-detail" });
      const chart = plot({
        height: 420, marginLeft: 60, marginBottom: 42,
        x: { type: "log", domain: xd, label: "checkpoint noise (relative sd over the last 20 % of the run) →", tickFormat: logTick },
        y: { type: "log", domain: yd, label: "↑ signal (relative sd of the final scores across design variants)", grid: true, tickFormat: logTick },
        color: hasDA ? { type: "linear", domain: [0.5, 1], range: [S.sizeRamp()[0], S.sizeRamp()[4]], legend: true, label: "DA-size against 1.7B" } : undefined,
        marks: [
          ...iso.map(({ k, seg }) => Plot.line(seg, { x: "x", y: "y", stroke: k === 1 ? css("--viz-bad") : css("--viz-muted"), strokeDasharray: k === 1 ? null : "3,3", strokeWidth: k === 1 ? 1.5 : 1 })),
          ...iso.map((o) => Plot.text([o], { x: o.seg[1].x, y: o.seg[1].y, text: () => `SNR ${o.k}`, dx: -4, dy: -6, textAnchor: "end", fontSize: 10, fill: o.k === 1 ? css("--viz-bad") : css("--viz-muted") })),
          Plot.dot(noDA, { x: "x", y: "y", r: 3, fill: css("--viz-s1"), fillOpacity: s.q ? 0.25 : 0.7 }),
          hasDA ? Plot.dot(withDA, { x: "x", y: "y", r: 3, fill: "da", fillOpacity: s.q ? 0.25 : 0.85 }) : null,
          Plot.dot(hits, { x: "x", y: "y", r: 5.5, fill: "none", stroke: css("--viz-s2"), strokeWidth: 2 }),
          Plot.tip(pts, Plot.pointer({ x: "x", y: "y",
            title: (p) => `${name(p)} (${p.task})\nsignal ${fmt(p.y, 3)} · noise ${fmt(p.x, 4)} · SNR ${fmt(p.snr, 1)}${p.da == null ? "" : `\nDA-size ${pct(p.da)}`}` })),
        ].filter(Boolean),
      });
      chart.addEventListener("input", () => { if (chart.value) { pinned = chart.value.task; detail(chart.value, panel, raw); } });
      const start = pts.find((p) => p.task === pinned) ?? hits[0] ?? pts.find((p) => p.task === "hellaswag") ?? pts[Math.floor(pts.length / 2)];
      if (start) detail(start, panel, raw);
      return [
        tiles([[pts.length, "tasks with an SNR here"], [fmt(med), "median log₁₀ SNR"], [`${below} (${pct(below / pts.length)})`, "below SNR 1 (under the red line)"],
          [s.q ? hits.length : "–", "match the search"]]),
        chart,
        note(`Each dot is one task at ${s.size}. Up-left is good: the design variants end far apart and each run is quiet. The red diagonal is SNR = 1, where the spread between variants equals one run's checkpoint wobble. ` +
          (hasDA ? "The colour is the task's DA-size: how often its ranking of the variants at this size matches 1.7B (legend above). " : "") +
          "Switch to the raw sd to see how much of the old noise was the WSD decay's own rise, now taken out by the line through the window. Hover a dot for its curves below."),
        panel,
      ];
    });
  };

  // ---------- one design effect against two noises ----------
  const DECISIONS = { arch: "depth (deep vs shallow)", scheme_B: "data scheme A vs B", scheme_C: "data scheme A vs C", temperature: "temperature T 1 vs 3" };
  const POPS = [{ value: "benchmark", label: "benchmarks" }, { value: "bpb", label: "per-language BPB" }, { value: "bpb_macro", label: "BPB, all languages" }, { value: "loss", label: "training loss" }];

  S.VIEWS.noiseSnrEffect = async (el) => {
    const d = await load("noise_snr_effect");
    const sizes = S.LADDER.filter((z) => d.some((r) => r.size === z && r.noise === "seed"));
    mount(el, [
      { key: "size", label: "Model size (with seed replicates)", type: "seg", options: sizes, value: sizes.includes("600M") ? "600M" : sizes[0] },
      { key: "pop", label: "Measurement", type: "seg", options: POPS },
    ], (s) => {
      const rows = d.filter((r) => r.size === s.size && r.population === s.pop && r.median != null && r.median > 0)
        .map((r) => ({ ...r, label: DECISIONS[r.decision] ?? r.decision, yard: r.noise === "seed" ? "÷ seed noise" : "÷ checkpoint noise" }));
      if (!rows.length) return h("div", { class: "viz-empty" }, "No cell with seed replicates for this measurement at this size.");
      const labels = Object.values(DECISIONS).filter((l) => rows.some((r) => r.label === l));
      const pairs = labels.map((l) => {
        const a = rows.find((r) => r.label === l && r.noise === "ckpt"), b = rows.find((r) => r.label === l && r.noise === "seed");
        return a && b ? { label: l, x1: Math.min(a.median, b.median), x2: Math.max(a.median, b.median) } : null;
      }).filter(Boolean);
      const xs = rows.map((r) => r.median);
      const xd = [Math.min(0.5, d3.min(xs) / 1.3), Math.max(4, d3.max(xs) * 1.3)];
      return [plot({
        height: 60 * labels.length + 70, marginLeft: 170,
        x: { type: "log", label: "median |effect of the decision| ÷ noise (log)", domain: xd, grid: true,
          ticks: [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256].filter((v) => v >= xd[0] && v <= xd[1]), tickFormat: (v) => `${v}×` },
        y: { domain: labels, label: null },
        color: { domain: ["÷ checkpoint noise", "÷ seed noise"], range: [S.css("--viz-s1"), S.css("--viz-s2")], legend: true },
        marks: [
          Plot.ruleX([1], { stroke: css("--viz-bad") }),
          Plot.ruleX([2], { stroke: css("--viz-muted"), strokeDasharray: "4,3" }),
          Plot.text([1, 2], { x: (v) => v, frameAnchor: "top", dy: -2, dx: 4, textAnchor: "start", text: (v) => (v === 1 ? "as big as the noise" : "2× the noise"), fontSize: 10, fill: css("--viz-muted") }),
          Plot.link(pairs, { x1: "x1", x2: "x2", y1: "label", y2: "label", stroke: css("--viz-grid"), strokeWidth: 4 }),
          Plot.dot(rows, { x: "median", y: "label", fill: "yard", r: 7, stroke: css("--viz-surface"), strokeWidth: 2, tip: true,
            title: (r) => `${r.label} at ${r.size} (${POPS.find((p) => p.value === r.population).label})\nmedian effect = ${fmt(r.median)}× the ${r.noise === "seed" ? "seed" : "checkpoint"} noise` }),
        ],
      }), note("Each row is one design decision, measured on the same cells against two yardsticks. Blue: divided by the wobble of one run's late checkpoints, the noise the SNR uses. Orange: divided by the spread across three seeds of the same recipe. Where orange sits left of blue, re-rolling the seed moves the score more than the late checkpoints do, so a checkpoint-noise SNR flatters the decision. Left of the red line, the decision is not distinguishable from a re-roll.")];
    });
  };

  // ---------- where the 600M misses come from ----------
  const CATS = [
    { key: "share_not_a_not_b", label: "agrees, within seed noise" },
    { key: "share_a_not_b", label: "agrees, clearly separated" },
    { key: "share_not_a_b", label: "flips, within seed noise" },
    { key: "share_a_b", label: "flips, clearly separated" },
  ];
  const GROUPS = [
    ["mono-axis", "arch", "depth (deep vs shallow)"], ["mono-axis", "L", "language count"], ["mono-axis", "scheme", "data scheme"],
    ["mono-axis", "T", "temperature"], ["mono-axis", "activation", "activation"], ["mono-axis", "optimizer", "optimizer"],
    ["mono-axis", "all", "all one-axis pairs"], ["multi-axis", "all", "multi-axis pairs"],
    ["multi-axis", "moves arch", "multi-axis, depth moved"], ["multi-axis", "arch held", "multi-axis, depth held"],
  ];

  S.VIEWS.noiseSnrFlips = async (el) => {
    const d = await load("noise_snr_flips");
    // agree / flip x within / beyond the seed noise: light and dark steps of a blue and an orange
    const colors = () => (S.dark() ? ["#2f4f75", "#3987e5", "#8a4a2c", "#d95926"] : ["#b7d3f6", "#2a78d6", "#f7b99b", "#c4501b"]);
    mount(el, [
      { key: "size", label: "Proxy size", type: "seg", options: S.LADDER.filter((z) => d.axes.some((r) => r.size === z)), value: "600M" },
      { key: "pop", label: "Measurement", type: "seg", options: [{ value: "bpb", label: "per-language BPB" }, { value: "accuracy", label: "benchmark accuracy" }] },
      { key: "thr", label: "\"Clearly separated\" means", type: "seg", options: [{ value: "1 sd", label: "> 1 seed sd" }, { value: "sqrt2 sd", label: "> √2 seed sd" }] },
    ], (s) => {
      const rows = GROUPS.map(([axes, axis, label]) => {
        const r = d.axes.find((x) => x.population === s.pop && x.axes === axes && x.axis === axis && x.size === s.size && x.threshold === s.thr);
        return r && r.n_comparisons ? { ...r, label: `${label}${r.size_matched ? "" : " *"}` } : null;
      }).filter((r) => r && r.share_a_b != null);
      if (!rows.length) return h("div", { class: "viz-empty" }, "No comparison here.");
      const segs = rows.flatMap((r) => {
        let x = 0;
        return CATS.map((c) => { const w = r[c.key] ?? 0; const o = { ...r, cat: c.label, x1: x, x2: x + w, w }; x += w; return o; });
      });
      const seedRows = d.depth_by_seed.filter((r) => r.n >= 10).map((r) => ({ ...r, share: r.flips / r.n, label: `${r.size} L${r.L} · deep seed ${r.deep_seed}` }));
      const seedLabels = uniq(seedRows.map((r) => r.label));
      return [
        plot({
          height: 34 * rows.length + 70, marginLeft: 190,
          x: { domain: [0, 1], tickFormat: "%", label: "share of the proxy's comparisons against 1.7B" },
          y: { domain: rows.map((r) => r.label), label: null },
          color: { domain: CATS.map((c) => c.label), range: colors(), legend: true },
          marks: [
            Plot.rectX(segs, { x1: "x1", x2: "x2", y: "label", fill: "cat", inset: 1, insetLeft: 0.5, insetRight: 0.5, tip: true,
              title: (o) => `${o.label}: ${o.cat}\n${pct(o.w)} of ${o.n_comparisons} comparisons (${o.n_pairs} design pairs)\nseed sd: ${o.sd_source}` }),
            Plot.text(segs.filter((o) => o.w >= 0.08 && o.cat.includes("clearly")), { x: (o) => (o.x1 + o.x2) / 2, y: "label", text: (o) => pct(o.w), fill: "white", fontSize: 10 }),
            Plot.text(segs.filter((o) => o.w >= 0.08 && !o.cat.includes("clearly")), { x: (o) => (o.x1 + o.x2) / 2, y: "label", text: (o) => pct(o.w), fill: css("--md-default-fg-color"), fontSize: 10 }),
          ],
        }),
        note("Every comparison is one design pair (in one language) ranked at the proxy and at 1.7B. Orange: the proxy picks the other winner (a DA-size miss); blue: it agrees. Strong shade: the proxy's difference exceeds its own seed noise, so a miss there is a real reversal; muted shade: the proxy could not have told the two variants apart. * the pairs are not size-matched (600M shallow has 3.7 % more parameters)."),
        h("h4", {}, "The depth reversal is the draw of one seed"),
        plot({
          height: 26 * seedLabels.length + 60, marginLeft: 170, marginRight: 60,
          x: { domain: [0, 1], tickFormat: "%", label: "share of the trained languages whose deep − shallow BPB has the other sign than at 1.7B", grid: true },
          y: { domain: seedLabels, label: null },
          marks: [
            Plot.barX(seedRows, { x: "share", y: "label", fill: colors()[3], rx: 3, insetTop: 3, insetBottom: 3, tip: true,
              title: (r) => `${r.size}, L${r.L}, deep seed ${r.deep_seed} vs shallow seed 1904\n${r.flips} of ${r.n} languages flip\nmedian deep − shallow BPB ${fmt(r.proxy, 4)} (1.7B: ${fmt(r.ref, 4)})` }),
            Plot.text(seedRows, { x: "share", y: "label", text: (r) => `${r.flips} of ${r.n}`, dx: 6, textAnchor: "start", fill: css("--md-default-fg-color"), fontSize: 10 }),
            Plot.ruleX([0], { stroke: css("--viz-muted") }),
          ],
        }),
        note("Shallow fixed at seed 1904, deep at each of its three seeds. The same 600M L50 pair flips on 30 %, 86 % or 100 % of its languages depending on the deep run's seed, and one 1B seed flips all 30 L30 languages where the other two flip none: the reversal is a draw of the seed, not a property of 600M."),
      ];
    });
  };
})();

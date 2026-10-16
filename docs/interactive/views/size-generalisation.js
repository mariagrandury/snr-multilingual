/* Additional interactive views of the size-generalisation page (registered on SNRSITE.VIEWS; see app.js).
 * Data: size_generalisation_extra.json (scripts/site_data/size_generalisation.py). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, note, css, fmt, pct, tiles } = S;
  const SCORE = { accuracy: "benchmark accuracy", bbpb: "bBPB (benchmark bits per byte)", bpb: "per-language BPB" };
  const PAIRS = [{ value: "mono-axis", label: "one design axis" }, { value: "multi-axis", label: "any pair" }];
  const chance = () => [
    Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
    Plot.text([0.5], { x: (v) => v, frameAnchor: "top", dy: -22, text: () => "chance", fill: css("--viz-muted"), textAnchor: "middle" }),
  ];

  // ---------- 1. the same decisions read against 1.7B and against 3B ----------
  S.VIEWS.sgReference = async (el) => {
    const x = await S.load("size_generalisation_extra");
    const REFS = ["against the 1.7B final", "against the 3B final"];
    S.mount(el, [
      { key: "kind", label: "Score", type: "seg", options: Object.entries(SCORE).map(([value, label]) => ({ value, label })), value: "bpb" },
      { key: "axes", label: "Design pairs", type: "seg", options: PAIRS },
    ], (s) => {
      const rows = x.reference.filter((r) => r.kind === s.kind && r.axes === s.axes)
        .map((r) => ({ ...r, ref: REFS[r.reference === "3B" ? 1 : 0] }));
      const sizes = S.SIZES.filter((z) => rows.some((r) => r.size === z));
      const span = d3.rollups(rows, (v) => d3.extent(v, (r) => r.da), (r) => r.size).map(([size, [lo, hi]]) => ({ size, lo, hi }));
      const top = rows.find((r) => r.size === "1.7B");
      return [
        tiles([
          [top ? pct(top.da) : "–", `of the 3B decisions the 1.7B model already makes (${SCORE[s.kind]})`],
          [`${pct(d3.min(rows.filter((r) => r.size !== "1.7B"), (r) => r.da))}–${pct(d3.max(rows.filter((r) => r.size !== "1.7B"), (r) => r.da))}`, "range over the proxies 90M–1B and both references"],
        ]),
        plot({
          height: 44 * sizes.length + 70, marginLeft: 60, marginTop: 30,
          x: { domain: [0.4, 1], label: "share of design pairs ordered like the reference (DA-size)", tickFormat: "%", grid: true },
          y: { domain: sizes, label: "proxy size" },
          color: { domain: REFS, range: [css("--viz-s12"), css("--viz-s1")], legend: true },
          marks: [
            ...chance(),
            Plot.ruleY(span, { y: "size", x1: "lo", x2: "hi", stroke: css("--viz-grid"), strokeWidth: 6, strokeLinecap: "round" }),
            Plot.dot(rows, { x: "da", y: "size", fill: "ref", r: 6, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.size} proxy ${r.ref}\n${r.n_matching} of ${r.n_pairs} decisions (${fmt(r.da)}) over ${r.n_tasks} task${r.n_tasks === 1 ? "" : "s"}` }),
          ],
        }),
        note(s.kind === "bpb"
          ? `Six configurations reach 3B, so this rests on ${d3.min(rows, (r) => r.n_tasks)}–${d3.max(rows, (r) => r.n_tasks)} BPB tasks (the languages every such configuration trains, English and the macro average). Per-language BPB keeps its ranking from 1.7B to 3B, and the proxies agree with either reference.`
          : `Benchmark tasks above chance at the proxy, at 1.7B and at 3B. The two dots of a row are the same decisions, so the gap between them is the reference alone: ${s.kind === "accuracy" ? `accuracy reads ${fmt(d3.min(rows, (r) => r.da))}–${fmt(d3.max(rows, (r) => r.da))} against either one` : "the bBPB twins move by a few points either way"}, and the 1.7B model itself orders the 3B decisions no better than chance.`),
      ];
    });
  };

  // ---------- 2. what 3B lifts above chance, per benchmark ----------
  S.VIEWS.sgGate = async (el) => {
    const x = await S.load("size_generalisation_extra");
    const g = x.gate;
    const tot = { n: d3.sum(g, (r) => r.n_tasks), ref: d3.sum(g, (r) => r.above_ref), top: d3.sum(g, (r) => r.above) };
    const STATES = ["share above chance at 1.7B", "share above chance at 3B"];
    S.mount(el, [{ key: "show", label: "Benchmarks", type: "seg", options: [
      { value: "moved", label: "those that change" }, { value: "all", label: `all ${g.length}` }] }], (s) => {
      const rows = g.filter((r) => s.show === "all" || r.gained + r.lost > 0)
        .sort((a, b) => b.delta - a.delta || b.share_above - a.share_above || a.benchmark.localeCompare(b.benchmark));
      const order = rows.map((r) => r.benchmark);
      const dots = rows.flatMap((r) => [
        { b: r.benchmark, state: STATES[0], share: r.share_ref, k: r.above_ref, n: r.n_tasks },
        { b: r.benchmark, state: STATES[1], share: r.share_above, k: r.above, n: r.n_tasks }]);
      return [
        tiles([[`${tot.ref} → ${tot.top}`, `of ${tot.n} benchmark tasks above chance, 1.7B → 3B (same six configurations)`],
          [`+${pct((tot.top - tot.ref) / tot.ref)}`, "more of the suite measurable at 3B"]]),
        plot({
          height: 20 * order.length + 80, marginLeft: 200, marginTop: 40,
          x: { domain: [0, 1], label: "share of the benchmark's tasks above chance", tickFormat: "%", grid: true, axis: "top" },
          y: { domain: order, label: null, tickSize: 0 },
          color: { domain: STATES, range: [css("--viz-s12"), css("--viz-s1")], legend: true },
          marks: [
            Plot.link(rows.filter((r) => r.share_above !== r.share_ref), { x1: "share_ref", x2: "share_above", y1: "benchmark", y2: "benchmark",
              stroke: css("--viz-s1"), strokeOpacity: 0.5, markerEnd: "arrow" }),
            Plot.dot(dots, { x: "share", y: "b", fill: "state", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.b}: ${r.k} of ${r.n} tasks (${pct(r.share)}) ${r.state.replace("share ", "")}` }),
          ],
        }),
        note("One row per benchmark, its tasks being its languages and subsets. The original letter-format knowledge benchmarks (Global-MMLU, INCLUDE, Belebele, CulturalBench) cross into measurability at 3B; their reformulated versions (rf_, rfgm_) were already above chance at 1.7B."),
      ];
    });
  };

  // ---------- 3. how much the choice of reference moves DA-size ----------
  S.VIEWS.sgSeeds = async (el) => {
    const x = await S.load("size_generalisation_extra");
    const BLOCK = { "1B seeds": "three seeds of a 1B reference", "3B K=30": "the 3B model at K = 30" };
    S.mount(el, [
      { key: "block", label: "Reference", type: "seg", options: Object.entries(BLOCK).map(([value, label]) => ({ value, label })) },
      { key: "axes", label: "Design pairs", type: "seg", options: PAIRS },
    ], (s) => {
      const rows = x.seeds.filter((r) => r.block === s.block && r.axes === s.axes && r.da != null);
      const readings = [...new Set(rows.map((r) => r.reading))];
      const sizes = S.SIZES.filter((z) => rows.some((r) => r.size === z));
      const span = d3.rollups(rows, (v) => d3.extent(v, (r) => r.da), (r) => r.size).map(([size, [lo, hi]]) => ({ size, lo, hi, gap: hi - lo }));
      const widest = d3.greatest(span, (r) => r.gap);
      return [
        tiles([[widest ? fmt(widest.gap) : "–", `widest spread across references (${widest?.size ?? "–"} proxy)`]]),
        plot({
          height: 44 * sizes.length + 70, marginLeft: 60, marginTop: 30,
          x: { domain: [0.35, 0.85], label: "DA-size of benchmark accuracy (with the bBPB twins)", tickFormat: "%", grid: true },
          y: { domain: sizes, label: "proxy size" },
          color: { domain: readings, range: S.series(readings.length), legend: true },
          marks: [
            ...chance(),
            Plot.ruleY(span, { y: "size", x1: "lo", x2: "hi", stroke: css("--viz-grid"), strokeWidth: 6, strokeLinecap: "round" }),
            Plot.dot(rows, { x: "da", y: "size", fill: "reading", r: 6, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.size} proxy, reference ${r.reading}\n${r.n_matching} of ${r.n_pairs} decisions (${fmt(r.da)}) over ${r.n_tasks} tasks` }),
          ],
        }),
        note(s.block === "1B seeds"
          ? "The four designs trained with three seeds at 1B, read against each 1B seed and against the 1.7B model. Within a row only the reference changes: its seed alone moves the agreement by up to 0.13, more than replacing the 1B reference with the 1.7B one."
          : "The decisions between the one 3B model at K = 30 and the five other 3B models, read against 3B and against 1.7B, and every decision among the six 3B models. Benchmark accuracy stays near chance whichever reference is used."),
      ];
    });
  };
})();

/* Additional interactive views of the size-generalisation page (registered on SNRSITE.VIEWS; see app.js).
 * Data: size_generalisation_extra.json (scripts/site_data/size_generalisation.py). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, note, css, fmt, pct, tiles } = S;
  const row = (...nodes) => h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;justify-content:center" }, ...nodes);
  const titled = (title, node) => h("div", { style: `display:flex;flex-direction:column;align-items:center;max-width:${node.getAttribute("width") ?? 720}px` },
    h("div", { style: "font-size:.7rem;font-weight:600;margin:.4em 0 .1em;text-align:center" }, title), node);
  const PAIRS = [{ value: "mono-axis", label: "one design axis" }, { value: "multi-axis", label: "any pair" }];
  const chance = () => [
    Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
    Plot.text([0.5], { x: (v) => v, frameAnchor: "top", dy: -22, text: () => "chance", fill: css("--viz-muted"), textAnchor: "middle" }),
  ];

  // ---------- 1. the same decisions read against 1.7B and against 3B (paper App. E.6) ----------
  S.VIEWS.sgReference = async (el) => {
    const x = await S.load("size_generalisation_extra");
    const REFS = ["against the 1.7B final", "against the 3B final"];
    // The paper pools benchmark accuracy with bBPB and keeps per-language BPB apart (App. E.6).
    const PANELS = {
      paper: [["benchmark accuracy (with bBPB)", ["accuracy", "bbpb"]], ["per-language BPB", ["bpb"]]],
      split: [["benchmark accuracy", ["accuracy"]], ["bBPB", ["bbpb"]], ["per-language BPB", ["bpb"]]],
    };
    S.mount(el, [
      { key: "split", label: "Scores", type: "seg", options: [
        { value: "paper", label: "accuracy with bBPB, and BPB (paper)" }, { value: "split", label: "accuracy, bBPB and BPB apart" }] },
      { key: "axes", label: "Design pairs", type: "seg", options: [
        { value: "mono-axis", label: "single-axis (paper)" }, { value: "multi-axis", label: "every pair (multi-axis)" }] },
    ], (s) => {
      const panels = PANELS[s.split];
      const rows = panels.flatMap(([panel, kinds]) => d3.rollups(
        x.reference.filter((r) => r.axes === s.axes && kinds.includes(r.kind)),
        (v) => ({ n_matching: d3.sum(v, (r) => r.n_matching), n_pairs: d3.sum(v, (r) => r.n_pairs), n_tasks: d3.sum(v, (r) => r.n_tasks) }),
        (r) => r.size, (r) => r.reference,
      ).flatMap(([size, refs]) => refs.map(([reference, t]) => ({ panel, size, ref: REFS[reference === "3B" ? 1 : 0], ...t, da: t.n_matching / t.n_pairs }))));
      const sizes = S.SIZES.filter((z) => rows.some((r) => r.size === z));
      const span = d3.rollups(rows, (v) => d3.extent(v, (r) => r.da), (r) => r.panel, (r) => r.size)
        .flatMap(([panel, v]) => v.map(([size, [lo, hi]]) => ({ panel, size, lo, hi })));
      const at = (panel, pred) => rows.filter((r) => r.panel === panel && pred(r));
      const range = (panel) => {
        const v = at(panel, (r) => r.size !== "1.7B" && r.ref === REFS[1]);
        return `${fmt(d3.min(v, (r) => r.da))}–${fmt(d3.max(v, (r) => r.da))}`;
      };
      const top = (panel) => at(panel, (r) => r.size === "1.7B")[0];
      const acc = panels[0][0], bpb = panels[panels.length - 1][0];
      return [
        tiles([
          [`${fmt(top(acc)?.da)} / ${fmt(top(bpb)?.da)}`, `DA-size of the 1.7B models against 3B: ${acc} / per-language BPB`],
          [range(acc), `${acc}: DA-size of the proxies 90M–1B against 3B`],
          [range(bpb), "per-language BPB: DA-size of the proxies 90M–1B against 3B"],
        ]),
        row(Plot.legend({ color: { domain: REFS, range: [css("--viz-s12"), css("--viz-s1")] } })),
        ...panels.map(([panel], i) => row(titled(panel, plot({
          width: 640, height: 30 * sizes.length + (i === panels.length - 1 ? 50 : 30), marginLeft: 60, marginTop: 10,
          x: { domain: [0.4, 1], label: i === panels.length - 1 ? "share of design pairs ordered like the reference (DA-size)" : null, tickFormat: "%", grid: true },
          y: { domain: sizes, label: null },
          color: { domain: REFS, range: [css("--viz-s12"), css("--viz-s1")] },
          marks: [
            Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
            Plot.ruleY(span.filter((r) => r.panel === panel), { y: "size", x1: "lo", x2: "hi", stroke: css("--viz-grid"), strokeWidth: 6, strokeLinecap: "round" }),
            Plot.dot(rows.filter((r) => r.panel === panel), { x: "da", y: "size", fill: "ref", r: 5.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.panel}, ${r.size} ${r.size === "1.7B" ? "model" : "proxy"} ${r.ref}\n${r.n_matching} of ${r.n_pairs} decisions (${fmt(r.da)}) over ${r.n_tasks} task${r.n_tasks === 1 ? "" : "s"}` }),
          ],
        })))),
        note(`The six deep configurations trained to 3B at seed 1904 (scheme A at K ∈ {8, 15, 30, 50}, scheme B at K ∈ {8, 15}), on the ${s.axes === "mono-axis" ? "single-axis" : "multi-axis"} pairs and the tasks above chance at the proxy, at 1.7B and at 3B. `
          + "Each row reads the same decisions against the 1.7B and the 3B final checkpoint, so its two dots differ in the reference alone. "
          + `Against 3B the proxies reach ${range(acc)} on ${acc === "benchmark accuracy" ? "accuracy" : "accuracy (pooled with bBPB, which five of the six 3B models have)"} and ${range(bpb)} on per-language BPB, `
          + `and the 1.7B models themselves only ${fmt(top(acc)?.da)} and ${fmt(top(bpb)?.da)}: a larger reference does not make the proxies agree more.`),
      ];
    });
  };

  // ---------- 2. what 3B lifts above chance, per benchmark (paper App. E.6) ----------
  S.VIEWS.sgGate = async (el) => {
    const x = await S.load("size_generalisation_extra");
    const g = x.gate;
    const tot = { n: d3.sum(g, (r) => r.n_tasks), ref: d3.sum(g, (r) => r.above_ref), top: d3.sum(g, (r) => r.above) };
    const STATES = ["share above chance at 1.7B", "share above chance at 3B"];
    S.mount(el, [{ key: "show", label: "Benchmarks", type: "seg", options: [
      { value: "moved", label: "those that change" }, { value: "all", label: `all ${g.length}` }] }], (s) => {
      const rows = g.filter((r) => s.show === "all" || r.gained + r.lost > 0)
        .sort((a, b) => b.delta - a.delta || b.share_above - a.share_above || a.name.localeCompare(b.name));
      const order = rows.map((r) => r.name);
      const dots = rows.flatMap((r) => [
        { b: r.name, state: STATES[0], share: r.share_ref, k: r.above_ref, n: r.n_tasks },
        { b: r.name, state: STATES[1], share: r.share_above, k: r.above, n: r.n_tasks }]);
      return [
        tiles([[`${tot.ref} → ${tot.top}`, `of ${tot.n} benchmark tasks above chance, 1.7B → 3B (the same six configurations)`],
          [`+${pct((tot.top - tot.ref) / tot.ref)}`, "more of the suite measurable at 3B"]]),
        row(plot({
          height: 20 * order.length + 80, marginLeft: 200, marginTop: 40,
          x: { domain: [0, 1], label: "share of the benchmark's tasks above chance", tickFormat: "%", grid: true, axis: "top" },
          y: { domain: order, label: null, tickSize: 0 },
          color: { domain: STATES, range: [css("--viz-s12"), css("--viz-s1")], legend: true },
          marks: [
            Plot.link(rows.filter((r) => r.share_above !== r.share_ref), { x1: "share_ref", x2: "share_above", y1: "name", y2: "name",
              stroke: css("--viz-s1"), strokeOpacity: 0.5, markerEnd: "arrow" }),
            Plot.dot(dots, { x: "share", y: "b", fill: "state", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.b}: ${r.k} of ${r.n} tasks (${pct(r.share)}) ${r.state.replace("share ", "")}` }),
          ],
        })),
        note("One row per benchmark, its tasks being its languages and subsets; the above-chance threshold is recomputed on the six configurations trained to 3B. Most of the gain is the letter-format Belebele, Global-MMLU and INCLUDE tasks, which cross the threshold only at 3B; their RF and LLM-RF versions were already above chance at 1.7B."),
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

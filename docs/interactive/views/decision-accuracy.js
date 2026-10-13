/* Additional interactive views of the decision-accuracy page (registered on SNRSITE.VIEWS; see app.js).
 * Data: decision_accuracy_pairs.json and decision_accuracy_extra.json
 * (scripts/site_data/decision_accuracy.py). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, note, css, fmt, pct, tiles } = S;

  // ---------- local helpers ----------
  const sizeColors = (n) => {
    const r = S.sizeRamp();
    return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), Math.max(n, 2));
  };
  const howTo = (text) => h("p", { class: "viz-note", style: "margin-top:0;font-style:italic" }, text);
  const AXIS = { L: "language count", arch: "depth", activation: "activation", optimizer: "optimizer", scheme: "data scheme", T: "sampling temperature" };
  const chanceMarks = (y0) => [
    Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
    Plot.text([0.5], { x: (v) => v, frameAnchor: "top", dy: y0 ?? -14, text: () => "chance", fill: css("--viz-muted"), textAnchor: "middle" }),
  ];

  /* A control bar the caller fills, an output area, and a redraw hook (for
     controls that depend on each other, which S.mount does not do). */
  function frame(el, controls, draw) {
    const out = h("div");
    const render = () => out.replaceChildren(...[draw()].flat().filter(Boolean));
    el.replaceChildren(h("div", { class: "viz-controls" }, controls(render)), out);
    S.REDRAW.push(render);
    render();
  }
  const select = (label, options, value, set) => {
    const sel = h("select", { onchange: (e) => set(e.target.value) },
      options.map(([v, text]) => h("option", { value: v }, text)));
    sel.value = value;
    return h("label", {}, label, sel);
  };
  const seg = (label, options, value, set) => {
    const btns = options.map(([v, text]) => h("button", {
      type: "button", "aria-pressed": String(v === value),
      onclick: (e) => { btns.forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget))); set(v); },
    }, text));
    return h("label", {}, label, h("span", { class: "viz-seg" }, btns));
  };

  // ---------- 1. the checkpoint slider: agreeing with the reference vs with oneself ----------
  S.VIEWS.daCheckpoints = async (el) => {
    const x = await S.load("decision_accuracy_extra");
    const SERIES = ["against the 1.7B final (DA-goal)", "against its own final (DA-ckpt)", "two seeds of one design (seed null)"];
    const rowsAt = (score, f) => {
      const rows = [];
      const grp = score === "bpb" ? "bpb" : "all benchmarks";
      for (const r of x.early.filter((r) => r.group === grp && Math.abs(r.frac - f) < 1e-6)) {
        rows.push({ size: r.proxy_size, series: SERIES[r.proxy_size === "1.7B" ? 1 : 0], da: r.da, n: `${r.tasks} ${score === "bpb" ? "languages" : "tasks"}` });
      }
      if (f < 1) {
        if (score === "bpb") {
          for (const r of x.ckpt_bpb.filter((r) => r.proxy_size !== "1.7B" && Math.abs(r.frac - f) < 1e-6))
            rows.push({ size: r.proxy_size, series: SERIES[1], da: r.da, n: `${r.n_languages} languages` });
        } else {
          for (const r of x.seed_null.filter((r) => Math.abs(r.frac - f) < 1e-6)) {
            rows.push({ size: r.size, series: SERIES[1], da: r.da_real, n: `${r.tasks} tasks` });
            rows.push({ size: r.size, series: SERIES[2], da: r.da, n: `${r.tasks} tasks` });
          }
        }
      }
      return rows;
    };
    const st = { f: 0.5, score: "bench" };
    const out = h("div");
    const valOut = h("output", {}, "50 %");
    const render = () => {
      const rows = rowsAt(st.score, st.f);
      const sizes = S.LADDER.filter((z) => rows.some((r) => r.size === z));
      const span = d3.rollups(rows, (v) => [d3.min(v, (r) => r.da), d3.max(v, (r) => r.da)], (r) => r.size).map(([size, [lo, hi]]) => ({ size, lo, hi }));
      const colors = [css("--viz-s1"), css("--viz-s2"), css("--viz-s12")];
      out.replaceChildren(plot({
        height: 46 * sizes.length + 70, marginLeft: 70, marginTop: 30,
        x: { domain: [0.4, 1], label: "share of design pairs ordered the same way", tickFormat: "%", grid: true },
        y: { domain: sizes, label: null, tickFormat: (s) => (s === "1.7B" ? "1.7B (ref.)" : s) },
        color: { domain: SERIES, range: colors, legend: true },
        marks: [
          ...chanceMarks(-22),
          Plot.ruleY(span, { y: "size", x1: "lo", x2: "hi", stroke: css("--viz-grid"), strokeWidth: 6, strokeLinecap: "round" }),
          Plot.dot(rows, { x: "da", y: "size", fill: "series", r: 6, stroke: css("--viz-surface"), strokeWidth: 2, tip: true,
            title: (r) => `${r.size}, checkpoint at ${pct(st.f)} of the run\n${r.series}: ${pct(r.da)} (${fmt(r.da)}) over ${r.n}` }),
        ],
      }), note(st.f === 1
        ? "At the final checkpoint the comparison with the run's own final is trivially 100 % and the seed null is not defined: what is left is DA-size, the proxy's final against the 1.7B final."
        : st.score === "bpb"
          ? "Per-language BPB, mean over the 50 trained languages. The proxies' early checkpoints already order the pairs like the 1.7B final, except 600M, whose ranking settles early (high agreement with its own final) on an order the reference does not share."
          : "Benchmarks: accuracy and the bBPB twins, mean over the gated tasks. A proxy's checkpoint agrees with its own final more and more as the run goes on, but two seeds of the same design, which have nothing to decide, agree almost as much: most of that rise is persistence within a run. Against the 1.7B final the proxies stay near 0.6 at every checkpoint. The seed replicates exist at 175M, 600M and 1B only."));
    };
    el.replaceChildren(h("div", { class: "viz-controls" },
      h("label", {}, h("span", {}, "Checkpoint, share of the run ", valOut),
        h("input", { type: "range", min: 0.1, max: 1, step: 0.1, value: 0.5,
          oninput: (e) => { st.f = Math.round(+e.target.value * 10) / 10; valOut.textContent = pct(st.f); render(); } })),
      seg("Score", [["bench", "benchmarks"], ["bpb", "per-language BPB"]], st.score, (v) => { st.score = v; render(); })), out);
    S.REDRAW.push(render);
    render();
  };

  // ---------- 2. which score ranks like the reference: one dot per proxy size ----------
  S.VIEWS.daChannels = async (el) => {
    const x = await S.load("decision_accuracy_extra");
    const sizes = x.channels.map((r) => r.size);
    const bpbLang = Object.fromEntries(x.early.filter((r) => r.group === "bpb" && r.frac === 1).map((r) => [r.proxy_size, r.da]));
    const atChance = Object.fromEntries(x.acc_gate.filter((r) => r.population === "acc at chance at the proxy").map((r) => [r.size, r]));
    const macro = Object.fromEntries(x.acc_gate.filter((r) => r.population === "acc above chance at the proxy").map((r) => [r.size, r.bpb_macro]));
    const chanceBbpb = d3.extent(Object.values(atChance), (r) => r.bbpb_bbpb);
    S.mount(el, [{ key: "pop", label: "Benchmark tasks", type: "seg", options: [
      { value: "all", label: "every gated task" }, { value: "chance", label: "accuracy still at chance at the proxy" }] }], (s) => {
      const R = s.pop === "all"
        ? [["accuracy → 1.7B accuracy", (z) => [x.channels.find((r) => r.size === z).acc_acc, x.channels.find((r) => r.size === z).acc_acc_n]],
          ["bBPB → 1.7B accuracy", (z) => [x.channels.find((r) => r.size === z).bbpb_acc, x.channels.find((r) => r.size === z).bbpb_acc_n]],
          ["bBPB → 1.7B bBPB", (z) => [x.channels.find((r) => r.size === z).bbpb_bbpb, x.channels.find((r) => r.size === z).bbpb_bbpb_n]]]
        : [["accuracy → 1.7B accuracy", (z) => [atChance[z]?.acc_acc, atChance[z]?.acc_acc_n]],
          ["bBPB → 1.7B bBPB", (z) => [atChance[z]?.bbpb_bbpb, atChance[z]?.bbpb_bbpb_n]]];
      R.push(["per-language BPB (50 languages)", (z) => [bpbLang[z], 50]], ["BPB, macro average (one task)", (z) => [macro[z], 1]]);
      const rows = R.flatMap(([reading, get]) => sizes.map((size) => {
        const [da, n] = get(size);
        return { reading, size, da, n };
      })).filter((r) => r.da != null);
      return [plot({
        height: 52 * R.length + 80, marginLeft: 220, marginTop: 30,
        x: { domain: [0.4, 1], label: "decision accuracy against the 1.7B final (DA-size)", tickFormat: "%", grid: true },
        y: { domain: R.map((r) => r[0]), label: null },
        color: { domain: sizes, range: sizeColors(sizes.length), legend: true, label: "proxy size" },
        marks: [
          ...chanceMarks(-22),
          Plot.ruleY(d3.rollups(rows, (v) => d3.extent(v, (r) => r.da), (r) => r.reading).map(([reading, [lo, hi]]) => ({ reading, lo, hi })),
            { y: "reading", x1: "lo", x2: "hi", stroke: css("--viz-grid"), strokeWidth: 6, strokeLinecap: "round" }),
          Plot.dot(rows, { x: "da", y: "reading", fill: "size", r: 6, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (r) => `${r.reading}\n${r.size} proxy: ${pct(r.da)} (${fmt(r.da)}), mean over ${r.n} task${r.n === 1 ? "" : "s"}` }),
        ],
      }), note(s.pop === "all"
        ? "Each dot: one proxy size, mean DA-size over the tasks above chance at the proxy and at 1.7B (bBPB needs the reference above chance only). Accuracy sits just above chance at every size; scoring the same items by the gold answer's bits per byte (bBPB) gains a few points; the per-language BPB of the training languages agrees with the reference on more than 90 % of the decisions, except at 600M, where the depth pairs are not size-matched. A shuffled proxy scores 0.49 on benchmarks and 0.50 on BPB (permutation null)."
        : `The tasks a proxy cannot yet answer above chance. Their accuracy ranks the designs below chance, while their bBPB still ranks them like the 1.7B bBPB on ${pct(chanceBbpb[0]).replace(" %", "")}–${pct(chanceBbpb[1])} of the decisions: scoring by log-likelihood lifts these items above chance, but not to the level of BPB.`)];
    });
  };

  // ---------- 3. from chance to decisive pairs ----------
  S.VIEWS.daDecisive = async (el) => {
    const x = await S.load("decision_accuracy_extra");
    const POP = { "all benchmarks": "benchmarks (accuracy + bBPB)", bpb: "per-language BPB" };
    S.mount(el, [{ key: "k", label: "Keep the pairs the 1.7B reference separates by more than", type: "seg", options: [
      { value: 1, label: "1 seed sd" }, { value: 2, label: "2 seed sd" }] }], (s) => {
      const rows = [];
      for (const r of x.decisive.filter((r) => r.axes === "multi-axis" && (r.k === 0 || r.k === s.k))) {
        const nl = x.null.find((n) => n.axes === r.axes && n.population === r.population && n.size === r.size);
        rows.push({ pop: POP[r.population], size: r.size, kind: r.k === 0 ? "every pair" : "decisive pairs only", da: r.da, kept: r.share_kept, tasks: r.n_tasks, nullm: nl?.null_mean });
      }
      const sizes = S.LADDER.filter((z) => rows.some((r) => r.size === z));
      const KINDS = ["every pair", "decisive pairs only"];
      const nulls = rows.filter((r) => r.kind === KINDS[0] && r.nullm != null);
      const kept = rows.filter((r) => r.kind === KINDS[1]);
      const share = (p) => kept.find((r) => r.pop === p)?.kept;
      const das = (pop, k) => x.decisive.filter((r) => r.axes === "multi-axis" && r.population === pop && r.k === k).map((r) => r.da);
      const f3 = (v) => fmt(v, v >= 0.995 ? 3 : 2);
      const span = (pop, k) => `${f3(d3.min(das(pop, k)))}–${f3(d3.max(das(pop, k)))}`;
      const links = sizes.flatMap((z) => Object.values(POP).map((p) => {
        const a = rows.find((r) => r.size === z && r.pop === p && r.kind === KINDS[0]);
        const b = rows.find((r) => r.size === z && r.pop === p && r.kind === KINDS[1]);
        return a && b ? { size: z, pop: p, x1: a.da, x2: b.da } : null;
      }).filter(Boolean));
      return [tiles([[pct(share(POP["all benchmarks"])), `of the benchmark pairs are decisive at ${s.k} seed sd`],
        [pct(share(POP.bpb)), `of the BPB pairs are decisive at ${s.k} seed sd`]]),
      plot({
        height: 2 * (28 * sizes.length) + 120, marginLeft: 60, marginTop: 30,
        fy: { domain: Object.values(POP), label: null },
        x: { domain: [0.4, 1], label: "DA-size against the 1.7B final, pooled over pairs", tickFormat: "%", grid: true },
        y: { domain: sizes, label: null },
        color: { domain: KINDS, range: [css("--viz-s12"), css("--viz-s1")], legend: true },
        marks: [
          Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.tickX(nulls, { x: "nullm", y: "size", fy: "pop", stroke: css("--viz-muted"), strokeWidth: 2, tip: true,
            title: (r) => `${r.pop}, ${r.size}\nshuffled proxy (no-signal null): ${fmt(r.nullm)}` }),
          Plot.link(links, { x1: "x1", x2: "x2", y1: "size", y2: "size", fy: "pop", stroke: css("--viz-s1"), strokeOpacity: 0.5, markerEnd: "arrow" }),
          Plot.dot(rows, { x: "da", y: "size", fy: "pop", fill: "kind", r: 5.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (r) => `${r.pop}, ${r.size}\n${r.kind}: ${fmt(r.da)}` + (r.kind === KINDS[1] ? ` (${pct(r.kept)} of the pairs kept)` : "") + ` · ${r.tasks} tasks` }),
        ],
      }), note(`Dark tick: what a proxy scores with its final scores shuffled across the designs (the no-signal null, measured cell by cell). Grey dot: every pair. Blue dot: only the pairs whose 1.7B gap exceeds k √2 seed standard deviations, the pairs the reference itself decides. On benchmarks DA rises from about ${fmt(d3.mean(das("all benchmarks", 0)))} to ${span("all benchmarks", 1)} (1 sd) and ${span("all benchmarks", 2)} (2 sd), but the decisive pairs are only half and a quarter of the pairs. BPB reaches ${span("bpb", 1)} and ${span("bpb", 2)}, and its 600M dip shrinks.`)];
    });
  };

  // ---------- 4. the pair explorer ----------
  S.VIEWS.daPairs = async (el) => {
    await S.languages();
    const d = await S.load("decision_accuracy_pairs");
    const F = d.families;
    const byKey = new Map(d.pairs.map((p) => [`${p.a}-${p.b}`, p]));
    const find = (a, b) => (a < b ? { p: byKey.get(`${a}-${b}`), flip: false } : { p: byKey.get(`${b}-${a}`), flip: true });
    const idx = (label) => F.findIndex((f) => f.label === label);
    const st = { a: Math.max(0, idx("L8 · deep · list A")), b: Math.max(0, idx("L8 · shallow · list A")), score: "bpb" };
    if (st.a === st.b) st.b = (st.a + 1) % F.length;
    const SCORE = { bpb: "per-language BPB", bbpb: "bBPB", acc: "accuracy" };
    const UNIT = { bpb: "languages", bbpb: "tasks", acc: "tasks" };
    const langLabel = (t) => (t === "dclm" ? "English (DCLM)" : S.fwName(t));
    // DA over every pair at each size, for the "average pair" tick
    const pooled = { acc: [], bbpb: [], bpb: [] };
    d.sizes.forEach((_, i) => {
      for (const k of ["acc", "bbpb"]) pooled[k][i] = d3.sum(d.pairs, (p) => p[k][i][0]) / d3.sum(d.pairs, (p) => p[k][i][1]);
      const ag = d3.sum(d.pairs, (p) => (p.bpb[i].match(/1/g) || []).length);
      pooled.bpb[i] = ag / (ag + d3.sum(d.pairs, (p) => (p.bpb[i].match(/0/g) || []).length));
    });
    const optsB = () => F.map((f, i) => [i, f]).filter(([i]) => i !== st.a).map(([i, f]) => {
      const { p } = find(st.a, i);
      return [String(i), `${f.label}${p ? ` (${p.axes.map((a) => AXIS[a] ?? a).join(" + ")})` : ""}`];
    });
    frame(el, (render) => {
      let selB;
      const selA = select("Design A", F.map((f, i) => [String(i), f.label]), String(st.a), (v) => {
        st.a = +v;
        if (st.b === st.a) st.b = (st.a + 1) % F.length;
        const nb = select("Design B", optsB(), String(st.b), (w) => { st.b = +w; render(); });
        selB.replaceWith(nb); selB = nb;
        render();
      });
      selB = select("Design B", optsB(), String(st.b), (v) => { st.b = +v; render(); });
      return [selA, selB, seg("Score", [["bpb", SCORE.bpb], ["bbpb", SCORE.bbpb], ["acc", SCORE.acc]], st.score, (v) => { st.score = v; render(); })];
    }, () => {
      const { p, flip } = find(st.a, st.b);
      if (!p) return h("div", { class: "viz-empty" }, "These two designs do not form a pair.");
      const A = F[st.a].label, B = F[st.b].label, k = st.score;
      // the 1.7B verdict, in A/B terms
      let refA, refB, refN;
      if (k === "bpb") {
        const s = p.ref.bpb.replace(/\./g, "");
        const ca = (s.match(/a/g) || []).length, cb = (s.match(/b/g) || []).length;
        [refA, refB, refN] = flip ? [cb, ca, s.length] : [ca, cb, s.length];
      } else {
        const [ca, cb, ct] = p.ref[k];
        [refA, refB, refN] = flip ? [cb, ca, ca + cb + ct] : [ca, cb, ca + cb + ct];
      }
      const rows = d.sizes.map((size, i) => {
        let agree, n;
        if (k === "bpb") {
          agree = (p.bpb[i].match(/1/g) || []).length;
          n = agree + (p.bpb[i].match(/0/g) || []).length;
        } else [agree, n] = p[k][i];
        return { size, agree, n, share: n ? agree / n : null, avg: pooled[k][i] };
      }).filter((r) => r.n > 0);
      if (!rows.length) return h("div", { class: "viz-empty" }, `No ${SCORE[k]} comparison for this pair: ${k === "bpb" ? "the two designs share no trained language with a BPB score at both sizes (the Muon runs have no BPB pair yet)" : "no task is above chance for both designs"}.`);
      const bars = rows.flatMap((r) => [
        { size: r.size, part: "same winner as 1.7B", v: r.share, k: r.agree, n: r.n },
        { size: r.size, part: "opposite winner", v: 1 - r.share, k: r.n - r.agree, n: r.n }]);
      const flags = [];
      if (p.axes.includes("arch")) flags.push("At 600M the depth pair is not size-matched: the shallow model has 3.7 % more non-embedding parameters and about 15 % more compute.");
      const win = refA === refB ? "neither design" : refA > refB ? "A" : "B";
      const nodes = [
        tiles([
          [p.axes.map((a) => AXIS[a] ?? a).join(" + "), p.axes.length === 1 ? "the one design axis this pair moves" : "the design axes this pair moves"],
          [`${refA} : ${refB}`, `${UNIT[k]} on which 1.7B prefers A : B (${SCORE[k]}${refN - refA - refB ? `, ${refN - refA - refB} tied` : ""})`],
          [win === "neither design" ? "tie" : win, "the 1.7B reference's pick on most " + UNIT[k]],
        ]),
        plot({
          height: 34 * rows.length + 80, marginLeft: 60, marginRight: 110, marginTop: 30,
          x: { domain: [0, 1], label: `share of ${UNIT[k]} on which the proxy picks the same winner as 1.7B`, tickFormat: "%", grid: true },
          y: { domain: rows.map((r) => r.size), label: "proxy size" },
          color: { domain: ["same winner as 1.7B", "opposite winner"], range: [css("--viz-s1"), css("--viz-s2")], legend: true },
          marks: [
            Plot.barX(bars, Plot.stackX({ x: "v", y: "size", fill: "part", order: ["same winner as 1.7B", "opposite winner"], insetTop: 4, insetBottom: 4, insetRight: 1, rx: 2, tip: true,
              title: (r) => `${r.size}: ${r.part} on ${r.k} of ${r.n} ${UNIT[k]} (${pct(r.v)})` })),
            Plot.tickX(rows, { x: "avg", y: "size", stroke: css("--md-default-fg-color"), strokeWidth: 2, strokeDasharray: "2 2",
              title: (r) => `average over all ${d.pairs.length} pairs at ${r.size}: ${pct(r.avg)}` }),
            Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
            Plot.text(rows, { x: 1, y: "size", dx: 8, textAnchor: "start", fill: css("--md-default-fg-color"), text: (r) => `${r.agree} of ${r.n}` }),
          ],
        }),
      ];
      if (k === "bpb") {
        // one square per language: does each proxy pick that language's 1.7B winner?
        const langs = d.bpb_tasks.map((t, j) => ({ t, j })).filter(({ j }) => p.ref.bpb[j] !== ".");
        const order = langs.map(({ t }) => langLabel(t)).sort((x, y) => x.localeCompare(y));
        const pick = (c) => (c === "=" ? "tie" : (c === "a") !== flip ? "A" : "B");
        const cells = d.sizes.flatMap((size, i) => langs.map(({ t, j }) => {
          const c = p.bpb[i][j], refPick = pick(p.ref.bpb[j]);
          return { size, lang: langLabel(t), state: c === "1" ? "same winner as 1.7B" : c === "0" ? "opposite winner" : "not comparable",
            refPick, own: c === "1" ? refPick : refPick === "A" ? "B" : refPick === "B" ? "A" : "a winner" };
        }));
        nodes.push(plot({
          height: 26 * d.sizes.length + 150, marginLeft: 60, marginBottom: 110, width: Math.max(360, 18 * langs.length + 80),
          x: { domain: order, label: null, tickRotate: -60 },
          y: { domain: d.sizes, label: "proxy size" },
          color: { domain: ["same winner as 1.7B", "opposite winner", "not comparable"], range: [css("--viz-s1"), css("--viz-s2"), css("--viz-neutral")] },
          marks: [Plot.cell(cells, { x: "lang", y: "size", fill: "state", inset: 1, rx: 2, tip: true,
            title: (c) => `${c.lang} at ${c.size}: ${c.state}\n1.7B prefers ${c.refPick === "tie" ? "neither (tie)" : `design ${c.refPick}`}` +
              (c.state === "not comparable" ? "" : `; the ${c.size} proxy prefers ${c.state === "same winner as 1.7B" ? `design ${c.refPick}` : c.own === "a winner" ? "one of them" : `design ${c.own}`}`) })],
        }));
        nodes.push(note(`One square per language that both designs train (the BPB of a language neither design trains is not compared). Lower BPB wins. A = ${A}; B = ${B}.`));
      }
      nodes.push(note(`Bars: on how many ${UNIT[k]} each proxy's final checkpoint picks the same winner as the 1.7B final. Dashed tick: the same share averaged over all ${d.pairs.length} pairs of designs, the DA-size of the page's figures (pooled). ${k === "acc" ? "Accuracy counts the tasks above chance at the proxy and at 1.7B; a pair tied on one side counts as a miss." : k === "bbpb" ? "bBPB: bits per byte of the gold answer on the same benchmark items." : ""} ${flags.join(" ")}`));
      return nodes;
    });
  };
})();

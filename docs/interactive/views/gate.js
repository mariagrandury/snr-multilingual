/* Additional interactive views of the gate page (registered on SNRSITE.VIEWS; see app.js).
 * Data: gate.json (scripts/build_site_data.py) and gate_extra.json
 * (scripts/site_data/gate.py). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, note, css, fmt, pct, uniq } = S;

  // ---------- local helpers ----------
  const sizeColors = (n) => {
    const r = S.sizeRamp();
    return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), Math.max(n, 2));
  };
  const name = (names, b) => names?.[b] ?? b;
  const isTwin = (b) => /^rf(gm)?_/.test(b);
  // The gate's lower bound: one-sided 95 % Wilson bound of an accuracy p over n items.
  const Z = 1.6448536269514722;
  const wilson = (p, n) => {
    const z2 = Z * Z;
    return (p + z2 / (2 * n) - Z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  };
  // The smallest accuracy whose bound over n items clears chance c (null: none does).
  const needed = (n, c) => {
    for (let k = Math.max(0, Math.floor(c * n)); k <= n; k++) if (wilson(k / n, n) > c) return k / n;
    return null;
  };
  const tok = (x) => (x >= 1e9 ? `${+(x / 1e9).toPrecision(2)} B` : x >= 1e6 ? `${+(x / 1e6).toPrecision(2)} M` : `${+(x / 1e3).toPrecision(2)} k`);
  const howTo = (text) => h("p", { class: "viz-note", style: "margin-top:0;font-style:italic" }, text);

  /* A control bar the caller fills, an output area, and a redraw hook. */
  function frame(el, controls, draw) {
    const out = h("div");
    const render = () => out.replaceChildren(...[draw()].flat().filter(Boolean));
    el.replaceChildren(h("div", { class: "viz-controls" }, controls(render)), out);
    S.REDRAW.push(render);
    render();
  }

  // ---------- 1. the first-size map, condensed to one bar per benchmark ----------
  S.VIEWS.gateFirstShare = async (el) => {
    const g = await S.languages().then(() => S.load("gate"));
    const fs = g.first_size;
    const levels = [...fs.levels, "never"];
    S.mount(el, [
      { key: "twins", label: "Versions", type: "seg", options: [{ value: "orig", label: "as published" }, { value: "all", label: "with rewritten versions" }] },
    ], (s) => {
      const cells = fs.cells.filter(([b]) => s.twins === "all" || !isTwin(b));
      const counts = d3.rollup(cells, (v) => v.length, ([b]) => b, ([, , lv]) => lv);
      const total = (b) => d3.sum(counts.get(b).values());
      const share = (b, lv) => (counts.get(b).get(lv) ?? 0) / total(b);
      const order = [...counts.keys()].sort((a, b) =>
        share(a, "never") - share(b, "never") || share(b, fs.levels[0]) - share(a, fs.levels[0]) || name(g.names, a).localeCompare(name(g.names, b)));
      const rows = order.flatMap((b) => levels.filter((lv) => counts.get(b).get(lv))
        .map((lv) => ({ b, level: lv, k: counts.get(b).get(lv), n: total(b), share: share(b, lv) })));
      return [plot({
        height: 17 * order.length + 80, marginLeft: 190, marginRight: 70, marginTop: 40,
        x: { domain: [0, 1], tickFormat: "%", label: "share of the benchmark's languages", axis: "top", grid: true },
        y: { domain: order, label: null, tickFormat: (b) => name(g.names, b), tickSize: 0 },
        color: { domain: levels, range: [...sizeColors(fs.levels.length), css("--viz-bad")], legend: true, label: "above chance from" },
        marks: [
          Plot.barX(rows, Plot.stackX({ x: "share", y: "b", fill: "level", order: levels, insetTop: 1.5, insetBottom: 1.5, insetRight: 1,
            tip: true, title: (r) => `${name(g.names, r.b)}: ${r.k} of ${r.n} languages ` +
              (r.level === "never" ? "never above chance up to 1.7B" : `above chance from ${r.level} on`) })),
          Plot.text(order, { y: (b) => b, x: () => 1, dx: 6, textAnchor: "start", fill: css("--viz-muted"),
            text: (b) => `${total(b)} lang.` }),
        ],
      }), note("One bar per benchmark: how its languages split by the smallest model size from which they stay above chance (light: from the smallest models; red: never up to 1.7B). Benchmarks are sorted from the fewest to the most languages never above chance. The same data as the map above, without the language detail.")];
    });
  };

  // ---------- 2. every considered benchmark: above / at chance / never, per size ----------
  S.VIEWS.gateFunnel = async (el) => {
    const x = await S.load("gate_extra");
    const groups = Object.keys(x.population);
    const keys = [["above", "above chance at this size"], ["elsewhere", "at chance here, above at another size"], ["never", "never above chance"]];
    S.mount(el, [{ key: "g", label: "Tasks", options: groups, value: "Analysed", string: true }], (s) => {
      const p = x.population[s.g];
      const r = S.sizeRamp();
      const rows = p.flatMap((d) => keys.map(([k, label]) => ({ size: d.size, label, v: d[k], n: d.n })));
      const last = p[p.length - 1], first = p[0];
      return [S.tiles([
        [d3.format(",")(last.n), "tasks in this group"],
        [`${pct(first.above / first.n)} → ${pct(last.above / last.n)}`, `above chance, ${first.size} → ${last.size}`],
        [last.never, `never above chance at ${first.size}–${last.size}`],
      ]), plot({
        height: 300, marginLeft: 55,
        x: { domain: p.map((d) => d.size), label: "model size", padding: 0.3 },
        y: { domain: [0, last.n], label: "tasks", grid: true },
        color: { domain: keys.map((k) => k[1]), range: [r[3], r[0], css("--viz-bad")], legend: true },
        marks: [
          Plot.barY(rows, Plot.stackY({ x: "size", y: "v", fill: "label", order: keys.map((k) => k[1]), insetTop: 1, rx: 2,
            tip: true, title: (d) => `${d.size}: ${d3.format(",")(d.v)} of ${d3.format(",")(d.n)} tasks (${pct(d.v / d.n)})\n${d.label}` })),
          Plot.text(p.filter((d) => d.above / d.n > 0.08), { x: "size", y: "above", dy: 10, fill: "white", fontWeight: 600,
            text: (d) => pct(d.above / d.n) }),
        ],
      }), note(s.g.startsWith("Analysed")
        ? `Analysed: the ${d3.format(",")(x.analysed.tasks)} tasks of the ${x.analysed.benchmarks} benchmarks the gate reads, in the ${x.analysed.languages} trained languages, gated at every size from 90M to 1.7B.`
        : "Screened and discarded benchmarks were evaluated only at the final checkpoint of the 600M, 1B and 1.7B runs, so they have no verdict below 600M; \"never\" means never above chance at those three sizes.")];
    });
  };

  // ---------- 3. token thresholds: drag the token count ----------
  S.VIEWS.gateTokenSlider = async (el) => {
    const x = await S.load("gate_extra");
    const T = x.thresholds;
    const find = (b, sizes, level) => T.find((r) => r.benchmark === b && r.sizes === sizes && r.level === level);
    const key50 = (b) => { const r = find(b, "all", 0.5); return r?.bin_lo ?? (r?.status === "never" ? 1e13 : 1e14); };
    const benches = uniq(T.map((r) => r.benchmark)).sort((a, b) => key50(a) - key50(b) || name(x.names, a).localeCompare(name(x.names, b)));
    const st = { t: 9, sizes: "all", level: 0.5 };
    const SC = { reached: "reached", notyet: "not yet", never: "never reached", nodata: "too few cells" };
    frame(el, (render) => {
      const out = h("output", {}, tok(10 ** st.t));
      return [
        S.control({ type: "seg", label: "Model size", options: [{ value: "all", label: "pooled" }, ...x.sizes], value: "all" }, (v) => { st.sizes = v; render(); }),
        S.control({ type: "seg", label: "Share of cells above chance", options: [{ value: 0.5, label: "≥ 50 %" }, { value: 0.9, label: "≥ 90 %" }], value: 0.5 }, (v) => { st.level = v; render(); }),
        h("label", {}, h("span", {}, "Tokens of the language seen: ", out),
          h("input", { type: "range", min: 6, max: 11, step: 0.05, value: st.t,
            oninput: (e) => { st.t = +e.target.value; out.textContent = tok(10 ** st.t); render(); } })),
      ];
    }, () => {
      const t = 10 ** st.t;
      const rows = benches.map((b) => {
        const r = find(b, st.sizes, st.level) ?? { status: "no bin" };
        const status = r.status === "never" ? SC.never : r.bin_lo == null ? SC.nodata : r.bin_lo <= t ? SC.reached : SC.notyet;
        return { ...r, b, raw: r.status, status };
      });
      const n = (k) => rows.filter((r) => r.status === k).length;
      const dom = [SC.reached, SC.notyet, SC.never];
      return [S.tiles([[`${n(SC.reached)} of ${rows.length}`, `benchmark formats with ≥ ${st.level * 100} % of cells above chance by ${tok(t)} tokens`],
        [n(SC.notyet), "reach it with more tokens"], [n(SC.never), "never reach it in the sweep"]]),
      plot({
        height: 17 * benches.length + 70, marginLeft: 190, marginRight: 80, marginTop: 30,
        x: { type: "log", domain: [1e6, 1.5e11], label: "tokens of the language seen (log)", tickFormat: tok, ticks: [1e6, 1e7, 1e8, 1e9, 1e10, 1e11], axis: "top", grid: true },
        y: { domain: benches, label: null, tickFormat: (b) => name(x.names, b), tickSize: 0 },
        color: { domain: dom, range: [css("--viz-s1"), css("--viz-muted"), css("--viz-bad")], legend: true, label: "at the chosen token count" },
        marks: [
          Plot.ruleY(rows.filter((r) => r.lowest_tokens != null), { y: "b", x1: "lowest_tokens", x2: (r) => Math.min(r.highest_tokens, 1.5e11), stroke: css("--viz-grid"), strokeWidth: 7, strokeLinecap: "round" }),
          Plot.ruleX([t], { stroke: css("--md-default-fg-color"), strokeWidth: 1.5 }),
          Plot.dot(rows.filter((r) => r.bin_lo != null), { x: "bin_lo", y: "b", r: 5, fill: "status", stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (r) => `${name(x.names, r.b)} (${r.n_languages} languages, ${r.sizes === "all" ? "sizes pooled" : r.sizes})\n` +
              `≥ ${st.level * 100} % of cells above chance from ${r.raw === "<= lowest bin" ? "≤ " : ""}${tok(r.bin_lo)} tokens of the language on\n` +
              `margin a run needs over chance: ${fmt(r.margin_median, 3)} (${d3.format(",")(r.n_items_median)} items)` }),
          Plot.text(rows.filter((r) => r.status === SC.never || r.status === SC.nodata), { y: "b", frameAnchor: "right", dx: 74, textAnchor: "end",
            fill: (r) => (r.status === SC.never ? css("--viz-bad") : css("--viz-muted")), text: (r) => r.status,
            tip: true, title: (r) => `${name(x.names, r.b)}: ${r.status === SC.never ? `never ${st.level * 100} % of cells above chance at any token count the sweep reached` : "fewer than 40 non-English cells at this size"}` }),
        ],
      }), note("Each row is one benchmark in one format. The grey bar spans the token counts the sweep covers for it; the dot is the token count from which at least the chosen share of its cells stays above chance in every higher bin. Drag the slider: a dot left of the black line is reached. Pick a model size to see how far the threshold moves with size.")];
    });
  };

  // ---------- 4. reformulation: dumbbell original → twin ----------
  S.VIEWS.gateReformDumbbell = async (el) => {
    const g = await S.load("gate");
    const rows = g.reformulation;
    const langs = (f) => d3.max(rows.filter((r) => r.family === f), (r) => r.languages);
    const fams = uniq(rows.map((r) => r.family)).sort((a, b) => langs(b) - langs(a) || name(g.names, a).localeCompare(name(g.names, b)));
    const sizes = g.sizes.filter((z) => rows.some((r) => r.size === z));
    const kinds = ["as published", "RF: answer text", "LLM-RF: rewritten"];
    S.mount(el, [{ key: "size", label: "Model size", type: "seg", options: sizes, value: "1.7B" }], (s) => {
      const r = rows.filter((d) => d.size === s.size);
      const fs = fams.filter((f) => r.some((d) => d.family === f));
      const orig = fs.map((f) => ({ family: f, share: r.find((d) => d.family === f).share_original, kind: kinds[0] }));
      const tw = r.map((d) => ({ ...d, kind: d.set === "rf" ? kinds[1] : kinds[2] }));
      const tip = (d) => `${name(g.names, d.family)} at ${s.size} (${d.languages} languages)\nas published: ${pct(d.share_original)} above chance\n` +
        `${d.kind}: ${pct(d.share_twin)}\nlanguages only the twin passes : only the original passes = ${d.twin_only} : ${d.original_only}` +
        `\nMcNemar p = ${d.p_mcnemar < 0.001 ? d.p_mcnemar.toExponential(1) : fmt(d.p_mcnemar, 3)}${d.p_mcnemar < 0.05 ? " (significant)" : ""}`;
      return [plot({
        height: 24 * fs.length + 80, marginLeft: 200, marginRight: 30, marginTop: 40,
        x: { domain: [0, 1], tickFormat: "%", label: "share of the benchmark's languages above chance", grid: true, axis: "top" },
        y: { domain: fs, label: null, tickSize: 0, tickFormat: (f) => `${name(g.names, f)} (${langs(f)})` },
        color: { domain: kinds, range: [css("--viz-muted"), css("--viz-s1"), css("--viz-s2")], legend: true },
        marks: [
          Plot.ruleY(tw.filter((d) => d.set === "rf"), { y: "family", x1: "share_original", x2: "share_twin", stroke: "kind", strokeOpacity: 0.45, strokeWidth: 2.5 }),
          Plot.ruleY(tw.filter((d) => d.set !== "rf"), { y: "family", x1: "share_original", x2: "share_twin", stroke: "kind", strokeOpacity: 0.45, strokeWidth: 2.5, dy: 7 }),
          Plot.dot(orig, { x: "share", y: "family", fill: "kind", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5,
            tip: true, title: (d) => `${name(g.names, d.family)} as published at ${s.size}: ${pct(d.share)} of its languages above chance` }),
          Plot.dot(tw.filter((d) => d.set === "rf"), { x: "share_twin", y: "family", fill: "kind", r: 5.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true, title: tip }),
          Plot.dot(tw.filter((d) => d.set !== "rf"), { x: "share_twin", y: "family", fill: "kind", symbol: "diamond", r: 6, dy: 7, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true, title: tip }),
          Plot.text(tw.filter((d) => d.p_mcnemar < 0.05 && d.set === "rf"), { x: "share_twin", y: "family", text: () => "*", dx: 9, dy: 4, fontSize: 14 }),
          Plot.text(tw.filter((d) => d.p_mcnemar < 0.05 && d.set !== "rf"), { x: "share_twin", y: "family", text: () => "*", dx: 9, dy: 11, fontSize: 14 }),
        ],
      }), note("Each row is one letter-format benchmark (its language count in brackets). The grey dot is the share of its languages above chance as published; the line runs to the same share once the answers are scored as text (RF, blue) or the questions are rewritten (LLM-RF, orange diamond). Star: McNemar p < 0.05 on the languages where only one version passes. Some probe benchmarks were scored from 600M only.")];
    });
  };

  // ---------- 5. majority-class chance ----------
  S.VIEWS.gateMajority = async (el) => {
    const x = await S.load("gate_extra");
    const lab = { uniform: "uniform guessing, 1/m (the gate)", majority: "majority class, every task", majority_fixed: "majority class, fixed-label tasks only" };
    const dom = Object.values(lab), colors = () => [css("--viz-s1"), css("--viz-s2"), css("--viz-s3")];
    S.mount(el, [{ key: "v", label: "View", type: "seg", options: [{ value: "sizes", label: "per model size" }, { value: "bench", label: "per benchmark" }] }], (s) => {
      if (s.v === "sizes") {
        const rows = x.majority.filter((r) => r.population === "trained" && r.size !== "never").map((r) => ({ ...r, label: lab[r.chance] }));
        const last = rows.filter((r) => r.size === x.sizes[x.sizes.length - 1]);
        return [plot({
          height: 320, marginRight: 50,
          x: { type: "point", domain: x.sizes, label: "model size", padding: 0.3 },
          y: { domain: [0, rows[0].n_tasks], label: `trained tasks above chance (of ${rows[0].n_tasks})`, grid: true },
          color: { domain: dom, range: colors(), legend: true, label: "chance level" },
          marks: [
            Plot.line(rows, { x: "size", y: "n_pass", stroke: "label", strokeWidth: 2 }),
            Plot.dot(rows, { x: "size", y: "n_pass", fill: "label", r: 4.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.size}, chance = ${r.label}\n${r.n_pass} of ${r.n_tasks} trained tasks above chance (${pct(r.share)})` }),
            Plot.text(last.filter((r) => r.chance !== "majority_fixed"), { x: "size", y: "n_pass", text: (r) => r.n_pass, dx: 10, dy: -5, textAnchor: "start", fill: css("--md-default-fg-color") }),
            Plot.text(last.filter((r) => r.chance === "majority_fixed"), { x: "size", y: "n_pass", text: (r) => r.n_pass, dx: 10, dy: 8, textAnchor: "start", fill: css("--md-default-fg-color") }),
          ],
        }), note(`Each line counts the ${d3.format(",")(rows[0].n_tasks)} trained tasks above chance at each size under one chance level. Blue is the gate the site uses. Orange applies the majority class (the accuracy of always giving the most frequent correct position) to every task; green applies it only where a constant answer is possible, i.e. where each position carries a fixed label (A/B/C/D, yes/no).`)];
      }
      const rows = x.majority_by_benchmark.filter((r) => r.flipped_cells > 0).sort((a, b) => b.flipped_cells - a.flipped_cells);
      const long = rows.flatMap((r) => [["cells_uniform", dom[0]], ["cells_majority", dom[1]], ["cells_majority_fixed", dom[2]]]
        .map(([k, label]) => ({ ...r, v: r[k], label })));
      return [plot({
        height: 20 * rows.length + 80, marginLeft: 190, marginTop: 40,
        x: { label: "passing (task, size) cells", grid: true, axis: "top" },
        y: { domain: rows.map((r) => r.family), label: null, tickSize: 0, tickFormat: (f) => name(x.names, f) },
        color: { domain: dom, range: colors(), legend: true, label: "chance level" },
        marks: [
          Plot.ruleY(rows, { y: "family", x1: "cells_majority", x2: "cells_uniform", stroke: css("--viz-grid"), strokeWidth: 3 }),
          Plot.dot(long, { x: "v", y: "family", fill: "label", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (r) => `${name(x.names, r.family)} (${r.n_tasks} tasks)\n${r.label}: ${r.v} passing cells\n` +
              `uniform chance ${fmt(r.uniform, 3)}, majority ${fmt(r.majority, 3)}; fixed labels on ${pct(r.fixed_options)} of its tasks` }),
        ],
      }), note("The benchmarks that lose a pass under the majority level, most first. Where the green dot sits on the blue one, the lost passes are on tasks whose answers change per item: the model scores each answer's text and cannot see its position, so a constant answer is impossible there.")];
    });
  };

  // ---------- 6. the early passes ----------
  S.VIEWS.gateEarly = async (el) => {
    const x = await S.languages().then(() => S.load("gate_extra"));
    const rows = x.early.map((r) => ({ ...r, label: `${r.benchmark} · ${r.language}` }));
    const kinds = [["pass_low", "trained, few tokens (< 10⁸)"], ["pass_mid", "trained, 10⁸–10¹⁰ tokens"], ["pass_untrained_unrelated", "never trained on the language"]];
    const long = rows.flatMap((r) => kinds.filter(([k]) => r[k] != null).map(([k, label]) => ({ ...r, v: r[k], kind: label,
      n: r[{ pass_low: "n_low", pass_mid: "n_mid", pass_untrained_unrelated: "n_untrained_unrelated" }[k]] })));
    S.mount(el, [], () => [plot({
      height: 26 * rows.length + 80, marginLeft: 230, marginRight: 120, marginTop: 40,
      x: { domain: [0, 1], tickFormat: "%", label: "share of runs above chance", grid: true, axis: "top" },
      y: { domain: rows.map((r) => r.label), label: null, tickSize: 0 },
      color: { domain: kinds.map((k) => k[1]), range: [css("--viz-s1"), css("--viz-muted"), css("--viz-s2")], legend: true },
      marks: [
        Plot.ruleY(rows, { y: "label", x1: (r) => d3.min(kinds, ([k]) => r[k]), x2: (r) => d3.max(kinds, ([k]) => r[k]), stroke: css("--viz-grid"), strokeWidth: 3 }),
        Plot.dot(long, { x: "v", y: "label", fill: "kind", r: 5.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
          title: (r) => `${r.label} (${r.n_items} items)\n${r.kind}: ${pct(r.v)} of ${r.n} runs above chance` }),
        Plot.text(rows, { y: "label", x: () => 1, dx: 12, textAnchor: "start", fill: css("--md-default-fg-color"),
          text: (r) => `${pct(r.share_constant_above)} constant`, tip: true,
          title: (r) => `Of the ${r.n_passing_low} passing low-token runs, ${pct(r.share_constant_above)} pick an option whose share of gold answers alone clears chance;\n` +
            `median share of items given the most-picked option: ${pct(r.top_pick_share_median)}; picks on the longest option ${pct(r.pick_longest_median)} (gold longest ${pct(r.gold_longest)})` }),
      ],
    }), note("The ten tasks that pass with few tokens of their language and fail later. Blue: runs that trained the language, at few tokens; grey: the same task at intermediate tokens; orange: runs that never saw the language, at the same size and checkpoint as the blue ones. Right: the share of passing low-token runs that a constant answer explains.")]);
  };

  // ---------- 7. excluding the cells below the token threshold ----------
  S.VIEWS.gateExclusion = async (el) => {
    const x = await S.load("gate_extra");
    const lab = { all: "every cell", tokens: "cells above the token threshold", gate: "cells above chance" };
    const dom = Object.values(lab), colors = () => [css("--viz-s1"), css("--viz-s2"), css("--viz-s3")];
    const proxies = x.sizes.slice(0, -1);
    S.mount(el, [
      { key: "v", label: "View", type: "seg", options: [{ value: "pooled", label: "pooled" }, { value: "bench", label: "per benchmark" }] },
      { key: "axes", label: "Design pairs", type: "seg", options: [{ value: "mono-axis", label: "one decision at a time" }, { value: "multi-axis", label: "every pair" }] },
    ], (s) => {
      if (s.v === "pooled") {
        const rows = x.exclusion.filter((r) => r.reference === "same policy" && r.axes === s.axes).map((r) => ({ ...r, label: lab[r.policy] }))
          .sort((a, b) => proxies.indexOf(a.size) - proxies.indexOf(b.size));
        return [plot({
          height: 320, marginRight: 50,
          x: { type: "point", domain: proxies, label: "proxy size (ranks the designs against the 1.7B ranking)", padding: 0.3 },
          y: { label: "DA-size of the aggregate score", grid: true, tickFormat: ".2f" },
          color: { domain: dom, range: colors(), legend: true, label: "cells averaged" },
          marks: [
            Plot.line(rows, { x: "size", y: "da", stroke: "label", strokeWidth: 2 }),
            Plot.dot(rows, { x: "size", y: "da", fill: "label", r: 4.5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
              title: (r) => `${r.size}, ${r.label}: DA-size ${fmt(r.da, 3)}\n(${r.n_matching} of ${r.n_pairs} design pairs ranked like 1.7B, ${r.n_benchmarks} benchmarks)` }),
          ],
        }), note(`A benchmark's aggregate score is the mean of its languages that a run trains. Each line is how often that aggregate, at a small proxy, ranks two training designs the way it does at 1.7B, over the ${uniq(x.exclusion_by_benchmark.filter((r) => r.policy === "tokens" && r.da != null).map((r) => r.benchmark)).length} benchmarks that keep an aggregate under every policy. Higher is better: blue (keep every language) stays on top.`)];
      }
      const f = x.exclusion_by_benchmark.filter((r) => r.axes === s.axes);
      const by = d3.rollup(f, (v) => Object.fromEntries(v.map((r) => [r.policy, r.da])), (r) => r.benchmark);
      const fams = [...by.keys()].sort((a, b) => {
        const A = by.get(a), B = by.get(b);
        return (A.tokens == null) - (B.tokens == null) || (B.all - (B.tokens ?? B.all)) - (A.all - (A.tokens ?? A.all)) || name(x.names, a).localeCompare(name(x.names, b));
      });
      const long = f.map((r) => ({ ...r, label: lab[r.policy] }));
      return [plot({
        height: 19 * fams.length + 80, marginLeft: 190, marginRight: 90, marginTop: 40,
        x: { label: "DA-size of the aggregate, pooled over 90M–1B", grid: true, axis: "top", tickFormat: ".1f" },
        y: { domain: fams, label: null, tickSize: 0, tickFormat: (b) => name(x.names, b) },
        color: { domain: dom, range: colors(), legend: true, label: "cells averaged" },
        marks: [
          Plot.ruleY(fams.filter((b) => by.get(b).tokens != null), { y: (b) => b, x1: (b) => by.get(b).all, x2: (b) => by.get(b).tokens, stroke: css("--viz-grid"), strokeWidth: 3 }),
          Plot.dot(long, { x: "da", y: "benchmark", fill: "label", r: 5, stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
            title: (r) => `${name(x.names, r.benchmark)}, ${r.label}: DA-size ${fmt(r.da, 3)}` }),
          Plot.text(fams.filter((b) => by.get(b).tokens == null), { y: (b) => b, frameAnchor: "right", dx: 86, textAnchor: "end", fill: css("--viz-muted"),
            text: () => "never reaches it" }),
        ],
      }), note("Per benchmark, how often its aggregate ranks the designs like 1.7B with every language (blue) and without the languages below its token threshold (orange). Sorted by how much the exclusion costs. The bottom rows never reach their threshold, so the exclusion removes them whole.")];
    });
  };

  // ---------- 8. floors: the ladder against public model lines ----------
  S.VIEWS.gateFloorsLines = async (el) => {
    const x = await S.load("gate_extra");
    const rows = x.external_lines.map((r) => ({ ...r, line: r.line === "ladder" ? "our ladder" : r.line }));
    const lines = ["our ladder", ...uniq(rows.map((r) => r.line)).filter((l) => l !== "our ladder").sort()];
    const ends = lines.map((l) => d3.greatest(rows.filter((r) => r.line === l), (r) => r.params));
    // two labels at the same point: move the second one down
    const crowded = (e, i) => ends.slice(0, i).some((o) => o.params === e.params && Math.abs(o.share - e.share) < 0.04);
    S.mount(el, [], () => [plot({
      height: 360, marginRight: 90,
      x: { type: "log", label: "parameters (log)", tickFormat: (v) => (v >= 1e9 ? `${v / 1e9}B` : `${v / 1e6}M`), ticks: [1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11] },
      y: { domain: [0, 1], label: "share of the shared tasks above chance", tickFormat: "%", grid: true },
      color: { domain: lines, range: S.series(lines.length), legend: true },
      marks: [
        Plot.ruleX([1.7e9], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
        Plot.text([1.7e9], { x: (v) => v, frameAnchor: "bottom", dy: -6, dx: 4, textAnchor: "start", fill: css("--viz-muted"), text: () => "1.7B" }),
        Plot.line(rows, { x: "params", y: "share", stroke: "line", sort: "params", strokeWidth: (r) => (r.line === "our ladder" ? 3 : 1.6) }),
        Plot.dot(rows, { x: "params", y: "share", fill: "line", r: (r) => (r.line === "our ladder" ? 4.5 : 3.5), stroke: css("--viz-surface"), strokeWidth: 1.5, tip: true,
          title: (r) => `${r.line === "our ladder" ? `our ladder at ${r.model}` : r.model}\n${pct(r.share)} of ${r.n_tasks} shared tasks above chance` }),
        Plot.text(ends.filter((e, i) => !crowded(e, i)), { x: "params", y: "share", text: "line", dx: 8, textAnchor: "start", fill: css("--md-default-fg-color") }),
        Plot.text(ends.filter(crowded), { x: "params", y: "share", text: "line", dx: 8, dy: 12, textAnchor: "start", fill: css("--md-default-fg-color") }),
      ],
    }), note(`Each line is one family of models, from small to large; each point is the share of the shared tasks (${uniq(rows.filter((r) => r.line === "our ladder").map((r) => r.n_tasks)).join("/")} for the ladder, ${uniq(rows.filter((r) => r.line !== "our ladder").map((r) => r.n_tasks)).join("/")} for the public models) a model reads above chance. The ladder's point is its verdict (half of the runs of that size); a public model's point is its own single run. At the same size, public models trained on far more tokens read many more tasks.`)]);
  };

  // ---------- 9. the margin a task needs, by item count ----------
  S.VIEWS.gateMarginCalc = async (el) => {
    const x = await S.load("gate_extra");
    const tasks = x.margin.filter((t) => t.need != null).map((t) => ({ ...t, m: (t.need - t.chance) * 100, ok: t.first !== "never" }));
    // The bar before it is rounded to whole items: the accuracy p at which the bound over n items meets chance.
    const bar = (n, c) => {
      if (wilson(1, n) <= c) return null;
      let lo = c, hi = 1;
      for (let i = 0; i < 40; i++) { const mid = (lo + hi) / 2; if (wilson(mid, n) > c) hi = mid; else lo = mid; }
      return hi;
    };
    const grid = d3.range(1, 4.35, 0.02).map((v) => 10 ** v);
    const st = { v: 2, opts: 4 };
    const dom = ["above chance at 1.7B", "at chance at 1.7B"];
    frame(el, (render) => {
      const out = h("output", {}, Math.round(10 ** st.v));
      return [
        h("label", {}, h("span", {}, "Items in the benchmark: ", out),
          h("input", { type: "range", min: 1, max: 4.3, step: 0.01, value: st.v,
            oninput: (e) => { st.v = +e.target.value; out.textContent = d3.format(",")(Math.round(10 ** st.v)); render(); } })),
        S.control({ type: "seg", label: "Answer options", options: [2, 3, 4, 5], value: 4 }, (v) => { st.opts = v; render(); }),
      ];
    }, () => {
      const n = Math.round(10 ** st.v), c = 1 / st.opts, nd = needed(n, c);
      const curve = grid.map((g) => { const b = bar(g, c); return b == null ? null : { n: g, m: (b - c) * 100 }; }).filter((d) => d && d.m <= 30);
      const read = nd == null
        ? `With ${n} items and ${st.opts} options (chance ${pct(c)}), not even a perfect score clears chance.`
        : `With ${d3.format(",")(n)} items and ${st.opts} options, chance is ${fmt(100 * c, 1)} % and a run needs ${fmt(100 * nd, 1)} %: ${fmt(100 * (nd - c), 1)} points above chance.`;
      return [h("p", { class: "viz-note", style: "font-size:0.85rem" }, h("b", {}, read)), plot({
        height: 340,
        x: { type: "log", domain: [10, 20000], label: "items in the task (log)", tickFormat: "~s", ticks: [10, 30, 100, 300, 1000, 3000, 10000] },
        y: { label: "points above chance a run needs", grid: true, domain: [0, 30] },
        color: { domain: dom, range: [css("--viz-s1"), css("--viz-muted")], legend: true, label: "trained tasks" },
        marks: [
          Plot.dot(tasks.filter((t) => t.m <= 30), { x: "n", y: "m", fill: (t) => (t.ok ? dom[0] : dom[1]), r: 2.5, fillOpacity: 0.55, tip: true,
            title: (t) => `${t.task} (${name(x.names, t.family)})\n${d3.format(",")(t.n)} items, chance ${fmt(t.chance, 3)}: a run needs ${fmt(t.need, 3)} (+${fmt(t.m, 1)} points)\n` +
              (t.ok ? `above chance from ${t.first} on` : "not above chance at 1.7B") }),
          Plot.line(curve, { x: "n", y: "m", stroke: css("--md-default-fg-color"), strokeWidth: 1.2, strokeDasharray: "4 2" }),
          Plot.text(curve.filter((d) => Math.abs(d.n - 1000) / 1000 < 0.024).slice(0, 1), { x: "n", y: "m", text: () => `any task with ${st.opts} options`, dy: -12, textAnchor: "start", fill: css("--md-default-fg-color") }),
          nd == null ? null : Plot.ruleX([n], { stroke: css("--viz-s2"), strokeWidth: 1.5 }),
          nd == null ? null : Plot.dot([{ n, m: (nd - c) * 100 }], { x: "n", y: "m", r: 6, fill: css("--viz-s2"), stroke: css("--viz-surface"), strokeWidth: 2 }),
        ].filter(Boolean),
      }), note(`Each dot is one of the ${d3.format(",")(tasks.length)} trained tasks with an item count: how far above chance a run must score for its 95 % lower bound to clear chance, given the task's item count. Dashed curve: the same bar for any task with the chosen number of options, before rounding to whole items. Drag the slider to read the bar for any item count; the orange point follows it.`)];
    });
  };
})();

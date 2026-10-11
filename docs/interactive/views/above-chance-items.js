/* Additional interactive views of the above-chance-items page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, css, fmt, pct, mount, note, tiles, uniq } = S;
  const SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"];
  const ORD = { full: "full benchmark", gate_then_items: "gate, then items", items_then_gate: "items, then gate" };
  const median = (xs) => {
    const s = xs.filter((x) => x != null && Number.isFinite(x)).sort((a, b) => a - b);
    return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
  };
  const mean = (xs) => (xs.length ? xs.reduce((a, x) => a + x, 0) / xs.length : null);
  const M = (x) => d3.format(".2~s")(x).replace("G", "B");

  async function optional(el, name) {
    try { return await S.load(name); } catch (e) {
      el.replaceChildren(h("div", { class: "viz-empty" }, "The results of this analysis are not in this build yet; they appear here once it lands."));
      return null;
    }
  }

  /* Where every item of the suite ends up under each ordering: scored, in a
     task the gate drops, or dropped as at chance at 1.7B. */
  S.VIEWS.itemsFunnel = async (el) => {
    const d = await optional(el, "above_chance_items");
    if (!d) return;
    const total = d.survival.find((r) => r.ordering === "full").n_items;
    const segs = Object.keys(ORD).flatMap((o) => {
      const steps = d.survival.filter((r) => r.ordering === o);
      if (!steps.length) return [];
      const last = steps[steps.length - 1];
      const row = `${ORD[o]} (${last.n_tasks} tasks)`;
      return [
        { row, seg: "scored", n: last.n_kept },
        { row, seg: "item at chance at 1.7B", n: last.n_items - last.n_kept },
        { row, seg: "task dropped by the gate", n: total - last.n_items },
      ];
    });
    const order = ["scored", "item at chance at 1.7B", "task dropped by the gate"];
    const keptBy = (o) => d.survival.filter((r) => r.ordering === o).at(-1)?.n_kept;
    const extra = keptBy("items_then_gate") - keptBy("gate_then_items");
    const draw = () => [
      plot({
        height: 190, marginLeft: 200, marginRight: 20,
        x: { label: "items of the suite", tickFormat: M, grid: true, domain: [0, total] },
        y: { label: null, domain: uniq(segs.map((s) => s.row)) },
        color: { domain: order, range: [css("--viz-s1"), css("--viz-s4"), css("--viz-neutral")], legend: true },
        marks: [
          Plot.barX(segs, Plot.stackX({ x: "n", y: "row", fill: "seg", order, inset: 1, rx: 3,
            tip: true, title: (s) => `${s.row}\n${s.seg}: ${d3.format(",")(s.n)} items (${pct(s.n / total)})` })),
          // labels on the same stacked data as the bars (a filtered copy would stack differently)
          Plot.text(segs, Plot.stackX({ x: "n", y: "row", order, z: "seg", fill: "#fff",
            text: (s) => (s.seg === "scored" && s.n / total > 0.08 ? pct(s.n / total) : "") })),
          Plot.text(segs, Plot.stackX({ x: "n", y: "row", order, z: "seg", fill: css("--md-default-fg-color"),
            text: (s) => (s.seg !== "scored" && s.n / total > 0.08 ? pct(s.n / total) : "") })),
        ],
      }),
      note(`Each bar is the ${M(total)} items of the ${d.survival[0].n_tasks} tasks with a chance level. Choosing the items first lets the recomputed gate keep almost every task, so ${extra > 0 ? M(extra) + " more" : "more"} items are scored, nearly all of them in tasks the full benchmark's gate rejects.`),
    ];
    mount(el, [], draw);
  };

  /* Per benchmark, one row per language: how many items survive, and what the
     selection does to the SNR and the DA-size of that task at a chosen size. */
  S.VIEWS.itemsExplorer = async (el) => {
    const d = await optional(el, "above_chance_items_tasks");
    if (!d) return;
    await S.languages();
    const K = Object.fromEntries(d.keys.map((k, i) => [k, i]));
    const name = (b) => d.names[b] ?? b;
    const benches = uniq(d.tasks.map((t) => t.b)).sort((a, b) => name(a).localeCompare(name(b)));
    mount(el, [
      { key: "b", label: "Benchmark", options: [{ value: "", label: "All benchmarks (one row each)" }, ...benches.map((b) => ({ value: b, label: name(b) }))], value: "hellaswag", string: true },
      { key: "show", label: "Show", type: "seg", options: [{ value: "kept", label: "items kept" }, { value: "snr", label: "SNR" }, { value: "da", label: "DA-size" }] },
      { key: "size", label: "Model size", type: "seg", options: SIZES, value: "350M" },
    ], (s) => {
      const val = (t, k) => t.sizes[s.size]?.[K[k]] ?? null;
      let rows;
      if (s.b) {
        rows = d.tasks.filter((t) => t.b === s.b).map((t) => ({
          label: `${S.langName(t.lang)} (${t.lang})`, items: t.items, kept: t.kept, share: t.kept / t.items, runs: t.runs,
          gate: t.gate_full ? "passes as is" : t.gate_items ? "passes only on the kept items" : "fails",
          a: s.show === "snr" ? val(t, "snr_full") : val(t, "da_full"), z: s.show === "snr" ? val(t, "snr_kept") : val(t, "da_kept"),
        }));
      } else {
        rows = benches.map((b) => {
          const ts = d.tasks.filter((t) => t.b === b);
          const items = ts.reduce((a, t) => a + t.items, 0), kept = ts.reduce((a, t) => a + t.kept, 0);
          const nPass = ts.filter((t) => t.gate_full).length;
          const k1 = s.show === "snr" ? "snr_full" : "da_full", k2 = s.show === "snr" ? "snr_kept" : "da_kept";
          return { label: `${name(b)} (${ts.length})`, items, kept, share: kept / items, runs: null,
            gate: nPass === ts.length ? "passes as is" : nPass ? `${nPass} of ${ts.length} tasks pass as is` : "fails",
            a: median(ts.map((t) => val(t, k1))), z: median(ts.map((t) => val(t, k2))) };
        });
      }
      rows.sort((x, y) => y.share - x.share);
      const gates = ["passes as is", "passes only on the kept items", "fails"];
      const gateColor = (g) => (g === "passes as is" ? css("--viz-s1") : g === "fails" ? css("--viz-muted") : g.includes("of") ? css("--viz-s1") : css("--viz-s4"));
      const tasks = s.b ? d.tasks.filter((t) => t.b === s.b) : d.tasks;
      const paired = tasks.map((t) => [val(t, "snr_full"), val(t, "snr_kept"), val(t, "da_full"), val(t, "da_kept")]);
      const snrRatio = median(paired.filter((p) => p[0] && p[1]).map((p) => p[1] / p[0]));
      const daGain = mean(paired.filter((p) => p[2] != null && p[3] != null).map((p) => p[3] - p[2]));
      const sumItems = tasks.reduce((a, t) => a + t.items, 0), sumKept = tasks.reduce((a, t) => a + t.kept, 0);
      const H = 20 * rows.length + 70;
      const ylab = { domain: rows.map((r) => r.label), label: null, tickSize: 0 };
      let chart;
      if (s.show === "kept") {
        chart = plot({
          height: H, marginLeft: 190, marginRight: 80,
          x: { label: "share of the items the 1.7B runs answer above chance", domain: [0, 1], grid: true, tickFormat: "%" },
          y: ylab,
          color: s.b ? { domain: gates, range: gates.map(gateColor), legend: true, label: "gate at 1.7B" } : undefined,
          marks: [
            Plot.barX(rows, { x: 1, y: "label", fill: css("--viz-grid"), rx: 3 }),
            Plot.barX(rows, { x: "share", y: "label", fill: s.b ? "gate" : css("--viz-s1"), rx: 3, tip: true,
              title: (r) => `${r.label}\n${d3.format(",")(r.kept)} of ${d3.format(",")(r.items)} items kept (${pct(r.share)})` +
                (r.runs ? `\nchosen on ${r.runs} runs of 1.7B` : "") + `\ngate at 1.7B: ${r.gate}` }),
            Plot.text(rows, { x: 1, y: "label", dx: 6, textAnchor: "start", text: (r) => `${M(r.kept)} / ${M(r.items)}`, fill: css("--viz-muted") }),
          ],
        });
      } else {
        const pr = rows.filter((r) => r.a != null && r.z != null);
        if (!pr.length) return [tiles([[pct(sumKept / sumItems), "of the items kept"]]), h("div", { class: "viz-empty" }, `No ${s.show === "snr" ? "SNR" : "DA-size"} value at ${s.size} (the task is at chance there${s.show === "da" && s.size === "1.7B" ? ", or 1.7B is the reference" : ""}).`)];
        chart = plot({
          height: 20 * pr.length + 70, marginLeft: 190, marginRight: 20,
          x: s.show === "snr" ? { label: `SNR at ${s.size} (log scale)`, type: "log", grid: true }
            : { label: `DA-size of ${s.size} against 1.7B`, grid: true },
          y: { ...ylab, domain: pr.map((r) => r.label) },
          marks: [
            s.show === "da" ? Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }) : null,
            Plot.dot(pr, { x: "a", y: "label", r: 3.5, fill: css("--viz-muted") }),
            ...[[pr.filter((r) => r.z >= r.a), css("--viz-s1")], [pr.filter((r) => r.z < r.a), css("--viz-s2")]].flatMap(([part, color]) => [
              Plot.link(part, { x1: "a", x2: "z", y1: "label", y2: "label", stroke: color, markerEnd: "arrow", strokeWidth: 1.5 }),
              Plot.dot(part, { x: "z", y: "label", r: 4, fill: color, tip: true,
                title: (r) => `${r.label}\nfull benchmark ${fmt(r.a)} → kept items ${fmt(r.z)}` }),
            ]),
          ],
        });
      }
      return [
        tiles([
          [pct(sumKept / sumItems), `of ${M(sumItems)} items kept`],
          [snrRatio == null ? "–" : `×${fmt(snrRatio)}`, `median SNR change at ${s.size}`],
          [daGain == null ? "–" : (daGain >= 0 ? "+" : "") + fmt(daGain, 3), `mean DA-size change at ${s.size}`],
        ]),
        chart,
        note(s.show === "kept"
          ? "Grey track: every item of the task; filled: the items its 1.7B runs answer above chance. Colour: whether the above-chance gate passes the task at 1.7B on all its items or only once it is cut to the kept ones."
          : `Grey dot: the full benchmark; arrow head: the same task on its kept items (gate, then items). Blue: it rose; orange: it fell. ${s.show === "da" ? "The kept items are chosen on the 1.7B runs the DA is scored against, so a rise is an upper bound." : ""}`),
      ];
    });
  };

  /* Per task at one proxy: the SNR change on the kept items against the
     DA-size change. The SNR rises almost everywhere; the DA-size does not follow. */
  S.VIEWS.itemsSnrVsDa = async (el) => {
    const d = await optional(el, "above_chance_items_tasks");
    if (!d) return;
    await S.languages();
    const K = Object.fromEntries(d.keys.map((k, i) => [k, i]));
    const name = (b) => d.names[b] ?? b;
    mount(el, [{ key: "size", label: "Proxy size", type: "seg", options: SIZES.slice(0, -1), value: "90M" }], (s) => {
      const pts = d.tasks.map((t) => {
        const v = t.sizes[s.size];
        if (!v) return null;
        const [sf, sk, df, dk] = [v[K.snr_full], v[K.snr_kept], v[K.da_full], v[K.da_kept]];
        if (!sf || !sk || df == null || dk == null) return null;
        return { t, x: sk / sf, y: dk - df, noise: v[K.noise], sig: v[K.sig] };
      }).filter(Boolean);
      if (!pts.length) return h("div", { class: "viz-empty" }, "No paired task at this size.");
      const up = pts.filter((p) => p.x > 1).length, better = pts.filter((p) => p.y > 0).length, worse = pts.filter((p) => p.y < 0).length;
      const mx = median(pts.map((p) => p.x)), my = mean(pts.map((p) => p.y));
      return [
        tiles([
          [pct(up / pts.length), "of the tasks gain SNR"],
          [`×${fmt(mx)}`, "median SNR change"],
          [`${better} / ${worse}`, "tasks whose DA-size rises / falls"],
          [(my >= 0 ? "+" : "") + fmt(my, 3), "mean DA-size change"],
        ]),
        plot({
          height: 360,
          x: { label: "SNR on the kept items ÷ SNR on the full benchmark (log scale)", type: "log", grid: true },
          y: { label: "DA-size on the kept items − full benchmark", grid: true },
          marks: [
            Plot.ruleX([1], { stroke: css("--viz-muted") }),
            Plot.ruleY([0], { stroke: css("--viz-muted") }),
            Plot.dot(pts, { x: "x", y: "y", r: 3, fill: css("--viz-s1"), fillOpacity: 0.45, tip: true,
              title: (p) => `${name(p.t.b)} · ${S.langName(p.t.lang)} (${p.t.lang})\nSNR ×${fmt(p.x)} (signal ×${fmt(p.sig)}, noise ×${fmt(p.noise)})\nDA-size ${p.y >= 0 ? "+" : ""}${fmt(p.y, 3)}` +
                `\n${d3.format(",")(p.t.kept)} of ${d3.format(",")(p.t.items)} items kept` }),
            Plot.dot([{ x: mx, y: my }], { x: "x", y: "y", r: 7, symbol: "cross", stroke: css("--viz-s2"), strokeWidth: 2.5,
              tip: true, title: () => `median SNR ×${fmt(mx)}, mean DA-size change ${fmt(my, 3)}` }),
          ],
        }),
        note(`One dot per task above chance at ${s.size} (${pts.length} tasks, gate then items, multi-axis pairs). Right of the vertical line the SNR rose; above the horizontal line the proxy ranks the designs more like 1.7B. ` +
          "The cloud sits to the right but straddles the horizontal line: the SNR rises because the noise falls, while the ranking barely changes. Orange cross: the centre."),
      ];
    });
  };
})();

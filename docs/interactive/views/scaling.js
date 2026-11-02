/* Additional interactive views of the scaling page (registered on SNRSITE.VIEWS; see app.js). */
(() => {
  const S = window.SNRSITE;
  const { h, plot, css, fmt, pct, uniq, note, tiles, sizeNum, LADDER, LS } = S;

  // The five scaling regimes of rq01, in plain words, one fixed colour each (series order, never cycled).
  const REGIMES = [
    ["predictable across both", "scales with size and along training", "--viz-s1"],
    ["predictable across size only", "scales with size only", "--viz-s2"],
    ["predictable during training only", "scales along training only", "--viz-s3"],
    ["weak / unpredictable in both", "no clear trend", "--viz-s4"],
    ["declines with size", "gets worse with size", "--viz-s5"],
  ];
  const plain = (r) => REGIMES.find((x) => x[0] === r)?.[1] ?? r;
  const regColor = (r) => css(REGIMES.find((x) => x[0] === r)?.[2] ?? "--viz-muted");
  const SPLIT = 0.5;   // rq01's heuristic quadrant boundary on both R² axes
  const PICK = "snr:scaling-pick";
  const lColor = (L) => S.series(6)[LS.indexOf(+L)];
  const sizeColors = () => { const r = S.sizeRamp(); return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), LADDER.length); };
  const famLabel = (f) => f.replace(/^rfgm_/, "LLM-reformulated ").replace(/^rf_/, "reformulated ").replace(/_/g, " ");
  const bpbName = (task) => (task === "bpb_dclm" ? "English (DCLM validation text)" : S.fwName(task.slice(4)));
  const taskLang = (lang) => (lang && S.langName(lang) !== lang ? `${S.langName(lang)} (${lang})` : lang ?? "");
  const swatch = (c, hollow) => h("i", { style: `display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.35em;vertical-align:-1px;${hollow ? `border:2px solid ${c}` : `background:${c}`}` });
  const legend = (items) => h("div", { class: "ladder-legend" }, items.map(([c, label, hollow]) => h("span", {}, swatch(c, hollow), label)));
  const median = (a) => d3.median(a);
  // Plot's tooltip box reads --plot-background (white by default): follow the theme's surface.
  const tipBg = (el) => el.style.setProperty("--plot-background", "var(--viz-surface)");

  // A control row the views fill themselves (their options change with the selection).
  const select = (label, options, value, onchange) => {
    const sel = h("select", { onchange: (e) => onchange(e.target.value) },
      options.map(([v, l]) => h("option", { value: v }, l)));
    sel.value = value;
    return h("label", {}, label, sel);
  };

  // ---------- 1. the quadrant, every point named on demand ----------
  S.VIEWS.scalingQuadrant = async (el) => {
    tipBg(el);
    const [d] = await Promise.all([S.load("scaling"), S.languages()]);
    const all = d.regimes.map((r) => ({ ...r, twin: r.task.startsWith("bbpb_") }));
    const pops = { acc: all.filter((r) => !r.twin), twin: all.filter((r) => r.twin) };
    const label = (r) => `${famLabel(r.family.replace(/^bbpb_/, ""))}${r.twin ? " bBPB" : ""} · ${r.language}`;
    S.mount(el, [
      { key: "pop", label: "Tasks", type: "seg", options: [{ value: "acc", label: `accuracy and BPB (${pops.acc.length})` }, { value: "twin", label: `bBPB twins (${pops.twin.length})` }] },
      { key: "y", label: "Compare size with", type: "seg", options: [{ value: "r2_trajectory", label: "training (R²)" }, { value: "rho_size", label: "direction (ρ)" }] },
      { key: "q", label: "Find a task", type: "text" },
    ], (s) => {
      const rows = pops[s.pop];
      const q = (s.q ?? "").trim().toLowerCase();
      const hit = (r) => q && [r.task, r.family, r.language, S.langName(r.language), famLabel(r.family)].some((v) => String(v).toLowerCase().includes(q));
      const hits = rows.filter(hit);
      const traj = s.y === "r2_trajectory";
      const x0 = Math.min(0, d3.min(rows, (r) => r.r2_size));
      const y0 = traj ? Math.min(0, d3.min(rows, (r) => r.r2_trajectory)) : -1;
      const count = (k) => rows.filter((r) => r.regime === k).length;
      const both = count(REGIMES[0][0]);
      const quads = traj ? [
        { k: REGIMES[0][0], x1: SPLIT, x2: 1, y1: SPLIT, y2: 1, tx: 1, ty: 1, a: "end" },
        { k: REGIMES[2][0], x1: x0, x2: SPLIT, y1: SPLIT, y2: 1, tx: x0, ty: 1, a: "start" },
        { k: REGIMES[1][0], x1: SPLIT, x2: 1, y1: y0, y2: SPLIT, tx: 1, ty: y0, a: "end" },
        { k: REGIMES[3][0], x1: x0, x2: SPLIT, y1: y0, y2: SPLIT, tx: x0, ty: y0, a: "start" },
      ] : [];
      const fig = plot({
        height: 420, marginLeft: 55, marginTop: 24,
        x: { label: "R² of the fit across model size →", domain: [x0, 1], grid: true },
        y: traj ? { label: "↑ R² of the fit along training", domain: [y0, 1], grid: true }
          : { label: "↑ Spearman ρ with model size (improves = +1)", domain: [-1, 1], grid: true },
        marks: [
          ...(traj ? [
            Plot.rect(quads, { x1: "x1", x2: "x2", y1: "y1", y2: "y2", fill: (q) => regColor(q.k), fillOpacity: 0.07 }),
            ...quads.map((q) => Plot.text([q], { x: "tx", y: "ty", textAnchor: q.a, dx: q.a === "end" ? -4 : 4, dy: q.ty === 1 ? 10 : -10,
              text: (r) => `${plain(r.k)}: ${count(r.k)}`, fill: css("--viz-muted"), fontWeight: 600, stroke: css("--viz-surface"), strokeWidth: 3, paintOrder: "stroke" })),
          ] : [
            Plot.ruleY([0], { stroke: css("--viz-muted") }),
            Plot.text(["gets worse with size"], { x: x0, y: -1, dy: -8, dx: 4, textAnchor: "start", fill: css("--viz-muted"), fontWeight: 600, text: (t) => `${t}: ${count(REGIMES[4][0])}` }),
          ]),
          Plot.dot(rows, { x: "r2_size", y: s.y, r: q ? 2.5 : 3.5, fill: (r) => regColor(r.regime),
            fillOpacity: q ? 0.18 : 0.75, stroke: css("--viz-surface"), strokeWidth: 0.6 }),
          Plot.dot(hits, { x: "r2_size", y: s.y, r: 5.5, fill: (r) => regColor(r.regime), stroke: css("--md-default-fg-color"), strokeWidth: 1.2 }),
          Plot.text(hits.slice(0, 25), { x: "r2_size", y: s.y, text: label, dy: -10, fontSize: 10, stroke: css("--viz-surface"), strokeWidth: 3, paintOrder: "stroke" }),
          Plot.dot(rows, { x: "r2_size", y: s.y, r: 7, fill: "transparent", tip: true,
            title: (r) => `${r.task}\n${plain(r.regime)}\nR² across size ${fmt(r.r2_size)} (${r.n_size_fits} fits) · along training ${fmt(r.r2_trajectory)}\nρ with size ${fmt(r.rho_size)}` }),
        ],
      });
      // A click opens the pointed task in the curve explorer below (the tip's datum is the plot's value).
      fig.addEventListener("click", () => { if (fig.value?.task) window.dispatchEvent(new CustomEvent(PICK, { detail: fig.value.task })); });
      const open = (r) => h("button", { type: "button", class: "md-button", style: "font-size:.6rem;padding:.2em .7em;margin:.15em",
        onclick: () => window.dispatchEvent(new CustomEvent(PICK, { detail: r.task })) }, label(r));
      return [
        tiles([[`${both} of ${rows.length}`, "scale with size and along training"], [pct(both / rows.length), "share"],
          [fmt(median(rows.map((r) => r.r2_size))), "median size-fit R² over these tasks"], [count(REGIMES[4][0]), "get worse with size"]]),
        fig,
        legend(REGIMES.map(([k, p, v]) => [css(v), `${p} (${count(k)})`])),
        q ? h("div", {}, h("p", { class: "viz-note" }, hits.length ? `${hits.length} task${hits.length > 1 ? "s" : ""} match “${s.q}”${hits.length > 25 ? " (the first 25 named)" : ""}. Open one in the curve explorer below:` : `No task matches “${s.q}”.`),
          hits.slice(0, 12).map(open)) : null,
      ];
    });
    const input = el.querySelector("input[type=text]");
    if (input) input.placeholder = "hellaswag, truthfulqa, German…";
  };

  // ---------- 2. pick a benchmark: its accuracy across model size and along training ----------
  // The paper's Figures 9 and 10 (one benchmark at a time, its formats side by side), and, as a second
  // mode, one task's own curves with the per-K fits behind the medians of the scaling-predictability figure.
  const row = (...nodes) => h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;justify-content:center" }, ...nodes);
  const BUILDS = ["A", "AT3", "B", "ZH", "ES", "DCLMP", "FWEB"];
  const LADDERS = ["deep", "shallow", "swiglu", "muon"];
  const LADDER_NAME = { deep: "Deep", shallow: "Shallow", swiglu: "SwiGLU", muon: "Muon" };
  // The paper figures' encoding: line width = ladder, dash = data build (analysis/style.py), colour = K or size.
  const WIDTH = { deep: 1.5, shallow: 0.9, swiglu: 2.4, muon: 3.3 };
  const DASH = { A: null, B: "6 3", AT3: "1.5 2.5", ZH: "6 2 1.5 2", ES: "3 1.5 1 1.5", DCLMP: "7 2 2 2", FWEB: "5 1.5 1 1.5 1 1.5" };
  const kColors = () => { const r = S.sizeRamp(); return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), LS.length); };
  const lineKey = (label, { color, width = 1.5, dash } = {}) => {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    const ln = document.createElementNS(NS, "line");
    Object.entries({ width: 30, height: 10, style: "margin-right:.35em;vertical-align:-1px" }).forEach(([k, v]) => svg.setAttribute(k, v));
    Object.entries({ x1: 1, x2: 29, y1: 5, y2: 5, stroke: color ?? css("--md-default-fg-color"), "stroke-width": width, ...(dash ? { "stroke-dasharray": dash } : {}) })
      .forEach(([k, v]) => ln.setAttribute(k, v));
    svg.append(ln);
    return h("span", {}, svg, label);
  };

  // A seg control choosing between modes, each drawn the first time it is shown.
  const modes = (el, label, options) => {
    const boxes = options.map(() => h("div"));
    const started = new Set();
    const seg = S.control({ type: "seg", label, options: options.map((o, i) => ({ value: i, label: o.label })) }, (i) => show(i));
    const show = (i) => {
      seg.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
      boxes.forEach((b, j) => { b.hidden = i !== j; });
      if (!started.has(i)) {
        started.add(i);
        Promise.resolve(options[i].draw(boxes[i])).catch((e) => { boxes[i].replaceChildren(h("div", { class: "viz-empty" }, `Could not draw this view (${e.message}).`)); console.error(e); });
      }
    };
    el.replaceChildren(h("div", { class: "viz-controls" }, seg), ...boxes);
    show(0);
    return show;
  };

  // Figures 9 and 10: every run, the mean over the tasks in the languages it trains, not filtered.
  async function paperCurves(box) {
    tipBg(box);
    const idx = await S.load("scaling_paper");
    const got = {};
    const fmtLabel = (b) => `${b.name}${b.formats.length > 1 ? ` (${b.formats.join(", ")})` : ""}`;
    const def = idx.benchmarks.find((b) => b.base === "include_base_44") ?? idx.benchmarks[0];
    const { render } = S.mount(box, [
      { key: "b", label: "Benchmark", string: true, value: def.base, options: idx.benchmarks.map((b) => ({ value: b.base, label: fmtLabel(b) })) },
      { key: "x", label: "Accuracy against", type: "seg", options: [{ value: "size", label: "model size" }, { value: "train", label: "training progress" }] },
      { key: "K", label: "K", options: [{ value: "all", label: "every K (paper)" }, ...LS.map((K) => ({ value: K, label: `K = ${K}` }))] },
      { key: "ladder", label: "Ladder", options: [{ value: "all", label: "every ladder (paper)" }, ...LADDERS.map((l) => ({ value: l, label: LADDER_NAME[l] }))] },
      { key: "data", label: "Data build", options: [{ value: "all", label: "every build (paper)" }, ...BUILDS.map((b) => ({ value: b, label: `Data ${b}` }))] },
    ], (s) => {
      const d = got[s.b];
      if (!d) {
        S.load(`scaling_paper/${s.b}`).then((x) => { got[s.b] = x; render(); });
        return h("div", { class: "viz-empty" }, "Loading…");
      }
      const bySize = s.x === "size";
      const keep = (r) => (s.K === "all" || r.K === +s.K) && (s.ladder === "all" || r.ladder === s.ladder) && (s.data === "all" || r.data === s.data);
      const formats = Object.keys(d);
      const kc = kColors(), sc = sizeColors();
      const color = bySize ? { type: "ordinal", domain: LS, range: kc } : { type: "ordinal", domain: LADDER, range: sc };
      const desc = (r) => `K = ${r.K} · ${LADDER_NAME[r.ladder]} · Data ${r.data} · seed ${r.seed}`;
      // one facet per format, on a shared accuracy axis
      const rows = formats.flatMap((f) => (bySize
        ? d[f].size.map(([K, ladder, data, seed, size, y]) => ({ f, K, ladder, data, seed, size, N: sizeNum(size), y })).filter(keep)
        : d[f].train.filter(keep).flatMap((r) => r.y.map((y, j) => (y == null ? null : { ...r, f, x: (j + 1) / 2, y })).filter(Boolean))));
      const chance = formats.filter((f) => d[f].chance != null).map((f) => ({ f, y: d[f].chance }));
      const z = bySize ? (r) => `${r.f}|${r.K}|${r.ladder}|${r.data}|${r.seed}` : (r) => `${r.f}|${r.size}|${r.K}|${r.ladder}|${r.data}|${r.seed}`;
      const lines = d3.groups(rows, (r) => r.ladder, (r) => r.data).flatMap(([ladder, g]) => g.map(([data, rs]) =>
        Plot.line(rs, { fx: "f", x: bySize ? "N" : "x", y: "y", z, stroke: bySize ? "K" : "size", strokeWidth: WIDTH[ladder] ?? 1.2,
          strokeOpacity: 0.85, ...(bySize ? { sort: "N" } : {}), ...(DASH[data] ? { strokeDasharray: DASH[data] } : {}) })));
      const panels = rows.length ? plot({
        width: formats.length === 1 ? 560 : 720, height: 320, marginLeft: 48, marginRight: 12, marginTop: 34, color,
        fx: { domain: formats, label: null, tickFormat: (f) => d[f].name, padding: 0.14 },
        x: bySize
          ? { type: "log", label: "Non-embedding parameters", ticks: LADDER.map(sizeNum), tickFormat: (n) => LADDER.find((z2) => sizeNum(z2) === n) ?? "" }
          : { domain: [0.5, 5], ticks: [1, 2, 3, 4, 5], tickFormat: (c) => `${c}C`, label: "Training progress (Chinchilla multiples)" },
        y: { label: bySize ? "Final accuracy over trained-language tasks" : "Mean accuracy over trained-language tasks", grid: true,
          domain: d3.extent(rows.map((r) => r.y).concat(chance.map((c) => c.y))), nice: true },
        marks: [
          ...lines,
          bySize ? Plot.dot(rows, { fx: "f", x: "N", y: "y", fill: "K", r: 1.6 }) : null,
          Plot.ruleY(chance, { fx: "f", y: "y", stroke: css("--viz-bad"), strokeDasharray: "2 3", strokeWidth: 1.4 }),
          Plot.dot(rows, { fx: "f", x: bySize ? "N" : "x", y: "y", r: 6, fill: "transparent", tip: true,
            title: (r) => `${d[r.f].name}\n${desc(r)}\n${r.size}${bySize ? ", final checkpoint" : ` at ${r.x}C`}: ${fmt(r.y, 3)}` }),
        ].filter(Boolean),
      }) : h("div", { class: "viz-empty" }, "No run with this K, ladder and data build.");
      const present = (key, all) => all.filter((k) => formats.some((f) => (bySize ? d[f].size.some((r) => r[{ K: 0, ladder: 1, data: 2 }[key]] === k) : d[f].train.some((r) => r[key] === k))));
      const sizesShown = LADDER.filter((z2) => formats.some((f) => d[f].train.some((r) => r.size === z2)));
      const keyRow = h("div", { class: "ladder-legend", style: "justify-content:center" },
        ...(bySize ? present("K", LS).map((K) => lineKey(`K = ${K}`, { color: kc[LS.indexOf(K)], width: 2.5 }))
          : sizesShown.map((z2) => lineKey(z2, { color: sc[LADDER.indexOf(z2)], width: 2.5 }))),
        ...present("ladder", LADDERS).map((l) => lineKey(LADDER_NAME[l], { color: css("--viz-muted"), width: WIDTH[l] })),
        ...present("data", BUILDS).map((b) => lineKey(`Data ${b}`, { width: 1.2, dash: DASH[b] })),
        lineKey("Chance", { color: css("--viz-bad"), width: 1.3, dash: "2 3" }));
      return [row(panels), keyRow,
        note(bySize
          ? "Final-checkpoint accuracy against non-embedding parameters, averaged over the tasks in the languages each configuration trains: one line per (K, ladder, data build, seed), colour = K, width = ladder, dash = data build. Not filtered by the above-chance threshold. Hover a point for its run."
          : "Accuracy at the ten evaluated tenths of each run against Chinchilla multiples of its token budget (5C = the full run), averaged over the tasks in the languages the run trains (every seed and data build, 90M–1.7B): colour = size, width = ladder, dash = data build. Not filtered by the above-chance threshold.")];
    });
  }

  // One task: its per-K final scores with the log-linear fits, and its runs along training (deep, data A, seed 1904).
  let pendingPick = null;
  async function taskCurves(el) {
    tipBg(el);
    const [idx, reg, lad] = await Promise.all([S.load("scaling_index"), S.load("scaling"), S.load("ladder"), S.languages()]);
    const regime = Object.fromEntries(reg.regimes.map((r) => [r.task, r]));
    const tasks = Object.fromEntries(idx.tasks.map((t) => [t.task, t]));
    const famName = (f) => idx.names?.[f] ?? famLabel(f);
    const byFam = d3.group(idx.tasks, (t) => t.family);
    const fams = [...byFam.keys()].sort((a, b) => famName(a).localeCompare(famName(b)));
    const tokens = (size, L) => lad.cells.find((c) => c.size === size && c.L === +L && c.ladder === "deep" && c.scheme === "A" && c.seed === 1904)?.tokens
      ?? 100 * sizeNum(size);
    // A task reads as its language; two tasks of one language add their subtask (the task name without its benchmark).
    const sub = (t) => t.task.replace(new RegExp(`^${t.family}_?`), "").replace(/_/g, " ");
    const tLabel = (t) => {
      if (t.kind === "bpb") return bpbName(t.task);
      const twin = byFam.get(t.family).filter((x) => x.language === t.language).length > 1;
      return `${taskLang(t.language)}${twin && sub(t) ? ` · ${sub(t)}` : ""}`;
    };
    const st = { fam: "hellaswag", task: "hellaswag", L: "all" };
    if (pendingPick && tasks[pendingPick]) Object.assign(st, { task: pendingPick, fam: tasks[pendingPick].family });
    const bar = h("div", { class: "viz-controls" });
    const out = h("div");
    el.replaceChildren(bar, out);
    let ticket = 0;

    const controls = () => {
      const t = tasks[st.task];
      const Ls = Object.keys(t.fits).map(Number).sort((a, b) => a - b);
      bar.replaceChildren(
        select("Benchmark", fams.map((f) => [f, `${famName(f)} (${byFam.get(f).length})`]), st.fam, (v) => {
          st.fam = v;
          const list = byFam.get(v);
          st.task = (list.find((x) => x.language === "en") ?? list[0]).task;
          st.L = "all"; controls(); render();
        }),
        select("Task", byFam.get(st.fam).map((x) => [x.task, tLabel(x)]).sort((a, b) => a[1].localeCompare(b[1])), st.task, (v) => { st.task = v; st.L = "all"; controls(); render(); }),
        select("K", [["all", "every K"], ...Ls.map((L) => [String(L), `K = ${L}`])], st.L, (v) => { st.L = v; render(); }),
      );
    };

    const render = async () => {
      const my = ++ticket;
      const t = tasks[st.task];
      const curves = (await S.load(`scaling_curves/${t.family}`))[t.task] ?? {};
      if (my !== ticket) return;
      const bpb = t.kind === "bpb";
      const up = (i) => !t.above || t.above[i] == null || t.above[i] === 1;
      const Ls = Object.keys(t.fits).filter((L) => curves[L]).map(Number).sort((a, b) => a - b);
      const shown = st.L === "all" ? Ls : Ls.filter((L) => L === +st.L);
      // Final scores: the last of each run's ten evaluated tenths.
      const finals = shown.flatMap((L) => LADDER.map((size, i) => {
        const pts = curves[L]?.[size];
        const last = pts?.[pts.length - 1];
        return last && last[0] === 10 ? { L: `K${L}`, Lnum: L, size, N: sizeNum(size), y: last[1], above: up(i) } : null;
      }).filter(Boolean));
      // The fitted line over the sizes above chance: the fit's slope through the mean of those sizes.
      const lines = shown.flatMap((L) => {
        const [r2, , slope] = t.fits[L];
        const g = finals.filter((f) => f.Lnum === L && f.above);
        if (r2 == null || g.length < 3) return [];
        const mx = d3.mean(g, (f) => Math.log10(f.N)), my2 = d3.mean(g, (f) => f.y);
        return d3.extent(g, (f) => f.N).map((N) => ({ L: `K${L}`, N, y: my2 + slope * (Math.log10(N) - mx) }));
      });
      const ys = finals.map((f) => f.y).concat(t.chance != null ? [t.chance] : []);
      const reg0 = regime[t.task];
      const fitted = Ls.filter((L) => t.fits[L][0] != null);
      const head = h("div", { class: "viz-card" },
        h("h4", {}, bpb ? `Bits per byte · ${bpbName(t.task)}` : `${famName(t.family)} · ${tLabel(t)}`),
        reg0 ? h("p", {}, swatch(regColor(reg0.regime)), h("b", {}, plain(reg0.regime)), ` — median R² ${fmt(reg0.r2_size)} across model-size fits (${reg0.n_size_fits} fits), ${fmt(reg0.r2_trajectory)} across training fits, ρ with size ${fmt(reg0.rho_size)}.`)
          : h("p", {}, fitted.length === 0
            ? "Above chance at too few sizes wherever it is trained: no fit (the above-chance filter removes it)."
            : `Fitted at ${fitted.length} value${fitted.length > 1 ? "s" : ""} of K only, or without enough training checkpoints, so no median and no regime.`));

      const sizeChart = plot({
        height: 300, marginLeft: 55,
        x: { type: "log", label: "Non-embedding parameters →", ticks: LADDER.map(sizeNum), tickFormat: (v) => LADDER.find((s) => sizeNum(s) === v) ?? "" },
        y: { label: bpb ? "↓ final bits per byte (lower is better)" : "↑ final accuracy", grid: true, domain: d3.extent(ys), nice: true },
        marks: [
          t.chance != null ? Plot.ruleY([t.chance], { stroke: css("--viz-bad"), strokeDasharray: "2 3" }) : null,
          t.chance != null ? Plot.text([t.chance], { x: sizeNum(LADDER[0]), y: (v) => v, dy: -7, textAnchor: "start", text: () => "chance", fill: css("--viz-bad") }) : null,
          Plot.line(lines, { x: "N", y: "y", z: "L", stroke: (l) => lColor(l.L.slice(1)), strokeWidth: 2 }),
          Plot.dot(finals, { x: "N", y: "y", stroke: (f) => lColor(f.Lnum), fill: (f) => (f.above ? lColor(f.Lnum) : css("--viz-surface")), r: 4.5, strokeWidth: 2 }),
          Plot.dot(finals, { x: "N", y: "y", r: 8, fill: "transparent", tip: true,
            title: (f) => `K = ${f.Lnum} · ${f.size}: ${fmt(f.y, 3)}${f.above ? "" : "\nat chance at this size: left out of the fit"}` }),
        ].filter(Boolean),
      });
      const sizeLegend = legend(shown.map((L) => {
        const [r2, , , n] = t.fits[L];
        return [lColor(L), r2 == null ? `K = ${L}: no fit (above chance at ${n} of ${LADDER.length} sizes)` : `K = ${L}: R² ${fmt(r2)} over ${n} sizes`];
      }));

      // Along training: one run per size at one K.
      const Lt = st.L === "all" ? Ls[Ls.length - 1] : +st.L;
      const runs = LADDER.flatMap((size, i) => (curves[Lt]?.[size] ?? []).map(([k, y]) => ({ size, k, T: (k / 10) * tokens(size, Lt), y, above: up(i) })));
      const sc = sizeColors();
      const trainChart = runs.length ? plot({
        height: 280, marginLeft: 55,
        x: { type: "log", label: "Training tokens seen →", tickFormat: (v) => d3.format("~s")(v).replace("G", "B") },
        y: { label: bpb ? "↓ bits per byte" : "↑ accuracy", grid: true },
        marks: [
          t.chance != null ? Plot.ruleY([t.chance], { stroke: css("--viz-bad"), strokeDasharray: "2 3" }) : null,
          Plot.line(runs.filter((r) => r.above), { x: "T", y: "y", z: "size", stroke: (r) => sc[LADDER.indexOf(r.size)], strokeWidth: 2 }),
          Plot.line(runs.filter((r) => !r.above), { x: "T", y: "y", z: "size", stroke: (r) => sc[LADDER.indexOf(r.size)], strokeWidth: 1.5, strokeOpacity: 0.4, strokeDasharray: "3 3" }),
          Plot.dot(runs, { x: "T", y: "y", fill: (r) => sc[LADDER.indexOf(r.size)], r: 2.5, fillOpacity: (r) => (r.above ? 1 : 0.35) }),
          Plot.dot(runs, { x: "T", y: "y", r: 7, fill: "transparent", tip: true,
            title: (r) => `${r.size} at ${r.k * 10} % of its run (${d3.format(".3~s")(r.T).replace("G", "B")} tokens): ${fmt(r.y, 3)}${r.above ? "" : "\nat chance at this size: left out of the training fit"}` }),
        ].filter(Boolean),
      }) : h("div", { class: "viz-empty" }, "No checkpoints of this task at this K.");

      out.replaceChildren(head,
        h("p", { class: "viz-note" }, "Across model size: the final accuracy of the deep models trained on data build A (seed 1904), one colour per K. Filled dots pass the above-chance filter and enter the fit (the line); hollow ones do not. The median of these per-K fits is the task's model-size R²."),
        row(sizeChart), sizeLegend,
        h("p", { class: "viz-note", style: "margin-top:1em" }, `Along training at K = ${Lt}${st.L === "all" ? " (pick a K above to change it)" : ""}: one line per model size, light to dark blue from 90M to 1.7B. Dashed, faded runs are at chance at that size and stay out of the training fit.`),
        row(trainChart),
        legend(LADDER.filter((s) => runs.some((r) => r.size === s)).map((s) => [sc[LADDER.indexOf(s)], s])));
    };

    window.addEventListener(PICK, () => {
      const task = pendingPick;
      if (!tasks[task]) return;
      Object.assign(st, { task, fam: tasks[task].family, L: "all" });
      controls(); render();
    });
    S.REDRAW.push(render);
    controls();
    await render();
  }

  S.VIEWS.scalingCurves = async (el) => {
    const show = modes(el, "Curves", [
      { label: "every run, mean over trained-language tasks (paper)", draw: paperCurves },
      { label: "one task, with its fits", draw: taskCurves },
    ]);
    window.addEventListener(PICK, (e) => {
      let task = e.detail;
      if (task.startsWith("bbpb_")) task = task.slice(5);   // a bBPB score: open its accuracy task
      pendingPick = task;
      show(1);
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  // ---------- 3. predict the 1.7B BPB from the small models ----------
  S.VIEWS.scalingForecast = async (el) => {
    tipBg(el);
    const [d] = await Promise.all([S.load("scaling_forecast"), S.languages()]);
    const tasksF = uniq(d.fits.map((f) => f.task)).sort((a, b) => (a === "bpb_dclm" ? -1 : b === "bpb_dclm" ? 1 : bpbName(a).localeCompare(bpbName(b))));
    const TOPS = ["350M", "600M", "1B"];
    const REF = "1.7B";
    const medErr = Object.fromEntries(TOPS.map((top) => [top, median(d.fits.filter((f) => f.top === top).map((f) => Math.abs(f.error)))]));
    const signed = (e) => `${e > 0 ? "+" : ""}${fmt(100 * e, 1)} %`;
    S.mount(el, [
      { key: "task", label: "Language", options: tasksF.map((t) => ({ value: t, label: bpbName(t) })), string: true },
      { key: "L", label: "Language setting", type: "seg", options: LS.map((L) => ({ value: L, label: `K${L}` })), value: 30 },
      { key: "top", label: "Fit on the sizes up to", type: "seg", options: TOPS, value: "600M" },
    ], (s) => {
      const fits = d.fits.filter((f) => f.task === s.task && f.L === s.L);
      const fit = fits.find((f) => f.top === s.top);
      if (!fit) return h("div", { class: "viz-empty" }, `${bpbName(s.task)} is not a training language at K = ${s.L}: rq01 fits the BPB of trained languages only. Try another setting.`);
      const obs = d.observed.filter((o) => o.task === s.task && o.L === s.L).map((o) => ({ ...o, N: sizeNum(o.size), inFit: sizeNum(o.size) <= sizeNum(s.top) }));
      const law = (N) => fit.predicted * (N / sizeNum(REF)) ** -fit.alpha;
      const grid = d3.range(0, 41).map((i) => sizeNum(LADDER[0]) * (sizeNum(REF) / sizeNum(LADDER[0])) ** (i / 40));
      const curve = grid.map((N) => ({ N, y: law(N), part: N <= sizeNum(s.top) * 1.0001 ? "fit" : "extrapolated" }));
      const join = curve.findIndex((c) => c.part === "extrapolated");
      if (join > 0) curve.splice(join, 0, { ...curve[join - 1], part: "extrapolated" });
      const ref = [{ N: sizeNum(REF), y: fit.predicted, what: "predicted" }, { N: sizeNum(REF), y: fit.observed, what: "observed" }];
      return [
        tiles([[signed(fit.error), `error at 1.7B, fit to ${s.top}`], ...TOPS.filter((t) => t !== s.top).map((t) => {
          const f = fits.find((x) => x.top === t);
          return [f ? signed(f.error) : "–", `fit to ${t}`];
        }), [`${fmt(100 * medErr[s.top], 1)} %`, `median |error| over all ${d.fits.filter((f) => f.top === s.top).length} fits to ${s.top}`]]),
        plot({
          height: 320, marginLeft: 55, insetTop: 12, insetBottom: 16,
          x: { type: "log", label: "model size (non-embedding) →", ticks: LADDER.map(sizeNum), tickFormat: (v) => LADDER.find((z) => sizeNum(z) === v) ?? "" },
          y: { label: "↓ final bits per byte", grid: true },
          marks: [
            Plot.line(curve.filter((c) => c.part === "fit"), { x: "N", y: "y", stroke: css("--viz-s1"), strokeWidth: 2 }),
            Plot.line(curve.filter((c) => c.part === "extrapolated"), { x: "N", y: "y", stroke: css("--viz-s1"), strokeWidth: 2, strokeDasharray: "5 4" }),
            Plot.dot(obs, { x: "N", y: "bpb", r: 4.5, stroke: css("--md-default-fg-color"), strokeWidth: 1.5,
              fill: (o) => (o.inFit ? css("--md-default-fg-color") : css("--viz-surface")), tip: true,
              title: (o) => `${o.size}: ${fmt(o.bpb, 3)} BPB${o.inFit ? " (in the fit)" : " (not in the fit)"}` }),
            Plot.dot(ref.slice(0, 1), { x: "N", y: "y", symbol: "diamond", r: 6, fill: css("--viz-s1"), stroke: css("--viz-surface"), strokeWidth: 1.5 }),
            ...ref.map((r) => Plot.text([r], { x: "N", y: "y", dx: -10, dy: r.y === d3.max(ref, (x) => x.y) ? -8 : 8, textAnchor: "end",
              text: (x) => `${x.what} ${fmt(x.y, 3)}`, fill: r.what === "predicted" ? css("--viz-s1") : css("--md-default-fg-color"),
              stroke: css("--viz-surface"), strokeWidth: 3, paintOrder: "stroke" })),
          ],
        }),
        note(`Black dots: the observed final BPB of each size (filled: the sizes the power law is fitted on). Blue: the power law, solid over the fitted sizes and dashed where it extrapolates to 1.7B (diamond). Positive error = the law predicts a higher BPB than 1.7B reaches: it over-predicts. Every deep data-A fit does.`),
      ];
    });
  };
})();

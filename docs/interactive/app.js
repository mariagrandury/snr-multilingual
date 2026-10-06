/* Interactive views of the showcase site.
 *
 * A page asks for a view with <div class="viz" data-viz="NAME"></div>; every
 * view reads one JSON file of docs/interactive/data/, written by
 * scripts/build_site_data.py from the committed analysis tables. Charts are
 * Observable Plot; colours come from the CSS tokens in app.css, so a theme
 * toggle only redraws.
 */
(() => {
  const DATA = new URL("data/", document.currentScript.src);
  const SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"];
  const PROXIES = ["175M", "350M", "600M", "1B", "1.7B"];
  const LS = [1, 2, 8, 15, 30, 50];
  const HF_ORG = "https://huggingface.co/msnr";
  const WANDB = "https://wandb.ai/mariagrandury-epflnlp/msnr";
  const REDRAW = [];
  const POP = { benchmark: "benchmarks", bpb_macro: "BPB, all languages", bpb_trained: "BPB, trained languages", loss: "training loss" };
  const cache = {};

  // ---------- helpers ----------
  const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();
  const series = (n) => Array.from({ length: n }, (_, i) => css(`--viz-s${i + 1}`));
  const dark = () => document.body.getAttribute("data-md-color-scheme") === "slate";
  // Ordinal blue ramp for model size (steps 250→650 light, 150→500 dark).
  const sizeRamp = () => dark()
    ? ["#b7d3f6", "#86b6ef", "#5598e7", "#3987e5", "#256abf"]
    : ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"];
  const diverging = () => ({ type: "diverging", scheme: "RdBu", pivot: 0 });
  const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (+x).toFixed(d));
  const pct = (x) => (x == null ? "–" : `${Math.round(100 * x)} %`);
  const sizeNum = (s) => {
    const m = String(s).trim().match(/^([\d.]+)\s*([kmbt]?)/i);
    return m ? +m[1] * { "": 1, k: 1e3, m: 1e6, b: 1e9, t: 1e12 }[m[2].toLowerCase()] : NaN;
  };
  const byKey = (arr, key) => arr.reduce((m, r) => (m[key(r)] = r, m), {});
  const uniq = (a) => [...new Set(a)];

  function h(tag, attrs = {}, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v != null) el.setAttribute(k, v);
    }
    el.append(...kids.flat().filter((k) => k != null));
    return el;
  }

  const expand = (t) => t.data.map((r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i]])));

  async function load(name) {
    if (!cache[name]) cache[name] = fetch(new URL(`${name}.json`, DATA)).then((r) => r.json());
    return cache[name];
  }

  function plot(opts) {
    return Plot.plot({
      style: { background: "transparent", color: css("--md-default-fg-color"), fontSize: "11px" },
      marginLeft: 50, width: 720, ...opts,
    });
  }

  /* A control row plus an output area; `draw(state)` returns the nodes to show. */
  function mount(el, controls, draw) {
    const state = {};
    const bar = h("div", { class: "viz-controls" });
    const out = h("div");
    el.replaceChildren(...(controls.length ? [bar] : []), out);
    const render = () => out.replaceChildren(...[draw(state)].flat().filter(Boolean));
    for (const c of controls) {
      state[c.key] = c.value ?? (c.options ? optValue(c.options[0]) : undefined);
      bar.append(control(c, (v) => { state[c.key] = v; render(); }));
    }
    REDRAW.push(render);
    render();
    return { state, render };
  }
  const optValue = (o) => (typeof o === "object" ? o.value : o);
  const optLabel = (o) => (typeof o === "object" ? o.label : o);

  function control(c, set) {
    if (c.type === "seg") {
      const btns = c.options.map((o) => h("button", {
        type: "button", "aria-pressed": String(optValue(o) === (c.value ?? optValue(c.options[0]))),
        onclick: (e) => {
          btns.forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
          set(optValue(o));
        },
      }, optLabel(o)));
      return h("label", {}, c.label, h("span", { class: "viz-seg" }, btns));
    }
    if (c.type === "range") {
      const val = h("output", {}, c.value);
      const inp = h("input", {
        type: "range", min: c.min, max: c.max, step: c.step, value: c.value,
        oninput: (e) => { val.textContent = e.target.value; set(+e.target.value); },
      });
      return h("label", {}, h("span", {}, c.label, " ", val), inp);
    }
    if (c.type === "text") return h("label", {}, c.label, h("input", { type: "text", placeholder: "name, task, note…", oninput: (e) => set(e.target.value) }));
    const sel = h("select", { onchange: (e) => set(isNaN(e.target.value) || c.string ? e.target.value : +e.target.value) },
      c.options.map((o) => h("option", { value: optValue(o) }, optLabel(o))));
    if (c.value != null) sel.value = c.value;
    return h("label", {}, c.label, sel);
  }

  const note = (text) => h("p", { class: "viz-note" }, text);
  const tiles = (items) => h("div", { class: "viz-tiles" },
    items.map(([v, label]) => h("div", { class: "viz-tile" }, h("b", { class: String(v).length > 8 ? "small" : null }, v), h("span", {}, label))));

  function table(rows, cols, { sort } = {}) {
    let key = sort ?? cols[0].key, dir = -1;
    const body = h("tbody");
    const fill = () => {
      const sorted = [...rows].sort((a, b) => {
        const x = a[key], y = b[key];
        return (x == null) - (y == null) || dir * (x < y ? -1 : x > y ? 1 : 0);
      });
      body.replaceChildren(...sorted.map((r) => h("tr", {}, cols.map((c) =>
        h("td", { class: c.num ? "num" : null }, c.render ? c.render(r) : c.num ? fmt(r[c.key], c.digits ?? 2) : r[c.key] ?? "–")))));
    };
    const head = h("tr", {}, cols.map((c) => h("th", {
      title: c.title ?? "sort", onclick: () => { dir = key === c.key ? -dir : -1; key = c.key; fill(); },
    }, c.label)));
    fill();
    return h("div", { class: "viz-scroll" }, h("table", { class: "viz-table" }, h("thead", {}, head), body));
  }

  let LANG = null;
  async function languages() {
    if (!LANG) {
      const d = await load("ladder");
      LANG = { iso: d.languages, fw: d.fineweb_iso2 };
    }
    return LANG;
  }
  const langName = (code) => LANG?.iso[code]?.language ?? code;
  const fwName = (fw) => langName(LANG?.fw[fw.split("_")[0]] ?? fw) + ` (${fw})`;

  // ---------- views ----------
  const VIEWS = {};

  VIEWS.hero = async (el) => {
    const f = await load("facts");
    const rows = Object.entries(f.gate.per_size).map(([size, n]) => ({ size, n, share: n / f.gate.gated_tasks }));
    mount(el, [], () => [
      plot({
        height: 220, marginLeft: 60,
        x: { domain: PROXIES, label: "model size (non-embedding)" },
        y: { label: "tasks above chance", domain: [0, f.gate.gated_tasks], grid: true },
        marks: [
          Plot.barY(rows, { x: "size", y: "n", fill: css("--viz-s1"), rx: 4, tip: true,
            title: (d) => `${d.size}: ${d.n} of ${f.gate.gated_tasks} tasks (${pct(d.share)}) beat chance` }),
          Plot.ruleY([f.gate.gated_tasks], { strokeDasharray: "4 3", stroke: css("--viz-muted") }),
          Plot.text([f.gate.gated_tasks], { x: PROXIES[0], y: (d) => d, dy: -8, textAnchor: "start", text: () => `${f.gate.gated_tasks} (task × language) pairs evaluated` }),
          Plot.text(rows, { x: "size", y: "n", dy: -8, text: (d) => pct(d.share) }),
        ],
      }),
      note("Most multilingual benchmark tasks are at chance on small models: half of the suite carries no signal for the ablation you run at 175M."),
    ]);
  };

  VIEWS.ladder = async (el) => {
    const d = await languages().then(() => load("ladder"));
    const variants = ["deep A", "shallow A", "deep B", "shallow B", "deep AT3", "shallow AT3", "deep ZH", "deep ES"];
    const color = (c) => series(8)[variants.indexOf(`${c.arch} ${c.scheme}`)];
    let selected = d.cells.find((c) => c.size === "1.7B" && c.benchmarks) ?? d.cells[0];
    const status = (c) => (c.benchmarks ? "bench" : d.final_bpb[c.name] ? "bpb" : "");
    const trainedSet = (c) => {
      if (c.L === 1) return new Set();
      const sets = d.language_sets[c.scheme === "AT3" ? "A" : c.scheme] ?? {};
      return new Set(sets[`FW_L${c.L}`] ?? []);
    };

    const detail = (c) => {
      const trained = trainedSet(c);
      const bpb = Object.entries(d.final_bpb[c.name] ?? {}).map(([fw, v]) => ({ fw, v, trained: trained.has(fw) }));
      const langs = ["English", ...[...trained].map(fwName)];
      return h("div", { class: "ladder-detail" },
        h("h4", {}, c.name),
        h("dl", {},
          h("dt", {}, "Size"), h("dd", {}, `${c.size} non-embedding (${(c.n_non_emb / 1e6).toFixed(0)} M; ${(c.params / 1e6).toFixed(0)} M with embeddings), d_model ${c.d_model}, ${c.arch}`),
          h("dt", {}, "Data"), h("dd", {}, `50 % English (DCLM) + 50 % FineWeb-2 over ${c.L} language${c.L > 1 ? "s" : ""}, scheme ${c.scheme}, sampling T = ${c.T}`),
          h("dt", {}, "Budget"), h("dd", {}, `${(c.tokens / 1e9).toFixed(1)} B tokens (100 × N, 5× Chinchilla), ${c.n_ckpts} checkpoints, seed ${c.seed}`),
          h("dt", {}, "Status"), h("dd", {}, { bench: "benchmarks + BPB in the analysis", bpb: "per-language BPB in the analysis", "": "not yet in the published report" }[status(c)]),
          h("dt", {}, "Languages"), h("dd", {}, langs.length > 12 ? `${langs.slice(0, 12).join(", ")} … (+${langs.length - 12})` : langs.join(", ")),
          h("dt", {}, "Links"), h("dd", {},
            h("a", { href: `${HF_ORG}/${c.name}`, target: "_blank" }, "Hugging Face checkpoint"), " · ",
            h("a", { href: `${WANDB}?nw=&runSets=${encodeURIComponent(c.name)}`, target: "_blank" }, "W&B project")),
        ),
        bpb.length ? plot({
          height: 180, marginBottom: 30,
          x: { axis: null, domain: bpb.sort((a, b) => a.v - b.v).map((b) => b.fw), label: null },
          y: { label: "final bits per byte", grid: true },
          color: { domain: [true, false], range: [css("--viz-s1"), css("--viz-grid")] },
          marks: [Plot.barY(bpb, { x: "fw", y: "v", fill: "trained", tip: true,
            title: (b) => `${fwName(b.fw)}: ${fmt(b.v, 3)} BPB${b.trained ? " (trained)" : ""}` })],
        }) : null,
        bpb.length ? note("Final bits per byte on each of the 100 validation languages, sorted. Blue: languages in this model's training mix.") : null,
      );
    };

    mount(el, [
      { key: "arch", label: "Architecture", type: "seg", options: ["all", "deep", "shallow"] },
      { key: "scheme", label: "Data scheme", options: ["all", "A", "AT3", "B", "ZH", "ES"] },
      { key: "seeds", label: "Seeds", type: "seg", options: [{ value: "main", label: "1904" }, { value: "all", label: "all" }] },
    ], (s) => {
      const keep = d.cells.filter((c) => (s.arch === "all" || c.arch === s.arch)
        && (s.scheme === "all" || c.scheme === s.scheme) && (s.seeds === "all" || c.seed === 1904));
      const grid = h("div", { class: "ladder-grid", style: `grid-template-columns: 4em repeat(${LS.length}, 1fr)` },
        h("div", {}), LS.map((L) => h("div", { class: "hd" }, `L = ${L}`)));
      const detailBox = h("div");
      for (const size of SIZES) {
        grid.append(h("div", { class: "rowhd" }, size));
        for (const L of LS) {
          const chips = keep.filter((c) => c.size === size && c.L === L).map((c) => {
            const b = h("button", {
              class: `ladder-chip ${status(c)} ${c === selected ? "sel" : ""}`, style: `--chip:${color(c)}`,
              title: `${c.name} — ${status(c) ? "evaluated" : "not yet in the report"}`, "aria-label": c.name,
              onclick: () => { selected = c; grid.querySelectorAll(".sel").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); detailBox.replaceChildren(detail(c)); },
            });
            return b;
          });
          grid.append(h("div", { class: "cell" }, chips));
        }
      }
      detailBox.append(detail(selected));
      const legend = h("div", { class: "ladder-legend" },
        variants.map((v, i) => h("span", {}, h("i", { style: `background:${series(8)[i]};border-color:${series(8)[i]}` }), v)),
        h("span", {}, "filled = benchmarks + BPB · half = BPB only · outline = not yet in the report"));
      const n = keep.length, ev = keep.filter(status).length;
      return [tiles([[n, "runs in this view"], [ev, "in the analysis"], [uniq(keep.map((c) => c.size)).length, "sizes"], [uniq(keep.map((c) => c.L)).length, "language settings"]]),
        h("div", { style: "margin-top:1em" }, grid), legend, detailBox];
    });
  };

  VIEWS.bpb = async (el) => {
    const d = await languages().then(() => load("ladder"));
    const langs = uniq(Object.values(d.final_bpb).flatMap(Object.keys)).sort((a, b) => fwName(a).localeCompare(fwName(b)));
    mount(el, [
      { key: "lang", label: "Validation language", options: langs.map((l) => ({ value: l, label: fwName(l) })), value: "deu_Latn", string: true },
      { key: "arch", label: "Architecture", type: "seg", options: ["deep", "shallow"] },
    ], (s) => {
      const rows = d.cells.filter((c) => c.seed === 1904 && c.scheme === "A" && c.arch === s.arch && d.final_bpb[c.name]?.[s.lang] != null)
        .map((c) => {
          const sets = d.language_sets.A[`FW_L${c.L}`] ?? [];
          return { ...c, bpb: d.final_bpb[c.name][s.lang], trained: sets.includes(s.lang) || s.lang === "eng_Latn", Lname: `L${c.L}` };
        });
      if (!rows.length) return h("div", { class: "viz-empty" }, "No run of this architecture was scored on this language yet.");
      return [plot({
        height: 340,
        x: { type: "log", label: "model size (non-embedding)", ticks: uniq(rows.map((r) => r.n_non_emb)), tickFormat: (v) => rows.find((r) => r.n_non_emb === v)?.size },
        y: { label: "final bits per byte (lower is better)", grid: true },
        color: { domain: LS.map((L) => `L${L}`), range: series(6), legend: true },
        marks: [
          Plot.line(rows, { x: "n_non_emb", y: "bpb", stroke: "Lname", sort: "n_non_emb", strokeWidth: 2 }),
          Plot.dot(rows, { x: "n_non_emb", y: "bpb", stroke: "Lname", fill: css("--viz-surface"), r: 4.5, strokeWidth: 2 }),
          Plot.dot(rows.filter((r) => r.trained), { x: "n_non_emb", y: "bpb", fill: "Lname", r: 4.5 }),
          Plot.dot(rows, { x: "n_non_emb", y: "bpb", r: 8, fill: "transparent", tip: true,
            title: (r) => `${r.name}\n${fmt(r.bpb, 3)} BPB — ${r.trained ? "language in the training mix" : "never trained on this language"}` }),
        ],
      }), note("Filled dots: the language is in that model's training mix. Hollow: never trained on it. More training languages lowers BPB on the ones added, at no visible cost on English (pick English to check).")];
    });
  };

  VIEWS.gate = async (el) => {
    const g = await languages().then(() => load("gate"));
    const tasks = expand(g.tasks).filter((t) => t.random_baseline != null);  // BPB, loss, LAMBADA have no chance level
    const families = uniq(tasks.map((t) => t.family)).sort();
    mount(el, [
      { key: "mode", label: "View", type: "seg", options: [{ value: "family", label: "all families" }, { value: "lang", label: "one family, per language" }] },
      { key: "family", label: "Family", options: families, value: "hellaswag", string: true },
    ], (s) => {
      const long = tasks.filter((t) => s.mode === "family" || t.family === s.family)
        .flatMap((t) => g.sizes.map((size) => ({ ...t, size, share: t[size] }))).filter((r) => r.share != null);
      const rowKey = s.mode === "family" ? "family" : "language";
      const cells = Object.values(long.reduce((m, r) => {
        const k = `${r[rowKey]}|${r.size}`;
        (m[k] ??= { row: r[rowKey], size: r.size, sum: 0, n: 0, opts: r.n_options }).sum += r.share;
        m[k].n += 1;
        return m;
      }, {})).map((c) => ({ ...c, share: c.sum / c.n }));
      const order = uniq(cells.map((c) => c.row)).sort((a, b) =>
        d3.mean(cells.filter((c) => c.row === b), (c) => c.share) - d3.mean(cells.filter((c) => c.row === a), (c) => c.share));
      return [plot({
        height: 24 * order.length + 60, marginLeft: 170,
        x: { domain: g.sizes, label: "model size", axis: "top" },
        y: { domain: order, label: null, tickFormat: (r) => (rowKey === "language" ? `${langName(r)} (${r})` : r.startsWith("rf_") ? `${r.slice(3)} · answer text` : r) },
        color: { type: "linear", domain: [0, 1], range: [css("--viz-neutral"), css("--viz-s1")], legend: true, label: "share of runs above chance" },
        marks: [
          Plot.cell(cells, { x: "size", y: "row", fill: "share", inset: 1, rx: 3, tip: true,
            title: (c) => `${c.row} at ${c.size}: ${pct(c.share)} of runs above chance${c.n > 1 ? ` (mean over ${c.n} tasks)` : ""}` }),
          Plot.text(cells, { x: "size", y: "row", text: (c) => pct(c.share), fill: (c) => (c.share > 0.6 ? "white" : css("--md-default-fg-color")) }),
        ],
      }), note(s.mode === "family"
        ? "Each cell: of the runs at that size, the share whose score clears the chance level with confidence (rq00's gate). Letter-format knowledge benchmarks (Global-MMLU, INCLUDE, Belebele) stay near 10 %; scored on the answer text (\u201canswer text\u201d rows) the same questions clear chance."
        : "Per language: signal emerges language by language, and later for lower-resource ones.")];
    });
  };

  VIEWS.reformulation = async (el) => {
    const g = await load("gate");
    const rows = g.reformulation.filter((r) => r.panel !== "rf − original");
    mount(el, [], () => [plot({
      height: 240, marginLeft: 60,
      fx: { label: null }, x: { domain: PROXIES, label: null },
      y: { label: "accuracy − chance", grid: true },
      color: { domain: uniq(rows.map((r) => r.panel)), range: [css("--viz-muted"), css("--viz-s1")], legend: true },
      marks: [
        Plot.ruleY([0], { stroke: css("--viz-muted") }),
        Plot.line(rows, { fx: "row", x: "col", y: "value", stroke: "panel", strokeWidth: 2 }),
        Plot.dot(rows, { fx: "row", x: "col", y: "value", fill: "panel", r: 4, tip: true, title: (r) => `${r.row} ${r.col}\n${r.panel}: ${fmt(r.value, 3)}` }),
      ],
    }), note("Scoring the answer text instead of the letter A–D lifts the same questions above chance at small sizes: the letter format measures the convention, not the knowledge.")]);
  };

  VIEWS.scaling = async (el) => {
    const d = await load("scaling");
    const fams = ["all", ...uniq(d.regimes.map((r) => r.family)).sort()];
    mount(el, [{ key: "family", label: "Family", options: fams, string: true }], (s) => {
      const rows = d.regimes.filter((r) => s.family === "all" || r.family === s.family);
      const regimes = uniq(d.regimes.map((r) => r.regime)).sort();
      return [plot({
        height: 380, marginLeft: 55,
        x: { label: "R² of the fit across model size", domain: [Math.min(0, d3.min(d.regimes, (r) => r.r2_size)), 1], grid: true },
        y: { label: "R² of the fit along training", domain: [Math.min(0, d3.min(d.regimes, (r) => r.r2_trajectory)), 1], grid: true },
        color: { domain: regimes, range: series(3), legend: true },
        marks: [
          Plot.dot(rows, { x: "r2_size", y: "r2_trajectory", stroke: "regime", fill: "regime", fillOpacity: 0.5, r: 5, tip: true,
            channels: { task: "task", family: "family", language: "language" } }),
        ],
      }), note("One dot per task. Top right: the score follows a log-linear trend both across sizes and along each run, so a small-model measurement has a trend to extrapolate from.")];
    });
  };

  VIEWS.da = async (el) => {
    const d = await load("decision_accuracy");
    mount(el, [{ key: "group", label: "Measurement", type: "seg", options: [{ value: "all benchmarks", label: "benchmarks" }, { value: "bpb", label: "bits per byte" }] }], (s) => {
      const rows = d.early_small.filter((r) => r.group === s.group);
      return [plot({
        height: 320,
        x: { label: "fraction of the proxy's training run", tickFormat: "%" },
        y: { label: "decision accuracy vs the 1.7B final ranking", domain: [0.3, 1], grid: true },
        color: { domain: PROXIES, range: sizeRamp(), legend: true, label: "proxy size" },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.text([0.5], { x: 1, y: (v) => v, dy: 9, textAnchor: "end", text: () => "coin flip", fill: css("--viz-muted") }),
          Plot.line(rows, { x: "frac", y: "da", stroke: "proxy_size", strokeWidth: 2 }),
          Plot.dot(rows, { x: "frac", y: "da", fill: "proxy_size", r: 3.5, tip: true,
            title: (r) => `${r.proxy_size} at ${pct(r.frac)} of its run\nDA ${fmt(r.da)} over ${r.tasks} tasks` }),
        ],
      }), note("Read a line left to right: how early a proxy of that size ranks the design variants the way the 1.7B model does at the end. Bits per byte reach high agreement from small, early checkpoints; the benchmark average stays near a coin flip.")];
    });
  };

  VIEWS.daBench = async (el) => {
    const d = await languages().then(() => load("decision_accuracy"));
    const rows = expand(d.per_task);
    const comps = { "DA-size": uniq(rows.filter((r) => r.da_def === "DA-size").map((r) => r.comparison)), "DA-ckpt": uniq(rows.filter((r) => r.da_def === "DA-ckpt").map((r) => r.comparison)) };
    const langs = ["all", ...uniq(rows.map((r) => r.language)).sort()];
    mount(el, [
      { key: "comp", label: "Comparison", options: [...comps["DA-size"].filter((c) => c.endsWith("1.7B")).map((c) => ({ value: c, label: `size: ${c}` })), ...comps["DA-ckpt"].map((c) => ({ value: c, label: `checkpoint: ${c.replace("@", " % of run at ").replace("f", "")}` }))], value: "350M→1.7B", string: true },
      { key: "lang", label: "Language", options: langs.map((l) => ({ value: l, label: l === "all" ? "all" : `${langName(l)} (${l})` })), string: true },
    ], (s) => {
      const sel = rows.filter((r) => r.comparison === s.comp && (s.lang === "all" || r.language === s.lang) && r.decision_acc != null);
      const order = d3.groupSort(sel, (g) => -d3.mean(g, (r) => r.decision_acc), (r) => r.benchmark);
      return [plot({
        height: 26 * order.length + 60, marginLeft: 170,
        x: { domain: [0, 1], label: "decision accuracy", grid: true },
        y: { domain: order, label: null },
        marks: [
          Plot.ruleX([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.dot(sel, { x: "decision_acc", y: "benchmark", r: 3.5, fill: css("--viz-s1"), fillOpacity: 0.35, tip: true,
            title: (r) => `${r.task} (${langName(r.language)})\nDA ${fmt(r.decision_acc)}` }),
          Plot.tickX(sel, Plot.groupY({ x: "mean" }, { x: "decision_acc", y: "benchmark", stroke: css("--viz-s2"), strokeWidth: 3 })),
        ],
      }), note("Dots: one task (one language). Orange tick: the family mean. DA is quantised when few design variants share a size, so read families, not single dots.")];
    });
  };

  // An SNR simulator: the textbook picture behind every other page.
  VIEWS.snrToy = (el) => {
    let seed = 7;
    const rng = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
    const { render } = mount(el, [
      { key: "gap", label: "True gap between recipes", type: "range", min: 0, max: 0.1, step: 0.005, value: 0.03 },
      { key: "noise", label: "Checkpoint noise (sd)", type: "range", min: 0.002, max: 0.04, step: 0.002, value: 0.01 },
      { key: "n", label: "Recipes", type: "range", min: 2, max: 8, step: 1, value: 5 },
    ], (s) => {
      seed = 7;
      const means = d3.range(s.n).map((i) => 0.45 + s.gap * (s.n === 1 ? 0 : i / (s.n - 1)));
      const pts = means.flatMap((m, i) => d3.range(10).map((c) => ({ recipe: `recipe ${i + 1}`, ckpt: c, score: m + s.noise * gauss() })));
      const final = means.map((_, i) => pts.filter((p) => p.recipe === `recipe ${i + 1}`).at(-1).score);
      const signal = (d3.max(final) - d3.min(final)) / d3.mean(final);
      const noise = d3.mean(means, (_, i) => { const v = pts.filter((p) => p.recipe === `recipe ${i + 1}`).map((p) => p.score); return d3.deviation(v) / d3.mean(v); });
      let agree = 0, total = 0;
      for (let t = 0; t < 400; t++) {
        const draw = means.map((m) => m + s.noise * gauss());
        for (let a = 0; a < s.n; a++) for (let b = a + 1; b < s.n; b++) { total++; agree += Math.sign(draw[b] - draw[a]) === Math.sign(means[b] - means[a]) ? 1 : 0; }
      }
      const da = total ? agree / total : 1;
      return [
        tiles([[fmt(signal, 3), "signal: (max − min) / mean"], [fmt(noise, 3), "noise: sd over checkpoints / mean"], [fmt(signal / noise, 1), "SNR = signal / noise"], [pct(da), "decision accuracy of one checkpoint"]]),
        plot({
          height: 40 * s.n + 50, marginLeft: 70,
          x: { label: "benchmark score", grid: true },
          y: { label: null },
          marks: [
            Plot.dot(pts, { x: "score", y: "recipe", r: 4, fill: css("--viz-s1"), fillOpacity: 0.5, tip: true, title: (p) => `${p.recipe}, checkpoint ${p.ckpt + 1}: ${fmt(p.score, 3)}` }),
            Plot.tickX(means.map((m, i) => ({ m, recipe: `recipe ${i + 1}` })), { x: "m", y: "recipe", stroke: css("--viz-s2"), strokeWidth: 3 }),
          ],
        }),
        note("Dots: ten late checkpoints of each recipe. Orange tick: its true quality. Shrink the gap or raise the noise: when the SNR falls towards 1, one checkpoint ranks the recipes little better than a coin flip."),
      ];
    });
    return render;
  };

  VIEWS.snr = async (el) => {
    const d = await languages().then(() => load("snr"));
    const rows = expand(d.per_task).filter((r) => r.log_snr != null);
    mount(el, [{ key: "size", label: "Model size", type: "seg", options: PROXIES.filter((p) => rows.some((r) => r.size === p)), value: "1B" }], (s) => {
      const sel = rows.filter((r) => r.size === s.size).map((r) => ({ ...r, x: Math.max(-1.5, Math.min(2.5, r.log_snr)) }));
      const order = d3.groupSort(sel, (g) => -d3.median(g, (r) => r.log_snr), (r) => r.family);
      return [plot({
        height: 24 * order.length + 60, marginLeft: 190,
        x: { label: "log₁₀ SNR (clipped to [−1.5, 2.5])", grid: true },
        y: { domain: order, label: null },
        marks: [
          Plot.ruleX([0], { stroke: css("--viz-muted") }),
          Plot.dot(sel, { x: "x", y: "family", r: 3.5, fill: css("--viz-s1"), fillOpacity: 0.35, tip: true, title: (r) => `${r.task} (${langName(r.language)})\nlog₁₀ SNR ${fmt(r.log_snr)}` }),
          Plot.tickX(sel, Plot.groupY({ x: "median" }, { x: "x", y: "family", stroke: css("--viz-s2"), strokeWidth: 3 })),
        ],
      }), note("Only tasks that clear chance at this size get an SNR. Right of 0: the spread between recipes is larger than the checkpoint noise.")];
    });
  };

  VIEWS.surrogates = async (el) => {
    const d = await load("surrogates");
    mount(el, [{ key: "kind", label: "Measurement", type: "seg", options: uniq(d.rho.map((r) => r.kind)) }], (s) => {
      const rows = d.rho.filter((r) => r.kind === s.kind);
      return [plot({
        height: 30 * uniq(rows.map((r) => r.metric)).length + 70, marginLeft: 240,
        x: { domain: PROXIES.filter((p) => rows.some((r) => r.proxy === p)), label: "proxy size", axis: "top" },
        y: { label: null },
        color: { ...diverging(), domain: [-0.8, 0.8], legend: true, label: "Spearman ρ with decision accuracy" },
        marks: [
          Plot.cell(rows, { x: "proxy", y: "metric", fill: "rho", inset: 1, rx: 3, tip: true, title: (r) => `${r.metric} at ${r.proxy}\nρ = ${fmt(r.rho)} (p = ${fmt(r.p, 3)}, n = ${r.n})` }),
          Plot.text(rows, { x: "proxy", y: "metric", text: (r) => fmt(r.rho), fontWeight: (r) => (r.p < 0.05 ? "bold" : "normal") }),
        ],
      }), note("Can a number computed on the small model alone tell you whether its decision will hold? Blue: the statistic ranks tasks like their decision accuracy. Bold: p < 0.05. No single statistic wins everywhere.")];
    });
  };

  VIEWS.bestVariant = async (el) => {
    const d = await languages().then(() => load("surrogates"));
    const rows = d.best_variant.filter((r) => r.best_variant_da_size || r.best_variant_da_ckpt).map((r) => ({ ...r, name: langName(r.language) }));
    mount(el, [], () => table(rows, [
      { key: "name", label: "Language" },
      { key: "best_variant_da_size", label: "Best SNR for size decisions" },
      { key: "best_pearson_r_da_size", label: "r", num: true },
      { key: "best_variant_da_ckpt", label: "Best SNR for checkpoint decisions" },
      { key: "best_pearson_r_da_ckpt", label: "r", num: true },
    ], { sort: "best_pearson_r_da_size" }));
  };

  VIEWS.effect = async (el) => {
    const d = await load("design_decisions");
    mount(el, [], () => [plot({
      height: 280, marginLeft: 190,
      x: { type: "log", label: "median |effect| of the decision ÷ seed sd", grid: true, ticks: [0.3, 0.5, 1, 2, 5], tickFormat: (v) => `${v}×` },
      y: { label: null },
      color: { domain: ["benchmarks", "bits per byte"], range: series(2), legend: true },
      marks: [
        Plot.ruleX([1], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
        Plot.dot(d.effect, { x: "median_effect_over_seed_sd", y: "label", stroke: "population", fill: "population", fillOpacity: 0.5, r: 6, tip: true,
          title: (r) => `${r.label}, L = ${r.L} (${r.population}, ref ${r.reference_size})\neffect ${fmt(r.median_effect_over_seed_sd)}× seed sd, ${pct(r.share_above_2)} of tasks above 2×` }),
      ],
    }), note("Left of the dashed line: changing the seed moves the scores more than the design decision does. There is nothing reliable to rank, whatever the proxy.")]);
  };

  VIEWS.interventions = async (el) => {
    const d = await load("design_decisions");
    const pops = uniq(d.da.map((r) => r.population));
    mount(el, [{ key: "pop", label: "Measurement", type: "seg", options: pops.map((v) => ({ value: v, label: POP[v] ?? v })) }], (s) => {
      const rows = d.da.filter((r) => r.population === s.pop);
      const labels = uniq(d.da.map((r) => r.label));
      return [plot({
        height: 300,
        x: { domain: PROXIES.slice(0, 4), label: "proxy size" },
        y: { domain: [0, 1], label: "decision accuracy vs the reference", grid: true },
        color: { domain: labels, range: series(labels.length), legend: true },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.line(rows, { x: "proxy_size", y: "decision_acc", stroke: "label", strokeWidth: 2, sort: (r) => PROXIES.indexOf(r.proxy_size) }),
          Plot.dot(rows, { x: "proxy_size", y: "decision_acc", fill: "label", r: 4, tip: true, title: (r) => `${r.label} at ${r.proxy_size}\nDA ${fmt(r.decision_acc)} over ${r.cells} cells (ref ${r.refs})` }),
        ],
      })];
    });
  };

  VIEWS.early = async (el) => {
    const d = await load("design_decisions");
    mount(el, [
      { key: "iv", label: "Decision", type: "seg", options: uniq(d.early.map((r) => r.intervention)).map((v) => ({ value: v, label: d.early.find((r) => r.intervention === v).label })) },
      { key: "pop", label: "Measurement", options: uniq(d.early.map((r) => r.population)).map((v) => ({ value: v, label: POP[v] ?? v })), string: true },
    ], (s) => {
      const rows = d.early.filter((r) => r.intervention === s.iv && r.population === s.pop);
      return [plot({
        height: 300,
        x: { label: "fraction of the proxy's run", tickFormat: "%" },
        y: { domain: [0, 1], label: "decision accuracy vs the 1.7B reference", grid: true },
        color: { domain: PROXIES, range: sizeRamp(), legend: true, label: "proxy size" },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.line(rows, { x: "frac", y: "da", stroke: "proxy_size", strokeWidth: 2 }),
          Plot.dot(rows, { x: "frac", y: "da", fill: "proxy_size", r: 3.5, tip: true, title: (r) => `${r.proxy_size} at ${pct(r.frac)}: DA ${fmt(r.da)} (${r.items} items)` }),
        ],
      })];
    });
  };

  VIEWS.transferSummary = async (el) => {
    const d = await load("transfer");
    const rows = d.summary.flatMap((r) => [["transfer", "pooled exponent (transfer)"], ["own", "the language's own fit"], ["last", "largest proxy as is"]]
      .filter(([k]) => r[k] != null).map(([k, label]) => ({ k: r.k, err: r[k], method: label, group: r.trained ? "trained languages" : "never-trained languages", n: r.n })));
    mount(el, [], () => [plot({
      height: 280, marginLeft: 60,
      fx: { label: null }, x: { label: "proxy rungs observed", ticks: [1, 2, 3, 4] },
      y: { label: "median error at the reference", tickFormat: "%", grid: true },
      color: { domain: uniq(rows.map((r) => r.method)), range: series(3), legend: true },
      marks: [
        Plot.line(rows, { fx: "group", x: "k", y: "err", stroke: "method", strokeWidth: 2 }),
        Plot.dot(rows, { fx: "group", x: "k", y: "err", fill: "method", r: 4, tip: true, title: (r) => `${r.group}, ${r.k} rung(s)\n${r.method}: ${pct(r.err)} (n = ${r.n})` }),
      ],
    }), note("One small measurement plus the exponent pooled over the other languages predicts a language's BPB at the reference size within a few percent, even for languages the model never trained on.")]);
  };

  VIEWS.transfer = async (el) => {
    const d = await languages().then(() => load("transfer"));
    const rows = expand(d.per_language);
    mount(el, [
      { key: "L", label: "Training languages", options: uniq(rows.map((r) => r.L)).map((L) => ({ value: L, label: `L = ${L}` })), value: 8 },
      { key: "k", label: "Rungs observed", type: "seg", options: [1, 2, 3, 4] },
    ], (s) => {
      const sel = rows.filter((r) => r.L === s.L && r.k === s.k && r.pred_transfer != null)
        .map((r) => ({ ...r, lang: fwName(r.task.replace("bpb_", "")), status: r.trained ? "trained" : "never trained" }));
      const ext = d3.extent(sel.flatMap((r) => [r.observed, r.pred_transfer]));
      return [plot({
        height: 380, width: 480, marginLeft: 55,
        x: { label: "observed BPB at the reference", domain: ext, grid: true },
        y: { label: "predicted BPB", domain: ext, grid: true },
        color: { domain: ["trained", "never trained"], range: series(2), legend: true },
        marks: [
          Plot.line(ext.map((v) => [v, v]), { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.dot(sel, { x: "observed", y: "pred_transfer", stroke: "status", fill: "status", fillOpacity: 0.5, r: 4, tip: true,
            title: (r) => `${r.lang} (${r.status})\nobserved ${fmt(r.observed, 3)}, predicted ${fmt(r.pred_transfer, 3)} (${pct(Math.abs(r.err_transfer))} off)` }),
        ],
      })];
    });
  };

  VIEWS.external = async (el) => {
    const d = await load("external");
    const a = d.agreement[0] ?? {};
    const rows = [...d.ours.map((r) => ({ ...r, corpus: "this ladder" })), ...d.allenai.map((r) => ({ ...r, corpus: "AllenAI DataDecide" }))];
    mount(el, [], () => [
      tiles([[a.n_shared ?? "–", "shared English tasks"], [fmt(a.pearson_log_snr), "Pearson r of log SNR"], [fmt(a.spearman_rank), "Spearman ρ of the ranking"], [a.variant ?? "–", "SNR definition"]]),
      plot({
        height: 24 * uniq(rows.map((r) => r.task)).length + 70, marginLeft: 140, marginTop: 30,
        fx: { label: null }, x: { label: "SNR", grid: true }, y: { label: null },
        marks: [Plot.barX(rows, { fx: "corpus", x: "snr", y: "task", fill: css("--viz-s1"), rx: 4, sort: { y: "-x" }, tip: true })],
      }),
      note("With so few shared tasks the agreement is suggestive, not conclusive — the same benchmarks (HellaSwag, PIQA, ARC-Easy) lead on both corpora."),
    ]);
  };

  VIEWS.subsets = async (el) => {
    const rows = (await load("subsets")).filter((r) => r.best_snr != null);
    const cases = uniq(rows.map((r) => r.case));
    mount(el, [
      { key: "case", label: "Subset of", options: cases.map((c) => ({ value: c, label: c.replace(/^case\d_/, "").replaceAll("_", " ") })), string: true },
      { key: "size", label: "Model size", options: uniq(rows.map((r) => r.size)), string: true },
    ], (s) => {
      const sel = rows.filter((r) => r.case === s.case && r.size === s.size).sort((a, b) => b.snr_gain - a.snr_gain).slice(0, 25);
      if (!sel.length) return h("div", { class: "viz-empty" }, "No subset for this combination.");
      return [plot({
        height: 22 * sel.length + 60, marginLeft: 190,
        x: { label: "SNR", grid: true }, y: { domain: sel.map((r) => r.task), label: null },
        marks: [
          Plot.link(sel, { x1: "full_set_snr", x2: "best_snr", y1: "task", y2: "task", stroke: css("--viz-muted"), markerEnd: "arrow" }),
          Plot.dot(sel, { x: "full_set_snr", y: "task", fill: css("--viz-muted"), r: 4 }),
          Plot.dot(sel, { x: "best_snr", y: "task", fill: css("--viz-s1"), r: 4.5, tip: true, title: (r) => `${r.task}: ${fmt(r.full_set_snr)} → ${fmt(r.best_snr)} with ${r.best_n} subtasks\n${r.best_subset_short ?? ""}` }),
        ],
      }), note("Grey: the full benchmark. Blue: the best subset of its languages or subjects. Part of every gain is selection on the same numbers — rq08 tests it against random subsets of the same size.")];
    });
  };

  VIEWS.design = async (el) => {
    const rows = await load("benchmark_design");
    const feats = [["context_len_chars_median", "context length (chars)"], ["n_options", "answer options"], ["option_len_chars_median", "option length (chars)"], ["context_to_option_ratio", "context ÷ option length"]];
    mount(el, [{ key: "x", label: "Design feature", options: feats.map(([value, label]) => ({ value, label })), string: true }], (s) => {
      const cats = uniq(rows.map((r) => r.curation_category)).sort();
      return [plot({
        height: 340,
        x: { label: feats.find(([k]) => k === s.x)[1], grid: true, type: s.x === "n_options" ? "linear" : "log" },
        y: { label: "median SNR of the family's tasks", grid: true },
        marginRight: 80,
        color: { domain: cats, range: series(cats.length) },
        symbol: { domain: cats, legend: true },
        marks: [
          Plot.dot(rows, { x: s.x, y: "snr_median", fill: "curation_category", symbol: "curation_category", r: 6, tip: true, title: (r) => `${r.family}: median SNR ${fmt(r.snr_median)} over ${r.n_tasks} tasks\n${r.curation_process}` }),
          Plot.text(rows, { x: s.x, y: "snr_median", text: "family", dx: 9, textAnchor: "start", fontSize: 10 }),
        ],
      }), note("Only the families that clear chance are here — most are two-option. Too few remain to separate curation, format or length effects.")];
    });
  };

  // ---------- benchmark catalogue (configs/multilingual_benchmarks.csv) ----------
  VIEWS.benchmarks = async (el) => {
    await languages();
    const rows = (await d3.csv(new URL("benchmarks.csv", DATA).href)).map((r) => ({
      ...r, langs: r.languages ? r.languages.split(";") : [], n_languages: +r.n_languages || null,
      n_items: r.n_items === "" ? null : +r.n_items, cats: r.categories ? r.categories.split(";").map((c) => c.trim()) : [],
      sources: r.data_source ? r.data_source.split(";").map((c) => c.trim()) : [],
    }));
    const split = (key) => uniq(rows.flatMap((r) => r[key])).filter(Boolean).sort();
    const langs = split("langs").sort((a, b) => langName(a).localeCompare(langName(b)));
    const all = (label, values, fmtv = (v) => v) => [{ value: "", label: `all ${label}` }, ...values.map((v) => ({ value: v, label: fmtv(v) }))];
    const link = (url, text) => (url ? h("a", { href: url, target: "_blank", rel: "noopener" }, text) : "–");
    const csvOf = (list) => d3.csvFormat(list.map(({ langs: _l, cats: _c, sources: _s, ...r }) => r));
    mount(el, [
      { key: "q", label: "Search", type: "text", value: "" },
      { key: "lang", label: "Language", options: all("languages", langs, (l) => `${langName(l)} (${l})`), string: true },
      { key: "fw", label: "Framework", type: "seg", options: [{ value: "", label: "all" }, { value: "harness", label: "lm-eval-harness" }, { value: "lighteval", label: "lighteval" }] },
      { key: "format", label: "Format", type: "seg", options: [{ value: "", label: "all" }, { value: "mcqa", label: "MCQA" }, { value: "generative", label: "generative" }] },
      { key: "cat", label: "Category", options: all("categories", split("cats")), string: true },
      { key: "src", label: "Data source", options: all("sources", split("sources")), string: true },
    ], (s) => {
      const q = (s.q ?? "").toLowerCase();
      const sel = rows.filter((r) => (!q || `${r.name} ${r.id} ${r.suite} ${r.harness_tasks} ${r.lighteval_tasks} ${r.notes}`.toLowerCase().includes(q))
        && (!s.lang || r.langs.includes(s.lang)) && (!s.fw || r.frameworks.includes(s.fw))
        && (!s.format || r.format.includes(s.format)) && (!s.cat || r.cats.includes(s.cat)) && (!s.src || r.sources.includes(s.src)));
      const perLang = d3.rollups(sel.flatMap((r) => r.langs), (v) => v.length, (l) => l).sort((a, b) => b[1] - a[1]).slice(0, 40)
        .map(([l, n]) => ({ l, n, name: langName(l) }));
      return [
        tiles([[sel.length, `of ${rows.length} benchmarks`], [uniq(sel.flatMap((r) => r.langs)).length, "languages covered"],
          [d3.format(".3s")(d3.sum(sel, (r) => r.n_items) || 0).replace("G", "B"), "evaluation items"],
          [sel.filter((r) => r.frameworks.includes("harness") && r.frameworks.includes("lighteval")).length, "in both frameworks"]]),
        perLang.length > 1 ? plot({
          height: 200, marginBottom: 40, marginTop: 24,
          x: { domain: perLang.map((d) => d.l), label: null, tickRotate: -45 },
          y: { label: "benchmarks covering the language", grid: true },
          marks: [Plot.barY(perLang, { x: "l", y: "n", fill: css("--viz-s1"), rx: 3, tip: true, title: (d) => `${d.name} (${d.l}): ${d.n} benchmarks` })],
        }) : null,
        h("div", { class: "viz-controls", style: "margin-top:.6em" }, h("span", { class: "viz-seg" },
          h("button", { type: "button", onclick: () => { const a = h("a", { href: URL.createObjectURL(new Blob([csvOf(sel)], { type: "text/csv" })), download: "multilingual_benchmarks.csv" }); a.click(); } }, `download these ${sel.length} rows`))),
        table(sel, [
          { key: "name", label: "Benchmark", render: (r) => [h("b", {}, r.name), r.suite ? h("div", { class: "viz-note" }, r.suite) : null] },
          { key: "n_languages", label: "Languages", num: true, render: (r) => h("span", { title: r.langs.map((l) => `${langName(l)} (${l})`).join(", ") },
            r.langs.length <= 4 ? r.langs.join(", ") : `${r.langs.length} (${r.langs.slice(0, 3).join(", ")}, …)`) },
          { key: "frameworks", label: "Framework", render: (r) => h("span", { title: [r.harness_tasks && `harness: ${r.harness_tasks}`, r.lighteval_tasks && `lighteval: ${r.lighteval_tasks}`].filter(Boolean).join("\n") }, r.frameworks) },
          { key: "format", label: "Format" },
          { key: "n_options", label: "Options" },
          { key: "n_items", label: "Items", num: true, render: (r) => (r.n_items == null ? "–" : d3.format(",")(r.n_items)) },
          { key: "categories", label: "Categories" },
          { key: "data_source", label: "Data source" },
          { key: "hf_dataset", label: "Links", render: (r) => [...r.hf_dataset.split(";").filter(Boolean).flatMap((u, i) => [i ? " " : "", link(u.trim(), i ? `HF${i + 1}` : "HF")]), " · ", link(r.paper, "paper")] },
          { key: "notes", label: "Notes", render: (r) => h("span", { class: "viz-note" }, r.notes || "") },
        ], { sort: "n_languages" }),
      ];
    });
  };

  // ---------- recommendations ----------
  VIEWS.recommend = async (el) => {
    await languages();
    const [da, gate, snr] = await Promise.all([load("decision_accuracy"), load("gate"), load("snr")]);
    const daRows = expand(da.per_task);
    const gateBy = byKey(expand(gate.tasks), (r) => r.task);
    const snrBy = byKey(expand(snr.per_task), (r) => `${r.task}|${r.size}`);
    const langs = uniq(daRows.map((r) => r.language)).sort((a, b) => langName(a).localeCompare(langName(b)));
    mount(el, [
      { key: "lang", label: "Target language", options: langs.map((l) => ({ value: l, label: `${langName(l)} (${l})` })), value: "de", string: true },
      { key: "size", label: "Your proxy size", type: "seg", options: ["175M", "350M", "600M", "1B"], value: "350M" },
      { key: "q", label: "Your decision", options: [{ value: "size", label: "Will my small-model ranking hold at 1.7B?" }, { value: "ckpt", label: "Can I decide before the run ends?" }], string: true },
      { key: "frac", label: "Decide at (% of run)", options: [10, 20, 30, 40, 50, 60].map((v) => ({ value: v, label: `${v} %` })), value: 20 },
    ], (s) => {
      const comp = s.q === "size" ? `${s.size}→1.7B` : `f${s.frac}@${s.size}`;
      const rows = daRows.filter((r) => r.language === s.lang && r.comparison === comp).map((r) => {
        const share = r.benchmark === "bpb" ? 1 : gateBy[r.task]?.[s.size];
        const logSnr = snrBy[`${r.task}|${s.size}`]?.log_snr;
        const verdict = share != null && share < 0.5 ? "avoid" : r.decision_acc == null ? null : r.decision_acc >= 0.75 ? "use" : r.decision_acc >= 0.6 ? "care" : "avoid";
        const why = share != null && share < 0.5 ? "at chance at this size" : r.decision_acc >= 0.75 ? "ranks like the reference" : r.decision_acc >= 0.6 ? "often right, check with a second task" : "close to a coin flip";
        return { ...r, share, logSnr, verdict, why, rank: { use: 3, care: 2, avoid: 1 }[verdict] ?? 0 };
      });
      if (!rows.length) return h("div", { class: "viz-empty" }, "No measurement for this language and comparison.");
      const best = rows.filter((r) => r.verdict === "use").sort((a, b) => b.decision_acc - a.decision_acc);
      const badge = (r) => (r.verdict ? [h("span", { class: `badge ${r.verdict}` }, { use: "✓ use", care: "! with care", avoid: "✕ avoid" }[r.verdict]), h("div", { class: "viz-note" }, r.why)] : "–");
      const bpbEarly = da.early_small.filter((r) => r.group === "bpb" && r.proxy_size === s.size && r.da >= 0.75).sort((a, b) => a.frac - b.frac)[0];
      return [
        tiles([[best.length, `of ${rows.length} measurements to use`], [best[0]?.task ?? "none", "most reliable here"], [bpbEarly ? pct(bpbEarly.frac) : "not reached", `of a ${s.size} run for BPB to rank like 1.7B (DA ≥ 0.75, all languages)`]]),
        h("div", { style: "margin-top:1em" }, table(rows, [
          { key: "rank", label: "Verdict", render: badge },
          { key: "task", label: "Task" },
          { key: "benchmark", label: "Family" },
          { key: "share", label: "Runs above chance", num: true, render: (r) => pct(r.share), title: `share of ${s.size} runs above chance (rq00)` },
          { key: "logSnr", label: "log₁₀ SNR", num: true, title: `at ${s.size} (rq03)` },
          { key: "decision_acc", label: "Decision accuracy", num: true, title: `${comp} (rq02)` },
        ], { sort: "rank" })),
        note(`Decision accuracy here is ${s.q === "size" ? `the ranking of our design variants at ${s.size} against 1.7B` : `the ranking at ${s.frac} % of a ${s.size} run against its final checkpoint`}, on the ladder's own decisions (language count, depth, language list, temperature). Treat it as a prior; confirm on your own runs with the upload tool below.`),
      ];
    });
  };

  // Port of snr.metrics.decision_acc_fast: sign agreement over unordered pairs.
  function decisionAcc(small, target) {
    let agree = 0, n = 0;
    for (let i = 0; i < small.length; i++) for (let j = i + 1; j < small.length; j++) {
      n++; agree += Math.sign(small[i] - small[j]) === Math.sign(target[i] - target[j]) ? 1 : 0;
    }
    return n ? agree / n : null;
  }

  function parseCSV(text) {
    const rows = [[]];
    let cell = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') quoted = false; else cell += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { rows.at(-1).push(cell.trim()); cell = ""; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        rows.at(-1).push(cell.trim()); cell = ""; rows.push([]);
      } else cell += c;
    }
    rows.at(-1).push(cell.trim());
    const [head, ...body] = rows.filter((r) => r.some(Boolean));
    const cols = head.map((c) => c.toLowerCase());
    return body.map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i] ?? ""])));
  }

  /* Per task: signal and noise as in snr.metrics.signal_to_noise_ratio, DA-size
     (smallest vs largest size) and DA-ckpt (early vs final at the largest size). */
  function analyse(records, { tail, early }) {
    const out = [];
    for (const [task, recs] of d3.group(records, (r) => r.task)) {
      const lower = recs.some((r) => /^(1|true|yes)$/i.test(r.lower_is_better ?? ""));
      const sizes = uniq(recs.map((r) => r.size ?? "")).sort((a, b) => sizeNum(a) - sizeNum(b));
      const big = sizes.at(-1), small = sizes[0];
      const at = (size) => d3.group(recs.filter((r) => (r.size ?? "") === size), (r) => r.recipe);
      const bigBy = at(big);
      const recipes = [...bigBy.keys()].sort();
      const curve = (by, rc) => (by.get(rc) ?? []).map((r) => ({ step: +(r.step ?? 0), score: +r.score })).sort((a, b) => a.step - b.step);
      const final = (by, rc) => curve(by, rc).at(-1)?.score;
      const finals = recipes.map((rc) => final(bigBy, rc));
      const signal = (d3.max(finals) - d3.min(finals)) / d3.mean(finals);
      const noises = recipes.map((rc) => { const v = curve(bigBy, rc).slice(-tail).map((p) => p.score); return v.length > 1 ? d3.deviation(v) / d3.mean(v) : null; }).filter((v) => v != null);
      const noise = noises.length ? d3.mean(noises) : null;
      const sgn = lower ? -1 : 1;
      let daSize = null;
      if (small !== big) {
        const smallBy = at(small);
        const shared = recipes.filter((rc) => smallBy.has(rc));
        if (shared.length > 1) daSize = decisionAcc(shared.map((rc) => sgn * final(smallBy, rc)), shared.map((rc) => sgn * final(bigBy, rc)));
      }
      const earlyScores = recipes.map((rc) => {
        const c = curve(bigBy, rc); if (c.length < 2) return null;
        const target = early * c.at(-1).step;
        return c.reduce((b, p) => (Math.abs(p.step - target) < Math.abs(b.step - target) ? p : b)).score;
      });
      const daCkpt = earlyScores.every((v) => v != null) && recipes.length > 1 ? decisionAcc(earlyScores.map((v) => sgn * v), finals.map((v) => sgn * v)) : null;
      const chance = recs.find((r) => r.chance)?.chance;
      const above = chance == null || chance === "" ? null : d3.median(finals) > +chance;
      out.push({ task, recipes: recipes.length, sizes: sizes.length, signal, noise, snr: noise ? signal / noise : null, daSize, daCkpt, above, big, small });
    }
    const snrs = out.map((r) => r.snr).filter((v) => v != null).sort(d3.ascending);
    const cut = [d3.quantile(snrs, 1 / 3), d3.quantile(snrs, 2 / 3)];
    for (const r of out) {
      const da = r.daSize ?? r.daCkpt;
      r.verdict = r.above === false ? "avoid" : da != null ? (da >= 0.75 ? "use" : da >= 0.6 ? "care" : "avoid")
        : r.snr == null ? null : r.snr >= cut[1] ? "use" : r.snr >= cut[0] ? "care" : "avoid";
      r.rank = { use: 3, care: 2, avoid: 1 }[r.verdict] ?? 0;
    }
    return out;
  }

  function exampleCSV() {
    let s = 11;
    const rng = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
    const tasks = { hellaswag_de: [0.3, 0.04, 0.004], xnli_de: [0.4, 0.02, 0.012], belebele_deu: [0.235, 0.005, 0.01], multiblimp_deu: [0.7, 0.05, 0.006], bpb_deu: [1.2, -0.08, 0.003] };
    const lines = ["recipe,size,step,task,score,chance,lower_is_better"];
    for (const [task, [base, gap, noise]] of Object.entries(tasks))
      for (const [si, size] of ["150M", "1B"].entries())
        for (const [ri, recipe] of ["mix-a", "mix-b", "mix-c", "mix-d"].entries())
          for (let step = 1000; step <= 10000; step += 1000) {
            const quality = gap * ri / 3 * (task === "xnli_de" && si === 0 ? -1 : 1);
            const score = base + (task.startsWith("bpb") ? -0.1 * si : 0.05 * si) + quality * (step / 10000) + noise * 3 * rng();
            lines.push([recipe, size, step, task, score.toFixed(4), task === "belebele_deu" ? 0.25 : "", task.startsWith("bpb") ? 1 : 0].join(","));
          }
    return lines.join("\n");
  }

  VIEWS.upload = (el) => {
    const state = { tail: 5, early: 0.2, records: null, name: "" };
    const out = h("div");
    const input = h("input", { type: "file", accept: ".csv,text/csv", style: "display:none", onchange: (e) => read(e.target.files[0]) });
    const drop = h("div", { class: "dropzone", onclick: () => input.click(),
      ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); },
      ondragleave: () => drop.classList.remove("over"),
      ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); read(e.dataTransfer.files[0]); } },
    "Drop a CSV here or click to choose one. It stays in your browser: nothing is uploaded.");
    const read = (file) => file && file.text().then((t) => { state.records = parseCSV(t); state.name = file.name; render(); });
    const download = (name, text) => {
      const a = h("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
      a.click(); URL.revokeObjectURL(a.href);
    };
    const bar = h("div", { class: "viz-controls" },
      control({ key: "tail", label: "Checkpoints for noise", type: "range", min: 2, max: 10, step: 1, value: 5 }, (v) => { state.tail = v; render(); }),
      control({ key: "early", label: "Early checkpoint (% of run)", options: [10, 20, 30, 50].map((v) => ({ value: v / 100, label: `${v} %` })), value: 0.2 }, (v) => { state.early = v; render(); }),
      h("label", {}, "Try it", h("span", { class: "viz-seg" },
        h("button", { type: "button", onclick: () => { state.records = parseCSV(exampleCSV()); state.name = "example.csv"; render(); } }, "load example"),
        h("button", { type: "button", onclick: () => download("snr-template.csv", exampleCSV()) }, "download template"))));
    function render() {
      if (!state.records) return out.replaceChildren(note("Required columns: recipe, task, score. Optional: size, step, chance, lower_is_better."));
      const missing = ["recipe", "task", "score"].filter((c) => !(c in state.records[0]));
      if (missing.length) return out.replaceChildren(h("div", { class: "viz-empty" }, `Missing column(s): ${missing.join(", ")}.`));
      const rows = analyse(state.records, state);
      const withSnr = rows.filter((r) => r.snr != null).sort((a, b) => b.snr - a.snr);
      const badge = (r) => (r.verdict ? h("span", { class: `badge ${r.verdict}` }, { use: "✓ use", care: "! with care", avoid: "✕ avoid" }[r.verdict]) : "–");
      out.replaceChildren(
        tiles([[state.records.length, `rows in ${state.name}`], [rows.length, "tasks"], [d3.max(rows, (r) => r.recipes), "recipes compared"], [rows.filter((r) => r.verdict === "use").length, "tasks to keep"]]),
        withSnr.length ? plot({
          height: 26 * withSnr.length + 50, marginLeft: 150,
          x: { label: "SNR at the largest size", grid: true }, y: { domain: withSnr.map((r) => r.task), label: null },
          marks: [Plot.barX(withSnr, { x: "snr", y: "task", fill: css("--viz-s1"), rx: 4, tip: true, title: (r) => `${r.task}: SNR ${fmt(r.snr)} (signal ${fmt(r.signal, 3)}, noise ${fmt(r.noise, 4)})` })],
        }) : note("Add a step column with several late checkpoints per recipe to measure noise."),
        table(rows, [
          { key: "rank", label: "Verdict", render: badge },
          { key: "task", label: "Task" },
          { key: "recipes", label: "Recipes", num: true, digits: 0 },
          { key: "signal", label: "Signal", num: true, digits: 3 },
          { key: "noise", label: "Noise", num: true, digits: 4 },
          { key: "snr", label: "SNR", num: true, digits: 1 },
          { key: "above", label: "Above chance", render: (r) => (r.above == null ? "–" : r.above ? "yes" : "no") },
          { key: "daSize", label: "DA small → large", num: true, render: (r) => (r.daSize == null ? "–" : `${fmt(r.daSize)} (${r.small} → ${r.big})`) },
          { key: "daCkpt", label: "DA early → final", num: true },
        ], { sort: "rank" }),
        note("Signal = (max − min) / mean of the recipes' final scores at the largest size; noise = sd over each recipe's last checkpoints / mean, averaged over recipes (Heineman et al., 2025). The verdict follows decision accuracy when you give two sizes or several checkpoints (≥ 0.75 use, ≥ 0.6 with care), and otherwise the SNR tercile within your suite."),
      );
    }
    el.replaceChildren(bar, drop, input, out);
    REDRAW.push(render);
    render();
  };

  // ---------- boot ----------
  function boot() {
    for (const el of document.querySelectorAll("[data-viz]")) {
      const view = VIEWS[el.dataset.viz];
      if (!view) continue;
      el.replaceChildren(h("div", { class: "viz-empty" }, "Loading…"));
      Promise.resolve(view(el)).catch((e) => { el.replaceChildren(h("div", { class: "viz-empty" }, `Could not draw this view (${e.message}).`)); console.error(e); });
    }
    new MutationObserver(() => REDRAW.forEach((f) => f()))
      .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();

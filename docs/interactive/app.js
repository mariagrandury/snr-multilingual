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
  const LADDER = ["90M", "175M", "350M", "600M", "1B", "1.7B"];   // the analysis sizes, up to the reference
  const LS = [1, 2, 8, 15, 30, 50];
  // Double-blind review: no links to our HF / W&B pages. The sentence is the
  // anonymity.py REMOVED_LINK, injected by mkdocs_hooks.py as a <meta>.
  const ANON = document.querySelector('meta[name="anonymity-notice"]')?.content || "";
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

  // The intervention axes of the grid, one dropdown each. A cell carries one
  // field per axis (null where the axis does not apply, e.g. the second
  // language exists only at L = 2); an axis with a single level in the data
  // (the optimizer, for now) gets no dropdown until a second level is trained.
  const AXES = [
    { key: "list", label: "Language list", fmt: (v) => ({ A: "A · ranked by data size", B: "B · chosen for diversity" }[v] ?? v) },
    { key: "T", label: "Sampling temperature", fmt: (v) => `T = ${v}` },
    { key: "lang2", label: "Second language (L = 2)", fmt: (v) => langName(v) },
    { key: "english", label: "English data", fmt: (v) => v },
    { key: "arch", label: "Depth", fmt: (v) => v },
    { key: "activation", label: "Activation", fmt: (v) => ({ xielu: "xIELU", swiglu: "SwiGLU" }[v] ?? v) },
    { key: "optimizer", label: "Optimizer", fmt: (v) => ({ ademamix: "AdEMAMix" }[v] ?? v) },
  ];
  const BASE = { arch: "deep", activation: "xielu", list: "A", T: 1, lang2: "ru", english: "DCLM (edu-filtered)" };
  // What a run changes with respect to the baseline (deep, xIELU, list A, T = 1, Russian, DCLM-edu).
  const variantOf = (c) => {
    const diff = [];
    if (c.arch !== BASE.arch) diff.push(c.arch);
    if (c.activation !== BASE.activation) diff.push(AXES[5].fmt(c.activation));
    if (c.list && c.list !== BASE.list) diff.push(`list ${c.list}`);
    if (c.T && c.T !== BASE.T) diff.push(`T = ${c.T}`);
    if (c.lang2 && c.lang2 !== BASE.lang2) diff.push(`${langName(c.lang2)} at L = 2`);
    if (c.english !== BASE.english) diff.push(c.english);
    return diff.length ? diff.join(" + ") : "baseline";
  };
  const human = (n) => (n == null ? "–" : n >= 1e9 ? `${fmt(n / 1e9, n >= 1e10 ? 0 : 1)} B` : n >= 1e6 ? `${fmt(n / 1e6, n >= 1e7 ? 0 : 1)} M` : `${Math.round(n / 1e3)} k`);

  VIEWS.ladder = async (el) => {
    const d = await languages().then(() => load("ladder"));
    const ROWS = [...LS].reverse();   // L on the y axis: the largest setting on top
    const variants = Object.entries(d.cells.reduce((m, c) => ((m[variantOf(c)] = (m[variantOf(c)] ?? 0) + 1), m), {}))
      .sort((a, b) => (a[0] !== "baseline") - (b[0] !== "baseline") || b[1] - a[1]).map(([v]) => v);
    const color = (c) => series(12)[variants.indexOf(variantOf(c)) % 12];
    let selected = d.cells.find((c) => c.size === "1.7B" && c.L === 50 && variantOf(c) === "baseline" && c.seed === 1904) ?? d.cells[0];
    const trainedSet = (c) => new Set(c.L === 1 ? [] : d.language_sets[c.sets]?.[`FW_L${c.L}`] ?? []);

    const detail = (c) => {
      const trained = trainedSet(c);
      const bpb = Object.entries(d.final_bpb[c.name] ?? {}).map(([fw, v]) => ({ fw, v, trained: trained.has(fw) }));
      const langs = ["English", ...[...trained].map(fwName)];
      const fwHalf = c.L === 1 ? "" : ` + 50 % FineWeb-2 over ${c.L - 1} more language${c.L > 2 ? "s" : ""}`
        + `${c.list ? `, language list ${c.list}` : ""}, sampling T = ${c.T}${c.lang2 ? ` (second language: ${langName(c.lang2)})` : ""}`;
      return h("div", { class: "ladder-detail" },
        h("h4", {}, c.name),
        h("dl", {},
          h("dt", {}, "Size"), h("dd", {}, c.n_non_emb
            ? `${c.size} non-embedding (${(c.n_non_emb / 1e6).toFixed(0)} M; ${(c.params / 1e6).toFixed(0)} M with embeddings), d_model ${c.d_model}`
            : `${c.size} non-embedding`),
          h("dt", {}, "Model"), h("dd", {}, `${c.arch} · ${AXES[5].fmt(c.activation)} · ${AXES[6].fmt(c.optimizer)}`),
          h("dt", {}, "Data"), h("dd", {}, c.L === 1 ? `100 % English (${c.english})` : `50 % English (${c.english})${fwHalf}`),
          h("dt", {}, "Budget"), h("dd", {}, c.tokens
            ? `${(c.tokens / 1e9).toFixed(1)} B tokens (100 × N, 5× Chinchilla), ${c.n_ckpts} checkpoints, seed ${c.seed}`
            : `100 tokens per parameter (5× Chinchilla), seed ${c.seed}`),
          h("dt", {}, "Languages"), h("dd", {}, langs.length > 12 ? `${langs.slice(0, 12).join(", ")} … (+${langs.length - 12})` : langs.join(", ")),
          h("dt", {}, "Links"), h("dd", {}, h("em", {}, ANON)),
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

    const axes = AXES.map((a) => ({ ...a, levels: uniq(d.cells.map((c) => c[a.key]).filter((v) => v != null)).sort() }))
      .filter((a) => a.levels.length > 1);
    mount(el, [
      ...axes.map((a) => ({ key: a.key, label: a.label, string: a.key !== "T",
        options: [{ value: "all", label: "all" }, ...a.levels.map((v) => ({ value: v, label: a.fmt(v) }))] })),
      { key: "seeds", label: "Seeds", type: "seg", options: [{ value: "main", label: "1904" }, { value: "all", label: "all" }] },
    ], (s) => {
      const keep = d.cells.filter((c) => axes.every((a) => s[a.key] === "all" || c[a.key] === s[a.key])
        && (s.seeds === "all" || c.seed === 1904));
      const grid = h("div", { class: "ladder-grid", style: `grid-template-columns: 4em repeat(${SIZES.length}, 1fr)` });
      const detailBox = h("div");
      for (const L of ROWS) {
        grid.append(h("div", { class: "rowhd" }, `L = ${L}`));
        for (const size of SIZES) {
          const chips = keep.filter((c) => c.size === size && c.L === L).map((c) => {
            const b = h("button", {
              class: `ladder-chip ${c === selected ? "sel" : ""}`, style: `--chip:${color(c)}`,
              title: `${c.name} — ${variantOf(c)}`, "aria-label": c.name,
              onclick: () => { selected = c; grid.querySelectorAll(".sel").forEach((x) => x.classList.remove("sel")); b.classList.add("sel"); detailBox.replaceChildren(detail(c)); },
            });
            return b;
          });
          grid.append(h("div", { class: "cell" }, chips));
        }
      }
      grid.append(h("div", {}), ...SIZES.map((sz) => h("div", { class: "hd" }, sz)));
      detailBox.append(detail(selected));
      const shown = uniq(keep.map(variantOf));
      const legend = h("div", { class: "ladder-legend" },
        variants.filter((v) => shown.includes(v)).map((v) => {
          const c = series(12)[variants.indexOf(v) % 12];
          return h("span", {}, h("i", { style: `background:${c};border-color:${c}` }), v);
        }));
      return [tiles([[keep.length, "runs in this view"], [uniq(keep.map((c) => c.size)).length, "sizes"],
        [uniq(keep.map((c) => c.L)).length, "language settings"], [uniq(keep.map((c) => c.seed)).length, "seeds"]]),
        h("div", { style: "margin-top:1em" }, grid),
        h("p", { class: "viz-note" }, "Columns: model size (non-embedding parameters). Rows: number of training languages L. Each square is one run, coloured by what it changes with respect to the baseline. Picking a level in a dropdown hides the runs where that intervention does not apply (for example, the second language exists only at L = 2)."),
        legend, detailBox];
    });
  };

  VIEWS.bpb = async (el) => {
    const d = await languages().then(() => load("ladder"));
    const langs = uniq(Object.values(d.final_bpb).flatMap(Object.keys)).sort((a, b) => fwName(a).localeCompare(fwName(b)));
    mount(el, [
      { key: "lang", label: "Validation language", options: langs.map((l) => ({ value: l, label: fwName(l) })), value: "deu_Latn", string: true },
      { key: "arch", label: "Architecture", type: "seg", options: ["deep", "shallow"] },
    ], (s) => {
      const rows = d.cells.filter((c) => c.seed === 1904 && c.scheme === "A" && c.ladder === s.arch && d.final_bpb[c.name]?.[s.lang] != null)
        .map((c) => {
          const sets = d.language_sets.A[`FW_L${c.L}`] ?? [];
          return { ...c, bpb: d.final_bpb[c.name][s.lang], trained: sets.includes(s.lang) || s.lang === "eng_Latn", Lname: `L${c.L}` };
        });
      if (!rows.length) return h("div", { class: "viz-empty" }, "No run of this architecture was scored on this language.");
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

  // ---------- evaluation: score curves + the benchmark card below them ----------
  const sci = (x) => {
    const e = Math.floor(Math.log10(x)), m = x / 10 ** e;
    return `${m.toFixed(m < 10 ? 1 : 0)}e${e}`;
  };
  const tokensLabel = (x) => (x >= 1e9 ? `${fmt(x / 1e9, x >= 1e10 ? 0 : 1)}B` : `${fmt(x / 1e6, 0)}M`);

  function example(ex, title) {
    if (!ex) return h("div", { class: "viz-example" }, h("b", {}, title), h("p", { class: "viz-note" }, "No item of this benchmark is available in this language."));
    return h("div", { class: "viz-example" },
      h("b", {}, title),
      ex.prompt != null ? h("pre", { class: "viz-prompt" }, ex.prompt) : null,
      h("ol", { start: 1 }, ex.options.map((o, i) => h("li", { class: i === ex.gold ? "gold" : null }, o.trim(), i === ex.gold ? h("span", { class: "viz-gold" }, " correct") : null))),
      h("p", { class: "viz-note" },
        ex.prompt != null ? "The model reads the prompt and we compare how likely it finds each option as the continuation." : "Each option is a full sentence; we compare how likely the model finds each one.",
        ex.gold == null ? " The correct option is not marked for this item." : "",
        ` Task: ${ex.task}.`));
  }

  function benchmarkCard(b, lang, twin) {
    const src = (ex) => (ex?.anonymized ? ANON : ex?.source ?? "the benchmark's own release (no public dataset id recorded)");
    const names = b.languages.map((l) => (LANG?.iso[l] ? langName(l) : l));
    const own = lang !== "all" ? b.examples[lang] : null;
    return h("div", { class: "viz-card" },
      h("h4", {}, b.name, b.venue ? h("span", { class: "viz-muted" }, ` · ${b.venue}`) : null),
      b.format ? h("p", {}, b.format) : null,
      h("dl", { class: "viz-facts" },
        h("dt", {}, "Answer options"), h("dd", {}, b.n_options == null ? "– (no fixed options)" : `${b.n_options}${b.options_vary ? " (typical; it varies by item)" : ""}`),
        h("dt", {}, "Languages"), h("dd", {}, `${b.languages.length}: ${names.join(", ")}`),
        h("dt", {}, "Data source"), h("dd", {}, src(own ?? b.examples.en ?? Object.values(b.examples)[0]),
          b.paper ? [" · ", h("a", { href: b.paper, target: "_blank", rel: "noopener" }, "paper")] : null)),
      h("div", { class: "viz-examples" },
        lang !== "all" && lang !== "en" ? example(own, `Example in ${langName(lang)}`) : null,
        example(b.examples.en ?? (lang !== "all" ? twin?.examples[lang] : null), "Example in English")));
  }

  VIEWS.scoreCurves = async (el) => {
    await languages();
    const d = await load("evaluation");
    const fams = Object.keys(d.benchmarks).sort();
    const ramp = sizeRamp();
    mount(el, [
      { key: "lang", label: "Language", string: true, value: "all",
        options: d.languages.map((l) => ({ value: l, label: l === "all" ? "All trained languages" : `${langName(l)} (${l})` })) },
      { key: "fam", label: "Benchmark", string: true, value: fams.includes("belebele") ? "belebele" : fams[0],
        options: fams.map((f) => ({ value: f, label: `${d.benchmarks[f].name} [${f}]` })) },
      { key: "x", label: "x axis", type: "seg", options: [{ value: "tokens", label: "training tokens" }, { value: "compute", label: "compute" }] },
    ], (s) => {
      const c = d.curves[s.lang]?.[s.fam];
      const b = d.benchmarks[s.fam];
      const card = benchmarkCard(b, s.lang, b.english_twin && d.benchmarks[b.english_twin]);
      if (!c) {
        const have = Object.keys(d.curves[s.lang] ?? {}).map((f) => d.benchmarks[f]?.name ?? f);
        return [h("div", { class: "viz-empty" }, `No trained run is scored on this benchmark in ${langName(s.lang)}. `,
          `Benchmarks with ${langName(s.lang)} items: ${uniq(have).join(", ")}.`), card];
      }
      const sizes = Object.keys(d.sizes).filter((z) => c.sizes[z]);
      const pts = sizes.flatMap((z) => c.sizes[z].map((y, i) => {
        if (y == null) return null;
        const tokens = d.grid[i] / 5 * d.sizes[z].tokens;
        return { size: z, y, mult: d.grid[i], tokens, compute: 6 * d.sizes[z].params * tokens };
      }).filter(Boolean));
      const color = d3.quantize(d3.interpolateRgb(ramp[0], ramp[ramp.length - 1]), Math.max(sizes.length, 2));
      const xLabel = s.x === "tokens" ? "training tokens (log)" : "training compute, FLOPs (log)";
      return [plot({
        height: 360, marginBottom: 40,
        x: { type: "log", label: xLabel, tickFormat: s.x === "tokens" ? tokensLabel : sci, ticks: 6 },
        y: { label: "score", grid: true },
        color: { domain: sizes, range: color, legend: true, label: "model size" },
        marks: [
          c.chance != null ? Plot.ruleY([c.chance], { stroke: css("--viz-bad"), strokeDasharray: "4,3" }) : null,
          c.chance != null ? Plot.text([c.chance], { y: (v) => v, frameAnchor: "right", dy: -6, text: () => "chance", fill: css("--viz-bad"), fontSize: 10 }) : null,
          Plot.line(pts, { x: s.x, y: "y", stroke: "size", z: "size", strokeWidth: 2 }),
          Plot.dot(pts, { x: s.x, y: "y", fill: "size", r: 2.5 }),
          Plot.tip(pts, Plot.pointer({ x: s.x, y: "y",
            title: (p) => `${p.size}, ${fmt(p.mult, 2)}× Chinchilla\n${tokensLabel(p.tokens)} tokens · ${sci(p.compute)} FLOPs\nscore ${fmt(p.y, 3)}` })),
        ].filter(Boolean),
      }),
      note(s.lang === "all"
        ? "Each line is one model size: per run, the mean over the tasks in the languages that run trained on, then the mean over the runs of that size. The dashed line is chance."
        : `Each line is one model size: the mean over the runs whose training data include ${langName(s.lang)}. The dashed line is chance, the score of guessing.`
          + " The last point of each line is the end of the run (5× Chinchilla tokens: 100 tokens per parameter)."
          + (s.x === "compute" ? " Compute is 6 × parameters × tokens, with the parameters of the baseline (deep) model." : "")),
      card];
    });
  };

  // ---------- data mixtures: treemap + bars, colour = family, texture = script ----------
  // Pattern library for the script textures; Latin (the most common script)
  // stays plain. Each entry draws inside an 8 x 8 tile.
  const TEXTURES = [
    (g) => g.append("path").attr("d", "M-2,2 l4,-4 M0,8 l8,-8 M6,10 l4,-4"),               // diagonal /
    (g) => g.append("circle").attr("cx", 4).attr("cy", 4).attr("r", 1.4).attr("stroke", "none").attr("class", "dot"),
    (g) => g.append("path").attr("d", "M0,4 h8"),                                            // horizontal
    (g) => g.append("path").attr("d", "M4,0 v8"),                                            // vertical
    (g) => g.append("path").attr("d", "M-2,6 l4,4 M0,0 l8,8 M6,-2 l4,4"),                   // diagonal \
    (g) => g.append("path").attr("d", "M0,0 l8,8 M8,0 l-8,8"),                               // cross-hatch
    (g) => g.append("path").attr("d", "M0,4 h8 M4,0 v8"),                                    // grid
    (g) => g.append("path").attr("d", "M0,6 l2,-4 l2,4 l2,-4 l2,4"),                         // zigzag
    (g) => g.append("circle").attr("cx", 4).attr("cy", 4).attr("r", 2.4),                    // rings
    (g) => g.append("rect").attr("x", 0).attr("y", 0).attr("width", 4).attr("height", 4).attr("stroke", "none").attr("class", "dot"),  // checker
    (g) => g.append("path").attr("d", "M1,2 h3 M5,6 h3"),                                    // dashes
    (g) => g.append("path").attr("d", "M0,2 h8 M0,6 h8"),                                    // dense horizontal
    (g) => g.append("path").attr("d", "M2,0 v8 M6,0 v8"),                                    // dense vertical
    (g) => g.append("circle").attr("cx", 2).attr("cy", 2).attr("r", .9).attr("stroke", "none").attr("class", "dot"),
  ];
  let patternId = 0;

  VIEWS.mixtures = async (el) => {
    const [d, mx] = await Promise.all([languages().then(() => load("ladder")), load("mixtures")]);
    const info = (fw) => {
      const code = d.fineweb_iso2[fw.split("_")[0]] ?? fw;
      const m = d.languages[code] ?? {};
      return { fw, code, name: m.language ?? fw, family: m.family ?? "unknown", top: (m.family ?? "unknown").split(",")[0].trim(),
        script: m.script ?? fw.split("_")[1], hello: m.hello, thanks: m.thank_you, speakers: m.speakers };
    };
    // Every language any mixture trains on, identified by its FineWeb-2 subset.
    const all = uniq(mx.mixtures.flatMap((m) => Object.keys(m.shares))).map(info);
    const famOrder = Object.entries(all.reduce((m, l) => ((m[l.top] = (m[l.top] ?? 0) + 1), m), {})).sort((a, b) => b[1] - a[1]).map(([f]) => f);
    const scriptOrder = Object.entries(all.reduce((m, l) => ((m[l.script] = (m[l.script] ?? 0) + 1), m), {})).sort((a, b) => b[1] - a[1]).map(([f]) => f);
    const famColor = (f) => series(12)[famOrder.indexOf(f) % 12];
    const SCHEME_LABEL = { A: "A · ranked by data size", B: "B · chosen for diversity", ZH: "A with Chinese at L = 2", ES: "A with Spanish at L = 2" };
    const schemes = uniq(mx.mixtures.map((m) => m.scheme));
    const tip = h("div", { class: "viz-tip", hidden: "" });

    const defs = (svg, id) => {
      const stroke = dark() ? "rgba(255,255,255,.55)" : "rgba(0,0,0,.42)";
      const defsEl = svg.append("defs");
      scriptOrder.forEach((s, i) => {
        if (i === 0) return;   // the most common script (Latin) stays plain
        const p = defsEl.append("pattern").attr("id", `${id}-${i}`).attr("width", 8).attr("height", 8).attr("patternUnits", "userSpaceOnUse");
        const g = p.append("g").attr("fill", "none").attr("stroke", stroke).attr("stroke-width", 1.2);
        TEXTURES[(i - 1) % TEXTURES.length](g);
        g.selectAll(".dot").attr("fill", stroke);
      });
      return (script) => { const i = scriptOrder.indexOf(script); return i > 0 ? `url(#${id}-${i})` : "none"; };
    };
    const showTip = (ev, r) => {
      const box = el.getBoundingClientRect();
      tip.replaceChildren(
        h("b", { class: "hello" }, r.hello ? `“${r.hello}”` : r.name),
        h("span", {}, `${r.name} (${r.code})`),
        h("span", {}, `${human(r.tokens)} training tokens (${fmt(100 * r.share, r.share < 0.01 ? 2 : 1)} %)`),
        h("span", {}, `Family: ${r.family}`),
        h("span", {}, `Script: ${r.script}`),
        h("span", {}, `Speakers: ${r.speakers ? `≈ ${human(r.speakers)}` : "–"}`),
        h("span", {}, r.thanks ? `“${r.thanks}”` : ""));
      tip.hidden = false;
      const x = ev.clientX - box.left + el.scrollLeft + 14, y = ev.clientY - box.top + 14;
      tip.style.left = `${Math.min(x, el.scrollWidth - 230)}px`;
      tip.style.top = `${y}px`;
    };
    const hideTip = () => { tip.hidden = true; };
    const hover = (sel) => sel.on("mousemove", (ev, r) => showTip(ev, r)).on("mouseleave", hideTip);

    const treemap = (rows) => {
      const W = 720, H = 380, id = `mx${patternId++}`;
      const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("role", "img")
        .attr("aria-label", "Treemap of training tokens per language");
      const tex = defs(svg, id);
      const root = d3.treemap().size([W, H]).paddingInner(1.5).round(true)(
        d3.hierarchy({ children: rows }).sum((r) => r.share ?? 0).sort((a, b) => b.value - a.value));
      const g = svg.append("g").selectAll("g").data(root.leaves()).join("g").attr("transform", (n) => `translate(${n.x0},${n.y0})`);
      const w = (n) => n.x1 - n.x0, hh = (n) => n.y1 - n.y0;
      g.append("rect").attr("width", w).attr("height", hh).attr("rx", 2).attr("fill", (n) => famColor(n.data.top));
      g.append("rect").attr("width", w).attr("height", hh).attr("rx", 2).attr("fill", (n) => tex(n.data.script));
      g.filter((n) => w(n) > 26 && hh(n) > 14).append("text").attr("x", 4).attr("y", 13)
        .attr("fill", "white").attr("font-size", 11).attr("font-weight", 600).attr("paint-order", "stroke")
        .attr("stroke", "rgba(0,0,0,.35)").attr("stroke-width", 2).text((n) => n.data.code);
      hover(g.datum((n) => n.data));
      return svg.node();
    };

    const bars = (rows, logScale) => {
      const W = 720, H = 300, m = { t: 10, r: 10, b: 60, l: 56 }, id = `mx${patternId++}`;
      const svg = d3.create("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("role", "img")
        .attr("aria-label", "Training tokens per language");
      const tex = defs(svg, id);
      const x = d3.scaleBand(rows.map((r) => r.code), [m.l, W - m.r]).padding(0.15);
      const pos = rows.filter((r) => r.tokens > 0);
      const y = logScale
        ? d3.scaleLog([d3.min(pos, (r) => r.tokens) / 2, d3.max(pos, (r) => r.tokens)], [H - m.b, m.t]).nice()
        : d3.scaleLinear([0, d3.max(pos, (r) => r.tokens)], [H - m.b, m.t]).nice();
      const fg = css("--md-default-fg-color"), muted = css("--viz-muted");
      const ticks = logScale ? y.ticks().filter((v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9) : y.ticks(5);
      const yAxis = svg.append("g").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).tickValues(ticks).tickFormat((v) => human(v)));
      yAxis.selectAll("text").attr("fill", fg);
      yAxis.selectAll("line,path").attr("stroke", muted);
      svg.append("g").attr("stroke", css("--viz-grid")).selectAll("line").data(ticks).join("line")
        .attr("x1", m.l).attr("x2", W - m.r).attr("y1", y).attr("y2", y);
      const xAxis = svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickSize(0));
      xAxis.selectAll("text").attr("fill", fg).attr("transform", "rotate(-90)").attr("text-anchor", "end").attr("dx", "-.6em").attr("dy", "-.55em");
      xAxis.select("path").attr("stroke", muted);
      svg.append("text").attr("x", 12).attr("y", m.t + 2).attr("fill", muted).attr("font-size", 11)
        .attr("transform", `rotate(-90 12 ${m.t + 2})`).attr("text-anchor", "end").text("training tokens");
      const g = svg.append("g").selectAll("g").data(pos).join("g");
      const base = H - m.b;
      g.append("rect").attr("x", (r) => x(r.code)).attr("width", x.bandwidth()).attr("y", (r) => y(r.tokens))
        .attr("height", (r) => base - y(r.tokens)).attr("fill", (r) => famColor(r.top));
      g.append("rect").attr("x", (r) => x(r.code)).attr("width", x.bandwidth()).attr("y", (r) => y(r.tokens))
        .attr("height", (r) => base - y(r.tokens)).attr("fill", (r) => tex(r.script));
      // a wide invisible hit area, so short bars are easy to hover
      g.append("rect").attr("x", (r) => x(r.code)).attr("width", x.bandwidth()).attr("y", m.t)
        .attr("height", base - m.t).attr("fill", "transparent");
      hover(g);
      return svg.node();
    };

    const legend = (rows) => {
      const fams = famOrder.filter((f) => rows.some((r) => r.top === f));
      const scripts = scriptOrder.filter((s) => rows.some((r) => r.script === s));
      const swatch = (script) => {
        const svg = d3.create("svg").attr("width", 14).attr("height", 14).attr("viewBox", "0 0 14 14");
        const tex = defs(svg, `mx${patternId++}`);
        svg.append("rect").attr("width", 14).attr("height", 14).attr("rx", 2).attr("fill", css("--viz-grid"));
        svg.append("rect").attr("width", 14).attr("height", 14).attr("rx", 2).attr("fill", tex(script));
        return svg.node();
      };
      return h("div", { class: "mix-legend" },
        h("div", {}, h("b", {}, "Family (colour)"), fams.map((f) => h("span", {}, h("i", { style: `background:${famColor(f)}` }), f))),
        h("div", {}, h("b", {}, "Script (texture)"), scripts.map((s) => h("span", {}, swatch(s), s))));
    };

    mount(el, [
      { key: "scheme", label: "Scheme", string: true, value: "A", options: schemes.map((s) => ({ value: s, label: SCHEME_LABEL[s] ?? s })) },
      { key: "L", label: "Languages (L)", value: 50, options: uniq(mx.mixtures.map((m) => m.L)).sort((a, b) => a - b).map((L) => ({ value: L, label: `L = ${L}` })) },
      { key: "T", label: "Temperature", value: 3, options: uniq(mx.mixtures.map((m) => m.T)).sort().map((T) => ({ value: T, label: `T = ${T}` })) },
      { key: "scale", label: "Bar scale", type: "seg", options: ["linear", "log"] },
    ], (s) => {
      const mix = mx.mixtures.find((m) => m.scheme === s.scheme && m.L === s.L && m.T === s.T);
      if (!mix) {
        const have = mx.mixtures.filter((m) => m.scheme === s.scheme).map((m) => `L = ${m.L}, T = ${m.T}`);
        return [h("div", { class: "viz-empty" }, `Scheme ${s.scheme} has no mixture at L = ${s.L} with T = ${s.T}. It exists at: ${have.join(" · ")}.`), tip];
      }
      const rows = Object.entries(mix.shares).map(([fw, share]) => ({ ...info(fw), share, tokens: share * mix.ref_tokens }))
        .sort((a, b) => b.share - a.share);
      // The bar plot keeps a slot for every language any mixture trains on, so mixtures line up.
      const barRows = [...rows, ...all.filter((l) => !mix.shares[l.fw]).map((l) => ({ ...l, share: 0, tokens: 0 }))
        .sort((a, b) => a.name.localeCompare(b.name))];
      return [
        tiles([[rows.length, "languages"], [human(mix.ref_tokens), `training tokens of the ${mix.ref_size} reference run`],
          [`${fmt(100 * (mix.shares.eng_Latn ?? 0), 0)} %`, "English"], [`T = ${mix.T}`, "sampling temperature"]]),
        h("div", { style: "margin-top:1em" }, treemap(rows)),
        bars(barRows, s.scale === "log"),
        legend(rows),
        note(`Area and bar height: each language's training tokens in a run of the ${mix.ref_size} reference model (${human(mix.ref_tokens)} tokens; every size uses the same shares, scaled to its own budget). English is fixed at ${fmt(100 * (mix.shares.eng_Latn ?? 0), 0)} %; the rest is split over the other languages as the data builder's plan splits it. Hover a language for its details.`),
        tip,
      ];
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
        color: { domain: LADDER, range: sizeColors(LADDER), legend: true, label: "proxy size" },
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
    mount(el, [{ key: "size", label: "Model size", type: "seg", options: LADDER.filter((p) => rows.some((r) => r.size === p)), value: "1B" }], (s) => {
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
        x: { domain: LADDER.filter((p) => rows.some((r) => r.proxy === p)), label: "proxy size", axis: "top" },
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
        x: { domain: LADDER.slice(0, 5), label: "proxy size" },
        y: { domain: [0, 1], label: "decision accuracy vs the reference", grid: true },
        color: { domain: labels, range: series(labels.length), legend: true },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.line(rows, { x: "proxy_size", y: "decision_acc", stroke: "label", strokeWidth: 2, sort: (r) => LADDER.indexOf(r.proxy_size) }),
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
        color: { domain: LADDER, range: sizeColors(LADDER), legend: true, label: "proxy size" },
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

  /* Recommender look-up table: your proxy size, mixture and token budget give
   * the tokens of each language the proxy sees; lookup.json says from how many
   * tokens of the language each (evaluation, language) stays above chance. */
  VIEWS.lookup = async (el) => {
    await languages();
    const [lk, mx] = await Promise.all([load("lookup"), load("mixtures")]);
    const by = byKey(expand(lk.table), (r) => `${r.benchmark}|${r.language}|${r.size}`);
    const measured = uniq(expand(lk.table).map((r) => r.language)).sort((a, b) => langName(a).localeCompare(langName(b)));
    const iso = (fw) => LANG.fw[fw.split("_")[0]] ?? fw;
    // A preset's shares are keyed by FineWeb subset; the table by language (Arabic varieties add up).
    const preset = (m) => Object.entries(m.shares).reduce((o, [fw, s]) => ((o[iso(fw)] = (o[iso(fw)] ?? 0) + 100 * s), o), {});
    const presets = mx.mixtures.map((m, i) => ({ value: i, label: `Scheme ${m.scheme} · L = ${m.L}${m.T !== 1 ? ` · T = ${m.T}` : ""}` }));
    const start = Math.max(0, mx.mixtures.findIndex((m) => m.scheme === "A" && m.L === 8 && m.T === 1));
    const size0 = lk.sizes.includes("350M") ? "350M" : lk.sizes[0];
    const s = { size: size0, budget: lk.budget[size0] / 1e9, ownBudget: false, shares: preset(mx.mixtures[start]), only: false };

    const out = h("div"), mixBox = h("div", { class: "lk-mix" });
    const budgetIn = h("input", { type: "number", min: 0.1, step: 0.1, value: fmt(s.budget, 1),
      oninput: (e) => { s.budget = +e.target.value; s.ownBudget = true; render(); } });
    const bar = h("div", { class: "viz-controls" },
      control({ key: "size", label: "Proxy size", type: "seg", options: lk.sizes, value: s.size }, (v) => {
        s.size = v;
        if (!s.ownBudget) { s.budget = lk.budget[v] / 1e9; budgetIn.value = fmt(s.budget, 1); }
        render();
      }),
      h("label", {}, "Token budget (billions)", budgetIn),
      control({ key: "preset", label: "Start from our mixture", options: presets, value: start }, (i) => {
        s.shares = preset(mx.mixtures[i]); editor(); render();
      }));

    function editor() {
      const total = Object.values(s.shares).reduce((a, b) => a + b, 0);
      const add = h("select", { onchange: (e) => { if (e.target.value) { s.shares[e.target.value] = 1; editor(); render(); } } },
        h("option", { value: "" }, "+ add a language"),
        measured.filter((l) => !(l in s.shares)).map((l) => h("option", { value: l }, `${langName(l)} (${l})`)));
      mixBox.replaceChildren(
        h("div", { class: "lk-mix-rows" }, Object.entries(s.shares).sort((a, b) => b[1] - a[1]).map(([l, v]) =>
          h("label", {}, `${langName(l)} (${l})`, h("span", {},
            h("input", { type: "number", min: 0, step: 0.1, value: +v.toFixed(2), "aria-label": `share of ${langName(l)} in %`,
              oninput: (e) => { s.shares[l] = Math.max(0, +e.target.value || 0); render(); } }), " %",
            h("button", { type: "button", title: "remove", "aria-label": `remove ${langName(l)}`,
              onclick: () => { delete s.shares[l]; editor(); render(); } }, "×"))))),
        h("div", { class: "viz-controls" }, h("label", {}, "Mixture", add)),
        note(`Shares add up to ${fmt(total, 1)} %; they are rescaled to 100 % of the budget.`));
    }

    function cell(b, l, tokens) {
      const r = by[`${b}|${l}|${s.size}`];
      if (!r) return { k: "none" };
      const need = r.min_tokens;
      if (need != null && tokens >= need) return { k: "use", r, text: `✓ ${need === r.lo ? "≤ " : ""}${pct(need / tokens)}`, why: `above chance from ${human(need)} tokens of ${langName(l)}; your run reaches that at ${pct(need / tokens)} of training${need === r.lo ? " (or earlier: it is above chance from the first checkpoint we measured)" : ""}` };
      if (need == null && tokens > r.hi) return { k: "care", r, text: "? beyond", why: `at chance in every ${s.size} run we have, up to ${human(r.hi)} tokens of ${langName(l)}; your ${human(tokens)} is beyond what we measured` };
      if (need === r.lo && tokens < r.lo) return { k: "care", r, text: "? below", why: `above chance from the fewest tokens we measured (${human(r.lo)}); your ${human(tokens)} is below that` };
      if (need == null) return { k: "avoid", r, text: "✕ chance", why: `at chance in every ${s.size} run we have, up to ${human(r.hi)} tokens of ${langName(l)}` };
      return { k: "avoid", r, text: `✕ ${human(need)}`, why: `needs ${human(need)} tokens of ${langName(l)} to stay above chance; your run sees ${human(tokens)}` };
    }

    function render() {
      const total = Object.values(s.shares).reduce((a, b) => a + b, 0) || 1;
      const langs = Object.entries(s.shares).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
        .map(([l, v]) => ({ l, tokens: (v / total) * s.budget * 1e9 }));
      const rowsAll = lk.benchmarks.map((b) => ({ b, cells: langs.map(({ l, tokens }) => cell(b, l, tokens)) }))
        .filter((r) => r.cells.some((c) => c.k !== "none"));
      const shown = s.only ? rowsAll.filter((r) => r.cells.some((c) => c.k === "use")) : rowsAll;
      const flat = rowsAll.flatMap((r) => r.cells).filter((c) => c.k !== "none");
      const nUse = flat.filter((c) => c.k === "use").length;
      const langUse = langs.filter((_, i) => rowsAll.some((r) => r.cells[i].k === "use")).length;
      if (!langs.length) return out.replaceChildren(h("div", { class: "viz-empty" }, "Add at least one language to the mixture."));
      const head = h("tr", {}, h("th", {}, "Evaluation"), langs.map(({ l, tokens }) =>
        h("th", { title: `${langName(l)}: ${human(tokens)} training tokens` }, l, h("div", { class: "viz-note" }, human(tokens)))));
      const body = shown.map((r) => h("tr", {}, h("td", {}, lk.names[r.b] ?? r.b), r.cells.map((c) =>
        h("td", { title: c.why ?? "not measured for this language" }, c.k === "none" ? "–" : h("span", { class: `badge ${c.k}` }, c.text)))));
      out.replaceChildren(
        tiles([[`${nUse} / ${flat.length}`, `(evaluation, language) pairs above chance at ${s.size}`],
               [`${langUse} / ${langs.length}`, "languages with at least one usable evaluation"],
               [`${fmt(s.budget, 1)} B`, `training tokens${s.ownBudget ? "" : ` (our ${s.size} budget, 100 × N)`}`]]),
        h("div", { class: "viz-controls", style: "margin-top:1em" },
          control({ key: "only", label: "Show", type: "seg", options: [{ value: false, label: "all evaluations" }, { value: true, label: "usable only" }], value: s.only },
            (v) => { s.only = v; render(); })),
        h("div", { class: "viz-scroll" }, h("table", { class: "viz-table lk-table" }, h("thead", {}, head), h("tbody", {}, body))),
        note(`✓ x % = above chance from x % of your run on (≤ x %: already at our first measured checkpoint); x % is the tokens of the language it needs over the tokens your run gives it. ` +
             `✕ n = needs n tokens of the language first. ✕ chance = at chance in all our ${s.size} runs. ` +
             `? = outside the token range we measured, so we cannot say. – = not measured. ` +
             `Bits per byte has no chance level and is usable at every size.`));
    }
    el.replaceChildren(bar, mixBox, out);
    editor();
    REDRAW.push(render);
    render();
  };

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

  // ---------- RQ0: above chance ----------
  // The gate's lower bound (above_random.wilson_lcb): one-sided 95 % Wilson
  // bound of an accuracy over n items, the accuracy turned back into a count.
  const Z = 1.6448536269514722;   // the one-sided 95 % normal quantile (alpha = 0.10, two-sided)
  const wilson = (score, n) => {
    const p = Math.round(score * n) / n, z2 = Z * Z;
    return (p + z2 / (2 * n) - Z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  };
  const sizeColors = (sizes) => {
    const r = sizeRamp();
    return d3.quantize(d3.interpolateRgb(r[0], r[r.length - 1]), Math.max(sizes.length, 2));
  };
  const isTwin = (b) => /^rf(gm)?_/.test(b);
  const benchName = (names, b) => names?.[b] ?? b;

  /* A view whose data may not be in this build yet (an analysis not merged). */
  async function loadOptional(el, name) {
    try { return await load(name); } catch (e) {
      delete cache[name];
      el.replaceChildren(h("div", { class: "viz-empty" }, "The results of this analysis are not in this build yet; they appear here once it lands."));
      return null;
    }
  }

  VIEWS.firstSize = async (el) => {
    const g = await languages().then(() => load("gate"));
    const fs = g.first_size;
    const levels = [...fs.levels, "never"];
    mount(el, [
      { key: "twins", label: "Versions", type: "seg", options: [{ value: "orig", label: "as published" }, { value: "all", label: "with rewritten versions" }] },
    ], (s) => {
      const benches = fs.benchmarks.filter((b) => s.twins === "all" || !isTwin(b));
      const cells = fs.cells.filter(([b]) => benches.includes(b)).map(([b, l, level]) => ({ b, l, level }));
      const colors = [...sizeColors(fs.levels), css("--viz-bad")];
      return [plot({
        width: Math.max(720, 14 * fs.languages.length + 190), height: 15 * benches.length + 70,
        marginLeft: 180, marginBottom: 10, padding: 0.06,
        x: { domain: fs.languages, axis: "top", label: null, tickRotate: -90, tickSize: 0 },
        y: { domain: benches, label: null, tickFormat: (b) => benchName(g.names, b), tickSize: 0 },
        color: { domain: levels, range: colors, legend: true, label: "smallest model size above chance" },
        marks: [
          Plot.cell(cells, { x: "l", y: "b", fill: "level", inset: 0.5, rx: 2 }),
          Plot.tip(cells, Plot.pointer({ x: "l", y: "b",
            title: (c) => `${benchName(g.names, c.b)} · ${langName(c.l)} (${c.l})\n` +
              (c.level === "never" ? "never above chance, up to 1.7B" : `above chance from ${c.level} on`) })),
        ],
      }), note("Each square is one benchmark in one language. Its colour is the smallest model size from which at least half of the runs that trained on the language score above chance, and every larger size does too. Red: no size up to 1.7B gets there. White: the benchmark has no items in that language. Languages run from English, then from the most to the least training data; benchmarks from the most to the fewest languages above chance.")];
    });
  };

  VIEWS.passTokens = async (el) => {
    const g = await load("gate");
    const pts = g.pass_tokens;
    const benches = ["all", ...uniq(pts.map((p) => p.benchmark)).filter((b) => b !== "all")
      .sort((a, b) => benchName(g.names, a).localeCompare(benchName(g.names, b)))];
    mount(el, [
      { key: "b", label: "Benchmark", string: true, value: "all",
        options: benches.map((b) => ({ value: b, label: b === "all" ? "All benchmarks" : benchName(g.names, b) })) },
    ], (s) => {
      const rows = pts.filter((p) => p.benchmark === s.b);
      const sizes = SIZES.filter((z) => rows.some((r) => r.model_size === z));
      return [plot({
        height: 360, marginBottom: 40,
        x: { type: "log", label: "tokens of the language seen in training (log)", tickFormat: tokensLabel, ticks: 6 },
        y: { domain: [0, 1], label: "share of tasks above chance", tickFormat: "%", grid: true },
        color: { domain: sizes, range: sizeColors(sizes), legend: true, label: "model size" },
        r: { range: [2, 7] },
        marks: [
          Plot.line(rows, { x: "tokens", y: "share_above", stroke: "model_size", z: "model_size", strokeWidth: 1.6, sort: "tokens" }),
          Plot.dot(rows, { x: "tokens", y: "share_above", fill: "model_size", r: "n_cells" }),
          Plot.tip(rows, Plot.pointer({ x: "tokens", y: "share_above",
            title: (r) => `${r.model_size}, ${tokensLabel(r.bin_lo)}–${tokensLabel(r.bin_hi)} tokens of the language\n${pct(r.share_above)} of ${r.n_cells} (task, run) cells above chance` })),
        ],
      }), note("Every checkpoint of every run counts once per task: its x is how many tokens of the task's language the run had seen at that point (the language's share of the mixture times the tokens trained so far). Points pool the cells of a size in bins of the x axis; their size is the number of cells. Read along a line: the same model size clears chance on more tasks once it has seen more of the language.")];
    });
  };

  VIEWS.formatExample = async (el) => {
    const g = await load("gate");
    const f = g.format_example;
    if (!f.og) return el.replaceChildren(h("div", { class: "viz-empty" }, "The example items are not in this build."));
    const letters = "ABCDEFGH";
    const question = f.og.prompt.replace(/\s*Answer:\s*$/, "");
    const asLetters = {
      task: f.og.task, gold: f.og.gold, options: f.og.options.map((_, i) => ` ${letters[i]}`),
      prompt: `${question}\n${f.og.options.map((o, i) => `${letters[i]}. ${o.trim()}`).join("\n")}\nAnswer:`,
    };
    el.replaceChildren(
      h("div", { class: "viz-examples" },
        example(asLetters, "Letter format: the model must output the letter"),
        example(f.og, "Answer-text format: the model scores each answer")),
      f.en ? h("details", { class: "viz-details" }, h("summary", {}, "The same question in English (the benchmark's own translation)"),
        h("div", { class: "viz-examples" }, example(f.en, "English"))) : null,
      note("Both formats ask the same INCLUDE question from a Greek exam with the same four answers. In the letter format the model only compares the four letters, so it has to have learned the A/B/C/D convention first; in the answer-text format it compares the four answers themselves."));
  };

  VIEWS.llmRewrite = async (el) => {
    const g = await load("gate");
    const f = g.format_example;
    if (!f.letter || !f.llm) return el.replaceChildren(h("div", { class: "viz-empty" }, "The example items are not in this build."));
    el.replaceChildren(h("div", { class: "viz-examples viz-examples-3" },
      example(f.letter, "As published (letter format)"),
      f.rf ? example(f.rf, "Reformatted (RF): the answer text") : null,
      example(f.llm, "Rewritten by an LLM (LLM-RF)")),
    note("One item of the INCLUDE benchmark in Greek, in the three versions we evaluate. RF keeps the question and scores the answer texts. LLM-RF asks a language model to rewrite the question so that each answer reads as its natural continuation."));
  };

  VIEWS.reformulation = async (el) => {
    const g = await load("gate");
    const rows = g.reformulation;
    const fams = uniq(rows.map((r) => r.family)).sort((a, b) =>
      d3.max(rows.filter((r) => r.family === b), (r) => r.languages) - d3.max(rows.filter((r) => r.family === a), (r) => r.languages) || a.localeCompare(b));
    const rf = rows.filter((r) => r.set === "rf"), llm = rows.filter((r) => r.set === "rfgm");
    const sizes = SIZES.filter((z) => rows.some((r) => r.size === z));
    const tip = (r) => `${benchName(g.names, r.family)} at ${r.size}, ${r.languages} languages\n` +
      `${r.set === "rf" ? "RF" : "LLM-RF"} version: ${pct(r.share_twin)} above the threshold\nas published: ${pct(r.share_original)}\n` +
      `McNemar p = ${fmt(r.p_mcnemar, 3)}${r.p_mcnemar < 0.05 ? " (significant)" : ""}`;
    mount(el, [], () => [plot({
      height: 340, marginBottom: 60, marginLeft: 45, width: 760,
      fx: { domain: fams, label: null, tickRotate: -20, axis: "bottom",
            tickFormat: (f) => `${benchName(g.names, f)} (${d3.max(rf.filter((r) => r.family === f), (r) => r.languages)})` },
      x: { domain: sizes, axis: null, paddingInner: 0.15 },
      y: { domain: [0, 1.08], label: "share of languages above chance", tickFormat: "%", grid: true },
      color: { domain: sizes, range: sizeColors(sizes), legend: true, label: "RF version at model size" },
      marks: [
        Plot.barY(rf, { fx: "family", x: "size", y: "share_twin", fill: "size" }),
        Plot.tickY(rf, { fx: "family", x: "size", y: "share_original", stroke: css("--viz-muted"), strokeWidth: 2.5 }),
        Plot.dot(llm, { fx: "family", x: "size", y: "share_twin", symbol: "diamond", fill: css("--viz-s2"), r: 4.5 }),
        Plot.text(rf.filter((r) => r.p_mcnemar < 0.05), { fx: "family", x: "size", y: "share_twin", text: () => "*", dy: -6, fontSize: 13 }),
        Plot.tip([...rf, ...llm], Plot.pointer({ fx: "family", x: "size", y: "share_twin", title: tip })),
      ],
    }), note("Bars: the share of a benchmark's languages above chance once the answers are scored as text (RF), one bar per model size. Grey tick: the same share for the benchmark as published. Orange diamond: the LLM-rewritten version (LLM-RF). Star: the change from the published version is significant (McNemar test over the languages, p < 0.05).")]);
  };

  // The bins of above_random_external.py, smallest first.
  const FLOOR_BINS = ["≤ 600M", "1B–1.7B", "3B–4B", "7B–14B", "≥ 27B", "never"];

  VIEWS.floors = async (el) => {
    const g = await languages().then(() => load("gate"));
    const fl = g.floors;
    const binOrder = (b) => FLOOR_BINS.indexOf(b);
    mount(el, [
      { key: "set", label: "Tasks", type: "seg", options: [{ value: "never", label: "the ladder never reads" }, { value: "all", label: "every shared task" }] },
    ], (s) => {
      const tasks = fl.tasks.filter((t) => s.set === "all" || t.ladder_floor === "never");
      const fams = d3.groupSort(tasks, (v) => -v.length, (t) => t.family);
      const units = fams.flatMap((f) => tasks.filter((t) => t.family === f)
        .sort((a, b) => binOrder(a.external_bin) - binOrder(b.external_bin) || a.language.localeCompare(b.language))
        .map((t, i) => ({ ...t, x1: i, x2: i + 1, xm: i + 0.5, models: fl.readers[t.task] ?? [] })));
      const panel = h("div", { class: "viz-card" }, h("p", { class: "viz-note" }, "Hover over a block to list the public models that read that task."));
      const colors = [...sizeColors(FLOOR_BINS.slice(0, 5)), css("--viz-muted")];
      const chart = plot({
        height: 26 * fams.length + 60, marginLeft: 150,
        x: { label: "tasks (one block per language)", grid: true },
        y: { domain: fams, label: null, tickFormat: (f) => benchName(g.names, f) },
        color: { domain: FLOOR_BINS, range: colors, legend: true, label: "smallest public model above chance" },
        marks: [
          Plot.rectX(units, { x1: "x1", x2: "x2", y: "family", fill: "external_bin", inset: 0.6 }),
          Plot.tip(units, Plot.pointer({ x: "xm", y: "family",
            title: (u) => `${u.task} · ${langName(u.language)}\nladder: ${u.ladder_floor === "never" ? "never above chance up to 1.7B" : `above chance from ${u.ladder_floor}`}\n` +
              `public models: ${u.external_floor === "never" ? "none above chance up to 70B" : `from ${u.external_floor}`} (${u.models.length} listed below)` })),
        ],
      });
      chart.addEventListener("input", () => {
        const u = chart.value;
        if (!u) return;
        const list = u.models.length
          ? h("ol", { class: "viz-models" }, u.models.map(([m, size, post]) => h("li", {}, `${m} (${size}${post ? ", post-trained" : ""})`)))
          : h("p", {}, "No released public model clears chance on this task.");
        panel.replaceChildren(h("h4", {}, `${langName(u.language)} · ${benchName(g.names, u.family)} (${u.task})`),
          h("p", { class: "viz-note" }, "Public models whose own run clears the threshold, smallest first:"), list);
      });
      return [chart, panel, note("Each block is one task (a benchmark in one language) that both our ladder and public models are scored on. Its colour is the smallest public model that reads it above chance (same threshold). A task the ladder never reads but a public model of 1.7B or less does is a training-budget floor: those models saw 10 to 36 trillion tokens. A task that needs 3B or more, or that no model reads, is a floor of the benchmark or of the language's data.")];
    });
  };

  VIEWS.gateExample = async (el) => {
    const idx = await languages().then(() => load("gate_example"));
    const fams = Object.keys(idx.families).sort((a, b) => benchName(idx.names, a).localeCompare(benchName(idx.names, b)));
    const st = { fam: idx.families.include_v2_og ? "include_v2_og" : fams[0], task: null, hl: null };
    const famSel = h("select", {}, fams.map((f) => h("option", { value: f }, benchName(idx.names, f))));
    const taskSel = h("select", {});
    const out = h("div");
    el.replaceChildren(h("div", { class: "viz-controls" }, h("label", {}, "Benchmark", famSel), h("label", {}, "Language", taskSel)), out);
    famSel.value = st.fam;

    const fillTasks = () => {
      const tasks = idx.families[st.fam];
      const count = d3.rollup(Object.values(tasks), (v) => v.length, (l) => l);
      taskSel.replaceChildren(...Object.entries(tasks)
        .sort((a, b) => langName(a[1]).localeCompare(langName(b[1])))
        .map(([t, l]) => h("option", { value: t }, count.get(l) > 1 ? `${langName(l)} (${t})` : langName(l))));
      st.task = tasks.include_v2_og_hungarian_hungary ? "include_v2_og_hungarian_hungary" : taskSel.options[0].value;
      taskSel.value = st.task;
    };
    const render = async () => {
      const data = await load(`gate_example/${st.fam}`);
      const t = data[st.task];
      if (!t) return out.replaceChildren(h("div", { class: "viz-empty" }, "No run trains this language."));
      const sizes = idx.sizes, colors = sizeColors(sizes);
      const runs = t.runs.map(([m, size, score, lcb, above, traj]) => ({ model: idx.models[m], size, score, lcb, above, traj }));
      if (st.hl == null || !runs.some((r) => r.model === st.hl))
        st.hl = (t.need != null ? d3.least(runs, (r) => Math.abs(r.score - t.need)) : runs[0]).model;
      const pts = runs.flatMap((r) => r.traj.map((y, i) => (y == null ? null :
        { model: r.model, size: r.size, x: (i + 1) / 2, y, lcb: wilson(y, t.n_items), final: r })).filter(Boolean));
      const hl = pts.filter((p) => p.model === st.hl);
      const hlRun = runs.find((r) => r.model === st.hl);
      const short = (m) => m.replace(/^lm-/, "").replace(/-seed\d+$/, "");
      const yMin = d3.min([...pts.map((p) => p.lcb), t.chance]) - 0.005, yMax = d3.max(pts, (p) => p.y) + 0.005;
      const lines = (k) => [
        Plot.ruleY([t.chance], { stroke: css("--md-default-fg-color"), strokeDasharray: "1,3" }),
        t.need != null ? Plot.ruleY([t.need], { stroke: css("--viz-muted"), strokeDasharray: "5,3" }) : null,
        k ? Plot.text([t.chance], { y: (v) => v, frameAnchor: "left", dy: 7, dx: 3, text: () => `chance ${fmt(t.chance, 3)}`, fontSize: 10 }) : null,
        k && t.need != null ? Plot.text([t.need], { y: (v) => v, frameAnchor: "left", dy: -6, dx: 3, text: () => `score needed ${fmt(t.need, 3)}`, fill: css("--viz-muted"), fontSize: 10 }) : null,
      ];
      // (a) every run along training; click a line to show its lower bound
      const a = plot({
        height: 330, marginBottom: 40, width: 640,
        x: { label: "training tokens (× Chinchilla)", domain: [0.5, 5], ticks: [1, 2, 3, 4, 5], tickFormat: (v) => `${v}C` },
        y: { label: "accuracy", domain: [yMin, yMax], grid: true },
        color: { domain: sizes, range: colors, legend: true, label: "model size" },
        marks: [
          Plot.areaY(hl, { x: "x", y1: "lcb", y2: "y", fill: css("--viz-s2"), fillOpacity: 0.2 }),
          ...lines(true),
          Plot.line(pts, { x: "x", y: "y", z: "model", stroke: "size", strokeWidth: 1, strokeOpacity: 0.75 }),
          Plot.line(hl, { x: "x", y: "y", stroke: css("--viz-s2"), strokeWidth: 2.6 }),
          Plot.line(hl, { x: "x", y: "lcb", stroke: css("--viz-s2"), strokeWidth: 1, strokeDasharray: "3,2" }),
          Plot.tip(pts, Plot.pointer({ x: "x", y: "y",
            title: (p) => `${short(p.model)} at ${p.x}C: score ${fmt(p.y, 4)}, lower bound ${fmt(p.lcb, 4)}\n` +
              `final: score ${fmt(p.final.score, 4)}, bound ${fmt(p.final.lcb, 4)} → ${p.final.above ? "passes" : "fails"}\n(click to show its bound)` })),
        ].filter(Boolean),
      });
      a.addEventListener("click", () => { if (a.value) { st.hl = a.value.model; render(); } });

      // (b) per size: each run's final score and its lower bound
      const fin = sizes.flatMap((z) => runs.filter((r) => r.size === z).sort((p, q) => p.score - q.score).map((r, i) => ({ ...r, i })));
      const verdict = sizes.map((z, k) => {
        const rs = runs.filter((r) => r.size === z), pass = rs.filter((r) => r.above).length;
        return { size: z, n: rs.length, pass, ok: t.mask[k] };
      });
      const b = plot({
        height: 330, marginBottom: 40, marginTop: 34, width: 640,
        fx: { domain: sizes, label: "model size", axis: "bottom" },
        x: { axis: null, domain: d3.range(d3.max(verdict, (v) => v.n) || 1) },
        y: { label: "final accuracy, with its lower bound", domain: [yMin, Math.max(yMax, d3.max(fin, (r) => r.score) + 0.005)], grid: true },
        marks: [
          ...lines(false),
          Plot.ruleX(fin, { fx: "size", x: "i", y1: "lcb", y2: "score", stroke: (r) => (r.above ? css("--viz-s1") : css("--viz-muted")), strokeWidth: 1.5 }),
          Plot.dot(fin, { fx: "size", x: "i", y: "score", r: (r) => (r.model === st.hl ? 4.5 : 3),
            fill: (r) => (r.model === st.hl ? css("--viz-s2") : r.above ? css("--viz-s1") : css("--viz-muted")) }),
          Plot.text(verdict, { fx: "size", frameAnchor: "top", dy: -24, lineWidth: 8, fontSize: 10,
            text: (v) => (v.n ? `${v.pass} of ${v.n} pass\n${v.ok ? "above chance" : "at chance"}` : "no run"),
            fill: (v) => (v.ok ? css("--viz-good") : css("--viz-muted")), fontWeight: (v) => (v.ok ? 600 : 400) }),
          Plot.tip(fin, Plot.pointer({ fx: "size", x: "i", y: "score",
            title: (r) => `${short(r.model)}\nfinal score ${fmt(r.score, 4)}, lower bound ${fmt(r.lcb, 4)}\n${r.above ? "bound above chance: passes" : "bound at or below chance: fails"}` })),
        ],
      });
      b.addEventListener("click", () => { if (b.value) { st.hl = b.value.model; render(); } });

      // (c) what the verdict does: rq02's decision-accuracy cells of this task
      const ref = sizes.length - 1;
      const cells = sizes.flatMap((z, k) => [
        k < ref ? { kind: "DA-size", size: z, gated: t.mask[k] === 0 || t.mask[ref] === 0, da: t.da_size[z] } : null,
        { kind: "DA-ckpt", size: z, gated: t.mask[k] === 0, da: t.da_ckpt[z] },
      ].filter(Boolean));
      const c = plot({
        height: 150, marginLeft: 70, width: 640,
        x: { domain: sizes, label: null, axis: "top" },
        y: { domain: ["DA-size", "DA-ckpt"], label: null },
        color: { type: "linear", domain: [0.5, 1], range: [css("--viz-neutral"), css("--viz-s1")] },
        marks: [
          Plot.cell(cells, { x: "size", y: "kind", inset: 2, rx: 3,
            fill: (q) => (q.gated || q.da == null ? null : q.da), stroke: (q) => (q.gated ? css("--viz-muted") : null), strokeDasharray: "2,2" }),
          Plot.text(cells, { x: "size", y: "kind", text: (q) => (q.gated ? "gated" : q.da == null ? "–" : fmt(q.da, 2)),
            fill: (q) => (q.gated ? css("--viz-muted") : css("--md-default-fg-color")) }),
          Plot.tip(cells, Plot.pointer({ x: "size", y: "kind",
            title: (q) => (q.kind === "DA-size"
              ? `Does ${q.size} rank the designs like 1.7B?\n${q.gated ? "Not asked: the task is at chance at " + (t.mask[sizes.indexOf(q.size)] === 0 ? q.size : "1.7B") : q.da == null ? "too few design pairs" : `decision accuracy ${fmt(q.da)}`}`
              : `Does ${q.size} at 90 % of its run rank like its own end?\n${q.gated ? `Not asked: the task is at chance at ${q.size}` : q.da == null ? "too few design pairs" : `decision accuracy ${fmt(q.da)}`}`) })),
        ],
      });
      const head = h("p", { class: "viz-note" },
        `${t.n_items.toLocaleString()} items, ${Math.round(1 / t.chance)} options, so chance is ${fmt(t.chance, 3)}. ` +
        (t.need != null ? `A run needs ${fmt(t.need, 3)} (${fmt(t.need - t.chance, 3)} over chance) for its lower bound to clear chance. ` : "Even a perfect score cannot clear chance with this few items. ") +
        `Highlighted run: ${short(hlRun.model)}, final ${fmt(hlRun.score, 4)}, bound ${fmt(hlRun.lcb, 4)} → ${hlRun.above ? "passes" : "fails"}.`);
      out.replaceChildren(head,
        h("h4", { class: "viz-sub" }, "(a) Every run that trained on the language, along training"), a,
        h("h4", { class: "viz-sub" }, "(b) The verdict per model size"), b,
        h("h4", { class: "viz-sub" }, "(c) What the verdict changes downstream"), c);
    };
    famSel.addEventListener("change", () => { st.fam = famSel.value; st.hl = null; fillTasks(); render(); });
    taskSel.addEventListener("change", () => { st.task = taskSel.value; st.hl = null; render(); });
    fillTasks();
    REDRAW.push(render);
    await render();
  };

  // ---------- RQ10 / RQ11: past the reference, the recipe ----------
  VIEWS.recipe = async (el) => {
    const d = await load("recipe");
    const rows = d.recommendation.map((r) => ({ ...r, name: benchName(d.names, r.benchmark) }));
    el.replaceChildren(table(rows, [
      { key: "name", label: "benchmark" },
      { key: "variant", label: "evaluate it as" },
      { key: "median_safe_size", label: "reliable from (median over languages)" },
      { key: "safe_by_1B", label: "languages reliable by 1B", render: (r) => `${pct(r.safe_by_1B)} of ${r.languages}` },
      { key: "mean_da_size", label: "mean DA-size", num: true },
    ], { sort: "mean_da_size" }),
    note(`Per benchmark, the version (as published, RF or LLM-RF) and the scoring (accuracy, or bits per byte of the correct answer: bBPB) that reaches a reliable decision (decision accuracy ≥ ${d.tau} against the 1.7B model) from the smallest proxy. "never": no proxy up to 1B is reliable on the median language.`));
  };

  VIEWS.recipeSizes = async (el) => {
    const d = await load("recipe");
    const pops = uniq(d.overview.map((r) => r.population));
    mount(el, [{ key: "pop", label: "Tasks", type: "seg", options: pops.map((p) => ({ value: p, label: p === "paired" ? "tasks with every version" : p })) }], (s) => {
      const rows = d.overview.filter((r) => r.population === s.pop && r.variant !== "all variants");
      const variants = uniq(rows.map((r) => r.variant));
      return [plot({
        height: 320,
        x: { domain: SIZES.filter((z) => rows.some((r) => r.size === z)), label: "proxy size" },
        y: { label: "mean decision accuracy against 1.7B", grid: true },
        color: { domain: variants, range: series(variants.length), legend: true },
        marks: [
          Plot.ruleY([d.tau], { stroke: css("--viz-good"), strokeDasharray: "4,3" }),
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "1,3" }),
          Plot.line(rows, { x: "size", y: "mean_da_size", stroke: "variant", strokeWidth: 2 }),
          Plot.dot(rows, { x: "size", y: "mean_da_size", fill: "variant", r: 3.5, tip: true,
            title: (r) => `${r.variant} at ${r.size}\nmean DA ${fmt(r.mean_da_size)} over ${r.n_tasks} tasks of ${r.n_benchmarks} benchmarks\n${pct(r.reliable_share)} of the tasks reliable` }),
        ],
      }), note(`One line per way of posing and scoring a benchmark. Green dashed line: the reliability threshold (${d.tau}); dotted: a coin flip (0.5).`)];
    });
  };

  VIEWS.aboveReference = async (el) => {
    const d = await load("recipe");
    const ar = d.above_reference;
    mount(el, [
      { key: "ref", label: "Reference", type: "seg", options: [{ value: "3B", label: "the 3B model" }, { value: "1.7B", label: "the 1.7B model, same pairs" }] },
      { key: "channel", label: "Measured with", type: "seg", options: uniq(ar.map((r) => r.channel)).map((c) => ({ value: c, label: c === "bpb" ? "bits per byte" : c })) },
      { key: "axes", label: "Design pairs", type: "seg", options: uniq(ar.map((r) => r.axes)) },
    ], (s) => {
      const rows = ar.filter((r) => r.reference === s.ref && r.channel === s.channel && r.axes === s.axes).map((r) => ({ ...r, x: r.frac * 5 }));
      const sizes = SIZES.filter((z) => rows.some((r) => r.size === z));
      return [plot({
        height: 320,
        x: { label: "proxy's training tokens (× Chinchilla)", ticks: [1, 2, 3, 4, 5], tickFormat: (v) => `${v}C` },
        y: { label: `decision accuracy against ${s.ref}`, domain: [0.3, 1], grid: true },
        color: { domain: sizes, range: sizeColors(sizes), legend: true, label: "proxy size" },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.line(rows, { x: "x", y: "da", stroke: "size", strokeWidth: 2 }),
          Plot.dot(rows, { x: "x", y: "da", fill: "size", r: 3, tip: true,
            title: (r) => `${r.size} at ${r.x}C\nDA ${fmt(r.da)} over ${r.n_pairs} pairs, ${r.n_tasks} tasks` }),
        ],
      }), note("Does a proxy rank the designs like the 3B model as well as like the 1.7B model? Both references are read on the same design pairs (the four 3B cells), so the two settings differ only in the reference.")];
    });
  };

  // ---------- RQ12 / RQ13 (shown once their analyses are merged) ----------
  VIEWS.itemsDA = async (el) => {
    const d = await loadOptional(el, "above_chance_items");
    if (!d) return;
    const orderings = uniq(d.da_size.map((r) => r.ordering));
    mount(el, [{ key: "axes", label: "Design pairs", type: "seg", options: uniq(d.da_size.map((r) => r.axes)) }], (s) => {
      const rows = d.da_size.filter((r) => r.axes === s.axes);
      return [plot({
        height: 300,
        x: { domain: SIZES.filter((z) => rows.some((r) => r.size === z)), label: "proxy size" },
        y: { label: "decision accuracy against 1.7B", grid: true },
        color: { domain: orderings, range: series(orderings.length), legend: true },
        marks: [
          Plot.ruleY([0.5], { stroke: css("--viz-muted"), strokeDasharray: "4 3" }),
          Plot.ruleX(rows, { x: "size", y1: "lo", y2: "hi", stroke: "ordering", strokeOpacity: 0.5, dx: 0 }),
          Plot.line(rows, { x: "size", y: "reliability", stroke: "ordering", strokeWidth: 2 }),
          Plot.dot(rows, { x: "size", y: "reliability", fill: "ordering", r: 3.5, tip: true,
            title: (r) => `${r.ordering} at ${r.size}\nDA ${fmt(r.reliability)} (${fmt(r.lo)}–${fmt(r.hi)}) over ${r.n_tasks} tasks` }),
        ],
      }), note("'full': every item. The other lines keep only the items the 1.7B models answer above chance, choosing the items before or after the gate.")];
    });
  };

  VIEWS.itemsSurvival = async (el) => {
    const d = await loadOptional(el, "above_chance_items");
    if (!d) return;
    el.replaceChildren(table(d.survival, [
      { key: "ordering", label: "ordering" }, { key: "step", label: "step" },
      { key: "n_tasks", label: "tasks", num: true, digits: 0 }, { key: "n_items", label: "items", num: true, digits: 0 },
      { key: "n_kept", label: "items kept", num: true, digits: 0 },
    ], { sort: "n_kept" }));
  };

  VIEWS.englishVerdict = async (el) => {
    const d = await loadOptional(el, "english_only");
    if (!d) return;
    el.replaceChildren(table(d.verdict, [
      { key: "expectation", label: "expectation" }, { key: "l1", label: "English-only runs" },
      { key: "comparator", label: "against the multilingual runs" }, { key: "verdict", label: "holds?" },
    ], { sort: "expectation" }));
  };

  VIEWS.englishGap = async (el) => {
    const d = await loadOptional(el, "english_only");
    if (!d) return;
    const opt = (k) => uniq(d.scores.map((r) => r[k]));
    mount(el, [
      { key: "scoring", label: "Scoring", type: "seg", options: opt("scoring") },
      { key: "arch", label: "Architecture", type: "seg", options: opt("arch") },
    ], (s) => {
      const rows = d.scores.filter((r) => r.scoring === s.scoring && r.arch === s.arch && !r.thin);
      const comps = uniq(rows.map((r) => r.comparator));
      return [plot({
        height: 300,
        x: { domain: SIZES.filter((z) => rows.some((r) => r.size === z)), label: "model size" },
        y: { label: "English-only minus multilingual (mean over English tasks)", grid: true },
        color: { domain: comps, range: series(comps.length), legend: true, label: "against" },
        marks: [
          Plot.ruleY([0], { stroke: css("--viz-muted") }),
          Plot.line(rows, { x: "size", y: "mean_gap", stroke: "comparator", strokeWidth: 2 }),
          Plot.dot(rows, { x: "size", y: "mean_gap", fill: "comparator", r: 3.5, tip: true,
            title: (r) => `${r.size}, against ${r.comparator}\nmean gap ${fmt(r.mean_gap, 4)}, English-only ahead on ${pct(r.win_share)} of ${r.n_tasks} tasks` }),
        ],
      }), note("Above zero: the run trained only on English scores higher on the English benchmarks than the multilingual run of the same size.")];
    });
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

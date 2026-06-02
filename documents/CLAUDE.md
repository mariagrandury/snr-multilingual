# CLAUDE.md — `documents/`

Slidev presentation for the Signal-Aware Multilingual Evaluation
project. The deck lives in [slides.md](slides.md) and uses the
[`slidev-theme-scholarly`](https://github.com/maxnatamo/slidev-theme-scholarly)
theme. Deck-wide style overrides live in [style.css](style.css).

## Commands

```bash
# from documents/
npx slidev --port 3030      # dev server with hot-reload → http://localhost:3030/
npx slidev build            # static SPA into dist/
npx slidev export           # PDF (needs the playwright-chromium devDep)
```

`npm run dev` (= `slidev --open`), `build`, and `export` are defined in
[package.json](package.json).

> Hot-reload caveat: edits to `slides.md` and to an **existing**
> `style.css` reload live. **Creating** `style.css` (or other auto-loaded
> entry files) only registers on server start — restart after adding one.

## How a Slidev deck is structured

- Slides are separated by `---` on its own line.
- The **first** frontmatter block is the **headmatter** — global deck
  config (theme, `themeConfig`, `aspectRatio`, `transition`, `authors`…).
  See the top of [slides.md](slides.md).
- Every other frontmatter block is **per-slide**: `layout:` plus that
  layout's props.
- `<!-- ... -->` after a slide's content becomes **presenter notes**.
- Markdown can embed Vue components and HTML; the theme ships components
  like `<Block type="success" title="...">`, `<Columns>`, `<Cite>`,
  `<Steps>`, `<Theorem>` (see `node_modules/slidev-theme-scholarly/components/`).
- `$frontmatter`, KaTeX math (`$...$`, `$$...$$`), and code fences all work.
- Slidev docs: https://sli.dev/ — theme layouts/components are documented
  in the theme repo and, authoritatively, in
  `node_modules/slidev-theme-scholarly/layouts/`.

## Layouts

Set per slide with `layout: <name>`. The scholarly theme provides the
layouts below (they override/extend Slidev built-ins like `center`,
`cover`, `default`, `two-cols`). Props come from each layout's
`defineProps` block in `node_modules/slidev-theme-scholarly/layouts/<name>.vue`
— check there when unsure.

### Used in this deck

| Layout | Props | Notes |
|---|---|---|
| `section` | `sectionMode: light \| dark` | Divider slide. **In this deck restyled white via [style.css](style.css)** (see below) — the `sectionMode` prop does **not** take effect here. |
| `bullets` | `icon` (bullet char, default `▸`) | Title/subtitle come from `title:`/`subtitle:` frontmatter. Most common layout in the deck. |
| `figure` | `image` (use this, **not** `src` — `src` is reserved by Slidev), `caption`, `label`, `title`, `subtitle`, `height` (default `60%`), `fit: contain \| cover` | Single captioned figure. |
| `focus` | `color: primary \| blue \| green \| amber \| red \| purple`, `icon` (emoji) | Big centered statement with an accent border. |
| `default` | `density: auto \| compact \| normal \| relaxed`, `title`, `subtitle` | Title auto-extracted from first `#` if `title:` unset. |
| `image-left` | `image`, `ratio` (image:content, e.g. `2:3`), `fit: cover \| contain \| fill`, `position` | Image column on the left. |
| `image-right` | same as `image-left` but `ratio` is content:image | Image column on the right. |
| `timeline` | `title`, `items` (array of `{year/title/description}`) | Horizontal timeline. |
| `compare` | `title`, `subtitle`, `leftLabel`, `rightLabel`, `leftColor`/`rightColor: red \| blue \| green \| gray` | Side-by-side comparison; content goes in `::left::` / `::right::` slots. |
| `agenda` | `title`, `items` (string array) | Numbered agenda list. |

### Also available (not yet used)

| Layout | Props |
|---|---|
| `cover` | `authors`, `footerLeft`, `footerMiddle` (title slide) |
| `intro` | `align: left \| center`, `density` |
| `toc` | `title` (or `false` to hide), `showNumbers`, `highlightCurrent`, `sections` |
| `two-cols` | `gap`, `ratio`; content in `::left::` / `::right::` |
| `split-image` | `images` (array), `captions` (array) |
| `fact` / `statement` | `color` (fact) / `author` (statement) — large emphasis slides |
| `quote` | `author`, `source` |
| `methodology` | `ratio`, `title`, `subtitle` |
| `results` | `cols` (number/string), `title`, `subtitle` |
| `center` / `auto-center` / `auto-size` | centered content; `auto-*` autofit text (`minFontSize`/`maxFontSize`, `density`) |
| `acknowledgments` | `title`, `funders`, `collaborators` |
| `references` | `page`, `title`, `minFontSize`, `maxFontSize` (needs `references.bib`) |
| `end` | `thankYou`, `subtitle`, `email`, `website`, `qrcode`, `qrcodeLabel` |

Example (from the deck):

```markdown
---
layout: figure
image: /results/belebele.png
caption: Per-language SNR on Belebele
fit: contain
---

---
layout: focus
color: green
icon: 🎯
---

## Which benchmarks provide reliable signal at each training stage?
```

Slidev also has built-in `image`, `full`, and `none` layouts (see
https://sli.dev/builtin/layouts).

## Updating the style

Three levers, from broadest to narrowest:

### 1. `themeConfig` in the headmatter (theme presets)

In the first frontmatter block of [slides.md](slides.md):

```yaml
themeConfig:
  colorTheme: classic-blue   # classic-blue, cambridge-green, nordic-blue, yale-blue,
                             # oxford-burgundy, princeton-orange, warm-sepia,
                             # high-contrast, monochrome
  fontTheme: contemporary    # classic, contemporary, modern, traditional, elegant,
                             # humanist, technical, sans-default
  colorMode: dark
  sectionMode: dark          # default appearance of `section` slides (overridden below)
```

These map to `[data-color-theme="…"]` / `[data-font-theme="…"]` rule
sets in `node_modules/slidev-theme-scholarly/styles/`.

### 2. `style.css` — deck-wide CSS overrides ([style.css](style.css))

Slidev auto-loads `documents/style.css`. This is the place for changes
that should apply to **every** slide. It currently does two things:

- **White section slides with theme-blue titles.** The theme's
  `sectionMode` prop relies on an attribute that doesn't reach the
  `.slidev-layout.section` element, so it's a no-op here; we override the
  CSS directly instead (`background`, `h1 color: var(--slidev-theme-primary)`).
- **Avenir everywhere.** The theme routes all fonts through three CSS
  variables (`--scholarly-font-serif`, `--scholarly-font-sans`,
  `--scholarly-font-body`); we redefine all three to an Avenir stack on a
  `:root:root` selector (doubled `:root` outweighs the theme's
  `[data-font-theme]` rule regardless of load order). Code blocks keep
  their hardcoded monospace stack.

When overriding theme rules, prefer setting the theme's own CSS variables
(grep `node_modules/slidev-theme-scholarly/styles/` for `--scholarly-*`)
over `!important`; bump specificity (e.g. `:root:root`,
`.slidev-layout.section`) only when load order isn't enough.

### 3. Per-slide `<style>` — scoped to one slide

A `<style>` block inside a single slide is auto-scoped to that slide
only. Use it for one-off tweaks; use `style.css` for anything deck-wide.

```markdown
# A special slide

<style>
.slidev-layout { background: #000; color: #fff; }
</style>
```

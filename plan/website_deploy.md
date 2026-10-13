# Website: serve the docs (no slides) at the Netlify URL, and rename the site

Goal for the double-blind review: `https://snr-multilingual.netlify.app/`
serves the MkDocs site (Home, Model ladder, Findings, Benchmarks,
Recommender), and the Slidev deck is reachable nowhere.

## What is wrong today (checked 2026-10-13)

- Every path of the live site (`/`, `/1`, `/findings/gate/`, `/slides/`)
  returns the Slidev deck: the deployed bundle is a bare `slidev build`
  (`assets/index-*.js` at the root, plus the catch-all `_redirects` Slidev
  writes into `documents/dist/`, which sends every path to the deck).
- The repo does not build that. The root `netlify.toml` runs `bash build.sh`,
  which builds MkDocs into `site/` and skips the slides unless
  `BUILD_SLIDES=1`.
- So Netlify is not using the root `netlify.toml`. The likely cause is the
  site's UI build settings: **Base directory = `documents`** (Netlify then
  looks for `documents/netlify.toml`, finds none, and falls back to the UI's
  build command and publish directory, e.g. `npm run build` → `dist`).
  Other possibilities to rule out on the way: the production branch is not
  `main`, or every build since the docs landed has failed and the live deploy
  is an old one.

## 1. Point Netlify at the docs build

In the Netlify dashboard, open the site, then **Site configuration → Build &
deploy → Continuous deployment**.

1. **Repository**: the snr-multilingual repo. **Production branch**: `main`
   (the branch you merge the paper work into).
2. **Build settings → Edit settings**:
   - Base directory: **empty** (the repo root, so `netlify.toml` is read)
   - Package directory: **empty**
   - Build command: `bash build.sh` (or empty: `netlify.toml` sets it, and
     `netlify.toml` wins over the UI anyway once the base directory is the
     root)
   - Publish directory: `site`
   - Functions directory: empty
3. **Environment variables** (Site configuration → Environment variables):
   make sure `BUILD_SLIDES` is **not** set (or is `0`). `NODE_VERSION` and
   `PYTHON_VERSION` come from `netlify.toml`.
4. **Branches and deploy contexts**: set **Branch deploys** to "None" and
   **Deploy previews** to "None" (or "Any pull request" only if you accept
   that a PR preview builds the same anonymous site). A branch deploy of an
   old branch could otherwise publish the deck under
   `<branch>--snr-multilingual.netlify.app`.
5. **Deploys → Trigger deploy → Clear cache and deploy site**.
6. Open the deploy log and check, in order: `==> Building MkDocs`,
   `==> Skipping Slidev slides (set BUILD_SLIDES=1 after review)`,
   `==> Build complete`. No line should say `Building Slidev slides`.

Note on the ignore rule: `netlify.toml` skips a build when nothing under
`docs/`, `*.md`, `mkdocs.yml`, `mkdocs_hooks.py`, `anonymity.py` or
`configs/multilingual_benchmarks.csv` changed. A deploy listed as "Canceled
(ignored)" is that rule, not a failure; "Clear cache and deploy" always builds.

## 2. Verify (no dashboard needed)

```bash
for u in / /findings/gate/ /recommend/ /slides/ /1; do
  printf '%-16s ' "$u"
  curl -s -o /tmp/p.html -w '%{http_code}  ' "https://snr-multilingual.netlify.app$u"
  grep -o '<title>[^<]*</title>' /tmp/p.html | head -1; echo
done
```

Expected: `/`, `/findings/gate/` and `/recommend/` titled "… - SNR
Multilingual" (MkDocs); `/slides/` and `/1` return **404** (MkDocs's
`404.html`), never a "- Slidev" title. Also open `/findings/gate/` in a
browser and check the token-threshold table draws.

## 3. The old slide deploys

Every past deploy stays reachable at its own permalink
(`https://<deploy-id>--snr-multilingual.netlify.app`). The IDs are random
and appear nowhere public, so a reviewer cannot reach the deck through
them. If you want them gone anyway, the Netlify API has a delete-deploy
endpoint (`DELETE https://api.netlify.com/api/v1/deploys/<deploy-id>`, with
a personal access token); check the dashboard first, since the currently
published deploy cannot be deleted.

## 4. Rename the site

### The URL (Netlify subdomain)

**Site configuration → Site details → Change site name** → type the new name
→ Save. The site is then at `https://<new-name>.netlify.app/`; the old
`snr-multilingual.netlify.app` stops working at once (no redirect), so
change it before the URL goes into the paper. Then:

1. `mkdocs.yml`: `site_url: https://<new-name>.netlify.app/` (it feeds the
   sitemap and canonical links).
2. `docs/CLAUDE.md`: the URL if it is quoted.
3. The paper: wherever the site is cited (today no section cites it:
   `grep -rn netlify documents/paper/sections`).
4. Commit, push to `main`, re-run the check of step 2 on the new URL.

For anonymity, the name must not identify the authors or the institution
(`snr-multilingual` is fine; the Netlify team name does not appear in the
URL).

### A custom domain (only if you want `snr-multilingual.app`)

1. Buy the domain with WHOIS privacy on (the registrant's name must not be
   public during review). `.app` is HTTPS-only (HSTS-preloaded), which
   Netlify handles.
2. **Domain management → Add a domain** → `snr-multilingual.app` → follow the
   prompt: either move the domain's nameservers to Netlify DNS, or at your
   registrar add an `ALIAS`/`ANAME` (or `A` to Netlify's load balancer IP
   shown in the dashboard) for the apex and a `CNAME www →
   <site-name>.netlify.app`.
3. **HTTPS → Verify DNS configuration → Provision certificate** (Let's
   Encrypt, a few minutes after DNS resolves).
4. Set it as the **primary domain** so `*.netlify.app` redirects to it, then
   update `site_url` in `mkdocs.yml` as above.

### The title shown on the pages

`mkdocs.yml` → `site_name` (today "SNR Multilingual", shown in the header and
in every tab title) and `site_description`. Keep it free of names and
affiliations; `mkdocs_hooks.py`'s anonymity check runs on every build
(`mkdocs build --strict` fails on a leak).

## 5. After the review

Set `BUILD_SLIDES=1` in the Netlify environment variables (the deck then
builds into `/slides/`), uncomment the Slides and Docs nav entries in
`mkdocs.yml`, and put back the links `anonymity.py` replaced.

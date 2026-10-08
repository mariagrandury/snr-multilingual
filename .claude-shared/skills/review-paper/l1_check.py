#!/usr/bin/env python3
"""L1 mechanical checks for one version of the paper (review-paper skill).

Walks the \\input tree from main.tex and reports, one issue per line:

    severity <TAB> check <TAB> file:line <TAB> message

severity is blocking | major | minor | info (info is never counted by the
stop rule). Checks: broken \\input / \\includegraphics / \\bibliography,
undefined and duplicate labels, unreferenced figures and tables, cite keys
missing from the .bib files, suspicious bib entries, dashes and ';' as prose
punctuation (RULES.md "Paper punctuation"), "from X to Y" ranges, AI lingo,
anonymity leaks, leftover draft macros and placeholders, K vs L notation,
repeated words, mixed British/American spelling, the ARR structure
(review option, Limitations, \\bibliographystyle, abstract length), and, for
an Overleaf export, generated constants and generated appendix pages that
differ from the repo copy.

Works on an Overleaf export (main.tex at the root) and on documents/paper
(main.tex under sections/). An export without its .bib files or figures is
checked against documents/paper and says so. LaTeX comments, \\iffalse
blocks and the preamble are not prose.

    python3 l1_check.py <folder>                        # TSV + summary
    python3 l1_check.py <folder> --json <folder>/review-l1.json \\
        --prev <previous folder>/review-l1.json         # + fixed/open/new
    python3 l1_check.py <folder> --dump-dir <scratch>   # cited.tsv, numbers.tsv

Runs under the system python3 (3.6) and in the snr env; stdlib only.
"""
import argparse
import datetime
import fnmatch
import json
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

REPO_FALLBACK = Path("/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual")
_here = Path(__file__).resolve().parent
REPO = _here.parents[2] if (_here.parents[2] / "documents" / "paper").is_dir() else REPO_FALLBACK
REPO_PAPER = REPO / "documents" / "paper"

# Stop rule after L1 (SKILL.md). Distinct issues: a label, cite key or path
# counts once however often it is referenced; a style scan counts once per file.
STOP_BLOCKING = 5
STOP_BLOCKING_MAJOR = 20
STOP_TOTAL = 60

# check_rules.GENERATED_TEX: paper sections written whole by a generator.
GENERATED_TEX = ("app_rq*.tex", "app_fixed_populations.tex", "app_02_table_languages.tex")
GRAPHIC_EXTS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")
ABSTRACT_MAX_WORDS = 200  # ARR / ACL style guide

AI_LINGO = [
    "delve", "delves", "delving", "crucial", "crucially", "notably", "leverage", "leverages",
    "leveraging", "landscape", "pivotal", "intricate", "intricacies", "nuanced", "underscore",
    "underscores", "underscoring", "showcase", "showcases", "showcasing", "realm", "tapestry",
    "multifaceted", "paramount", "seamless", "seamlessly", "holistic", "foster", "fosters",
    "harnessing", "unlock", "unlocks", "unveil", "unveils", "navigate", "navigating",
    "meticulous", "meticulously", "noteworthy", "commendable", "testament", "groundbreaking",
    "cutting-edge", "game-changer", "game-changing", "ever-evolving", "ever-growing", "elevate",
    "embark", "endeavor", "endeavour", "plethora", "myriad", "boasts", "invaluable", "utilize",
    "utilizes", "utilizing", "facilitate", "facilitates", "remarkable", "remarkably", "vital",
    "robustly", "streamline", "synergy", "bolster", "bolsters", "garner", "insightful",
]
AI_PHRASES = [
    r"it is (?:worth|important to) not(?:e|ing)", r"plays? a (?:crucial|key|pivotal|vital) role",
    r"in the realm of", r"shed(?:s|ding)? light on", r"pav(?:e|es|ing) the way",
    r"a deeper understanding", r"in today'?s", r"serves? as a testament",
    r"not (?:only|just|merely) [^.,;]{1,60}, but", r"(?:this|these) (?:findings|results) highlight",
]
MISSPELLINGS = {
    "teh": "the", "recieve": "receive", "seperate": "separate", "occured": "occurred",
    "accross": "across", "dependant": "dependent", "existance": "existence", "wich": "which",
    "langauge": "language", "langauges": "languages", "lenght": "length", "occurence": "occurrence",
    "relevent": "relevant", "succesful": "successful", "thier": "their", "untill": "until",
    "acheive": "achieve", "begining": "beginning", "consistant": "consistent",
    "definately": "definitely", "enviroment": "environment", "independant": "independent",
    "neccessary": "necessary", "perfomance": "performance", "preformance": "performance",
    "referece": "reference", "reponse": "response", "similiar": "similar", "threshhold": "threshold",
    "tokenizaton": "tokenization", "evalution": "evaluation", "benchamrk": "benchmark",
    "multilingal": "multilingual", "accuarcy": "accuracy", "corrlation": "correlation",
    "checkpont": "checkpoint", "paramters": "parameters", "parmeters": "parameters",
}
SPELLING_PAIRS = [("behaviour", "behavior"), ("colour", "color"), ("modelling", "modeling"),
                  ("labelled", "labeled"), ("favour", "favor"), ("centre", "center"),
                  ("analyse", "analyze"), ("generalisation", "generalization")]

ANON_BLOCKING = re.compile(
    r"iopsstor|capstor|/users/|/scratch/|mariagrandury|aromanou|grandury|romanou|bosselut|"
    r"\bhasler\b|\bEPFL\b|\bCSCS\b|clariden|\bAlps\b|snr-models|snr-multilingual|github\.com|"
    r"wandb\.ai|\bW&B\b|\ba139\b|\ba145\b|our (?:previous|prior|earlier) work|"
    r"we (?:previously|recently) (?:showed|proposed|introduced)", re.I)
ANON_CHECK = re.compile(r"huggingface\.co/(?!docs/|papers/)|swiss[- ]ai", re.I)
ANON_INFO = re.compile(r"\bApertus\b")  # a public model: citing it is fine, naming it often is a hint
DRAFT_MACRO = re.compile(r"\\(AR|CL|TODO|todo|citeneeded|checknum|fix|new|update|marginpar|hl|"
                         r"note|missing|placeholder)\b(?![a-zA-Z@])")
PLACEHOLDER = re.compile(r"\[(?:X+\s*\\?%?|CITE|TBD|TODO|REF|N)\]|\bTBD\b|\bTODO\b|\bFIXME\b|"
                         r"\bXXX*\b|\?\?")
DEF_LINE = re.compile(r"\\(?:re)?newcommand|\\providecommand|\\def\b|\\definecolor|\\newtcolorbox|"
                      r"\\DeclareRobustCommand")
L_NOTATION = re.compile(r"\$L\s*(?:=\s*\d|\\in)|\$L\$|\bL\s?=\s?\d|\bL(?:1|2|8|15|30|50)\b")
REF_CMD = re.compile(r"\\(?:ref|cref|Cref|autoref|eqref|pageref|nameref|vref|labelcref)\*?\{([^}]*)\}")
CITE_CMD = re.compile(r"\\(?:no)?cite(?!needed)[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
LABEL_CMD = re.compile(r"\\label\{([^}]+)\}")
INPUT_CMD = re.compile(r"\\(input|include|subfile)\{([^}]+)\}")
GRAPHIC_CMD = re.compile(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]+)\}")
BIB_CMD = re.compile(r"\\bibliography\{([^}]+)\}|\\addbibresource(?:\[[^\]]*\])?\{([^}]+)\}")
ARXIV_ID = re.compile(r"(?:arxiv(?:\.org)?[:./\s]*(?:abs/|pdf/)?|10\.48550/arxiv\.)(\d{2})(\d{2})\.\d{4,5}", re.I)
MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline",
             "multline*", "eqnarray", "eqnarray*", "displaymath", "math")
SKIP_ENVS = ("verbatim", "lstlisting", "minted", "comment")


def strip_comment(line):
    return re.sub(r"(?<!\\)%.*", "", line)


def prose_view(code):
    """Code line with math, command arguments that hold keys/paths, and URLs blanked."""
    s = re.sub(r"\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]", " MATH ", code)
    s = re.sub(r"\\(?:S\s*)?(?:ref|cref|Cref|autoref|eqref|pageref)\*?\{[^}]*\}", " 0 ", s)  # ref ranges read as numeric
    s = re.sub(r"(?<!\\)\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(?:url|label|ref|cref|Cref|autoref|eqref|pageref|input|include|includegraphics|"
               r"href|bibliography|graphicspath|begin|end|cite[a-zA-Z]*|nocite|textcolor|color|"
               r"cellcolor|rowcolor|definecolor|setlength|hspace|vspace|newcommand)\*?"
               r"(?:\[[^\]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    return s


class Paper:
    """The compiled \\input tree of one paper folder."""

    def __init__(self, folder, issues):
        self.folder = folder.resolve()
        self.issues = issues
        self.main = self._find_main()
        self.bases = []
        for b in (self.main.parent, self.folder):
            if b not in self.bases:
                self.bases.append(b)
        self.is_repo = self.folder == REPO_PAPER.resolve()
        self.lines = []           # (relpath, lineno, raw, code, in_body, in_math_env)
        self.files = []
        self.graphicspath = [""]
        self.bibs = []
        self._walk(self.main, set())

    def rel(self, p):
        try:
            return p.resolve().relative_to(self.folder).as_posix()
        except ValueError:
            return str(p)

    def _find_main(self):
        for cand in (self.folder / "main.tex", self.folder / "sections" / "main.tex"):
            if cand.is_file():
                return cand
        for cand in sorted(self.folder.rglob("*.tex")):
            if "\\documentclass" in cand.read_text(errors="replace"):
                return cand
        sys.exit("no main.tex (or file with \\documentclass) under %s" % self.folder)

    def resolve(self, name, exts=("", ".tex"), extra_dirs=("",)):
        for base in self.bases:
            for d in extra_dirs:
                for ext in exts:
                    p = base / d / (name + ext)
                    if p.is_file():
                        return p
        return None

    def _walk(self, path, seen):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        self.files.append(path)
        rel = self.rel(path)
        in_body = path != self.main.resolve()
        skip_depth, iffalse, math_env = 0, 0, 0
        for i, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = strip_comment(raw)
            if path == self.main.resolve() and "\\begin{document}" in code:
                in_body = True
            if re.search(r"\\iffalse\b", code):
                iffalse += 1
            if iffalse:
                if re.search(r"\\fi\b", code):
                    iffalse -= 1
                continue
            for env in SKIP_ENVS:
                if "\\begin{%s}" % env in code:
                    skip_depth += 1
                if "\\end{%s}" % env in code:
                    skip_depth -= 1
                    code = ""
            if skip_depth:
                continue
            if code.count("$$") % 2:  # a $$ display block opens or closes on this line
                math_env = 0 if math_env else 1
                self.lines.append((rel, i, raw, code, in_body, True))
                continue
            opened = any("\\begin{%s}" % e in code for e in MATH_ENVS) or "\\[" in code and "\\]" not in code
            if opened:
                math_env += 1
            self.lines.append((rel, i, raw, code, in_body, math_env > 0))
            if any("\\end{%s}" % e in code for e in MATH_ENVS) or "\\]" in code and "\\[" not in code:
                math_env = max(0, math_env - 1)
            m = re.search(r"\\graphicspath\{(.*)\}", code)
            if m:
                self.graphicspath = [""] + re.findall(r"\{([^}]*)\}", m.group(1))
            for bm in BIB_CMD.finditer(code):
                names = bm.group(1) or bm.group(2)
                for name in [n.strip() for n in names.split(",") if n.strip()]:
                    self.bibs.append((name if name.endswith(".bib") else name + ".bib", rel, i))
            optional = "\\IfFileExists" in code
            for im in INPUT_CMD.finditer(code):
                self._input(im.group(2).strip(), rel, i, optional, seen)

    def _input(self, name, rel, i, optional, seen):
        target = self.resolve(name)
        if target is not None:
            self._walk(target, seen)
            return
        if optional:
            self.issues.add("info", "input-optional-absent", rel, i, name,
                            "\\IfFileExists target %s absent (skipped by LaTeX too)" % name)
            return
        stem = Path(name).name
        found = [p for p in list(self.folder.rglob(stem + ".tex")) + list(self.folder.rglob(stem)) if p.is_file()]
        if found:
            self.issues.add("blocking", "input-misplaced", rel, i, name,
                            "\\input{%s} does not resolve; the file is at %s (fix the path or move it in "
                            "Overleaf). Its content is still checked below."
                            % (name, self.rel(found[0])))
            self._walk(found[0], seen)
        else:
            self.issues.add("blocking", "input-missing", rel, i, name,
                            "\\input{%s}: no such file in the folder; nothing of it reaches the PDF" % name)


class Issues:
    def __init__(self):
        self.items = []

    def add(self, severity, check, rel, line, key, message, grouped=False):
        self.items.append(OrderedDict([
            ("severity", severity), ("check", check), ("file", rel), ("line", line),
            ("key", "%s|%s" % (check, Path(rel).name if grouped else key)), ("message", message)]))

    def distinct(self, severity):
        return len({it["key"] for it in self.items if it["severity"] == severity})


# --------------------------------------------------------------------------- bib

def parse_bib(text):
    """{key: {field: value, '_type': type}} with brace-balanced values."""
    out, dups = OrderedDict(), []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype = m.group(1).lower()
        if etype in ("comment", "string", "preamble"):
            continue
        depth, j = 1, m.end()
        start = m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        body = text[start:j - 1]
        fields = {"_type": etype}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            k, pos = fm.group(1).lower(), fm.end()
            if pos >= len(body):
                continue
            if body[pos] == "{":
                d, q = 1, pos + 1
                while q < len(body) and d:
                    d += {"{": 1, "}": -1}.get(body[q], 0)
                    q += 1
                val = body[pos + 1:q - 1]
            elif body[pos] == '"':
                q = body.find('"', pos + 1)
                val = body[pos + 1:q]
            else:
                val = re.match(r"[^,\s}]*", body[pos:]).group(0)
            fields.setdefault(k, " ".join(val.split()))
        if m.group(2) in out:
            dups.append(m.group(2))
        out[m.group(2)] = fields
    return out, dups


def clean(s):
    return re.sub(r"[{}\\]", "", s or "").strip()


def check_bibs(paper, issues):
    entries = {}
    used_fallback = False
    for name, rel, i in paper.bibs:
        path = paper.resolve(name, exts=("",))
        if path is None and not paper.is_repo and (REPO_PAPER / name).is_file():
            path, used_fallback = REPO_PAPER / name, True
            issues.add("info", "bib-not-in-folder", rel, i, name,
                       "%s is not in the folder; cite keys checked against the repo copy %s"
                       % (name, path))
        if path is None:
            issues.add("blocking", "bib-missing", rel, i, name, "%s not found: every \\cite is unresolved" % name)
            continue
        parsed, dups = parse_bib(path.read_text(errors="replace"))
        for k in dups:
            issues.add("major", "bib-duplicate-key", name, 0, k, "key %s defined twice in %s" % (k, name))
        for k, v in parsed.items():
            if k in entries:
                issues.add("major", "bib-duplicate-key", name, 0, k, "key %s also defined in another .bib" % k)
            entries[k] = v
    cites = defaultdict(list)
    for rel, i, raw, code, body, _ in paper.lines:
        for cm in CITE_CMD.finditer(code):
            for key in [k.strip() for k in cm.group(1).split(",") if k.strip()]:
                if key != "*":
                    cites[key].append("%s:%d" % (rel, i))
    year_now = datetime.date.today().year
    for key, locs in sorted(cites.items()):
        rel, line = locs[0].rsplit(":", 1)
        if key not in entries:
            issues.add("blocking", "cite-missing", rel, int(line), key,
                       "\\cite key %s is in no .bib (%d use%s, renders as '?')"
                       % (key, len(locs), "" if len(locs) == 1 else "s"))
            continue
        e = entries[key]
        missing = [f for f in ("title", "author", "year") if not e.get(f) and not (f == "year" and e.get("date"))]
        if missing:
            issues.add("major", "bib-incomplete", rel, int(line), key,
                       "%s lacks %s" % (key, ", ".join(missing)))
        year = re.search(r"\d{4}", e.get("year", "") or e.get("date", ""))
        year = int(year.group(0)) if year else None
        if year and not 1800 <= year <= year_now:
            issues.add("major", "bib-year", rel, int(line), key, "%s has year %d" % (key, year))
        blob = " ".join(e.get(f, "") for f in ("eprint", "url", "doi", "note", "journal"))
        am = ARXIV_ID.search(blob) or (re.match(r"(\d{2})(\d{2})\.\d{4,5}$", e.get("eprint", "")))
        if am and year and year < 2000 + int(am.group(1)):
            issues.add("major", "bib-year", rel, int(line), key,
                       "%s: year %d is earlier than its arXiv id (20%s)" % (key, year, am.group(1)))
        km = re.search(r"_(\d{4})[a-z]?$", key)
        if km and year and int(km.group(1)) != year:
            issues.add("minor", "bib-key-year", rel, int(line), key,
                       "key says %s, entry says %d (check which version is cited)" % (km.group(1), year))
        venue = any(e.get(f) for f in ("journal", "booktitle", "publisher", "howpublished", "eprint",
                                        "doi", "url", "school", "institution", "note"))
        if not venue:
            issues.add("minor", "bib-no-venue", rel, int(line), key, "%s has no venue, DOI, arXiv id or URL" % key)
        title = clean(e.get("title"))
        if title and title.upper() == title and len(title) > 12:
            issues.add("minor", "bib-title-case", rel, int(line), key, "%s title is all caps" % key)
    return entries, cites, used_fallback


# --------------------------------------------------------------------------- checks

def check_refs(paper, issues):
    labels = defaultdict(list)
    refs = defaultdict(list)
    for rel, i, raw, code, body, _ in paper.lines:
        if DEF_LINE.search(code):
            continue
        for lm in LABEL_CMD.finditer(code):
            labels[lm.group(1).strip()].append((rel, i))
        for rm in REF_CMD.finditer(code):
            for key in [k.strip() for k in rm.group(1).split(",") if k.strip()]:
                refs[key].append((rel, i))
        tm = re.search(r"\b(Figure|Fig\.|Table|Tab\.|Section|Sec\.|Appendix|App\.|Equation|Eq\.)"
                       r"\s+\\(?:ref|cref)\{", code)
        if tm and body:
            issues.add("minor", "ref-tie", rel, i, "", "'%s \\ref' without '~' (line break can split them)"
                       % tm.group(1), grouped=True)
    for lab, locs in sorted(labels.items()):
        if len(locs) > 1:
            issues.add("blocking", "label-duplicate", locs[1][0], locs[1][1], lab,
                       "label %s defined %d times (%s): refs land on the last one"
                       % (lab, len(locs), ", ".join("%s:%d" % l for l in locs)))
    for key, locs in sorted(refs.items()):
        if key not in labels:
            issues.add("blocking", "ref-undefined", locs[0][0], locs[0][1], key,
                       "\\ref{%s} has no \\label in the compiled tree (%d use%s, renders as '??')"
                       % (key, len(locs), "" if len(locs) == 1 else "s"))
    for lab, locs in sorted(labels.items()):
        if lab not in refs and re.match(r"(fig|tab)[:_-]", lab):
            issues.add("minor", "float-unreferenced", locs[0][0], locs[0][1], lab,
                       "%s is never referenced in the text" % lab)
    return labels, refs


def check_graphics(paper, issues):
    has_images = any(p.suffix.lower() in GRAPHIC_EXTS[1:] and p.stem != paper.main.stem  # not main.pdf
                     for p in paper.folder.rglob("*.*"))
    if not has_images:
        issues.add("info", "folder-without-figures", paper.rel(paper.main), 0, "figures",
                   "the folder holds no image files: figures are checked against documents/paper, and "
                   "one missing there too may still exist in the Overleaf project")
    for rel, i, raw, code, body, _ in paper.lines:
        for gm in GRAPHIC_CMD.finditer(code):
            name = gm.group(1).strip()
            if paper.resolve(name, exts=GRAPHIC_EXTS, extra_dirs=paper.graphicspath):
                continue
            if not paper.is_repo and any((REPO_PAPER / d / (name + e)).is_file()
                                         for d in paper.graphicspath for e in GRAPHIC_EXTS):
                issues.add("info", "figure-not-in-folder", rel, i, name,
                           "%s is not in the folder (present in documents/paper)" % name)
            elif has_images:
                issues.add("blocking", "figure-missing", rel, i, name, "\\includegraphics{%s}: file not found" % name)
            else:
                issues.add("major", "figure-unverified", rel, i, name,
                           "\\includegraphics{%s}: not in the folder nor in documents/paper (ask whether the "
                           "Overleaf project has it)" % name)


def check_structure(paper, issues):
    main_text = paper.main.read_text(errors="replace")
    main_code = "\n".join(strip_comment(l) for l in main_text.splitlines())
    mrel = paper.rel(paper.main)
    m = re.search(r"\\usepackage\[([^\]]*)\]\{acl\}", main_code)
    if not m:
        issues.add("blocking", "acl-option", mrel, 0, "acl", "no \\usepackage[review]{acl}: not ARR format")
    elif "review" not in m.group(1):
        issues.add("blocking", "acl-option", mrel, 0, "acl",
                   "\\usepackage[%s]{acl}: submission needs [review] (anonymous, line numbers)" % m.group(1))
    for bm in re.finditer(r"\\bibliographystyle\{([^}]*)\}", main_code):
        issues.add("major", "bibstyle", mrel, main_code[:bm.start()].count("\n") + 1, "bibstyle",
                   "\\bibliographystyle{%s}: acl.sty already sets acl_natbib, a second style is a BibTeX "
                   "error; delete the line" % bm.group(1))
    body = [(rel, i, code) for rel, i, raw, code, b, _ in paper.lines if b]
    if not any(re.search(r"\\section\*?\{\s*Limitations\s*\}", c) for _, _, c in body):
        issues.add("blocking", "limitations", mrel, 0, "limitations",
                   "no \\section*{Limitations} in the compiled tree (ARR desk-rejects without one)")
    for rel, i, code in body:
        if re.search(r"\\section\*?\{\s*Acknowledge?ments?\s*\}", code, re.I) or \
                re.search(r"\\input\{[^}]*acknowledg", code):
            issues.add("blocking", "anonymity", rel, i, "acknowledgements",
                       "acknowledgements are compiled into a review version")
        sm = re.search(r"\\(?:sub)*section\*?\{([^}]*)\}", code)
        if sm and re.search(r"\b[A-Z]{3,}\s+[A-Z]{2,}\s+[A-Z]{2,}", sm.group(1)):
            issues.add("major", "draft-heading", rel, i, sm.group(1), "all-caps draft heading: %s" % sm.group(1))
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_code, re.S)
    if am:
        text = re.sub(r"\\[a-zA-Z]+\*?", " w ", am.group(1))
        words = len(re.findall(r"[A-Za-z0-9][\w.,%-]*", re.sub(r"[{}$\\~]", " ", text)))
        if words > ABSTRACT_MAX_WORDS:
            issues.add("major", "abstract-length", mrel, main_code[:am.start()].count("\n") + 1, "abstract",
                       "abstract has ~%d words (ARR/ACL ask for at most %d)" % (words, ABSTRACT_MAX_WORDS))


def check_prose(paper, issues):
    vocab = Counter()
    for rel, i, raw, code, body, math in paper.lines:
        if not body and not code.lstrip().startswith("\\title"):
            continue
        if DEF_LINE.search(code):
            continue
        for dm in DRAFT_MACRO.finditer(code):
            issues.add("blocking", "draft-macro", rel, i, "\\" + dm.group(1),
                       "leftover \\%s renders in the PDF" % dm.group(1), grouped=True)
        if re.search(r"\\(?:textcolor|color)\{", code) and not DRAFT_MACRO.search(code):
            issues.add("minor", "coloured-text", rel, i, "", "coloured text (draft marker or legend swatch?)",
                       grouped=True)
        pv = prose_view(code)
        if not math:
            for pm in PLACEHOLDER.finditer(pv):
                issues.add("blocking", "placeholder", rel, i, "", "placeholder '%s' in the text" % pm.group(0),
                           grouped=True)
        # anonymity: on the code (math kept), \cite arguments removed
        acode = CITE_CMD.sub(" ", code)
        for am in ANON_BLOCKING.finditer(acode):
            issues.add("blocking", "anonymity", rel, i, am.group(0).lower(),
                       "identifying string '%s'" % am.group(0))
        for am in ANON_CHECK.finditer(acode):
            issues.add("major", "anonymity-check", rel, i, "",
                       "'%s': confirm it does not identify the authors (third-person self-citation is fine)"
                       % am.group(0), grouped=True)
        for am in ANON_INFO.finditer(acode):
            issues.add("info", "anonymity-mention", rel, i, "", "'%s' named (fine as a cited recipe)"
                       % am.group(0), grouped=True)
        if L_NOTATION.search(code) and body:
            issues.add("major", "notation-K-vs-L", rel, i, "",
                       "'%s': the paper writes the number of languages as $K$, the repo as L"
                       % L_NOTATION.search(code).group(0), grouped=True)
        if math:
            continue
        # punctuation (RULES.md "Paper punctuation")
        for d in re.finditer(r"---|\u2014|\u2013", pv):
            issues.add("minor", "dash", rel, i, "", "em/en dash '%s' as punctuation" % d.group(0), grouped=True)
        for d in re.finditer(r"(?<!-)--(?!-)", pv):
            left, right = pv[:d.start()], pv[d.end():]
            if re.search(r"(^|&)\s*$", left) and re.match(r"\s*(&|\\\\|$)", right):
                continue  # empty table cell
            if re.search(r"\d[\d.,]*\s*(?:[KMBkx%]|\\%|MATH|\})?\s*$", left) and \
                    re.match(r"\s*(?:\$|\{|\\[a-z]+\s*|[A-Za-z]{1,2})?[\d.]", right):
                continue  # numeric range
            if re.search(r"MATH\s*$", left) and re.match(r"\s*MATH", right):
                continue
            if re.search(r"\w$", left) and re.match(r"\w", right):
                issues.add("minor", "dash-compound", rel, i, "",
                           "en dash between words '%s--%s' (RULES allows -- only for numeric ranges and "
                           "empty cells; use a hyphen or 'and')"
                           % (re.search(r"\S*$", left).group(0)[-20:], re.match(r"\S*", right).group(0)[:20]),
                           grouped=True)
            else:
                issues.add("minor", "dash", rel, i, "", "'--' as prose punctuation", grouped=True)
        for _ in re.finditer(r"\s-\s", pv):
            issues.add("minor", "dash", rel, i, "", "spaced hyphen ' - ' as punctuation", grouped=True)
        for _ in re.finditer(r";", pv):
            issues.add("minor", "semicolon", rel, i, "", "';' in prose (RULES: comma, parentheses or new sentence)",
                       grouped=True)
        for rm in re.finditer(r"\bfrom \$?[\d.,]+\$?\s*(?:[KMB]\b|\\?%)?\s*to \$?\d", pv):
            issues.add("minor", "range-style", rel, i, "",
                       "'%s...': RULES writes ranges as X--Y" % rm.group(0), grouped=True)
        low = pv.lower()
        for w in re.findall(r"[a-z][a-z-]+", low):
            vocab[w] += 1
        for w in AI_LINGO:
            for _ in re.finditer(r"(?<![\w-])%s(?![\w-])" % re.escape(w), low):
                issues.add("minor", "ai-lingo", rel, i, "", "AI-lingo word '%s'" % w, grouped=True)
        for ph in AI_PHRASES:
            for pm in re.finditer(ph, low):
                issues.add("minor", "ai-lingo", rel, i, "", "AI-lingo phrase '%s'" % pm.group(0)[:60],
                           grouped=True)
        for rm in re.finditer(r"\b([A-Za-z]+)\s+\1\b", re.sub(r"\\[a-zA-Z]+", " ", pv), re.I):
            if rm.group(1).lower() not in ("that", "had"):
                issues.add("minor", "typo-repeated-word", rel, i, rm.group(0).lower(),
                           "repeated word '%s'" % rm.group(0))
        for w in re.findall(r"[a-z]+", low):
            if w in MISSPELLINGS:
                issues.add("minor", "typo", rel, i, w, "'%s' -> '%s'" % (w, MISSPELLINGS[w]))
    mixed = []
    for brit, amer in SPELLING_PAIRS:
        if any(w.startswith(brit) for w in vocab) and any(w.startswith(amer) for w in vocab):
            mixed.append("%s/%s" % (brit, amer))
    for w in vocab:
        m = re.match(r"(.+)is(e|ed|es|ing|ation|ations)$", w)
        if m and len(m.group(1)) > 3 and m.group(1) + "iz" + m.group(2) in vocab:
            mixed.append("%s/%s" % (w, m.group(1) + "iz" + m.group(2)))
    if mixed:
        issues.add("minor", "spelling-mixed", paper.rel(paper.main), 0, "spelling",
                   "both British and American spellings: %s (pick one)" % ", ".join(sorted(set(mixed))))


def check_against_repo(paper, issues):
    """Overleaf export only: generated constants and pages that differ from the repo copy."""
    if paper.is_repo or not (REPO_PAPER / "sections" / "main.tex").is_file():
        return
    def blocks(text):
        return {k: dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", b))
                for k, b in re.findall(r"% BEGIN generated: (\S+).*?\n(.*?)% END generated", text, re.S)}
    mine = blocks(paper.main.read_text(errors="replace"))
    repo = blocks((REPO_PAPER / "sections" / "main.tex").read_text(errors="replace"))
    for key, macros in mine.items():
        for name, val in macros.items():
            rv = repo.get(key, {}).get(name)
            if rv is not None and rv.strip() != val.strip():
                issues.add("major", "constant-stale", paper.rel(paper.main), 0, name,
                           "\\%s = %s here, %s in the repo's generated block '%s' (refresh one or the other)"
                           % (name, val.strip(), rv.strip(), key))
    repo_tex = {p.name: p for p in (REPO_PAPER / "sections").rglob("*.tex")}
    for path in paper.files:
        if any(fnmatch.fnmatch(path.name, pat) for pat in GENERATED_TEX) and path.name in repo_tex:
            a = path.read_text(errors="replace").splitlines()
            b = repo_tex[path.name].read_text(errors="replace").splitlines()
            if a != b:
                diff = len(set(a) ^ set(b))
                issues.add("info", "generated-differs", paper.rel(path), 0, path.name,
                           "generated page differs from the repo copy (%d lines differ): stale export or "
                           "newer analysis, check its numbers" % diff)


def orphans(paper, issues):
    compiled = {p.resolve() for p in paper.files}
    for p in sorted(paper.folder.rglob("*.tex")):
        if p.resolve() not in compiled and p.name != "main.tex":
            issues.add("info", "tex-not-compiled", paper.rel(p), 0, p.name, "not reached from main.tex")


# --------------------------------------------------------------------------- output

def dump(paper, entries, cites, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cited.tsv").open("w") as f:
        f.write("key\ttype\tyear\tauthors\ttitle\tvenue\tid\tn_cites\tfirst_use\n")
        for key, locs in sorted(cites.items()):
            e = entries.get(key, {})
            authors = [a.split(",")[0].strip() for a in clean(e.get("author")).split(" and ")][:3]
            venue = clean(e.get("booktitle") or e.get("journal") or e.get("publisher") or e.get("howpublished"))
            ident = clean(e.get("doi") or e.get("eprint") or e.get("url"))
            f.write("\t".join([key, e.get("_type", "MISSING"), clean(e.get("year")), "; ".join(authors),
                               clean(e.get("title")), venue, ident, str(len(locs)), locs[0]]) + "\n")
    with (out_dir / "numbers.tsv").open("w") as f:
        f.write("file\tline\tnumber\tcontext\n")
        for rel, i, raw, code, body, math in paper.lines:
            if not body:
                continue
            for nm in re.finditer(r"(?<![\w.{])(\d[\d,]*\.\d+|\d+(?:\.\d+)?\s*\\?%|\d{2,}[\d,]*)(?![\w}])", code):
                ctx = code[max(0, nm.start() - 70):nm.end() + 70].strip().replace("\t", " ")
                f.write("%s\t%d\t%s\t%s\n" % (rel, i, nm.group(0).strip(), ctx))


def compare(prev_path, items):
    prev = json.loads(Path(prev_path).read_text())["issues"]
    old = Counter(p["key"] for p in prev if p["severity"] != "info")
    new = Counter(it["key"] for it in items if it["severity"] != "info")
    return {"fixed": sorted(k for k in old if k not in new),
            "still_open": sorted(k for k in old if k in new),
            "new": sorted(k for k in new if k not in old)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Overleaf export folder or documents/paper")
    ap.add_argument("--json", type=Path, help="also write the issues and summary as JSON here")
    ap.add_argument("--prev", type=Path, help="review-l1.json of the previous version: list fixed/open/new")
    ap.add_argument("--dump-dir", type=Path, help="write cited.tsv and numbers.tsv here (for the L1 agents)")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    issues = Issues()
    paper = Paper(args.folder, issues)
    check_structure(paper, issues)
    check_refs(paper, issues)
    entries, cites, _ = check_bibs(paper, issues)
    check_graphics(paper, issues)
    check_prose(paper, issues)
    check_against_repo(paper, issues)
    orphans(paper, issues)

    order = {"blocking": 0, "major": 1, "minor": 2, "info": 3}
    items = sorted(issues.items, key=lambda it: (order[it["severity"]], it["check"], it["file"], it["line"]))
    if not args.quiet:
        for it in items:
            print("%s\t%s\t%s:%s\t%s" % (it["severity"], it["check"], it["file"], it["line"], it["message"]))
    raw = Counter(it["severity"] for it in items)
    dist = {s: issues.distinct(s) for s in order}
    by_check = Counter((it["severity"], it["check"]) for it in items)
    stop = (dist["blocking"] >= STOP_BLOCKING or dist["blocking"] + dist["major"] >= STOP_BLOCKING_MAJOR
            or dist["blocking"] + dist["major"] + dist["minor"] >= STOP_TOTAL)
    summary = OrderedDict([
        ("folder", str(paper.folder)), ("main", paper.rel(paper.main)), ("files_compiled", len(paper.files)),
        ("cite_keys", len(cites)), ("bib_entries", len(entries)),
        ("raw", {s: raw.get(s, 0) for s in order}), ("distinct", dist),
        ("by_check", {"%s/%s" % k: v for k, v in sorted(by_check.items())}),
        ("mechanical_gate", "STOP" if stop else "CONTINUE"),
        ("thresholds", "distinct blocking >= %d, blocking+major >= %d, or all >= %d"
         % (STOP_BLOCKING, STOP_BLOCKING_MAJOR, STOP_TOTAL))])
    print("# folder %s (main %s, %d files compiled, %d cite keys, %d bib entries)"
          % (paper.folder, summary["main"], len(paper.files), len(cites), len(entries)))
    print("# raw hits: " + ", ".join("%s=%d" % (s, raw.get(s, 0)) for s in order))
    print("# distinct: " + ", ".join("%s=%d" % (s, dist[s]) for s in order))
    print("# by check: " + ", ".join("%s=%d" % kv for kv in summary["by_check"].items()))
    print("# mechanical gate: %s (%s)" % (summary["mechanical_gate"], summary["thresholds"]))
    result = OrderedDict([("summary", summary), ("issues", items)])
    if args.prev:
        diff = compare(args.prev, items)
        result["vs_previous"] = diff
        print("# vs previous: fixed=%d still_open=%d new=%d"
              % (len(diff["fixed"]), len(diff["still_open"]), len(diff["new"])))
        for k in diff["fixed"]:
            print("# FIXED\t" + k)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    if args.dump_dir:
        dump(paper, entries, cites, args.dump_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

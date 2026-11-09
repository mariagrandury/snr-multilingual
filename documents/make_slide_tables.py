"""Render every table of the paper to a PNG for the slides (slides_paper.md), one
standalone LaTeX file per table, with the paper's own preamble and macros.
Captions, labels, citations and cross-references are dropped: the slide title
carries the caption. Needs tectonic and pdftoppm (Homebrew: tectonic, poppler).

    python3 make_slide_tables.py            # from documents/, writes public/tables/
"""
import pathlib, re, shutil, subprocess, tempfile
HERE = pathlib.Path(__file__).resolve().parent
paper, out = HERE / "paper", HERE / "public" / "tables"
build = pathlib.Path(tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True)
FILES = ["app_01_model_ladder", "rqs/app_02_table_languages", "app_03_evaluation", "app_04_chance_threshold",
         "app_05_ranking_agreement", "app_06_snr_new", "app_07_snr_monolingual", "rqs/app_rq09_benchmark_design",
         "app_hf_model_card"]

def nocomments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in s.splitlines())

def inputs(s):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: nocomments((paper / (m.group(1) + ".tex")).read_text()), s)

def drop_cmd(s, cmd):
    """remove \\cmd{...} with balanced braces"""
    out, i = [], 0
    while (j := s.find("\\" + cmd + "{", i)) >= 0:
        out.append(s[i:j]); k, depth = j + len(cmd) + 2, 1
        while depth: depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        i = k
    return "".join(out) + s[i:]

pre = (paper / "sections/main.tex").read_text().split("\\begin{document}")[0]
pre = nocomments(pre)
pre = pre.replace("\\documentclass[11pt]{article}", "\\documentclass[11pt,varwidth=30cm,border=4pt]{standalone}\n\\usepackage{xcolor}")
pre = re.sub(r"\\usepackage(\[[^]]*\])?\{(acl|hyperref)\}", "", pre)
pre = re.sub(r"\\title\{.*", "", pre, flags=re.S)
pre += ("\\usepackage{xcolor}\n\\setlength{\\textwidth}{16cm}\n"
        "\\newcommand{\\citep}[2][]{}\\newcommand{\\citet}[2][]{}\\newcommand{\\citealp}[2][]{}"
        "\\renewcommand{\\cite}[2][]{}\\renewcommand{\\citeneeded}[1]{}\\renewcommand{\\ref}[1]{}\\newcommand{\\href}[2]{#2}\n")

n = 0
for f in FILES:
    s = inputs(nocomments((paper / "sections" / (f + ".tex")).read_text()))
    for m in re.finditer(r"\\begin\{(table\*?)\}(?:\[[^]]*\])?(.*?)\\end\{\1\}", s, re.S):
        body = m.group(2)
        lab = (re.findall(r"\\label\{([^}]*)\}", body) or ["table"])[0]
        body = drop_cmd(drop_cmd(body, "caption"), "label")
        n += 1; name = f"table{n:02d}_{lab.split(':')[-1]}"
        tex = pre + "\\begin{document}\n" + body + "\n\\end{document}\n"
        (build / f"{name}.tex").write_text(tex)
        r = subprocess.run(["tectonic", "-c", "minimal", f"{name}.tex"], cwd=build, capture_output=True, text=True)
        if r.returncode:
            print("FAIL", name, r.stderr[-400:]); continue
        subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", f"{name}.pdf", str(out / name)], cwd=build)
        print("ok", name)
shutil.rmtree(build)

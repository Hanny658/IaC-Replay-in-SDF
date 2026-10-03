"""Build the arXiv submission bundle from report/preprint.tex.

    python scripts/build_arxiv_bundle.py

Writes report/arxiv_bundle/ (gitignored) containing main.tex, arxiv.sty, figs/ and
arxiv_bundle.zip, plus arxiv_metadata.txt with the fields the submission form asks for.

The working copy of preprint.tex is kept anonymous, because the repository feeds an
anonymous mirror.  This script produces the de-anonymised submission copy: it prepends \\pdfoutput=1 (so arXiv runs pdflatex), fills in
the author block, and swaps the anonymous code link for the public one.  Nothing else in the
document is altered.

The identifying values are NOT stored in the repository.  They are read from
report/arxiv_private.json (gitignored), which must look like

    {"authors": ["First Author", "email@example.org", "Affiliation, Country"],
     "code_url": "https://github.com/<owner>/<repo>"}

Each "authors" entry becomes one line of the \\author block.
"""
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "report", "preprint.tex")
OUT = os.path.join(ROOT, "report", "arxiv_bundle")
PRIVATE = os.path.join(ROOT, "report", "arxiv_private.json")

ANON_AUTHOR = """\\author{%
  Anonymous author(s)\\\\
  Affiliation withheld for double-blind review}"""
ANON_CODE = "https://anonymous.4open.science/r/IaC-CL-with-no-offline"


def load_private():
    if not os.path.isfile(PRIVATE):
        sys.exit(f"missing {PRIVATE}; see the module docstring for its format")
    cfg = json.load(open(PRIVATE, encoding="utf-8"))
    lines = cfg["authors"]
    block = "\\author{%\n" + " \\\\\n".join(f"  {l}" for l in lines) + "}"
    return block, lines, cfg["code_url"]

# The arXiv abstract field takes at most 1,920 characters, so the form gets this condensed
# version while the PDF keeps the full one.  Authored by hand; keep it in sync with the paper.
FORM_ABSTRACT = """Brains consolidate memories not only in sleep but also through local sleep: brief, use-dependent off-periods of individual circuits during wakefulness. Replay-based continual learning, by contrast, consolidates either in a dedicated offline phase or interleaved with the input stream. We ask whether a network trained by local, biologically constrained rules can consolidate with no offline phase. An isolation rule confines replay updates to hidden synapses invisible to the current input under k-winner-take-all dynamics, with optimiser state advanced only inside the mask; a refractory rotation rule makes units that have just fired sit out the next competition, widening the consolidable set. This inverts the usual direction of non-interfering continual learning: the hidden computation on the current input is held invariant (exactly on the proven channels, and for all but 0.3% of waking samples per update as implemented) while past memories are written into the degrees of freedom the current batch leaves unused. On class-incremental split-MNIST the system reaches 91.6+-0.3% with no offline phase, at or above the best offline night on two held-out splits and above experience replay, ER-ACE, A-GEM and unmasked local replay, each re-tuned under the same micro-batch schedule; against DER++ the comparison splits by protocol, DER++ leading by 1.5 points when the data is seen five times and the system by 1.6 when it is seen once, where the night falls to 76.9%. Measured while the stream runs, isolation holds the served prediction still (0.33% of predictions change per replay update against 4.39% unmasked) and rotation keeps the retention curve flat (a 1.5-point stability gap against about 16), while offline rehearsal carries 38. The advantage is largest at small buffers. Rotation also transfers to a backpropagation network with k-WTA hidden layers, where isolation is again free."""

CATEGORIES = "primary cs.LG; cross-list cs.NE, stat.ML (as submitted in v1; a replacement keeps them)"
V2_NOTE = ("v2: the published replay baselines are re-tuned and re-run under the "
           "paper's own micro-batch schedule, and a new section measures online "
           "accuracy, the stability gap and the churn of the served prediction while "
           "the stream runs; the short version of this work was accepted at the "
           "NeurIPS 2026 Workshop on Continual Learning in the Era of Foundation "
           "Models and Embodied Agents.")
LICENSE = "arXiv.org perpetual, non-exclusive license (or CC BY 4.0 if you want reuse)"


def detex(s):
    """LaTeX abstract to plain text, good enough for the submission form."""
    s = s.replace("\\kwta{}", "k-WTA").replace("\\%", "%")
    s = re.sub(r"\\(?:emph|textbf|textit|text)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\citep?\{[^}]*\}", "", s)
    s = s.replace("$k$", "k").replace("\\pm", "+-")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)          # strip inline math delimiters
    s = s.replace("\\,", " ").replace("\\ ", " ").replace("~", " ")
    s = s.replace("--", "-").replace("\\&", "&").replace("\\and", ";")
    s = re.sub(r"\\[a-zA-Z]+", "", s)             # any command left over
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def main():
    tex = open(SRC, encoding="utf-8").read()
    author_block, author_lines, public_code = load_private()

    # --- de-anonymise
    assert tex.count(ANON_AUTHOR) == 1, "author block not found; was preprint.tex edited?"
    tex = tex.replace(ANON_AUTHOR, author_block)
    assert tex.count(ANON_CODE) == 1, "anonymous code link not found"
    tex = tex.replace(ANON_CODE, public_code)
    assert not tex.lstrip().startswith("\\pdfoutput"), "pdfoutput already present"
    tex = "\\pdfoutput=1\n" + tex

    # --- lay out the bundle
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figs"))
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8", newline="\n").write(tex)
    shutil.copy2(os.path.join(ROOT, "report", "arxiv.sty"), os.path.join(OUT, "arxiv.sty"))
    figs = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{figs/([^}]+)\}", tex)))
    for f in figs:
        shutil.copy2(os.path.join(ROOT, "report", "figs", f), os.path.join(OUT, "figs", f))

    # --- metadata
    title = re.search(r"\\title\{(.+?)\}\n", tex).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    keywords = detex(re.search(r"\\keywords\{(.*?)\}\n", tex).group(1))
    full_abstract = detex(abstract)
    assert len(FORM_ABSTRACT) <= 1920, f"form abstract is {len(FORM_ABSTRACT)} characters"
    try:
        import pypdf
        pages = len(pypdf.PdfReader(os.path.join(ROOT, "report", "preprint.pdf")).pages)
    except Exception:
        pages = "?"
    n_fig = len(re.findall(r"\\label\{fig:", tex))
    meta = f"""TITLE
{title}

AUTHORS (as in main.tex)
{chr(10).join(author_lines)}

ABSTRACT FOR THE ARXIV FORM (condensed to the 1,920-character limit; the PDF keeps the full abstract)
{FORM_ABSTRACT}

FULL ABSTRACT (plain text, {len(full_abstract)} characters, too long for the form)
{full_abstract}

KEYWORDS
{keywords}

COMMENTS (suggested)
{pages} pages, {n_fig} figures. Code: {public_code}
{V2_NOTE}

CATEGORIES (suggested)
{CATEGORIES}

LICENSE (suggested)
{LICENSE}

BUNDLE
upload arxiv_bundle.zip (main.tex, arxiv.sty, figs/*.pdf); arXiv compiles with pdflatex because of \\pdfoutput=1
"""
    open(os.path.join(OUT, "arxiv_metadata.txt"), "w", encoding="utf-8", newline="\n").write(meta)

    # --- zip
    zpath = os.path.join(OUT, "arxiv_bundle.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(OUT, "main.tex"), "main.tex")
        z.write(os.path.join(OUT, "arxiv.sty"), "arxiv.sty")
        for f in figs:
            z.write(os.path.join(OUT, "figs", f), f"figs/{f}")
    print(f"wrote {OUT}")
    print(f"  main.tex  {os.path.getsize(os.path.join(OUT, 'main.tex')) / 1024:.0f} KB")
    print(f"  figs      {len(figs)}: {', '.join(figs)}")
    print(f"  zip       {os.path.getsize(zpath) / 1024:.0f} KB")
    print(f"  metadata  form abstract {len(FORM_ABSTRACT)} chars, {pages} pages, {n_fig} figures")


if __name__ == "__main__":
    main()

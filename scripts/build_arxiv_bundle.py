"""Build the arXiv submission bundle from report/preprint.tex.

    python scripts/build_arxiv_bundle.py

Writes report/arxiv_bundle/ (gitignored) containing main.tex, arxiv.sty, figs/ and
arxiv_bundle.zip, plus arxiv_metadata.txt with the fields the submission form asks for.

The working copy of preprint.tex is kept anonymous, because the workshop version is under
double-blind review and the public repository mirrors it.  This script produces the
de-anonymised submission copy: it prepends \\pdfoutput=1 (so arXiv runs pdflatex), fills in
the author block, and swaps the anonymous code link for the public one.  Nothing else in the
document is altered.
"""
import os
import re
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "report", "preprint.tex")
OUT = os.path.join(ROOT, "report", "arxiv_bundle")

AUTHOR_BLOCK = """\\author{%
  Zhang Yanhai \\\\
  zh0010ai@e.ntu.edu.sg \\\\
  Nanyang Technological University, Singapore}"""
ANON_AUTHOR = """\\author{%
  Anonymous author(s)\\\\
  Affiliation withheld for double-blind review}"""
ANON_CODE = "https://anonymous.4open.science/r/IaC-CL-with-no-offline"
PUBLIC_CODE = "https://github.com/Hanny658/IaC-Replay-in-SDF"

# The arXiv abstract field takes at most 1,920 characters, so the form gets this condensed
# version while the PDF keeps the full one.  Authored by hand; keep it in sync with the paper.
FORM_ABSTRACT = """Replay-based continual learning almost always consolidates in a dedicated offline phase or by interleaving replayed samples with the input stream, whereas brains also consolidate during wakefulness through local sleep, brief use-dependent off-periods of individual circuits. We ask whether a network trained by local, biologically constrained rules can consolidate with no offline phase at all. An isolation rule confines replay updates to hidden synapses invisible to the current input under k-winner-take-all dynamics, with optimiser state advanced only inside the mask; a refractory rotation rule makes units that have just fired sit out the next competition, widening the consolidable set; a homeostatic pressure and a relative-novelty gate decide when replay bursts fire and when rotation runs. This inverts the usual direction of non-interfering continual learning: the hidden computation on the current input is held invariant (exactly on the proven channels, and for all but 0.3% of waking samples per update elsewhere) while past memories are written into the degrees of freedom the current batch leaves unused. On class-incremental split-MNIST the system reaches 91.6+-0.3% with no offline phase, at or above the best offline-night schedule on two held-out splits, tied with DER++ and above experience replay, ER-ACE, A-GEM and unmasked local replay; in a single pass it leads DER++ (91.8% against 90.1%) while the night falls to 76.9%. The advantage is largest at small buffers and gives way to the backpropagation references at large ones; on split CIFAR-10 the system leads offline rehearsal and experience replay but trails ER-ACE and DER++. Rotation carries most of the gain; isolation adds the invariance guarantee. The mechanism is not tied to the local rule: under the same schedule a backpropagation network with k-WTA hidden layers gains from rotation, and isolation is again free on top of it."""

CATEGORIES = "primary cs.LG; cross-list cs.NE, q-bio.NC"
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

    # --- de-anonymise
    assert tex.count(ANON_AUTHOR) == 1, "author block not found; was preprint.tex edited?"
    tex = tex.replace(ANON_AUTHOR, AUTHOR_BLOCK)
    assert tex.count(ANON_CODE) == 1, "anonymous code link not found"
    tex = tex.replace(ANON_CODE, PUBLIC_CODE)
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
Zhang Yanhai
zh0010ai@e.ntu.edu.sg
Nanyang Technological University, Singapore

ABSTRACT FOR THE ARXIV FORM (condensed to the 1,920-character limit; the PDF keeps the full abstract)
{FORM_ABSTRACT}

FULL ABSTRACT (plain text, {len(full_abstract)} characters, too long for the form)
{full_abstract}

KEYWORDS
{keywords}

COMMENTS (suggested)
{pages} pages, {n_fig} figures. Code: {PUBLIC_CODE}

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

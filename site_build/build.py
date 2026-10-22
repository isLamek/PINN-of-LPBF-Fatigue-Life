"""
Build the static edition of the app into site/.

The Streamlit app stays as it is; this writes the same pages as plain HTML that GitHub
Pages serves directly, so they open at once instead of waiting for a server to wake.
Every sentence that carries a number is written here, from the same result files in
data/ the Streamlit pages read, and every chart is built with the same Plotly code.
The browser only draws, animates and, on the validation page, evaluates the three life
models the reader drives with sliders; build-time values of those models are written
alongside so the browser's arithmetic can be checked against Python's.

    .venv/Scripts/python.exe site_build/build.py
"""
import base64
import datetime
import html
import io
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots  # noqa: F401  (kept for parity with the pages)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("streamlit").setLevel(logging.ERROR)

from src import ct_viewer as cv  # noqa: E402
from src import ui  # noqa: E402  (registers the "thesis" chart template)
from src.data import (FOLD_DICE, GAUGE_ND, GAUGE_SA, HELDOUT_SIZE_R2,  # noqa: E402
                      STUDY_METRICS, load, splits)
from src.life_integral import C_PARIS, M_PARIS, Y_SURFACE, life_numeric  # noqa: E402
from src.ui import GREEN, GREY, MUTED, NAVY, RED, RULE, TITLE  # noqa: E402
from src.figstyle import H_AUX, H_MAIN, LEGEND, MEAS, PARITY, cycle_ticks, parity  # noqa: E402

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "site")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FONT = '"Source Sans 3", "Segoe UI", Arial, sans-serif'   # charts are apparatus: sans
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.1.1/plotly.min.js"
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
REPO = "https://github.com/isLamek/PINN-of-LPBF-Fatigue-Life"
STREAMLIT = "https://pinn-of-lpbf-fatigue-life.streamlit.app/"

pio.templates["thesis"].layout.font.family = FONT

PAGES = [("index", "Overview", None), ("defects", "Defects", 1),
         ("initiation", "Crack initiation", 2), ("life", "Fatigue life", 3),
         ("validation", "Validation", 4), ("conclusions", "Conclusions", None)]
CHAIN = [(1, "defects", "Find the defects"), (2, "initiation", "Where cracks start"),
         (3, "life", "The fatigue life"), (4, "validation", "Validation")]


# ====================================================================== helpers
def J(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def md(text):
    """A small Markdown subset: **bold**, lists, paragraphs. Maths in $...$ is kept
    verbatim for KaTeX."""
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return f"\x00{len(maths) - 1}\x00"

    t = re.sub(r"\$\$.+?\$\$|\$[^$\n]+?\$", keep, text.strip(), flags=re.S)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    out, buf, kind = [], [], None

    def flush():
        nonlocal buf, kind
        if not buf:
            return
        if kind == "p":
            out.append("<p>" + " ".join(buf) + "</p>")
        else:
            out.append(f"<{kind}>" + "".join(f"<li>{b}</li>" for b in buf) + f"</{kind}>")
        buf, kind = [], None

    for line in t.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        m_ul, m_ol = re.match(r"^- (.*)", s), re.match(r"^\d+\. (.*)", s)
        k = "ul" if m_ul else "ol" if m_ol else "p"
        if k != kind or k != "p" and False:
            flush()
            kind = k
        buf.append((m_ul or m_ol).group(1) if k != "p" else s)
    flush()
    res = "\n".join(out)
    return re.sub("\x00(\\d+)\x00", lambda m: maths[int(m.group(1))], res)


class Page:
    def __init__(self, slug):
        self.slug, self.parts, self.json = slug, [], {}
        self.n = 0
        self.sections = []        # (id, title) for the contents rail
        self.figno = 0            # figures are numbered in their captions
        self.after_chart = False

    def add(self, s):
        self.parts.append(s)
        self.after_chart = False

    def h1(self, t, dek=None):
        self.add(f"<h1>{t}</h1>" + (f'<p class="dek">{dek}</p>' if dek else ""))

    def h2(self, t):
        sid = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", t).lower()).strip("-")
        self.sections.append((sid, re.sub(r"<[^>]+>", "", t)))
        self.add(f'<h2 id="{sid}">{t}</h2>')

    def h3(self, t):
        self.add(f"<h3>{t}</h3>")

    def p(self, t):
        self.add(md(t))

    def caption(self, t):
        if self.after_chart:
            self.figno += 1
            t = f"<b>Figure {self.figno}.</b> {t}"
        self.add(f'<p class="caption">{t}</p>')

    def note(self, t):
        self.add(f'<div class="note">{md(t)}</div>')

    def finding(self, t):
        self.add(f'<div class="finding">{md(t)}</div>')

    def verdict(self, t, kind):
        self.add(f'<p class="verdict {kind}">{html.escape(t)}</p>')

    def spec(self, equation, scheme, verification, result):
        rows = (("Governing equation", equation), ("Numerical scheme", scheme),
                ("Verification", verification), ("Result", result))
        self.add('<dl class="spec">' + "".join(f"<dt>{a}</dt><dd>{md(b)}</dd>"
                                              for a, b in rows) + "</dl>")

    def uid(self, stem):
        self.n += 1
        return f"{stem}{self.n}"

    def data(self, obj, stem="d"):
        i = self.uid(stem)
        self.json[i] = json.dumps(obj, separators=(",", ":"), default=float)
        return i

    def fig_json(self, fig):
        i = self.uid("fig")
        self.json[i] = pio.to_json(fig, validate=False)
        return i

    def chart(self, fig, cid=None):
        fid = self.fig_json(fig)
        cid = cid or self.uid("chart")
        self.add(f'<div class="chart" id="{cid}" data-fig="{fid}" role="img" '
                 f'aria-label="chart"></div>')
        self.after_chart = True
        return cid

    def table(self, df, num=()):
        head = "".join(f'<th class="{"num" if c in num else ""}">{html.escape(str(c))}</th>'
                       for c in df.columns)
        rows = "".join("<tr>" + "".join(
            f'<td class="{"num" if c in num else ""}">{html.escape(str(v))}</td>'
            for c, v in zip(df.columns, r)) + "</tr>" for r in df.itertuples(index=False))
        self.add(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                 f"<tbody>{rows}</tbody></table></div>")

    def details(self, summary, inner):
        self.add(f"<details><summary>{summary}</summary>")
        inner()
        self.add("</details>")


def anim(fig, frame_ms, trans_ms=0, prefix=""):
    ui.play(fig, frame_ms=frame_ms, transition_ms=trans_ms, slider_prefix=prefix)
    fig.update_layout(meta=dict(frame_ms=frame_ms, trans_ms=trans_ms))
    return fig


def save_img(arr_or_im, rel, quality=84, width=None):
    from PIL import Image
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(arr_or_im)
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.convert("RGB").save(path, "JPEG", quality=quality, optimize=True)
    return rel.replace(os.sep, "/")


def chain_strip(n):
    items = []
    for k, slug, label in CHAIN:
        cur = ' aria-current="step"' if k == n else ""
        items.append(f'<li{cur}><a href="{slug}.html"><span class="k">{k}</span>'
                     f"{label}</a></li>")
    return f'<ol class="chain" aria-label="The model chain">{"".join(items)}</ol>'


def render_page(pg, title, desc):
    idx = [s for s, _, _ in PAGES].index(pg.slug)
    phase = PAGES[idx][2]
    nav = "".join(f'<a href="{s}.html"{" aria-current=" + chr(34) + "page" + chr(34) if s == pg.slug else ""}>'
                  f"{(str(n) + ' ') if n else ''}{lab}</a>" for s, lab, n in PAGES)
    prev = PAGES[idx - 1] if idx > 0 else None
    nxt = PAGES[idx + 1] if idx + 1 < len(PAGES) else None
    pn = '<nav class="pager">' + (
        f'<a class="prev" href="{prev[0]}.html"><span>Previous</span>{prev[1]}</a>'
        if prev else "<span></span>") + (
        f'<a class="next" href="{nxt[0]}.html"><span>Next</span>{nxt[1]}</a>'
        if nxt else "<span></span>") + "</nav>"
    scripts = "".join(f'<script type="application/json" id="{k}">{v}</script>'
                      for k, v in pg.json.items())
    built = datetime.date.today().strftime("%d %B %Y").lstrip("0")
    toc = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in pg.sections)
    favicon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'"
               "%3E%3Crect width='16' height='16' rx='3' fill='%231F4E79'/%3E%3Cpath d='M4 12 "
               "L8 4 L12 12' stroke='white' stroke-width='1.6' fill='none'/%3E%3C/svg%3E")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<link rel="icon" href="{favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="assets/site.css">
</head><body>
<a class="skip" href="#main">Skip to content</a>
<header class="masthead"><div class="inner">
<a class="mark" href="index.html">LPBF fatigue life</a>
<nav class="top" aria-label="Pages">{nav}</nav>
</div></header>
<div class="page">
<aside class="toc" aria-label="On this page"><p>On this page</p><ol>{toc}</ol></aside>
<article id="main">
{chain_strip(phase) if phase else ""}
{"".join(pg.parts)}
{pn}
</article>
</div>
<footer><p>Lamek Shali Indongo, University of Namibia, 2026. Every number on these pages is
read from the project's result files; built {built}.
<a href="{REPO}">Source</a> · <a href="{STREAMLIT}">Streamlit edition</a></p></footer>
{scripts}
<script src="{PLOTLY}"></script>
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"></script>
<script defer src="assets/site.js"></script>
</body></html>"""


# ====================================================================== CT images
def ct_frames():
    """Every slice of the two CT stacks as small JPEGs: raw, network overlay and the
    sized regions. Shared by the Phase 1 player, the Phase 2 viewer and the overview."""
    from src.sections.phase2 import _region_grid
    out = {}
    for sp in cv.available_specimens():
        rows = []
        for z in cv.list_slices(sp):
            ct = cv.load_ct_slice(sp, z)
            ov = cv.composite_slice(sp, z)
            if ct is None or ov is None:
                continue
            ab = cv.slice_abstraction(sp, z)
            mask = cv.load_mask_slice(sp, z)
            base = f"img/ct/{sp}_z{z:04d}"
            raw = save_img((ct * 255).astype("uint8"), base + "_ct.jpg", width=420)
            over = save_img(ov, base + "_ai.jpg", width=420)
            reg = (save_img(_region_grid(ab["rgb"], ab["regions"]) if ab and ab["regions"]
                            else ab["rgb"], base + "_reg.jpg", width=420) if ab else over)
            rows.append(dict(z=z, raw=raw, over=over, reg=reg, n=(ab or {}).get("n", 0),
                             px=int(mask.sum()) if mask is not None else 0,
                             blank=bool(ct.mean() < 0.05)))
        out[sp] = rows
        print(f"  CT {sp}: {len(rows)} slices")
    return out


SPEC_LABELS = {"SN_P017": "P017: largest governing defect of the 14",
               "SN_P059": "P059: most defects in its gauge"}


# ====================================================================== pages
def page_overview(ct):
    pg = Page("index")
    rows = load()
    cmp_d = J("comparison_model", "comparison_results.json") or {}
    h2 = J("phase2_peridynamics", "h2_first_bond.json")
    h4 = J("phase4_validation", "h4_calibration.json")
    bf = J("phase_mc_bayesian", "bayesian_results.json")
    mc = J("phase_mc_bayesian", "monte_carlo_results.json")
    lad = {r["model"].split()[0]: r for r in cmp_d.get("ladder", [])}
    path_a = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    p4 = pd.read_csv(os.path.join(DATA, "phase4_validation", "phase4_error_isolation.csv"))

    pg.h1(TITLE, "Predicting how long one 3D-printed metal part will last from the defects "
          "in its own CT scan, and testing that prediction against real fatigue tests.")
    pg.add('<p class="byline">Lamek Shali Indongo, BSc (Hons) Mechanical Engineering, '
           "University of Namibia, 2026</p>")
    pg.p("Metal parts made by laser powder bed fusion contain small internal defects. "
         "Fatigue cracks start at them, so parts printed the same way can last very "
         "different numbers of load cycles, and today a single safety factor covers the "
         "whole batch, set by the worst part ever tested. The pages that follow take one "
         "part's own defects to its own life, phase by phase, with each step's equation, "
         "numerical method and check against a known answer stated before its result.")

    pg.h2("The problem in the measured data")
    fig = go.Figure()
    for s, col in ((300, MEAS), (450, MEAS)):
        pts = sorted((r for r in rows if abs(r["smax"] - s) < 1), key=lambda r: r["Nf"])
        x = [s + (k / max(len(pts) - 1, 1) - 0.5) * 36 for k in range(len(pts))]
        fig.add_trace(go.Scatter(
            x=x, y=[r["Nf"] for r in pts], mode="markers", name=f"{s} MPa",
            marker=dict(size=8, color=col,
                        symbol=["triangle-up-open" if r["runout"] else "circle" for r in pts]),
            text=[f"{r['specimen']}: {r['Nf']:,} cycles" + (", run-out" if r["runout"] else "")
                  for r in pts], hoverinfo="text"))
    f450 = [r["Nf"] for r in rows if abs(r["smax"] - 450) < 1 and not r["runout"]]
    fig.update_layout(height=H_MAIN, yaxis_type="log", yaxis_title="cycles to failure",
                      xaxis=dict(tickvals=[300, 450], ticktext=["300 MPa", "450 MPa"],
                                 range=[255, 495], title="maximum stress"),
                      showlegend=False, margin=dict(t=10))
    pg.chart(fig)
    pg.p(f"All {len(rows)} fatigue-tested cylinders of the Oak Ridge dataset (open "
         f"triangles: run-outs, stopped unbroken at 10 million cycles). At 450 MPa, "
         f"{len(f450)} nominally identical cylinders failed between {min(f450):,} and "
         f"{max(f450):,} cycles: a **{max(f450) / min(f450):.0f}-fold** spread.")

    pg.h2("The method")
    shutil.copy2(os.path.join(ROOT, "assets", "framework_slide.png"),
                 os.path.join(OUT, "img", "framework.png"))
    pg.add('<figure><img src="img/framework.png" alt="The four-phase framework: CT scan, '
           'segmentation, defect catalogue, the life arm and the crack-path arm, and '
           'validation" loading="lazy" width="1600" height="900"></figure>')
    pg.p("One defect map feeds two arms. The **life arm** (navy) takes the size of the "
         "largest defect into a crack-growth integral and gives the fatigue life with its "
         "uncertainty. The **crack-path arm** (red) is a peridynamic model of the measured "
         "section; it shows where the crack starts and which way it runs, and is not used "
         "as a life. Every step is checked against a known answer before it touches "
         "measured data.")

    spec = "P017"
    pg.h2(f"One cylinder, from scan to life: {spec}")
    rec = {r["specimen"]: r for r in rows}[spec]
    pr = p4[p4["cylinder"] == spec].iloc[0]
    pred, meas = 10 ** float(pr["log10_N_pathA"]), rec["Nf"]
    sa_net = float(pr["largest_sqrt_area_um_ai_gauge"])
    h2row = next((r for r in h2["rows"] if r["specimen"] == f"SN_{spec}"), None) if h2 else None
    sl = [r for r in ct.get(f"SN_{spec}", []) if not r["blank"]]
    if sl:
        mid = sl[len(sl) // 2]
        pg.add(f'<div class="img-row two"><figure><img src="{mid["raw"]}" alt="X-ray CT '
               f'slice of {spec}" loading="lazy"><figcaption>X-ray CT slice</figcaption>'
               f'</figure><figure><img src="{mid["over"]}" alt="The same slice with the '
               f'defects the network found marked in red" loading="lazy"><figcaption>Defects '
               f"found by the network, in red</figcaption></figure></div>")
    ds = rec["ds"]
    dk = 0.65 * ds * np.sqrt(np.pi * sa_net * 1e-6)
    pg.p(f"1. **Scan.** {spec} is one of the 14 cylinders held out from all training.\n"
         f"2. **Defects.** The network finds {GAUGE_ND[spec]:.0f} defects in the gauge; the "
         f"largest measures √area = **{sa_net:,.0f} µm** (reference labels: "
         f"{GAUGE_SA[spec]:,} µm).\n"
         f"3. **Load.** Tested at {rec['smax']:.0f} MPa and R = {rec['R']:.1f}, so "
         f"Δσ = {ds:.0f} MPa and the driving force at that defect is "
         f"ΔK = 0.65 × {ds:.0f} × √(π × {sa_net / 1000:.3f} mm) = **{dk:.1f} MPa√m**.\n"
         f"4. **Life.** Predicted **{pred:,.0f}** cycles against **{meas:,}** measured "
         f"(ratio {pred / meas:.2f}), with nothing fitted to any measured life.\n" +
         (f"5. **Crack start.** The first bond breaks at a "
          f"{h2row['first_bond_defect_sqrt_area_um']:.0f} µm surface defect, not at the "
          f"{h2row['dominant_sqrt_area_um']:.0f} µm internal one that later carries the "
          "crack." if h2row else ""))

    pg.h2("Results against the hypotheses")
    res = []
    dice = STUDY_METRICS["Dice"]
    res.append(("H1  Segmentation accuracy", "mean Dice ≥ 0.80", f"{dice:.3f}",
                "met" if dice >= 0.80 else "not met"))
    if h2:
        res.append(("H2  Crack initiation site", "≥ 90% start at the largest defect",
                    f"{h2['hits_by_area']} of {h2['n']} ({h2['fraction_by_area']:.0%})",
                    "met" if h2["fraction_by_area"] >= h2["criterion"] else "not met"))
    if bf:
        r = bf["verify_recovery"]
        step = next((i for i, x in enumerate(r["history"], 1)
                     if abs(x["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
        ess = min(x["ess"] for x in r["history"])
        res.append(("H3  Bayesian convergence", "within 20% by inspection 3; ESS ≥ 250",
                    f"inspection {step}; ESS ≥ {ess:,.0f}",
                    "met" if step and step <= 3 and ess >= 250 else "partly met"))
    if h4:
        res.append(("H4  Predictive calibration", "≥ 90% inside; slope 0.90–1.10",
                    f"{h4['coverage']:.1%}; {h4['slope']:.3f}",
                    "met" if h4["slope_met"] and h4["coverage_met"] else "not met"))
    pg.table(pd.DataFrame(res, columns=["Hypothesis", "Criterion", "Result", "Outcome"]))
    if path_a and "B5" in lad and "C1" in lad:
        pg.p(f"The life prediction is within **{path_a:.2f}×** of the measured lives of the "
             f"14 held-out cylinders with nothing fitted to them. The best statistical fit "
             f"needs {lad['B5']['params']} parameters for {lad['B5']['loo_factor']:.2f}×, and "
             f"a physics-informed neural network {lad['C1']['params']} parameters for "
             f"{lad['C1']['loo_factor']:.2f}×.")
    v = (mc or {}).get("validation_independent")
    if v:
        pg.p(f"Propagated through the same model, the scatter in defect size and growth "
             f"rate reproduces the measured lives of the "
             f"{v['300']['n_tested'] + v['450']['n_tested']} cylinders not used to build it "
             f"(Kolmogorov–Smirnov p = {v['300']['ks_p']:.2f} at 300 MPa and "
             f"{v['450']['ks_p']:.2f} at 450 MPa).")
    pg.note("Every number on these pages is read from the project's result files when the "
            "site is built. Negative results are reported as they came out.")
    return pg, "LPBF fatigue life", ("Predicting the fatigue life of a laser powder bed "
                                    "fusion part from its own CT-measured defects.")


def page_defects(ct):
    pg = Page("defects")
    split = pd.read_csv(os.path.join(DATA, "00_docs", "Appendix_A_train_test_split.csv"))
    sp = split["split"].value_counts().to_dict()
    pg.h1("Phase 1: finding the defects",
          f"A neural network marks the defects in every CT slice, with a mean Dice of "
          f"{STUDY_METRICS['Dice']:.3f} against a criterion of 0.80.")
    pg.spec(
        r"A network $f_\theta$ maps a CT slice $x$ to a defect probability per pixel, "
        r"fitted by empirical risk minimisation, "
        r"$\theta^{*}=\arg\min_\theta \tfrac{1}{N}\sum_i \mathcal{L}\big(f_\theta(x_i),y_i\big)$, "
        r"with $\mathcal{L}$ the sum of Dice and cross-entropy losses. Agreement is scored by "
        r"$\mathrm{Dice}=2|P\cap G|/(|P|+|G|)$.",
        "nnU-Net v2, two-dimensional, trained 200 epochs on 16-bit slices (patch 768 × 768, "
        "batch 1). Every defect in the resulting mask is isolated and sized as √area.",
        f"Five-fold cross-validation (Dice {min(FOLD_DICE):.3f} to {max(FOLD_DICE):.3f} per "
        f"fold), and the governing defect size on {sp.get('test', 0)} cylinders never seen "
        f"in training against the reference labels: R² = {HELDOUT_SIZE_R2:.3f}.",
        f"Mean Dice **{STUDY_METRICS['Dice']:.3f}**: the 0.80 criterion of H1 is met.")
    pg.note("Dice measures the overlap between the predicted defect pixels and the reference "
            "labels, from 0 to 1. Defects fill a tiny fraction of each slice, which makes "
            "Dice a harsh score, so 0.80 already means close agreement.")

    pg.h2("The network on a held-out cylinder")
    specs = list(ct)
    opts = "".join(f'<option value="{pg.data([dict(src=[r["raw"], r["over"]], label=f"slice z{r["z"]:04d}") for r in ct[s] if not r["blank"]], "frames")}">'
                   f"{SPEC_LABELS.get(s, s)}</option>" for s in specs)
    pg.add(f'<div class="controls"><label for="p1-spec">Cylinder</label>'
           f'<select id="p1-spec" class="chooser" data-player="p1-player">{opts}</select></div>')
    first = re.search(r'value="([^"]+)"', opts).group(1)
    pg.add(f'<div class="player" id="p1-player" data-frames="{first}" data-ms="260">'
           '<div class="img-row two"><figure><img data-slot="0" alt="Raw CT slice">'
           '<figcaption>Raw CT</figcaption></figure><figure><img data-slot="1" alt="Defects '
           'the network finds, in red"><figcaption>Defects the network finds, in red'
           '</figcaption></figure></div><div class="controls"><button class="ctl" '
           'data-act="toggle" type="button">Pause</button><div class="slider-row">'
           '<input type="range" min="0" value="0" aria-label="Slice"><output></output>'
           "</div></div></div>")
    for s in specs:
        rows = [r for r in ct[s] if not r["blank"]]
        top = max(rows, key=lambda r: r["px"])
        pg.add(f'<p class="caption" data-show-for="p1-spec" data-key="'
               f'{re.findall(r"value=.([^\"]+).>" + re.escape(SPEC_LABELS.get(s, s)), opts)[0]}">'
               f"{len(rows)} slices of {s.replace('SN_', '')}, z{rows[0]['z']:04d} to "
               f"z{rows[-1]['z']:04d}. Defects appear and disappear as the slice passes "
               f"through them; the slice with the most defect pixels is z{top['z']:04d}.</p>")

    pg.h2("Accuracy on data it has never seen")
    pg.table(pd.DataFrame([
        ("Dice, five-fold mean", f"{STUDY_METRICS['Dice']:.3f}", "overlap with the reference"),
        ("Precision", f"{STUDY_METRICS['Precision']:.3f}",
         "share of predicted defect pixels that are defect"),
        ("Recall", f"{STUDY_METRICS['Recall']:.3f}", "share of true defect pixels found"),
        ("Intersection over union", f"{STUDY_METRICS['IoU']:.3f}", "a stricter overlap"),
        ("Held-out defect size, R²", f"{HELDOUT_SIZE_R2:.3f}",
         f"governing √area on {sp.get('test', 0)} unseen cylinders"),
    ], columns=["Metric", "Value", "Meaning"]), num=("Value",))
    mean = sum(FOLD_DICE) / len(FOLD_DICE)
    fig = go.Figure(go.Bar(x=[f"fold {i + 1}" for i in range(len(FOLD_DICE))], y=FOLD_DICE,
                           marker_color=NAVY, text=[f"{v:.3f}" for v in FOLD_DICE],
                           textposition="outside"))
    fig.add_hline(y=0.80, line_dash="dot", line_color=RED, annotation_text="criterion 0.80",
                  annotation_position="bottom right")
    fig.add_hline(y=mean, line_dash="dash", line_color=GREY,
                  annotation_text=f"mean {mean:.3f}", annotation_position="top left")
    fig.update_layout(yaxis_range=[0.75, 0.85], yaxis_title="Dice", xaxis_title="cross-validation fold", height=H_AUX,
                      margin=dict(t=20, b=10))
    pg.chart(fig)
    pg.caption("Dice by cross-validation fold. Every fold lands close to the mean: the "
               "result does not hang on one lucky split.")
    pg.h3("Why precision above recall is the safer error")
    pg.p(f"Precision ({STUDY_METRICS['Precision']:.3f}) is above recall "
         f"({STUDY_METRICS['Recall']:.3f}): the network misses slightly more true defect "
         "pixels than it invents. The next phase only needs the **largest** defect in the "
         "gauge. Trimming the edge of a large defect barely changes its √area, while "
         "inventing a large one that is not there would change the predicted life "
         "completely.")

    pg.h2("How the data were split")
    pg.p(f"The {len(split)} cylinders were split **before any training**: "
         f"**{sp.get('train', 0)} training**, **{sp.get('test', 0)} held out**, and "
         f"{sp.get('excluded_not_tested', 0)} not fatigue tested. The split is stratified "
         "by process regime (keyhole, spatter, clean) and by stress level (300 and "
         "450 MPa), so neither set is dominated by one condition. The held-out cylinders "
         "are used throughout Phases 2 to 4, and no model was ever fitted on them.")
    pg.p("**Labels.** The reference labels come from the originating team's own "
         "Laplacian-of-Gaussian, threshold and clustering pipeline, not from hand "
         "annotation, so the network is learning to reproduce a published method on "
         "unseen cylinders.")
    pg.caption("nnU-Net v2, 2-D, single-channel 16-bit input, two classes; RTX 3050 4 GB, "
               "patch 768 × 768, batch 1, 200 epochs, 5-fold cross-validation.")
    return pg, "Phase 1: finding the defects", "Segmenting defects in X-ray CT with nnU-Net."


def page_initiation(ct):
    from src.sections.phase2 import _section_frames, _star_xy
    pg = Page("initiation")
    PD = ("phase2_peridynamics",)
    h2 = J(*PD, "h2_first_bond.json")
    v1 = J(*PD, "verify_01_results.json")
    rows1 = v1["1A"] if v1 else None
    pg.h1("Phase 2: where cracks start",
          (f"A peridynamic model of each measured section finds where the crack starts: "
           f"at the largest defect in {h2['hits_by_area']} of {h2['n']} sections, not the "
           f"90% expected.") if h2 else None)
    pg.spec(
        r"Bond-based peridynamics: each point $x$ is held in equilibrium by bonds to every "
        r"point within its horizon $\delta$, "
        r"$\int_{H_x} c\,s\,\chi\,\frac{\xi+\eta}{|\xi+\eta|}\,dV_{x'} + b(x) = 0$, with bond "
        r"stretch $s=(|\xi+\eta|-|\xi|)/|\xi|$. Under cyclic load each bond loses life as "
        r"$d\lambda/dN = -A(\varepsilon-\varepsilon_{th})^{\beta}$ and breaks ($\chi=0$) "
        r"when $\lambda$ reaches zero.",
        "The measured section through the governing defect on a uniform grid, Δx = 40 µm, "
        "horizon δ = 3Δx, with defects as voids, in plane stress. Equilibrium is solved as "
        "a sparse linear system at each damage event; time jumps to the next bond failure.",
        (f"As the horizon is refined the model's elastic modulus converges to the classical "
         f"value: error {rows1[0]['err_pct']:.1f}% at m = {rows1[0]['m']:g} falling to "
         f"{rows1[-1]['err_pct']:.2f}% at m = {rows1[-1]['m']:g} (chart below)."
         if rows1 else "Horizon-refinement results not found."),
        (f"The first bond breaks at the largest defect in **{h2['hits_by_area']} of "
         f"{h2['n']}** measured sections ({h2['fraction_by_area']:.1%}); H2 asked for 90%."
         if h2 else "H2 result file not found."))
    pg.note("This phase has two jobs. It measures every defect, and the size of the largest "
            "goes to Phase 3 to become the life. Its crack model then shows where the crack "
            "starts and which way it runs. The crack model's cycle count is not used as a "
            "life: at attainable grid resolution it cannot carry one.")

    pg.h2("From a defect to a number")
    cat = pd.read_csv(os.path.join(DATA, "phase2_extraction", "phase2_defect_catalogue.csv"))
    cyl = pd.read_csv(os.path.join(DATA, "phase2_extraction", "phase2_cylinder_summary.csv"))
    pg.p(f"Every defect is isolated in three dimensions and reduced to the size fracture "
         f"mechanics uses: the square root of its area projected on the plane across the "
         f"load, √area (Murakami). The catalogue holds **{len(cat):,} defects** from "
         f"{cyl['cylinder'].nunique()} cylinders, the largest {cat['sqrt_area_um'].max():,.0f} "
         f"µm. Flatness separates lack-of-fusion flaws (about 0.1) from round gas pores "
         f"(about 1).")
    specs = list(ct)
    opts = ""
    for s in specs:
        fid = pg.data([dict(src=[r["raw"], r["over"], r["reg"]], label=f"z{r['z']:04d}",
                            caption=f"{r['n']} region(s) sized") for r in ct[s]], "frames")
        opts += f'<option value="{fid}">{SPEC_LABELS.get(s, s)}</option>'
    first = re.search(r'value="([^"]+)"', opts).group(1)
    n0 = len(ct[specs[0]])
    pg.add(f'<div class="controls"><label for="p2-spec">Cylinder</label>'
           f'<select id="p2-spec" class="chooser" data-player="p2-viewer">{opts}</select></div>'
           f'<div class="player" id="p2-viewer" data-frames="{first}" data-autoplay="0" '
           f'data-start="{n0 // 2}"><div class="controls"><div class="slider-row"><label>Slice'
           '</label><input type="range" min="0" value="0" aria-label="Slice"><output></output>'
           '</div></div><div class="img-row three"><figure><img data-slot="0" alt="Raw CT">'
           '<figcaption>Raw CT</figcaption></figure><figure><img data-slot="1" alt="Predicted '
           'defects"><figcaption>Predicted defects</figcaption></figure><figure><img '
           'data-slot="2" alt="Sized regions"><figcaption data-frame-caption></figcaption>'
           "</figure></div></div>")
    fig2 = go.Figure(go.Scatter(
        x=cat["sqrt_area_um"], y=cat["flatness"], mode="markers",
        marker=dict(size=5, color=NAVY, opacity=0.45), text=cat["cylinder"],
        hovertemplate="%{text}: √area %{x:.0f} µm, flatness %{y:.2f}<extra></extra>"))
    fig2.update_layout(height=H_MAIN, xaxis_title="√area (µm)", yaxis_title="flatness",
                       xaxis_type="log")
    pg.chart(fig2)
    pg.caption("Every catalogued defect by size and shape.")

    def summary():
        t = cyl.rename(columns={"largest_sqrt_area_um": "largest √area (µm)",
                                "aop24_porosity_pct": "porosity (%)"}).round(4)
        pg.table(t)
        r = np.corrcoef(cyl["aop24_flaw_vox"], cyl["pred_flaw_vox_scaled"])[0, 1]
        pg.p(f"Predicted and reference flaw volumes correlate at r = {r:.3f}.")
    pg.details("Cylinder summary and the cross-check against the reference flaw count",
               summary)

    pg.h2("Verification: the model converges to classical elasticity")
    if rows1:
        fig = go.Figure(go.Scatter(
            x=[r["m"] for r in rows1], y=[r["err_pct"] for r in rows1], mode="lines+markers",
            line=dict(color=NAVY), marker=dict(size=7),
            text=[f"{r['bonds_per_node']} bonds per node" for r in rows1],
            hovertemplate="m = %{x}: %{y:.2f}% (%{text})<extra></extra>"))
        fig.update_layout(height=H_AUX, xaxis_title="horizon multiplier m = δ/Δx",
                          yaxis_title="error in elastic modulus (%)", yaxis_type="log")
        pg.chart(fig)
        pg.p(f"Refining the horizon drives the error from {rows1[0]['err_pct']:.1f}% to "
             f"{rows1[-1]['err_pct']:.2f}% (fitted order {rows1[0]['fitted_order']:.2f}). The "
             "decline is not smooth because the number of bonds per node changes in steps. "
             "The bond-based form fixes Poisson's ratio at 1/3 in plane stress, against about "
             "0.30 for 316H; that is a property of the formulation, which is why a finite "
             "element model runs alongside as a cross-check.")

    pg.h2("The crack in a measured section")
    shutil.copy2(os.path.join(ROOT, "assets", "phase2_crack_growth.mp4"),
                 os.path.join(OUT, "img", "crack.mp4"))
    shutil.copy2(os.path.join(ROOT, "assets", "phase2_crack_growth.png"),
                 os.path.join(OUT, "img", "crack_poster.png"))
    pg.add('<video src="img/crack.mp4" poster="img/crack_poster.png" autoplay loop muted '
           'playsinline controls preload="metadata" aria-label="Peridynamic crack growth '
           'through the measured section of P017"></video>')
    pg.p("The measured section of P017 at 450 MPa, R = 0.1: its 22 defects at their real "
         "sizes and positions, modelled as voids (teal), and fatigue damage accumulating "
         "(orange). The crack leaves both tips of the large central defect. The video plays "
         "on a loop; use its controls to pause. Its cycle counter paces the playback and is "
         "not a fatigue life.")

    pg.h2("Hypothesis 2: does the crack start at the largest defect?")
    if h2:
        n, crit = h2["n"], h2["criterion"]
        met = h2["fraction_by_area"] >= crit
        pg.verdict(f"Criterion {'met' if met else 'not met'}: {h2['hits_by_area']} of {n} "
                   f"({h2['fraction_by_area']:.0%}) against {crit:.0%}. By Murakami severity, "
                   f"{h2['hits_by_severity']} of {n}.", "met" if met else "no")
        pg.p("Every bond starts with the same life and loses it fastest where it is "
             "stretched most, so the first bond to break is the most strained bond of the "
             "initial elastic solution: one solve per section, independent of the damage "
             "coefficient and of how many bonds the model breaks per step. The animation "
             "steps through all fourteen measured sections.")
        frames = _section_frames()

        def _title(f):
            r = f["row"]
            where = "surface" if r["first_bond_defect_is_surface"] else "internal"
            res = ("at the largest defect" if r["initiated_at_dominant"]
                   else "not at the largest defect")
            return (f"{f['spec'].replace('SN_', '')}, {r['sigma_max_MPa']} MPa: first bond "
                    f"{res} ({r['first_bond_defect_sqrt_area_um']:.0f} µm {where}; largest "
                    f"{r['dominant_sqrt_area_um']:.0f} µm)")

        def _data(f):
            x, y = _star_xy(f)
            return [go.Image(source=f["src"]),
                    go.Scatter(x=[x], y=[y], mode="markers",
                               marker=dict(symbol="star", size=20, color="#FFD54F",
                                           line=dict(color="#7A1F12", width=2)),
                               name="first bond", hoverinfo="name")]

        f0 = frames[0]
        fig = go.Figure(data=_data(f0))
        fig.frames = [go.Frame(data=_data(f), name=f["spec"].replace("SN_", ""),
                               layout=go.Layout(title_text=_title(f))) for f in frames]
        fig.update_layout(title=dict(text=_title(f0), x=0.0, font=dict(size=13)), height=330,
                          margin=dict(l=0, r=0, t=40, b=90),
                          xaxis=dict(visible=False, range=[0, max(f["w"] for f in frames)]),
                          yaxis=dict(visible=False, scaleanchor="x",
                                     range=[max(f["h"] for f in frames), 0]),
                          showlegend=False)
        anim(fig, 2200)
        pg.chart(fig)
        pg.caption("Grey: 316H. Teal: defects, the largest darker. Orange: damage at the end "
                   "of the run. Star: the first bond to break. Load axis left to right.")
        rows = h2["rows"]
        pg.table(pd.DataFrame([{
            "Cylinder": r["specimen"].replace("SN_", ""),
            "σmax (MPa)": r["sigma_max_MPa"], "Defects": r["n_defects"],
            "Largest √area (µm)": round(r["dominant_sqrt_area_um"]),
            "First bond at (µm)": round(r["first_bond_defect_sqrt_area_um"]),
            "Surface": "yes" if r["first_bond_defect_is_surface"] else "no",
            "At largest": "yes" if r["initiated_at_dominant"] else "no",
            "At most severe (ΔK)": "yes" if r["initiated_at_most_severe"] else "no",
        } for r in rows]), num=("σmax (MPa)", "Defects", "Largest √area (µm)",
                                "First bond at (µm)"))
        n_surf = sum(1 for r in rows if r["first_bond_defect_is_surface"])
        pg.finding(
            f"<b>Reported as found.</b> Size alone does not decide where a crack starts. "
            f"{n_surf} of {n} sections started at a surface defect, which Murakami's geometry "
            f"factors make {(0.65 / 0.50) ** 2:.2f} times as severe per unit area as a buried "
            f"one; weighting for that lifts the count only to {h2['hits_by_severity']} of "
            f"{n}, so neighbouring defects matter too. This concerns the severity criterion, "
            "not the segmentation or the life model, which uses defect size and is tested "
            "in Phase 4.")
    return pg, "Phase 2: where cracks start", "Peridynamic crack initiation in measured sections."


def page_life():
    pg = Page("life")
    MCB = ("phase_mc_bayesian",)
    p4 = pd.read_csv(os.path.join(DATA, "phase4_validation", "phase4_error_isolation.csv"))
    p4 = p4[np.isfinite(pd.to_numeric(p4["log10_N_pathA"], errors="coerce"))].copy()
    p4["predicted"] = 10 ** pd.to_numeric(p4["log10_N_pathA"])
    p4["measured"] = 10 ** pd.to_numeric(p4["measured_log10_cycles"])
    cmp_d = J("comparison_model", "comparison_results.json") or {}
    piml = J("comparison_model", "piml_results.json") or {}
    vi = piml.get("verify_integral")
    p4m = cmp_d.get("phase4", {})
    fac = p4m.get("log10_N_pathA", {}).get("factor")
    rates = [r["rate"] for r in vi["table"][1:4]] if vi else None

    _mc = J(*MCB, "monte_carlo_results.json") or {}
    _v = _mc.get("validation_independent", {})
    pg.h1("Phase 3: the fatigue life",
          (f"A crack grown from the largest defect predicts the life to within {fac:.2f}× "
           f"with nothing fitted, and its scatter reproduces the lives of "
           f"{sum(x['n_tested'] for x in _v.values())} further cylinders.") if fac and _v else None)
    pg.spec(
        r"The driving force on a crack of depth $a$ is $\Delta K = Y\,\Delta\sigma\sqrt{\pi a}$ "
        r"and the Paris law gives its growth rate, $da/dN = C(\Delta K-\Delta K_{th})^{m}$. "
        r"The life is the number of cycles to grow it from the governing defect, "
        r"$a_0=\sqrt{\text{area}}$, to the depth $a_f$ at which the section collapses: "
        r"$N=\int_{a_0}^{a_f} da\,/\,C(\Delta K(a)-\Delta K_{th})^{m}$.",
        "Composite Simpson's rule on a logarithmically spaced grid in $a$, since the "
        "integrand is concentrated at the lower limit. Scatter is carried by Monte Carlo "
        "over the defect size and the growth rate; inspections update one part's life "
        "through a particle filter.",
        (f"Against the closed form the error falls at order {min(rates):.1f} to "
         f"{max(rates):.1f}, as Simpson's rule should, reaching "
         f"{vi['table'][3]['rel_err']:.0e} at {vi['table'][3]['n']} points before round-off "
         "(table below)." if vi else "Integral verification file not found."),
        (f"Within **{fac:.2f}×** of the measured lives of the 14 held-out cylinders, with no "
         "constant fitted to any fatigue life." if fac else "Validation file not found."))
    pg.note("The constants are literature values for 316L: C = 6.25 × 10⁻¹⁰ and m = 3.94 "
            "(da/dN in mm per cycle, ΔK in MPa√m), and Y = 0.65, Murakami's factor for a "
            "surface defect. None of them is fitted to this dataset's fatigue lives.")

    pg.h2("The one modelling decision: the threshold")
    pb, sub, vt = piml.get("physics_bias"), piml.get("subthreshold_specimens", []), \
        piml.get("verify_threshold", [])
    if pb and vt:
        dks = [r["dK"] for r in vt]
        pg.p(f"The long-crack threshold for 316L is ΔK_th = 5 MPa√m. At the governing gauge "
             f"defect of the held-out cylinders the driving force lies between "
             f"{min(dks):.1f} and {max(dks):.1f} MPa√m, close to that threshold or below it. "
             f"Kept in, it predicts that {len(sub)} cylinder never fails and over-predicts "
             f"the other {pb['n']} by a median factor of {pb['median']:.1f}, although every "
             "one of them failed in test. Cracks from defects this small are short cracks, to "
             "which the long-crack threshold does not apply, so the validated prediction, "
             "path A, sets ΔK_th = 0.")
        sw = piml.get("threshold_sweep", [])
        fig = go.Figure(go.Scatter(
            x=[r["dkth"] for r in sw], y=[r["geometric_bias"] for r in sw],
            mode="lines+markers", line=dict(color=NAVY), marker=dict(size=7),
            text=[f"{r['n_finite']} of 14 predicted to fail" for r in sw],
            hovertemplate="ΔK_th = %{x} MPa√m: %{y:.2f}× (%{text})<extra></extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color=GREY)
        fig.update_layout(height=H_AUX, yaxis_type="log", xaxis_title="threshold ΔK_th (MPa√m)",
                          yaxis_title="predicted ÷ measured (geometric mean)")
        pg.chart(fig)
        pg.caption("The bias of the held-out predictions as the threshold is raised from zero "
                   "to the long-crack value.")

    # one crack, drawn in the browser from compact curves
    pg.h2("One crack, from the governing defect to collapse")
    m_paris, p_exp = 3.94, 1.0 - 3.94 / 2.0
    curves, texts = {}, {}
    for _, row in p4.iterrows():
        a0 = float(row["largest_sqrt_area_um_ai_gauge"]) / 1000.0
        ac, Nf = float(row["a_c_mm"]), float(row["predicted"])
        a = np.geomspace(a0, ac, 160)
        frac = (a ** p_exp - a0 ** p_exp) / (ac ** p_exp - a0 ** p_exp)
        N = frac * Nf
        dK = 0.65 * float(row["stress_level_MPa"]) * 0.9 * np.sqrt(np.pi * a / 1000.0)
        idx = np.searchsorted(frac, np.linspace(0, 1, 48)).clip(0, len(a) - 1)
        meas = float(row["measured"])
        c = row["cylinder"]
        curves[c] = dict(N=np.round(N, 1).tolist(), a=np.round(a, 6).tolist(),
                         dK=np.round(dK, 4).tolist(), idx=idx.tolist(), a0=a0, ac=ac, Nf=Nf,
                         meas=meas)
        half = float(np.interp(0.5, N / Nf, a))
        k2 = float(np.interp(2 * a0, a, N)) / Nf if 2 * a0 < ac else None
        texts[c] = (f"{c}'s governing defect is {a0 * 1000:,.0f} µm. Predicted life "
                    f"{Nf:,.0f} cycles against {meas:,.0f} measured. The crack hardly moves for "
                    f"most of that life: at half-life it is still only {half * 1000:,.0f} µm "
                    "deep, and " + (f"{k2:.0%} of the cycles pass before it doubles. "
                                    if k2 is not None else "it does not double before "
                                    "collapse. ") +
                    "ΔK grows with the crack, so growth accelerates and the end comes quickly. "
                    "This is why the starting defect size decides the life.")
    cid = pg.data(curves, "curves")
    order = list(p4["cylinder"])
    opts = "".join(f'<option value="{c}"{" selected" if c == "P017" else ""}>{c}, '
                   f"{int(p4.set_index('cylinder').loc[c, 'stress_level_MPa'])} MPa</option>"
                   for c in order)
    pg.add(f'<div class="controls"><label for="p3-cyl">Held-out cylinder</label><select '
           f'id="p3-cyl" class="chooser" data-growth="p3-growth">{opts}</select></div>'
           f'<div class="chart" id="p3-growth" data-curves="{cid}" role="img" '
           f'aria-label="crack depth against cycles"></div>')
    for c in order:
        pg.add(f'<div data-show-for="p3-cyl" data-key="{c}">{md(texts[c])}</div>')

    if vi:
        pg.h2("Verification of the integral")
        t = pd.DataFrame(vi["table"])
        t["rate"] = t["rate"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        t["rel_err"] = t["rel_err"].map(lambda v: f"{v:.2e}")
        t["value"] = t["value"].map(lambda v: f"{v:,.6f}")
        pg.table(t.rename(columns={"n": "Grid points", "value": "Integral (cycles)",
                                   "rel_err": "Relative error", "rate": "Observed order"}),
                 num=("Grid points", "Integral (cycles)", "Relative error", "Observed order"))
        pg.p(f"Exact value {vi['exact']:,.6f} cycles. Each doubling of the grid cuts the "
             "error sixteen-fold, order 4, until it reaches machine precision near 10⁻¹⁴; "
             "past that the observed order means nothing.")

    pg.h2("Every held-out cylinder, predicted then compared")

    def _fac(key):
        v = p4m.get(key)
        return v.get("factor") if isinstance(v, dict) else None
    rows = []
    if _fac("log10_N_pred") is not None:
        rows.append(("Threshold kept, whole-blank defect size",
                     f"{_fac('log10_N_pred'):.1f}×", f"{p4m['log10_N_pred']['bias']:+.2f}"))
    for key, lab in (("log10_N_pathA", "Path A: the network's defect sizes"),
                     ("log10_N_pathB", "Path B: the reference defect sizes")):
        if _fac(key) is not None:
            rows.append((lab, f"{_fac(key):.2f}×", f"{p4m[key]['bias']:+.2f}"))
    pg.table(pd.DataFrame(rows, columns=["Prediction", "Typical error", "Bias (decades)"]),
             num=("Typical error", "Bias (decades)"))
    pg.p("Path A and path B differ only in who measured the defects. Their closeness says "
         "the segmentation adds little error to the life.")
    p4s = p4.sort_values("measured").reset_index(drop=True)
    lo, hi = PARITY

    def frame_data(k):
        shown, new = p4s.iloc[:k + 1], p4s.iloc[[k]]
        return [
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                       line=dict(dash="dash", color=GREY), name="perfect agreement"),
            go.Scatter(x=[lo, hi], y=[2 * lo, 2 * hi], mode="lines",
                       line=dict(dash="dot", color=RULE), name="factor of 2"),
            go.Scatter(x=[lo, hi], y=[lo / 2, hi / 2], mode="lines",
                       line=dict(dash="dot", color=RULE), showlegend=False),
            go.Scatter(x=shown["measured"], y=shown["predicted"], mode="markers",
                       marker=dict(size=10, color=NAVY), name="held-out cylinder",
                       text=shown["cylinder"], hoverinfo="text+x+y"),
            go.Scatter(x=new["measured"], y=new["predicted"], mode="markers+text",
                       text=new["cylinder"], textposition="top center",
                       marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=RED, width=2)),
                       showlegend=False, hoverinfo="skip")]
    fig = go.Figure(data=frame_data(0))
    fig.frames = [go.Frame(data=frame_data(k), name=p4s.loc[k, "cylinder"])
                  for k in range(len(p4s))]
    parity(fig)
    fig.update_layout(margin=dict(t=20, b=110))
    anim(fig, 700, 200, "added ")
    pg.chart(fig)
    resid = np.log10(p4["predicted"]) - np.log10(p4["measured"])
    pg.caption(f"The {len(p4)} held-out cylinders added one at a time, shortest life first. "
               f"Every prediction is within a factor of {10 ** np.abs(resid).max():.1f} of its "
               "measured life.")
    pg.details("The numbers behind the chart", lambda: pg.table(pd.DataFrame({
        "Cylinder": p4s["cylinder"], "σmax (MPa)": p4s["stress_level_MPa"],
        "Measured": p4s["measured"].map(lambda v: f"{v:,.0f}"),
        "Predicted": p4s["predicted"].map(lambda v: f"{v:,.0f}"),
        "Ratio": (p4s["predicted"] / p4s["measured"]).map(lambda v: f"{v:.2f}")}),
        num=("σmax (MPa)", "Measured", "Predicted", "Ratio")))

    # ---------------------------------------------------------------- Monte Carlo
    pg.h2("Monte Carlo: a distribution of lives")
    mc = J(*MCB, "monte_carlo_results.json")
    dp, cp = mc["priors"]["defect"], mc["priors"]["C"]
    pg.p(f"Parts printed the same way differ. Each virtual part draws a governing defect from "
         f"a lognormal fitted to the 14 measured sizes, "
         f"$\\ln a_0 \\sim \\mathcal{{N}}({dp['mu']:.3f}, {dp['sigma']:.3f}^2)$ with $a_0$ in "
         f"µm, and a growth rate $C$ from the scatter of path A's own errors: the $C$ that "
         f"would have made each of the 14 predictions exact, "
         f"$\\sigma(\\ln C) = {cp['sigma_C']:.3f}$. It then goes through path A's life "
         "integral, threshold removed, so the uncertainty carried is that of the validated "
         "prediction.")
    opts, blocks = "", []
    for s in ("300", "450"):
        sn = mc["sn"][s]
        # the same figure as the Streamlit page and Figure 4.9 of the thesis: shared
        # axis, bins and vertical scale for both stresses
        from src.sections.phase3 import mc_figure
        fig, n_meas = mc_figure(mc, s)
        anim(fig, 600, 300)
        fid = pg.fig_json(fig)
        opts += f'<option value="{fid}">{s} MPa</option>'
        pct = 100 * sn["n_at_runout"] / sn["n_total"]
        v = mc.get("validation_independent", {}).get(s)
        txt = (f'<p class="caption">Simulated (navy) against measured (grey), same axes at '
               f"both stresses. Median {10 ** sn['log10_p50']:,.0f} cycles; run-outs "
               f"{pct:.1f}%.</p>")
        if v:
            txt += md(f"**Against the measured record.** Both priors come from the 14 held-out "
                      f"cylinders, so the other {v['n_tested']} tested at this stress are "
                      f"independent of the simulation. Their median life is "
                      f"{v['median_measured']:,.0f} cycles against {v['median_simulated']:,.0f} "
                      f"simulated; {v['n_inside_5_95']} of their {v['n_failed']} failures fall "
                      f"inside the simulated 5th to 95th percentile band, and a Kolmogorov–Smirnov "
                      f"test finds no difference (p = {v['ks_p']:.2f}).")
            if v["n_runout"]:
                txt += (f'<div class="finding">{md("<b>Where it falls short.</b> " + str(v["n_runout"]) + " of those cylinders ran out at 10 million cycles without failing; the simulation predicts no run-outs, because with the threshold removed every crack grows. A threshold that grows with crack length would be needed to reproduce the fatigue limit.")}</div>')
        blocks.append((fid, txt))
    pg.add(f'<div class="controls"><label for="mc-s">Stress level</label><select id="mc-s" '
           f'class="chooser" data-chart="mc-chart">{opts}</select></div>')
    pg.add(f'<div class="chart" id="mc-chart" data-fig="{blocks[0][0]}" role="img" '
           f'aria-label="life distribution"></div>')
    for fid, txt in blocks:
        pg.add(f'<div data-show-for="mc-s" data-key="{fid}">{txt}</div>')
    n_tot = mc["sn"]["300"]["n_total"]
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n_tot))
    m_need = int(np.ceil(np.log(2 / 0.05) / (2 * 0.05 ** 2)))
    pg.p("**Convergence.** The Dvoretzky–Kiefer–Wolfowitz inequality bounds how far the "
         "sampled distribution can lie from the true one: "
         r"$P\big(\sup_N |\hat F_M(N)-F(N)| > \varepsilon\big) \le 2e^{-2M\varepsilon^2}$. "
         f"For ε = 0.05 at 95% confidence it needs M ≥ {m_need:,} draws. The run used "
         f"M = {n_tot:,}, which gives ε = {eps:.4f}.")
    v2 = pd.DataFrame(mc["verify_convergence"]["table"])
    figv2 = go.Figure(go.Scatter(x=v2["n_samples"], y=v2["rmse"], mode="lines+markers",
                                 line=dict(color=NAVY), name="observed"))
    ref = v2["rmse"].iloc[0] * np.sqrt(v2["n_samples"].iloc[0] / v2["n_samples"])
    figv2.add_trace(go.Scatter(x=v2["n_samples"], y=ref, mode="lines", name="M^(−1/2)",
                               line=dict(color=GREY, dash="dot")))
    figv2.update_layout(xaxis_type="log", yaxis_type="log", height=H_AUX, xaxis_title="draws M",
                        yaxis_title="RMSE of the median (decades)", margin=dict(t=10),
                        legend=LEGEND)
    pg.chart(figv2)
    pg.caption(f"Each sample size repeated {int(v2['n_repeats'].iloc[0])} times. The error of "
               "the median falls close to the M^(−1/2) rate of Monte Carlo.")

    def prior_check():
        v1 = pd.DataFrame(mc["verify_extremes"]).sort_values("percentile")
        figv1 = go.Figure(go.Bar(x=v1["specimen"], y=v1["percentile"], marker_color=NAVY))
        figv1.update_layout(yaxis_title="leave-one-out percentile (%)", xaxis_title="held-out cylinder", height=H_AUX,
                            yaxis_range=[0, 100], margin=dict(t=10))
        pg.chart(figv1)
        pg.caption("Each specimen's own governing defect placed in a fit made without it. A "
                   "good fit spreads these evenly from 0 to 100%; the Kolmogorov–Smirnov "
                   f"statistic against a uniform spread is {mc['verify_extremes_ks']:.3f}.")
    pg.details("Check of the defect-size prior", prior_check)

    # ---------------------------------------------------------------- Bayes
    pg.h2("Particle filter: one part's life as inspections arrive")
    bf = J(*MCB, "bayesian_results.json")
    r = bf["verify_recovery"]
    step = next((i for i, h in enumerate(r["history"], 1)
                 if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
    ess = min(h["ess"] for h in r["history"])
    pg.p("Once a part is in service, each crack-length reading narrows its life. 4,000 "
         "particles, each a possible growth rate $C_i$, are reweighted by how well they "
         "predict the reading and then resampled. Crack size and rate separate in the Paris "
         "integral, so each particle's history is one shared function:")
    pg.add(r'<p class="eq">$$S(a) = \int_{a_0}^{a} \frac{da^\prime}{\big(\Delta K(a^\prime)'
           r'-\Delta K_{th}\big)^{m}}\quad\Rightarrow\quad a_i(N) = S^{-1}(C_i\,N)$$</p>')
    met = step is not None and step <= 3 and ess >= 250
    pg.verdict(f"Hypothesis 3 {'met' if met else 'partly met'} on the recovery trial: within "
               f"20% of the true life at inspection {step}, against 3 hypothesised; effective "
               f"sample size never below {ess:,.0f}, against 250.", "met" if met else "part")
    tails = bf.get("verify_tails")
    if tails:
        steps = [s for t in tails for s in t["steps"]]
        e_all = [e for t in tails for e in t["min_ess"]]
        pg.p("That trial's true growth rate fell near the centre of the prior, where the "
             "prior alone is already within 20%. Placed at the prior's 5th and 95th "
             f"percentiles instead, where the prior's life is off by "
             f"{min(t['prior_life_error'] for t in tails):.0%} to "
             f"{max(t['prior_life_error'] for t in tails):.0%}, the filter still came within "
             f"20% by inspection {max(steps)} in all {len(steps)} trials. The effective sample "
             f"size fell below 250 in {sum(e < 250 for e in e_all)} of them, to "
             f"{min(e_all):,.0f}, before resampling restored it: the speed condition holds "
             "throughout, the sample-size condition fails at the tails.")
    cases = {"Synthetic part with a known life": (r["history"], r["Nf_true"], True)}
    for spk, rs in bf["specimens"].items():
        cases[f"{spk}, measured {rs['Nf_measured']:,} cycles"] = (rs["history"],
                                                                 rs["Nf_measured"], False)
    opts, caps = "", []
    for name, (hist, truth, synth) in cases.items():
        H = pd.DataFrame(hist)
        ylo, yhi = min(H["Nf_p10"].min(), truth) / 1.3, max(H["Nf_p90"].max(), truth) * 1.3

        def bframe(k, H=H):
            h = H.iloc[:k + 1]
            return [go.Scatter(x=h["N_inspection"], y=h["Nf_p90"], mode="lines",
                               line=dict(width=0), showlegend=False, hoverinfo="skip"),
                    go.Scatter(x=h["N_inspection"], y=h["Nf_p10"], mode="lines", fill="tonexty",
                               fillcolor="rgba(31,78,121,0.15)", line=dict(width=0),
                               name="10th to 90th percentile"),
                    go.Scatter(x=h["N_inspection"], y=h["Nf_mean"], mode="lines+markers",
                               line=dict(color=NAVY, width=2.5), name="posterior mean")]
        fig = go.Figure(data=bframe(0))
        fig.frames = [go.Frame(data=bframe(k), name=f"inspection {k + 1}") for k in range(len(H))]
        fig.add_hline(y=truth, line_dash="dash", line_color=MEAS,
                      annotation_text="true life" if synth else "measured life")
        fig.update_layout(yaxis=dict(type="log", range=[np.log10(ylo), np.log10(yhi)],
                                     title="predicted cycles to failure"),
                          xaxis=dict(range=[0, H["N_inspection"].max() * 1.08],
                                     title="cycles inspected so far"),
                          height=H_MAIN, margin=dict(t=20, b=110), legend=LEGEND)
        anim(fig, 1000, 300)
        fid = pg.fig_json(fig)
        opts += f'<option value="{fid}">{html.escape(name)}</option>'
        last = hist[-1]
        caps.append((fid, f"One inspection added per frame. After the last, the posterior "
                          f"mean is {last['Nf_mean']:,.0f} cycles against {truth:,.0f}; the band "
                          f"narrowed {(hist[0]['Nf_p90'] - hist[0]['Nf_p10']) / max(last['Nf_p90'] - last['Nf_p10'], 1):.0f}-fold "
                          "from the first inspection to the last."))
    pg.add(f'<div class="controls"><label for="bf-case">Part</label><select id="bf-case" '
           f'class="chooser" data-chart="bf-chart">{opts}</select></div>'
           f'<div class="chart" id="bf-chart" data-fig="{caps[0][0]}" role="img" '
           f'aria-label="posterior life"></div>')
    for fid, c in caps:
        pg.add(f'<p class="caption" data-show-for="bf-case" data-key="{fid}">{c}</p>')
    return pg, "Phase 3: the fatigue life", "The Paris integral, Monte Carlo and the particle filter."


def pinn_export():
    """Train the network exactly as the Streamlit page does and export its weights."""
    from src.piml_nn import predict, train_pinn
    meas = splits(load())["meas"]
    X = np.column_stack([np.log([r["ds"] for r in meas]),
                         np.log([r["sqrt_area_gauge_um"] for r in meas]),
                         [r["flatness_median"] for r in meas]])
    y = np.log10([r["Nf"] for r in meas])
    model, std = train_pinn(X, y, lam_s=10.0, lam_a=10.0, lam_m=0.1, epochs=3000, seed=0)
    lin = [m for m in model.net if hasattr(m, "weight")]
    w = [(m.weight.detach().numpy().tolist(), m.bias.detach().numpy().tolist()) for m in lin]
    flat_med = float(np.median(X[:, 2]))
    return dict(W1=w[0][0], b1=w[0][1], W2=w[1][0], b2=w[1][1], W3=w[2][0], b3=w[2][1],
                mean=std.mean.tolist(), std=std.std.tolist(), flat_med=flat_med), \
        (lambda sa, ds: float(10 ** predict(model, std, np.array([[np.log(ds), np.log(sa),
                                                                       flat_med]]))[0])), meas


def page_validation():
    pg = Page("validation")
    D = ("comparison_model",)
    res, comp = J(*D, "piml_results.json"), J(*D, "comparison_results.json")
    fail, h4 = J(*D, "failure_criterion.json"), J("phase4_validation", "h4_calibration.json")
    lad = {r["model"].split()[0]: r for r in comp["ladder"]}
    pa = comp["phase4"]["log10_N_pathA"]["factor"]
    vr = res["verify_recovery"]
    rate = vr["table"][1].get("rate_s")

    pg.h1("Phase 4: validation",
          f"With nothing fitted, path A ({pa:.2f}×) beats every fitted model, including a "
          f"{lad['C1']['params']}-parameter physics-informed neural network "
          f"({lad['C1']['loo_factor']:.2f}×).")
    pg.spec(
        r"Theory fixes the form of a fatigue model. Neglecting the threshold, the life "
        r"integral is $N_f = a_0^{1-m/2}/\big[(m/2-1)\,C\,(Y\Delta\sigma\sqrt{\pi})^{m}\big]$, "
        r"so $\log_{10} N_f = b_0 + b_s\ln\Delta\sigma + b_a\ln\sqrt{\text{area}}$ with "
        r"$b_s=-m/\ln 10$ and $b_a=(1-m/2)/\ln 10$, both negative.",
        "A ladder of regressions from stress alone up to four parameters, the same "
        "regression with the signs of $b_s$ and $b_a$ bound (B3), a physics-informed neural "
        "network (C1) whose loss penalises its own gradients for leaving the theoretical "
        "slopes, and a correction factor φ with the physical signs built into its weights. "
        "Every model is scored leave-one-out on the 14 cylinders with a measured defect.",
        (f"Fitted to lives synthesised with known exponents, the regression recovers them "
         f"with error falling at order {rate:.2f} as the scatter falls, the rate of an "
         "unbiased linear estimator." if rate else
         "The fitting procedure recovers known exponents from synthetic data."),
        (f"Path A, with nothing fitted, is within **{pa:.2f}×**. The best fit needs "
         f"{lad['B5']['params']} parameters for {lad['B5']['loo_factor']:.2f}×; the network "
         f"{lad['C1']['params']} for {lad['C1']['loo_factor']:.2f}×." +
         (f" Hypothesis 4 on B3: {h4['n_inside']} of {h4['n']} inside its interval, slope "
          f"{h4['slope']:.3f}." if h4 else "")))
    pg.note("Leave-one-out: to score a model on one cylinder, fit it to the other thirteen, "
            "predict the one left out, and repeat for every cylinder. No prediction is made by "
            "a model that has seen its answer.")

    pg.h2("The data")
    pg.p(f"{comp['n_specimens']} cylinders of LPBF 316H were fatigue tested (ASTM E466, "
         f"R = 0.1, at 300 and 450 MPa): {comp['n_failures']} failed and {comp['n_runouts']} "
         f"ran out at 10 million cycles. Of the failures, {comp['n_gauge']} broke in the gauge "
         f"and {comp['n_radius']} at the fillet radius. {comp['n_measured']} have a governing "
         "defect measured from their CT scan; they carry every model below that uses defect "
         "size.")

    pg.h2("The comparison ladder")
    order = sorted(comp["ladder"], key=lambda r: (r["params"], r["model"]))
    names = [r["model"].split("  ")[0].strip() + "  " + r["model"].split("  ", 1)[-1].strip()
             for r in order]

    def rung(k):
        shown = order[:k + 1]
        cols = [RED if r["model"].startswith("C1") else NAVY if r["model"].startswith("B3")
                else GREY for r in shown]
        return [go.Bar(y=names[:k + 1], x=[r["loo_factor"] for r in shown], orientation="h",
                       marker_color=cols, text=[f"{r['loo_factor']:.2f}×, {r['params']} parameters"
                                                for r in shown],
                       textposition="outside", cliponaxis=False)]
    figl = go.Figure(data=rung(0))
    figl.frames = [go.Frame(data=rung(k), name=order[k]["model"].split()[0])
                   for k in range(len(order))]
    figl.add_vline(x=pa, line_color=NAVY, line_width=2, line_dash="dash",
                   annotation_text=f"path A, nothing fitted: {pa:.2f}×",
                   annotation_position="top right")
    figl.update_layout(xaxis=dict(range=[1.0, max(r["loo_factor"] for r in order) * 1.35],
                                  title="typical leave-one-out error (×), lower is better"),
                       yaxis=dict(categoryorder="array", categoryarray=names[::-1],
                                  range=[-0.5, len(names) - 0.5], automargin=True),
                       height=H_MAIN + 50, margin=dict(t=30, b=110, l=10), showlegend=False)
    anim(figl, 650, 250, "model ")
    pg.chart(figl)
    pg.caption("The ladder climbed from the fewest parameters to the most. A: all gauge "
               "failures, without a defect size. B: the 14 with a measured defect. C1: the "
               "physics-informed network.")
    df = pd.DataFrame(res["ladder"])[["model", "n", "params", "r2", "loo_rmse", "loo_factor"]]
    df = pd.DataFrame({"Model": df["model"], "n": df["n"], "Parameters": df["params"],
                       "R²": df["r2"].map(lambda v: f"{v:.3f}"),
                       "LOO RMSE (decades)": df["loo_rmse"].map(lambda v: f"{v:.3f}"),
                       "Typical error": df["loo_factor"].map(lambda v: f"{v:.2f}×")})
    pg.details("The ladder as a table", lambda: pg.table(df, num=(
        "n", "Parameters", "R²", "LOO RMSE (decades)", "Typical error")))
    pg.p(f"B3 and B4 score the same, {lad['B3']['loo_factor']:.3f}×: the unconstrained fit "
         "already has the physical signs, so the bound costs nothing. Adding defect size takes "
         f"stress-only from {lad['B1']['loo_factor']:.2f}× to {lad['B3']['loo_factor']:.2f}× on "
         f"the same cylinders. The network, with {lad['C1']['params']} parameters on 14 points, "
         "does no better than three. Path A matches the best fits without fitting anything to "
         "these lives.")

    # the brief
    feats, phi = J(*D, "defect_features.json"), J(*D, "phi_results.json")
    pg.h2("The physics-informed brief, item by item")
    pg.p("The brief for this comparison asked for three ways of putting physics into a data "
         "model: physical input features, a physics-constrained architecture, and a "
         "physics-informed loss. Each was built and scored by the same leave-one-out protocol.")
    c1 = next((r["loo_factor"] for r in res["ladder"] if r["model"].startswith("C1")), 0)
    pg.table(pd.DataFrame([
        ("Size, √area", "Measured on every defect (Phase 2)", "Drives every model"),
        ("Shape", "Circularity and aspect ratio of the projection across the load",
         "No measurable gain (below)"),
        ("Location", "Ligament from the defect to the gauge surface", "No measurable gain (below)"),
        ("Interaction", "Nearest-neighbour gap; Murakami's merging rule",
         "Merging worsens every affected prediction"),
        ("σa(1−R), σa/UTS, σa²/2E", "Computed for every cylinder", "Identical to stress here (below)"),
        ("Constrained architecture", "φ network with sign rules in its weights (D2, E2); "
         "sign-bounded regression (B3)", "Does not beat path A"),
        ("Physics-informed loss", "Network penalised off the Basquin and Paris slopes (C1)", f"{c1:.2f}×"),
        ("Paris crack-growth law", "Path A, literature constants", f"{pa:.2f}×, nothing fitted"),
    ], columns=["Brief item", "What was built", "Outcome"]))
    col = res.get("collinearity")
    if col:
        dist = res.get("collinearity_distinct", {})
        pg.h3("Engineered stress features")
        pg.p(f"Every cylinder was tested at R = 0.1 and at one of two stresses, so each "
             f"engineered feature takes only {max(dist.values()) if dist else 2} distinct "
             "values and is a fixed function of the stress amplitude. Their correlations with "
             "σa and with each other:")
        pg.table(pd.DataFrame([(c["a"], c["b"], f"{c['r']:.4f}") for c in col],
                              columns=["Feature", "Against", "Correlation"]), num=("Correlation",))
        pg.p("They cannot add information with this test matrix. They would on a dataset with "
             "varied R and more stress levels.")
    if feats:
        pg.h3("Shape, location and interaction of the governing defect")
        f = pd.DataFrame(feats["rows"])
        pg.table(pd.DataFrame({
            "Cylinder": f["cylinder"], "√area (µm)": f["sqrt_area_um"].round(0).astype(int),
            "Circularity": f["circularity"].map(lambda v: f"{v:.2f}"),
            "Aspect ratio": f["aspect_ratio"].map(lambda v: f"{v:.2f}"),
            "Ligament to surface (µm)": f["L_surface_um"].round(0).astype(int),
            "Gap to neighbour (µm)": f["nn_gap_um"].round(0).astype(int),
            "Merged √area (µm)": f["sqrt_area_eff_um"].round(0).astype(int)}),
            num=("√area (µm)", "Circularity", "Aspect ratio", "Ligament to surface (µm)",
                 "Gap to neighbour (µm)", "Merged √area (µm)"))
        u = feats["units"]
        pg.caption(f"Measured on the network's gauge-cropped defect map, the same governing "
                   f"defect path A grows. Pixels are {u['dxy_um']} µm across the load, but "
                   f"slices were predicted every {u['z_spacing_um']:.0f} µm along it, so true 3-D "
                   "sphericity cannot be measured; circularity of the projection, 4πA/P², is "
                   "used instead. Every calculator was first checked on shapes with known answers.")
        pg.p("Murakami's rule treats two defects closer than the smaller one's √area as a "
             "single defect. Applied here it makes every affected prediction worse. For P017 "
             "the merged flaw exceeds the depth at which the section collapses, a life of zero, "
             "while the single governing defect predicts its measured life almost exactly. The "
             "400 µm slice spacing biases the rule toward merging, but the data favour the "
             "single defect.")
    if phi:
        pg.h3("The correction factor φ")
        pg.p("φ multiplies a baseline life, N = N_base / φ. By the brief's physical intuition it "
             "must rise with defect size and fall with circularity and with distance from the "
             "surface. The network carries those signs in its weights: every weight between an "
             "input and φ passes through a softplus and is positive, and circularity and "
             "ligament enter with a minus sign. A model that breaks a sign rule cannot be "
             "represented.")
        pg.add(r'<p class="eq">$$N = \frac{N_{\text{base}}}{\varphi(\ln\sqrt{\text{area}},\,-\Psi,'
               r'\,-L)},\qquad \frac{\partial \ln\varphi}{\partial z_k}\ \ge 0$$</p>')
        lab = {"D1": "φ linear, on path A", "D2": "φ network, on path A",
               "E1": "φ linear, on stress alone", "E2": "φ network, on stress alone"}
        t = [("Path A, nothing fitted", 0, phi["pathA_factor"])] + [
            (lab[m["model"]], m["params"], m["loo_factor"]) for m in phi["models"]]
        pg.table(pd.DataFrame([(a, b, f"{c:.3f}×") for a, b, c in t],
                              columns=["Model", "Fitted parameters", "Leave-one-out error"]),
                 num=("Fitted parameters", "Leave-one-out error"))
        v = [r for r in phi["verification"] if r["wd"] == phi["weight_decay"]]
        pg.p("**Verification.** On synthetic lives built with a known φ, the trained network's "
             "error in log₁₀ φ falls from " + ", ".join(f"{r['rmse_log10_phi']:.3f} at n = {r['n']}"
                                                       for r in v) +
             f". Its weight decay ({phi['weight_decay']:g}) was fixed by that study before any "
             "real fit. The sign rules hold at 2,000 random test points.")
        pg.finding("<b>Reported as found.</b> On the 14 cylinders, neither φ improves on path A. "
                   "Fitted to all 14, the sign-bounded linear φ sets every coefficient to zero: "
                   "once the physical signs are enforced, the data ask for no correction for "
                   "size, shape or location beyond what the crack-growth law already contains. "
                   "With 14 specimens this cannot rule such effects out; it shows they are below "
                   "what this dataset can resolve.")

    pts = comp.get("points_B3") or []
    if h4 and pts:
        pg.h2("Hypothesis 4: are the predictions calibrated?")
        met = h4["slope_met"] and h4["coverage_met"]
        pg.verdict(f"Criterion {'met' if met else 'not met'}: {h4['n_inside']} of {h4['n']} "
                   f"({h4['coverage']:.1%}) inside the 90% interval, against 90%; slope "
                   f"{h4['slope']:.3f}, against 0.90 to 1.10.", "met" if met else "no")
        s = h4["sigma_decades"]
        lims = list(PARITY)
        k = 10 ** (1.645 * s)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lims, y=[l / k for l in lims], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lims, y=[l * k for l in lims], mode="lines", fill="tonexty",
                                 fillcolor="rgba(31,78,121,0.10)", line=dict(width=0),
                                 name="90% interval"))
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                 line=dict(dash="dash", color=GREY), name="perfect agreement"))
        out = set(h4.get("specimens_outside", []))
        fig.add_trace(go.Scatter(
            x=[p["measured"] for p in pts], y=[p["predicted"] for p in pts], mode="markers+text",
            text=[p["specimen"] if p["specimen"] in out else "" for p in pts],
            textposition="top center",
            marker=dict(size=10, color=[RED if p["specimen"] in out else NAVY for p in pts]),
            customdata=[p["specimen"] for p in pts],
            hovertemplate="%{customdata}: measured %{x:,.0f}, predicted %{y:,.0f}<extra></extra>",
            name="B3 leave-one-out prediction"))
        parity(fig)
        pg.chart(fig)
        pg.caption(f"The band is ±1.645 times the leave-one-out residual spread, σ = {s:.3f} "
                   "decades. Red: the cylinder outside it.")
        if h4["slope"] - 0.90 < 0.01:
            pg.p(f"The slope passes at the lower edge, {h4['slope']:.3f} against 0.90: B3 "
                 "slightly compresses the range of lives, over-predicting short ones and "
                 "under-predicting long ones.")

    def ver():
        tbl = pd.DataFrame(vr["table"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_s"], mode="lines+markers",
                                  name="stress slope", line=dict(color=NAVY)))
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_a"], mode="lines+markers",
                                  name="defect slope", line=dict(color=RED)))
        fig2.update_layout(xaxis_type="log", yaxis_type="log", height=H_AUX,
                           xaxis_title="scatter in synthetic lives (decades)",
                           yaxis_title="error in recovered slope", legend=LEGEND)
        pg.chart(fig2)
        pg.caption(f"True slopes b_s = {vr['b_s_true']:+.4f}, b_a = {vr['b_a_true']:+.4f} "
                   f"(m = {vr['m_true']}). Both errors fall in proportion to the scatter.")
        pv = J(*D, "pinn_verification.json")
        if pv:
            pg.p(f"For the network: {pv['n']} synthetic specimens with known slopes. As the "
                 "physics weight λ rises from zero, the network's own learned gradient should "
                 "move to the true slope.")
            t = pd.DataFrame(pv["table"])
            pg.table(pd.DataFrame({"λ": t["lam"].map(lambda v: f"{v:.2f}"),
                                   "Stress-slope residual": t["b_s_resid_rms"].map(lambda v: f"{v:.4f}"),
                                   "Defect-slope residual": t["b_a_resid_rms"].map(lambda v: f"{v:.4f}"),
                                   "LOO RMSE (decades)": t["loo_rmse"].map(lambda v: f"{v:.3f}")}),
                     num=("λ", "Stress-slope residual", "Defect-slope residual", "LOO RMSE (decades)"))
            pg.caption("λ = 10 was chosen from this table before the real fit was run.")
    pg.details("Verification of the fitting methods against known answers", ver)

    if fail:
        pg.h2("What ends the part")
        margin = fail["kic"] / fail["kmax_at_collapse"]
        pg.p(f"At the collapse depth the stress intensity reaches at most "
             f"{fail['kmax_at_collapse']:.1f} MPa√m, against a fracture toughness of about "
             f"{fail['kic']:.0f} MPa√m, a margin of {margin:.0f}. 316 steel is too tough to "
             "fracture fast in a 5 mm bar: the part fails when the cracked section can no "
             "longer carry the load, which is how path A defines the final crack depth.")
        loc = fail["fracture_location"]
        pg.table(pd.DataFrame({"Fracture location": list(loc), "Cylinders": list(loc.values())}),
                 num=("Cylinders",))
        pg.p("The fillet-radius failures start at a stress concentration rather than a defect "
             "and lie outside the defect-driven model.")

    # explorer
    pg.h2("Try the models")
    pg.p("The sliders drive the models themselves: path A integrates the Paris law with its "
         "literature constants, B3 uses its fitted coefficients, and the network is the "
         "trained network, evaluated in your browser.")
    pinn, pinn_py, meas = pinn_export()
    b3 = next(d for d in res["ladder"] if d["model"].startswith("B3"))
    params = dict(Y=Y_SURFACE, C=C_PARIS, m=M_PARIS, R=0.1, gauge_r_mm=2.54, sigma_uts=640.0,
                  b3=dict(b_ds=b3["coef"][0], b_sa=b3["coef"][1], b0=b3["coef"][2]), pinn=pinn,
                  measured=[dict(ds=r["ds"], sa=r["sqrt_area_gauge_um"], Nf=r["Nf"]) for r in meas])
    # build-time values the browser's arithmetic is checked against

    def af(smax):
        frac = 1 - smax / 640.0
        return 0.05 if frac <= 0 else float(np.clip(np.sqrt(2 * frac) * 2.54, 0.05, 2.54))
    checks = []
    for ds, sa in ((270, 345), (405, 1017), (300, 450), (450, 200)):
        checks.append(dict(ds=ds, sa=sa,
                           pathA=life_numeric(sa * 1e-3, af(ds / 0.9), ds, dkth=0.0,
                                              Y=Y_SURFACE, C=C_PARIS, m=M_PARIS),
                           b3=10 ** (b3["coef"][2] + b3["coef"][0] * np.log(ds) +
                                     b3["coef"][1] * np.log(sa)),
                           pinn=pinn_py(sa, ds)))
    params["checks"] = checks
    pid = pg.data(params, "params")
    pg.add(f'<div id="explorer" data-params="{pid}"><div class="controls">'
           '<div class="slider-row"><label for="ex-ds">Stress range Δσ</label><input id="ex-ds" '
           'type="range" min="200" max="500" step="5" value="405"><output id="ex-ds-out"></output>'
           '</div><div class="slider-row"><label for="ex-sa">Governing defect √area</label>'
           '<input id="ex-sa" type="range" min="150" max="1200" step="10" value="450">'
           '<output id="ex-sa-out"></output></div></div>'
           '<div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Predicted '
           'cycles</th></tr></thead><tbody><tr><td>Path A, nothing fitted</td><td class="num" '
           'id="ex-a"></td></tr><tr><td>B3, constrained regression</td><td class="num" '
           'id="ex-b3"></td></tr><tr><td>C1, physics-informed network</td><td class="num" '
           'id="ex-pinn"></td></tr></tbody></table></div><p class="caption" id="ex-cap"></p>'
           '<div class="chart" id="ex-chart" role="img" aria-label="life against defect size">'
           "</div></div>")

    pg.h2("Limits")
    pg.p("- Only 14 of the 60 cylinders have a measured defect size; the ladder's A rungs use "
         "bulk porosity, an average where the physics asks for the extreme.\n"
         "- With 14 points the leave-one-out scores carry real variance; read the ladder as "
         "indicative, not as a certified ranking.\n"
         "- Path A's threshold of zero is a short-crack argument, supported by the threshold "
         "sweep in Phase 3, not a measured short-crack law.")
    return pg, "Phase 4: validation", "Validation against measured lives and the comparison models."


def page_conclusions():
    pg = Page("conclusions")
    _h2 = J("phase2_peridynamics", "h2_first_bond.json")
    _bf = J("phase_mc_bayesian", "bayesian_results.json")
    _h4 = J("phase4_validation", "h4_calibration.json")
    _r = _bf["verify_recovery"]
    _st = next((i for i, h in enumerate(_r["history"], 1)
                if abs(h["Nf_mean"] - _r["Nf_true"]) / _r["Nf_true"] <= 0.20), None)
    met = {"H1": STUDY_METRICS["Dice"] >= 0.80,
           "H2": _h2["fraction_by_area"] >= _h2["criterion"],
           "H3": bool(_st and _st <= 3 and min(h["ess"] for h in _r["history"]) >= 250),
           "H4": _h4["slope_met"] and _h4["coverage_met"]}
    words = ["None", "One", "Two", "Three", "Four"]
    missed = [k for k, v in met.items() if not v]
    pg.h1("Conclusions", f"{words[sum(met.values())]} of the four hypotheses are met." + (
        " The one that is not, where cracks start, returns a clear negative that points at "
        "the severity criterion." if missed == ["H2"] else
        (f" Not met: {', '.join(missed)}." if missed else "")))
    pg.p("Each objective was tied to a hypothesis with an acceptance criterion fixed before "
         "testing. Each outcome below is read from the result files.")

    def hyp(code, name, criterion, outcome, kind, evidence):
        pg.h3(f"{code}. {name}")
        pg.p(f"Accepted only if {criterion}.")
        pg.verdict(outcome, kind)
        pg.p(evidence)

    dice = STUDY_METRICS["Dice"]
    ok = dice >= 0.80
    hyp("H1", "Segmentation accuracy", "the mean Dice on withheld data is at least 0.80",
        f"Criterion {'met' if ok else 'not met'}: Dice {dice:.3f}.", "met" if ok else "no",
        f"Mean over five folds, {min(FOLD_DICE):.3f} to {max(FOLD_DICE):.3f} per fold. The "
        "network finds the defects well enough for their size to be trusted.")
    h2 = J("phase2_peridynamics", "h2_first_bond.json")
    if h2:
        ok = h2["fraction_by_area"] >= h2["criterion"]
        hyp("H2", "Crack initiation site",
            f"the crack starts at the largest defect in at least {h2['criterion']:.0%} of sections",
            f"Criterion {'met' if ok else 'not met'}: {h2['hits_by_area']} of {h2['n']} "
            f"({h2['fraction_by_area']:.1%}).", "met" if ok else "no",
            f"Ranked by Murakami severity instead of size, {h2['hits_by_severity']} of "
            f"{h2['n']}. Size alone does not decide where a crack starts: closeness to the "
            "surface and neighbouring defects matter too.")
    bf = J("phase_mc_bayesian", "bayesian_results.json")
    if bf:
        r = bf["verify_recovery"]
        step = next((i for i, h in enumerate(r["history"], 1)
                     if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
        ess = min(h["ess"] for h in r["history"])
        ok_t, ok_e = step is not None and step <= 3, ess >= 250
        kind = "met" if ok_t and ok_e else "part" if (step or ok_e) else "no"
        word = {"met": "Criterion met", "part": "Partly met", "no": "Criterion not met"}[kind]
        hyp("H3", "Bayesian convergence",
            "the posterior is within 20% of the true life by the third inspection with an "
            "effective sample size of at least 250",
            f"{word}: within 20% at inspection {step}; effective sample size at least "
            f"{ess:,.0f}.", kind,
            "With the true growth rate placed in the tails of the prior instead, the filter "
            "still converges by the third inspection, but its effective sample size can dip "
            "below 250 before resampling restores it. The speed condition holds throughout; "
            "the sample-size condition fails where the prior is badly wrong.")
    h4 = J("phase4_validation", "h4_calibration.json")
    if h4:
        ok = h4["slope_met"] and h4["coverage_met"]
        hyp("H4", "Predictive calibration",
            "at least 90% of measured lives fall inside the 90% interval and the calibration "
            "slope lies between 0.90 and 1.10",
            f"Criterion {'met' if ok else 'not met'}: {h4['n_inside']} of {h4['n']} inside "
            f"({h4['coverage']:.1%}); slope {h4['slope']:.3f}.", "met" if ok else "no",
            f"Tested on {h4['model']}, with the interval built from its leave-one-out residual "
            "spread. The slope passes at the lower edge.")

    cmp_d = J("comparison_model", "comparison_results.json") or {}
    pa = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    lad = {r["model"].split()[0]: r for r in cmp_d.get("ladder", [])}
    pg.h2("What this work shows")
    pg.p(f"- A part's own CT scan predicts its own fatigue life to within **{pa:.2f}×** on "
         "held-out cylinders, with nothing fitted to the fatigue lives.\n"
         f"- The best statistical fit needs {lad['B5']['params']} parameters to reach "
         f"{lad['B5']['loo_factor']:.2f}×; a physics-informed neural network with "
         f"{lad['C1']['params']} reaches {lad['C1']['loo_factor']:.2f}×. Measuring the defect "
         "does the work that fitting cannot.\n"
         "- Propagated through the same model, the scatter in defect size and growth rate "
         "reproduces the measured life distribution of 46 further cylinders not used to build "
         "it.\n"
         "- Each numerical method was checked against a known answer before it was used on "
         "measured data, and a life can be followed by hand from defect size to cycles.")
    pg.note("A part with a benign defect population can be kept in service instead of scrapped "
            "on a batch rule, and a part with a severe one withdrawn before it fails, with the "
            "uncertainty stated rather than hidden in a safety factor.")
    mc = J("phase_mc_bayesian", "monte_carlo_results.json") or {}
    v300 = mc.get("validation_independent", {}).get("300")
    runout = (f"\n- With the threshold removed for short cracks, the model predicts no "
              f"run-outs, while {v300['n_runout']} of the {v300['n_tested']} independent "
              "cylinders at 300 MPa ran out." if v300 and v300["n_runout"] else "")
    pg.h2("Limits")
    pg.p("- One alloy and one dataset: 316H from one machine and one powder lot, with fourteen "
         "held-out cylinders carrying the validation.\n"
         "- Where a crack starts is not predicted by size alone (H2).\n"
         "- The peridynamic model cannot carry an absolute life at attainable grid resolution; "
         "it gives the site and the path, and the life comes from the crack-growth integral.\n"
         "- Fillet-radius failures start at a stress concentration, not a defect, and lie "
         "outside the defect-driven model." + runout)
    pg.h2("What comes next")
    pg.p("- A severity measure with surface and interaction terms, to re-test H2.\n"
         "- A threshold that grows from the short-crack to the long-crack value as the crack "
         "lengthens, so the population model reproduces the run-outs at 300 MPa.\n"
         "- A stress-concentration model at the fillet radius, so both failure locations are "
         "covered.\n"
         "- Each specimen's own Bayesian interval for H4, in place of the residual-based one.")
    return pg, "Conclusions", "The four hypotheses, what the work shows, its limits."


# ====================================================================== main
def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "assets"))
    os.makedirs(os.path.join(OUT, "img"))
    for f in os.listdir(STATIC):
        shutil.copy2(os.path.join(STATIC, f), os.path.join(OUT, "assets", f))
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    ct = ct_frames()
    for build in (lambda: page_overview(ct), lambda: page_defects(ct),
                  lambda: page_initiation(ct), page_life, page_validation, page_conclusions):
        pg, title, desc = build()
        doc = render_page(pg, title, desc)
        with open(os.path.join(OUT, pg.slug + ".html"), "w", encoding="utf-8") as fh:
            fh.write(doc)
        print(f"  {pg.slug}.html  {len(doc) / 1024:,.0f} kB, {len(pg.json)} data blocks")
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"site/: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

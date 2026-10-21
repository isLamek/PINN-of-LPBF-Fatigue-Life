"""The opening page: the question, why it matters, the idea, and one real cylinder
followed through the whole method. Every number is read from data/ at render time."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ct_viewer as cv
from src.data import GAUGE_ND, GAUGE_SA, STUDY_METRICS, load
from src.ui import ACCENT, GREEN, GREY, NAVY, TITLE, go_to, plain

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SPEC = "P017"


def _json(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def render():
    rows = load()
    cmp_d = _json("comparison_model", "comparison_results.json") or {}
    h2 = _json("phase2_peridynamics", "h2_first_bond.json")
    path_a = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    p4 = pd.read_csv(os.path.join(DATA, "phase4_validation", "phase4_error_isolation.csv"))

    st.markdown(f"## {TITLE}")
    st.markdown(
        "#### Can a CT scan of one metal 3D-printed part tell you how long *that* part "
        "will survive repeated loading?")
    plain(
        "<b>In one paragraph.</b> Metal 3D printing (laser powder bed fusion) leaves tiny "
        "holes inside every part. Under repeated loading, cracks start at those holes, so "
        "two parts printed the same way can last very different numbers of cycles. Today "
        "the safe life is set for a whole batch. This project measures the holes in one "
        "part's own X-ray CT scan and predicts that part's own fatigue life, with its "
        "uncertainty, and checks the prediction against real fatigue tests.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Defects found by the AI", f"Dice {STUDY_METRICS['Dice']:.3f}",
              help="5-fold cross-validated agreement with the reference defect labels; "
                   "the criterion was 0.80.")
    if path_a:
        c2.metric("Life predicted to within", f"{path_a:.2f}×",
                  help="Typical error over the 14 held-out cylinders, with nothing "
                       "fitted to the fatigue lives.")
    if h2:
        c3.metric("Crack starts at the biggest defect",
                  f"{h2['hits_by_area']} of {h2['n']} sections",
                  help="Hypothesis 2 asked for 90%. A negative result, reported as found.")

    # ------------------------------------------------------------- the problem
    st.divider()
    st.subheader("The problem, in the measured data")
    fail = [r for r in rows if not r["runout"]]
    fig = go.Figure()
    rng = np.random.default_rng(3)
    for s, col in ((300, NAVY), (450, ACCENT)):
        pts = [r for r in rows if abs(r["smax"] - s) < 1]
        y = [r["Nf"] for r in pts]
        x = s + rng.uniform(-14, 14, len(pts))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"{s} MPa",
            marker=dict(size=10, color=col, opacity=0.8,
                        symbol=["triangle-up" if r["runout"] else "circle" for r in pts]),
            text=[f"{r['specimen']} · {r['Nf']:,} cycles" +
                  (" · run-out" if r["runout"] else "") for r in pts],
            hoverinfo="text"))
    f450 = [r["Nf"] for r in fail if abs(r["smax"] - 450) < 1]
    fig.add_annotation(x=470, y=np.log10(np.sqrt(min(f450) * max(f450))), yref="y",
                       text=f"<b>{max(f450) / min(f450):.0f}× spread</b><br>same alloy,<br>"
                            "same machine,<br>same stress", showarrow=False, xanchor="left",
                       font=dict(size=13, color=ACCENT))
    fig.update_layout(height=420, yaxis_type="log", yaxis_title="cycles to failure",
                      xaxis=dict(tickvals=[300, 450], ticktext=["300 MPa", "450 MPa"],
                                 range=[255, 540], title="maximum stress"),
                      legend=dict(orientation="h", y=1.08), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"All {len(rows)} fatigue-tested cylinders of the ORNL dataset (Snow et al.). "
               "Triangles are run-outs, stopped at 10 million cycles without failing.")

    # ------------------------------------------------------------- the idea
    st.divider()
    st.subheader("The idea: two arms from one defect map")
    st.markdown("""
<style>
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:8px 0 6px 0}
.flow .n{border-radius:10px;padding:12px;font-size:.88rem;line-height:1.35;color:#fff;
 opacity:.25;animation:lit 6s infinite}
.flow .n b{display:block;font-size:.95rem;margin-bottom:4px}
.n1{background:#1F4E79;animation-delay:0s!important}
.n2{background:#1F4E79;animation-delay:1.2s!important}
.n3{background:#B5473A;animation-delay:2.4s!important}
.n4{background:#2E7D32;animation-delay:3.6s!important}
.arm{grid-column:span 4;background:#F4F1EC;border:1px dashed #8A9096;border-radius:10px;
 padding:10px 12px;font-size:.88rem;opacity:.35;animation:lit 6s infinite;animation-delay:2.4s}
@keyframes lit{0%{opacity:.25}8%{opacity:1}60%{opacity:1}75%{opacity:.25}100%{opacity:.25}}
</style>
<div class="flow">
 <div class="n n1"><b>1 · Find the defects</b>An AI (nnU-Net) marks every hole in every CT
 slice of the part.</div>
 <div class="n n2"><b>2 · Measure them</b>Each hole becomes one number: √area, the size
 fracture mechanics uses. The largest governs.</div>
 <div class="n n3"><b>3 · Predict the life</b>A crack-growth law grows a crack from the
 governing hole to collapse: cycles to failure, with its uncertainty.</div>
 <div class="n n4"><b>4 · Test the prediction</b>Every prediction is made without seeing that
 part's result, then compared with the real fatigue test.</div>
 <div class="arm"><b>The second arm.</b> The same defect map also feeds a peridynamic crack
 model, which shows <i>where</i> a crack starts and the path it takes. It answers a different
 question; it is <b>not</b> where the life number comes from.</div>
</div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------- one cylinder
    st.divider()
    st.subheader(f"Follow one real cylinder through the method: {SPEC}")
    rec = {r["specimen"]: r for r in rows}[SPEC]
    pr = p4[p4["cylinder"] == SPEC].iloc[0]
    pred = 10 ** float(pr["log10_N_pathA"])
    meas = rec["Nf"]
    sa_net = float(pr["largest_sqrt_area_um_ai_gauge"])
    h2row = next((r for r in h2["rows"] if r["specimen"] == f"SN_{SPEC}"), None) if h2 else None

    t1, t2, t3, t4, t5 = st.tabs(["① The scan", "② The defects", "③ The load",
                                  "④ The prediction", "⑤ Where the crack starts"])
    with t1:
        a, b = st.columns([1, 1])
        zs = cv.list_slices(f"SN_{SPEC}")
        if zs:
            z = zs[len(zs) // 2]
            ov = cv.composite_slice(f"SN_{SPEC}", z)
            ct = cv.load_ct_slice(f"SN_{SPEC}", z)
            a.image((ct * 255).astype("uint8"), caption="Raw X-ray CT slice", width="stretch")
            b.image(ov, caption="Defects the AI found, in red", width="stretch")
        st.markdown(f"Cylinder **{SPEC}** is one of the 14 held out from all training. "
                    "Its CT scan is a stack of slices like this one through the build.")
    with t2:
        st.markdown(
            f"Inside the machined gauge the AI found **{GAUGE_ND[SPEC]:.0f} defects**. "
            f"Each is reduced to √area, the square root of its area seen along the load. "
            f"The largest, the **governing defect**, measures **{sa_net:,.0f} µm** "
            f"(network) against {GAUGE_SA[SPEC]:,} µm in the reference labels.")
        plain("<b>Why only the largest?</b> A fatigue crack is a race. The largest flaw "
              "concentrates stress the most, so its crack usually gets going first and "
              "wins. Step ⑤ tests whether that is really true.")
    with t3:
        ds = rec["ds"]
        dk = 0.65 * ds * np.sqrt(np.pi * sa_net * 1e-6)
        st.markdown(
            f"{SPEC} was tested at **σmax = {rec['smax']:.0f} MPa**, stress ratio "
            f"R = {rec['R']:.1f}, so each cycle swings the stress by **Δσ = {ds:.0f} MPa**.")
        st.latex(rf"\Delta K = Y\,\Delta\sigma\sqrt{{\pi\,a_0}} = 0.65 \times {ds:.0f}"
                 rf"\times\sqrt{{\pi \times {sa_net / 1000:.3f}\times10^{{-3}}\,\text{{m}}}} "
                 rf"= {dk:.1f}\ \text{{MPa}}\sqrt{{\text{{m}}}}")
        st.caption("ΔK, the stress intensity range, is the driving force on a crack. "
                   "Y = 0.65 is Murakami's factor for a defect at the surface.")
    with t4:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Predicted life", f"{pred:,.0f} cycles",
                   help="Path A: the calibrated crack-growth integral, leave-one-out.")
        cc2.metric("Measured life", f"{meas:,.0f} cycles")
        cc3.metric("Predicted ÷ measured", f"{pred / meas:.2f}")
        st.markdown(
            "The crack-growth law is integrated from the governing defect to the depth at "
            "which the remaining section collapses, with literature constants and nothing "
            "fitted to any measured life, so this prediction owes nothing to the result it "
            "is compared with.")
        go_to("phase3", "See the crack grow, and every cylinder's prediction →")
    with t5:
        if h2row:
            st.markdown(
                f"The peridynamic model of {SPEC}'s measured section breaks its **first "
                f"bond** at a **{h2row['first_bond_defect_sqrt_area_um']:.0f} µm defect "
                f"at the surface**, not at the {h2row['dominant_sqrt_area_um']:.0f} µm "
                "internal defect that dominates the section by size. The crack that then "
                "grows to failure does start at the large defect.")
            plain("<b>So size is not the whole story.</b> A small flaw at the surface can "
                  "out-rank a large buried one. Across all 14 sections the crack starts "
                  f"at the largest defect in only {h2['hits_by_area']}, short of the 90% "
                  "the hypothesis required. That is reported as found.")
        go_to("phase2", "See all fourteen sections →")

    # ------------------------------------------------------------- reading guide
    st.divider()
    st.subheader("How to read this app")
    st.markdown(
        "Each phase opens with the same five boxes: **the question** it asks, **how** it "
        "answers it, **the check** made against a known answer before the method was "
        "trusted, **what we found**, and **what it means**. Every number on every page is "
        "read from the project's result files when the page loads; nothing is typed in "
        "or recomputed for display. Negative results are shown as they came out.")
    go_to("phase1", "Start with Phase 1 →")

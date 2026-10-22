"""The opening page: the question, the evidence that it matters, the method, one real
cylinder followed through it, and the results. Every number is read from data/."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ct_viewer as cv
from src.data import GAUGE_ND, GAUGE_SA, STUDY_METRICS, load
from src.figstyle import H_AUX, H_MAIN, LEGEND, MEAS, PARITY, cycle_ticks, parity
from src.ui import GREY, MUTED, NAVY, RED, TITLE, go_to, note, show

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
    h4 = _json("phase4_validation", "h4_calibration.json")
    lad = {r["model"].split()[0]: r for r in cmp_d.get("ladder", [])}
    path_a = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    p4 = pd.read_csv(os.path.join(DATA, "phase4_validation", "phase4_error_isolation.csv"))

    st.markdown(f"# {TITLE}")
    st.markdown('<div class="byline">Lamek Shali Indongo, BSc (Hons) Mechanical '
                'Engineering, University of Namibia, 2026</div>', unsafe_allow_html=True)
    st.markdown(
        "Metal parts made by laser powder bed fusion contain small internal defects. "
        "Fatigue cracks start at them, so parts printed the same way can last very "
        "different numbers of load cycles, and today a single safety factor covers the "
        "whole batch. This work predicts the fatigue life of an individual part from the "
        "defects measured in its own X-ray CT scan, states the uncertainty, and tests the "
        "prediction against real fatigue tests.")

    # ------------------------------------------------------------- the problem
    st.markdown("## The problem in the measured data")
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
    show(fig)
    st.markdown(
        f"All {len(rows)} fatigue-tested cylinders of the Oak Ridge dataset (open "
        f"triangles: run-outs, stopped unbroken at 10 million cycles). At 450 MPa, "
        f"{len(f450)} nominally identical cylinders failed between {min(f450):,} and "
        f"{max(f450):,} cycles: a **{max(f450) / min(f450):.0f}-fold** spread.")

    # ------------------------------------------------------------- the method
    st.markdown("## The method")
    fw = os.path.join(ROOT, "assets", "framework_slide.png")
    if os.path.exists(fw):
        st.image(fw, width="stretch")
    st.markdown(
        "One defect map feeds two arms. The **life arm** (navy) takes the size of the "
        "largest defect into a crack-growth integral and gives the fatigue life with its "
        "uncertainty. The **crack-path arm** (red) is a peridynamic model of the measured "
        "section; it shows where the crack starts and which way it runs, and is not used as "
        "a life. Every step is checked against a known answer before it touches measured "
        "data.")

    # ------------------------------------------------------------- one cylinder
    st.markdown(f"## One cylinder, from scan to life: {SPEC}")
    rec = {r["specimen"]: r for r in rows}[SPEC]
    pr = p4[p4["cylinder"] == SPEC].iloc[0]
    pred, meas = 10 ** float(pr["log10_N_pathA"]), rec["Nf"]
    sa_net = float(pr["largest_sqrt_area_um_ai_gauge"])
    h2row = next((r for r in h2["rows"] if r["specimen"] == f"SN_{SPEC}"), None) if h2 else None
    a, b = st.columns(2, gap="medium")
    zs = cv.list_slices(f"SN_{SPEC}")
    if zs:
        z = zs[len(zs) // 2]
        a.image((cv.load_ct_slice(f"SN_{SPEC}", z) * 255).astype("uint8"),
                caption="X-ray CT slice", width="stretch")
        b.image(cv.composite_slice(f"SN_{SPEC}", z),
                caption="Defects found by the network, in red", width="stretch")
    ds = rec["ds"]
    dk = 0.65 * ds * np.sqrt(np.pi * sa_net * 1e-6)
    st.markdown(
        f"1. **Scan.** {SPEC} is one of the 14 cylinders held out from all training.\n"
        f"2. **Defects.** The network finds {GAUGE_ND[SPEC]:.0f} defects in the gauge; the "
        f"largest measures √area = **{sa_net:,.0f} µm** (reference labels: "
        f"{GAUGE_SA[SPEC]:,} µm).\n"
        f"3. **Load.** Tested at {rec['smax']:.0f} MPa and R = {rec['R']:.1f}, so "
        f"Δσ = {ds:.0f} MPa and the driving force at that defect is "
        f"ΔK = 0.65 × {ds:.0f} × √(π × {sa_net / 1000:.3f} mm) = **{dk:.1f} MPa√m**.\n"
        f"4. **Life.** Predicted **{pred:,.0f}** cycles against **{meas:,}** measured "
        f"(ratio {pred / meas:.2f}), with nothing fitted to any measured life.\n" +
        (f"5. **Crack start.** The first bond breaks at a "
         f"{h2row['first_bond_defect_sqrt_area_um']:.0f} µm surface defect, not at the "
         f"{h2row['dominant_sqrt_area_um']:.0f} µm internal one that later carries the "
         "crack.\n" if h2row else ""))

    # ------------------------------------------------------------- results
    st.markdown("## Results against the hypotheses")
    res = []
    dice = STUDY_METRICS["Dice"]
    res.append(("H1  Segmentation accuracy", "mean Dice ≥ 0.80", f"{dice:.3f}",
                "met" if dice >= 0.80 else "not met"))
    if h2:
        res.append(("H2  Crack initiation site", "≥ 90% start at the largest defect",
                    f"{h2['hits_by_area']} of {h2['n']} ({h2['fraction_by_area']:.0%})",
                    "met" if h2["fraction_by_area"] >= h2["criterion"] else "not met"))
    bf = _json("phase_mc_bayesian", "bayesian_results.json")
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
    st.dataframe(pd.DataFrame(res, columns=["Hypothesis", "Criterion", "Result", "Outcome"]),
                 hide_index=True, width="stretch")
    if path_a and "B5" in lad and "C1" in lad:
        st.markdown(
            f"The life prediction is within **{path_a:.2f}×** of the measured lives of the "
            f"14 held-out cylinders with nothing fitted to them. The best statistical fit "
            f"needs {lad['B5']['params']} parameters for {lad['B5']['loo_factor']:.2f}×, "
            f"and a physics-informed neural network {lad['C1']['params']} parameters for "
            f"{lad['C1']['loo_factor']:.2f}×.")
    note("Every number on these pages is read from the project's result files when the "
         "page loads. Negative results are reported as they came out.")
    go_to("phase1", "Continue to Phase 1: finding the defects")

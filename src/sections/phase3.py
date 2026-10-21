import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import ACCENT, GREEN, GREY, NAVY, badge, plain, play, thinking

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PD_DATA = os.path.join(ROOT, "data", "phase2_peridynamics")
MCB_DATA = os.path.join(ROOT, "data", "phase_mc_bayesian")
CMP_DATA = os.path.join(ROOT, "data", "comparison_model")
P4_DATA = os.path.join(ROOT, "data", "phase4_validation")

# Path A's constants, as scripts/phase4_predict.py uses them for the validated result:
# literature values for 316L, Murakami's surface factor, threshold removed.
M_PARIS, Y_SURF, R_RATIO = 3.94, 0.65, 0.1


def _p4():
    p = os.path.join(P4_DATA, "phase4_error_isolation.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[np.isfinite(pd.to_numeric(d["log10_N_pathA"], errors="coerce"))].copy()
    d["predicted"] = 10 ** pd.to_numeric(d["log10_N_pathA"])
    d["measured"] = 10 ** pd.to_numeric(d["measured_log10_cycles"])
    return d


def _crack_growth(row, n=48):
    """The path A integral for one specimen, as a curve a(N).

    With the threshold removed the Paris integral has a closed form,
    N(a) = (a^p - a0^p) / (p C (Y ds sqrt(pi/1000))^m), p = 1 - m/2, so N(a)/N_f
    depends only on a0 and a_c; it is scaled here to the specimen's own path A life
    from the result file, so the curve ends exactly at the validated number."""
    a0 = float(row["largest_sqrt_area_um_ai_gauge"]) / 1000.0
    ac = float(row["a_c_mm"])
    Nf = float(row["predicted"])
    p = 1.0 - M_PARIS / 2.0
    a = np.geomspace(a0, ac, 400)
    frac = (a ** p - a0 ** p) / (ac ** p - a0 ** p)
    N = frac * Nf
    ds = float(row["stress_level_MPa"]) * (1 - R_RATIO)
    dK = Y_SURF * ds * np.sqrt(np.pi * a / 1000.0)
    t = np.linspace(0, 1, n)
    idx = np.searchsorted(frac, t).clip(0, len(a) - 1)
    return a0, ac, Nf, a, N, dK, idx


def _render_growth(p4):
    st.subheader("Watch one crack grow, from the governing defect to collapse")
    spec = st.selectbox(
        "Held-out cylinder", list(p4["cylinder"]),
        index=list(p4["cylinder"]).index("P017") if "P017" in set(p4["cylinder"]) else 0,
        format_func=lambda c: f"{c} · {int(p4.set_index('cylinder').loc[c, 'stress_level_MPa'])} MPa",
        key="p3_growth_spec")
    row = p4.set_index("cylinder").loc[spec]
    a0, ac, Nf, a, N, dK, idx = _crack_growth(row)
    meas = float(row["measured"])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("crack depth against cycles",
                                        "driving force ΔK against crack depth"))

    def data_for(k):
        j = idx[k]
        return [
            go.Scatter(x=N[:j + 1], y=a[:j + 1], mode="lines", line=dict(color=ACCENT, width=3),
                       name="crack depth", showlegend=False),
            go.Scatter(x=[N[j]], y=[a[j]], mode="markers",
                       marker=dict(size=13, color=ACCENT, line=dict(color="white", width=2)),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=a[:j + 1], y=dK[:j + 1], mode="lines", line=dict(color=NAVY, width=3),
                       showlegend=False, xaxis="x2", yaxis="y2"),
        ]

    for tr in data_for(0):
        fig.add_trace(tr, row=1, col=2 if tr.xaxis == "x2" else 1)
    fig.frames = [go.Frame(data=data_for(k), traces=[0, 1, 2],
                           name=f"{N[idx[k]]:,.0f}") for k in range(len(idx))]
    fig.add_hline(y=ac, line_dash="dot", line_color=GREY, row=1, col=1,
                  annotation_text="collapse depth", annotation_position="bottom right")
    fig.add_vline(x=meas, line_dash="dash", line_color=GREEN, row=1, col=1,
                  annotation_text="measured life", annotation_position="top left")
    fig.update_xaxes(range=[0, max(Nf, meas) * 1.08], title="cycles", row=1, col=1)
    fig.update_yaxes(type="log", range=[np.log10(a0 * 0.8), np.log10(ac * 1.3)],
                     title="crack depth a (mm)", row=1, col=1)
    fig.update_xaxes(type="log", range=[np.log10(a0 * 0.8), np.log10(ac * 1.3)],
                     title="crack depth a (mm)", row=1, col=2)
    fig.update_yaxes(range=[0, dK.max() * 1.1], title="ΔK (MPa√m)", row=1, col=2)
    fig.update_layout(height=430, margin=dict(t=40, b=110))
    play(fig, frame_ms=120, transition_ms=0, slider_prefix="cycles ")
    st.plotly_chart(fig, width="stretch")

    half = float(np.interp(0.5, N / Nf, a))
    k2 = float(np.interp(2 * a0, a, N)) / Nf if 2 * a0 < ac else 1.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Governing defect a₀", f"{a0 * 1000:,.0f} µm")
    c2.metric("Predicted life", f"{Nf:,.0f}", delta=f"measured {meas:,.0f}",
              delta_color="off")
    c3.metric("Life used before the crack doubles", f"{k2:.0%}")
    plain(
        f"<b>Read the curve.</b> The crack hardly moves for most of the life: "
        f"{k2:.0%} of {spec}'s predicted cycles pass before it doubles from "
        f"{a0 * 1000:,.0f} µm, and at half-life it is still only {half * 1000:,.0f} µm "
        "deep. ΔK grows with the crack, so growth accelerates and the end comes "
        "quickly. That is why the starting defect size decides the life, and why "
        "measuring it from each part's own CT scan matters.")


def render():
    st.title("Phase 3 · Predict the life")
    badge("VALIDATED ON HELD-OUT CYLINDERS", "real")
    p4 = _p4()
    cmp_path = os.path.join(CMP_DATA, "comparison_results.json")
    cmp_d = json.load(open(cmp_path)) if os.path.exists(cmp_path) else {}
    fac = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    thinking(
        "How many load cycles will this particular part survive?",
        "Grow a crack from its governing defect to collapse with the Paris law, using "
        "literature constants for 316L. Then propagate the scatter (Monte Carlo) and "
        "update it with inspections (Bayesian filter).",
        "The integral is checked against its closed form; Monte Carlo and the Bayesian "
        "filter are each checked on a problem with a known answer.",
        (f"Typical error {fac:.2f}× against the measured lives of the 14 held-out "
         "cylinders, with no constant fitted to the fatigue data." if fac else
         "Validation file not found."),
        "A part's own defect size, measured from its own CT scan, predicts its own life.")

    st.divider()
    st.subheader("The physics, in three lines")
    st.markdown("The driving force on a crack of depth $a$ grows with its size:")
    st.latex(r"\Delta K(a) = Y \cdot \Delta\sigma \cdot \sqrt{\pi a}")
    st.markdown("The Paris law says how fast the crack grows under that driving force:")
    st.latex(r"\frac{da}{dN} = C \cdot \big(\Delta K - \Delta K_{th}\big)^{m}")
    st.markdown("Add up the cycles from the governing defect $a_0 = \\sqrt{\\text{area}}$ to "
                "the depth $a_f$ at which the remaining section collapses:")
    st.latex(r"N = \int_{a_0}^{a_f} \frac{da}{C\,\big(\Delta K(a) - \Delta K_{th}\big)^{m}}")
    st.info(
        "**The constants, and the one decision.** $C$ and $m$ are literature values for "
        "316L (Merot et al.), $Y$ = 0.65 is Murakami's factor for a surface defect, and "
        "none of them is fitted to this dataset's fatigue lives. The decision is the "
        "threshold: at the literature value $\\Delta K_{th}$ = 5 MPa√m, most of these "
        "gauge defects sit below it and the model predicts they never fail, yet every one "
        "of these cylinders failed in test. Cracks growing from defects this small are in "
        "the short-crack regime, where the long-crack threshold does not hold, so the "
        "validated prediction (path A) is evaluated with $\\Delta K_{th}$ = 0.",
        icon="ℹ️")

    if p4 is not None:
        st.divider()
        _render_growth(p4)

    st.divider()
    st.subheader("Every held-out cylinder, predicted then compared")
    p4m = cmp_d.get("phase4", {})

    def _fac(key):
        v = p4m.get(key)
        return v.get("factor") if isinstance(v, dict) else None

    c1, c2, c3 = st.columns(3)
    if _fac("log10_N_pred") is not None:
        c1.metric("A priori: threshold kept", f"{_fac('log10_N_pred'):.1f}×",
                  help="The same physics with the long-crack threshold of 5 MPa√m and the "
                       "whole-blank defect size: badly over-predicts.")
    if _fac("log10_N_pathA") is not None:
        c2.metric("Path A · the AI's defect sizes", f"{_fac('log10_N_pathA'):.2f}×",
                  delta=f"bias {p4m['log10_N_pathA']['bias']:+.2f} decades",
                  delta_color="off")
    if _fac("log10_N_pathB") is not None:
        c3.metric("Path B · reference defect sizes", f"{_fac('log10_N_pathB'):.2f}×",
                  delta=f"bias {p4m['log10_N_pathB']['bias']:+.2f} decades",
                  delta_color="off",
                  help="The same physics fed the reference labels instead of the AI's.")
    if _fac("log10_N_pathA") and _fac("log10_N_pathB"):
        st.caption("Path A and path B differ only in who measured the defects, so their "
                   "closeness says the AI's segmentation adds little error to the life.")

    if p4 is not None:
        p4s = p4.sort_values("measured").reset_index(drop=True)
        lo = min(p4s["measured"].min(), p4s["predicted"].min()) / 2
        hi = max(p4s["measured"].max(), p4s["predicted"].max()) * 2

        def frame_data(k):
            shown = p4s.iloc[:k + 1]
            new = p4s.iloc[[k]]
            return [
                go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                           line=dict(dash="dash", color=GREY), name="perfect agreement"),
                go.Scatter(x=[lo, hi], y=[2 * lo, 2 * hi], mode="lines",
                           line=dict(dash="dot", color="#C8CDD2"), name="factor of 2"),
                go.Scatter(x=[lo, hi], y=[lo / 2, hi / 2], mode="lines",
                           line=dict(dash="dot", color="#C8CDD2"), showlegend=False),
                go.Scatter(x=shown["measured"], y=shown["predicted"], mode="markers",
                           marker=dict(size=11, color=ACCENT), name="held-out cylinder",
                           text=shown["cylinder"], hoverinfo="text+x+y"),
                go.Scatter(x=new["measured"], y=new["predicted"], mode="markers+text",
                           text=new["cylinder"], textposition="top center",
                           marker=dict(size=20, color="rgba(0,0,0,0)",
                                       line=dict(color=NAVY, width=3)),
                           showlegend=False, hoverinfo="skip"),
            ]

        fig = go.Figure(data=frame_data(0))
        fig.frames = [go.Frame(data=frame_data(k), name=p4s.loc[k, "cylinder"])
                      for k in range(len(p4s))]
        fig.update_layout(xaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)],
                                     title="measured cycles to failure"),
                          yaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)],
                                     title="predicted cycles to failure"),
                          height=520, margin=dict(t=20, b=110),
                          legend=dict(orientation="h", y=1.06))
        play(fig, frame_ms=700, transition_ms=200, slider_prefix="added ")
        st.plotly_chart(fig, width="stretch")
        resid = np.log10(p4["predicted"]) - np.log10(p4["measured"])
        st.caption(
            f"Press Play to add the {len(p4)} held-out cylinders one at a time, shortest "
            f"life first. Typical error {10 ** np.sqrt((resid ** 2).mean()):.2f}×; every "
            f"prediction is within a factor of "
            f"{10 ** np.abs(resid).max():.1f} of its measured life.")
        with st.expander("The numbers behind the chart"):
            st.dataframe(
                p4s[["cylinder", "stress_level_MPa", "measured", "predicted"]]
                .assign(ratio=p4s["predicted"] / p4s["measured"])
                .rename(columns={"cylinder": "specimen", "stress_level_MPa": "σmax (MPa)"})
                .style.format({"measured": "{:,.0f}", "predicted": "{:,.0f}",
                               "ratio": "{:.2f}×"}),
                width='stretch', hide_index=True)

    _render_monte_carlo()
    _render_bayesian()


# ============================================================== Monte Carlo
def _render_monte_carlo():
    st.divider()
    st.header("Monte Carlo: a life distribution, not a single number")
    mc = json.load(open(os.path.join(MCB_DATA, "monte_carlo_results.json")))
    dp, cp = mc["priors"]["defect"], mc["priors"]["C"]
    thinking(
        "Parts vary. What spread of lives should we expect at each stress?",
        "Draw thousands of virtual parts: a governing defect from the 14 measured sizes "
        "and a crack-growth rate from its calibration scatter, then run each through the "
        "life integral.",
        "The defect fit is checked leave-one-out, and the result is checked to settle as "
        "the number of draws grows.",
        f"At 300 MPa {100 * mc['sn']['300']['n_at_runout'] / mc['sn']['300']['n_finite']:.0f}% "
        "of virtual parts outlive the 10-million-cycle run-out; at 450 MPa almost none.",
        "Near the fatigue limit a single life is the wrong answer: a survival "
        "probability is the right one.")

    smax = st.radio("Stress level", ["300 MPa", "450 MPa"], horizontal=True,
                    key="mc_stress")
    sn = mc["sn"][smax.split()[0]]
    logN = np.log10(np.asarray(sn["N_samples"], float))
    bins = np.linspace(np.floor(logN.min() * 4) / 4, 7.0 + 1e-9, 41)
    steps = [n for n in (25, 50, 100, 200, 400, 800, 1600, len(logN)) if n <= len(logN)]
    ymax = np.histogram(logN, bins)[0].max() * 1.15

    def hist_frame(n):
        h, _ = np.histogram(logN[:n], bins)
        centres = 0.5 * (bins[1:] + bins[:-1])
        return [go.Bar(x=centres, y=h, width=np.diff(bins) * 0.95, marker_color=ACCENT)]

    fig = go.Figure(data=hist_frame(steps[0]))
    fig.frames = [go.Frame(data=hist_frame(n), name=f"{n:,} draws") for n in steps]
    for p, label in ((sn["log10_p5"], "p5"), (sn["log10_p50"], "median"),
                     (sn["log10_p95"], "p95")):
        fig.add_vline(x=p, line_dash="dash", line_color="#5A6067", annotation_text=label)
    fig.update_layout(xaxis_title="log₁₀(cycles to failure)", yaxis_title="virtual parts",
                      yaxis_range=[0, ymax], height=420, margin=dict(t=20, b=110),
                      bargap=0)
    play(fig, frame_ms=600, transition_ms=300, slider_prefix="")
    st.plotly_chart(fig, width="stretch")
    pct = 100 * sn["n_at_runout"] / sn["n_finite"]
    st.caption(
        f"Press Play to watch the distribution build up as draws arrive (the app ships "
        f"{len(logN):,} of the {sn['n_total']:,} draws). The run-out share, {pct:.1f}%, is "
        f"counted on all {sn['n_total']:,}. Median life about "
        f"{10 ** sn['log10_p50']:,.0f} cycles.")
    if pct > 2:
        plain(
            f"<b>The spike at the right edge is real.</b> {pct:.0f}% of virtual parts at "
            f"{smax} never accumulate enough crack growth to fail inside 10 million "
            "cycles, the limit at which the real tests were stopped too. The model was "
            "not told about a fatigue limit; this pile-up falls out of the physics, the "
            "same behaviour real S-N data show near the endurance limit.")

    with st.expander("The checks behind the distribution"):
        st.markdown(
            f"**Defect size**: lognormal fitted to the 14 measured governing defects, "
            f"$\\ln(\\mu m) \\sim \\mathcal{{N}}({dp['mu']:.3f}, {dp['sigma']:.3f}^2)$. Each "
            "specimen's real value is checked against a fit made without it; a good fit "
            "spreads those percentiles evenly from 0 to 100%.")
        v1 = pd.DataFrame(mc["verify_extremes"]).sort_values("percentile")
        figv1 = go.Figure(go.Bar(x=v1["specimen"], y=v1["percentile"], marker_color=NAVY))
        figv1.update_layout(yaxis_title="leave-one-out percentile (%)", height=300,
                            yaxis_range=[0, 100], margin=dict(t=10))
        st.plotly_chart(figv1, width='stretch')
        st.caption(f"Kolmogorov–Smirnov statistic {mc['verify_extremes_ks']:.3f} against a "
                   "uniform spread: no clustering at either end.")
        st.markdown(
            f"**Crack-growth rate**: lognormal about the calibrated value, "
            f"$\\sigma(\\ln C) = {cp['sigma_C']:.3f}$, from {cp['n_used']} of "
            f"{cp['n_total']} calibration residuals (one near-threshold artefact trimmed).")
        v2 = pd.DataFrame(mc["verify_convergence"]["table"])
        figv2 = go.Figure(go.Scatter(x=v2["n_samples"], y=v2["rmse"], mode="lines+markers",
                                     line=dict(color=NAVY)))
        figv2.update_layout(xaxis_type="log", yaxis_type="log", height=300,
                            xaxis_title="draws", yaxis_title="RMSE of the median (decades)",
                            margin=dict(t=10))
        st.plotly_chart(figv2, width='stretch')
        st.caption("Each sample size repeated 20 times: the error falls at close to the "
                   "theoretical rate of one over the square root of the draws.")


# ============================================================== Bayesian filter
def _render_bayesian():
    st.divider()
    st.header("Bayesian filter: sharpening one part's life as inspections arrive")
    bf = json.load(open(os.path.join(MCB_DATA, "bayesian_results.json")))
    r = bf["verify_recovery"]
    step = next((i for i, h in enumerate(r["history"], 1)
                 if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
    thinking(
        "Once a part is in service and inspected, how much better can we predict it?",
        "4,000 particles, each a possible crack-growth rate, are reweighted after every "
        "crack-length reading and resampled.",
        "A synthetic part with a known true life: the filter starts from the population "
        "prior, which does not know it, and must find it.",
        f"It reaches the true life (within 20%) at inspection {step}, not by the 3rd, and "
        f"keeps at least {min(h['ess'] for h in r['history']):,.0f} effective particles.",
        "H3 is partly supported: the filter converges, later than hypothesised.")
    st.latex(r"S(a) = \int_{a_0}^{a} \frac{da'}{(\Delta K(a') - \Delta K_{th})^{m}}"
             r"\quad\Rightarrow\quad a_i(N) = S^{-1}(C_i \cdot N)")
    st.caption("Crack size and growth rate separate in the Paris integral, so each "
               "particle's whole crack history follows from one shared function and its own "
               "rate Cᵢ: a lookup, not a simulation.")

    cases = {"Synthetic part (known true life)": (r["history"], r["Nf_true"])}
    for spec, rs in bf["specimens"].items():
        cases[f"{spec} · measured {rs['Nf_measured']:,} cycles"] = (rs["history"],
                                                                    rs["Nf_measured"])
    which = st.selectbox("Part", list(cases), key="bayes_case")
    hist, truth = cases[which]
    H = pd.DataFrame(hist)
    ylo = min(H["Nf_p10"].min(), truth) / 2
    yhi = max(H["Nf_p90"].max(), truth) * 2

    def bframe(k):
        h = H.iloc[:k + 1]
        return [
            go.Scatter(x=h["N_inspection"], y=h["Nf_p90"], mode="lines", line=dict(width=0),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=h["N_inspection"], y=h["Nf_p10"], mode="lines", fill="tonexty",
                       fillcolor="rgba(181,71,58,0.18)", line=dict(width=0),
                       name="10th–90th percentile"),
            go.Scatter(x=h["N_inspection"], y=h["Nf_mean"], mode="lines+markers",
                       line=dict(color=ACCENT, width=3), name="posterior mean"),
        ]

    fig = go.Figure(data=bframe(0))
    fig.frames = [go.Frame(data=bframe(k), name=f"inspection {k + 1}")
                  for k in range(len(H))]
    fig.add_hline(y=truth, line_dash="dash", line_color=GREEN,
                  annotation_text="true life" if "Synthetic" in which else "measured life")
    fig.update_layout(yaxis=dict(type="log", range=[np.log10(ylo), np.log10(yhi)],
                                 title="predicted cycles to failure"),
                      xaxis=dict(range=[0, H["N_inspection"].max() * 1.08],
                                 title="cycles inspected so far"),
                      height=430, margin=dict(t=20, b=110),
                      legend=dict(orientation="h", y=1.08))
    play(fig, frame_ms=1000, transition_ms=300, slider_prefix="")
    st.plotly_chart(fig, width="stretch")
    last = hist[-1]
    st.caption(
        f"Press Play to add one inspection at a time. After the last, the posterior mean "
        f"is {last['Nf_mean']:,.0f} cycles against {truth:,.0f}; the band narrowed "
        f"{(hist[0]['Nf_p90'] - hist[0]['Nf_p10']) / max(last['Nf_p90'] - last['Nf_p10'], 1):.0f}× "
        "from first to last inspection.")
    if "P045" in which:
        st.warning(
            "**P045 does not converge, a genuine prognostics finding.** Its crack grows so "
            "late in life (the Paris exponent makes growth strongly back-loaded) that "
            "inspections through 90% of its life see growth smaller than the measurement "
            "noise. Slow growth is then the most probable explanation, so the filter "
            "predicts a run-out. Real inspection programmes face exactly this limit.",
            icon="🔍")

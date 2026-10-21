import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ui import GREEN, GREY, NAVY, RED, RULE, finding, go_to, note, play, show, spec, verdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MCB_DATA = os.path.join(ROOT, "data", "phase_mc_bayesian")
CMP_DATA = os.path.join(ROOT, "data", "comparison_model")
P4_DATA = os.path.join(ROOT, "data", "phase4_validation")

# Path A's constants, as scripts/phase4_predict.py uses them for the validated result:
# literature values for 316L, Murakami's surface factor, threshold removed.
M_PARIS, Y_SURF, R_RATIO = 3.94, 0.65, 0.1


def _json(*parts):
    p = os.path.join(*parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


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


def render():
    st.title("Phase 3: the fatigue life")
    p4 = _p4()
    cmp_d = _json(CMP_DATA, "comparison_results.json") or {}
    piml = _json(CMP_DATA, "piml_results.json") or {}
    vi = piml.get("verify_integral")
    p4m = cmp_d.get("phase4", {})
    fac = p4m.get("log10_N_pathA", {}).get("factor")
    rates = [r["rate"] for r in vi["table"][1:4]] if vi else None
    spec(
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
         f"{vi['table'][3]['rel_err']:.0e} at {vi['table'][3]['n']} points before "
         "round-off (table below)." if vi else "Integral verification file not found."),
        (f"Within **{fac:.2f}×** of the measured lives of the 14 held-out cylinders, "
         "with no constant fitted to any fatigue life." if fac else
         "Validation file not found."))
    note("The constants are literature values for 316L: C = 6.25 × 10⁻¹⁰ and m = 3.94 "
         "(da/dN in mm per cycle, ΔK in MPa√m), and Y = 0.65, Murakami's factor for a "
         "surface defect. None of them is fitted to this dataset's fatigue lives.")

    # ------------------------------------------------------------ the threshold
    st.markdown("## The one modelling decision: the threshold")
    pb = piml.get("physics_bias")
    sub = piml.get("subthreshold_specimens", [])
    vt = piml.get("verify_threshold", [])
    if pb and vt:
        dks = [r["dK"] for r in vt]
        st.markdown(
            f"The long-crack threshold for 316L is ΔK_th = 5 MPa√m. At the governing gauge "
            f"defect of the held-out cylinders the driving force lies between "
            f"{min(dks):.1f} and {max(dks):.1f} MPa√m, close to that threshold or below it. "
            f"Kept in, it predicts that {len(sub)} cylinder never fails and over-predicts "
            f"the other {pb['n']} by a median factor of {pb['median']:.1f}, although every "
            "one of them failed in test. Cracks from defects this small are short cracks, "
            "to which the long-crack threshold does not apply, so the validated "
            "prediction, path A, sets ΔK_th = 0.")
        sw = piml.get("threshold_sweep", [])
        if sw:
            fig = go.Figure(go.Scatter(
                x=[r["dkth"] for r in sw], y=[r["geometric_bias"] for r in sw],
                mode="lines+markers", line=dict(color=NAVY), marker=dict(size=7),
                text=[f"{r['n_finite']} of 14 predicted to fail" for r in sw],
                hovertemplate="ΔK_th = %{x} MPa√m: %{y:.2f}× (%{text})"))
            fig.add_hline(y=1, line_dash="dot", line_color=GREY)
            fig.update_layout(height=320, yaxis_type="log",
                              xaxis_title="threshold ΔK_th (MPa√m)",
                              yaxis_title="predicted ÷ measured (geometric mean)")
            show(fig)
            st.caption("The bias of the held-out predictions as the threshold is raised "
                       "from zero to the long-crack value.")

    if p4 is not None:
        _render_growth(p4)

    # ------------------------------------------------------------ verification
    if vi:
        st.markdown("## Verification of the integral")
        t = pd.DataFrame(vi["table"])
        t["rate"] = t["rate"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        t["rel_err"] = t["rel_err"].map(lambda v: f"{v:.2e}")
        t["value"] = t["value"].map(lambda v: f"{v:,.6f}")
        st.dataframe(t.rename(columns={"n": "grid points", "value": "integral (cycles)",
                                       "rel_err": "relative error",
                                       "rate": "observed order"}),
                     width="stretch", hide_index=True)
        st.markdown(
            f"Exact value {vi['exact']:,.6f} cycles. Each doubling of the grid cuts the "
            "error sixteen-fold, order 4, until it reaches machine precision near "
            "10⁻¹⁴; past that the observed order means nothing.")

    # ------------------------------------------------------------ held-out comparison
    st.markdown("## Every held-out cylinder, predicted then compared")

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
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Prediction", "Typical error",
                                                 "Bias (decades)"]),
                     width="stretch", hide_index=True)
        st.markdown("Path A and path B differ only in who measured the defects. Their "
                    "closeness says the segmentation adds little error to the life.")

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
                           line=dict(dash="dot", color=RULE), name="factor of 2"),
                go.Scatter(x=[lo, hi], y=[lo / 2, hi / 2], mode="lines",
                           line=dict(dash="dot", color=RULE), showlegend=False),
                go.Scatter(x=shown["measured"], y=shown["predicted"], mode="markers",
                           marker=dict(size=10, color=NAVY), name="held-out cylinder",
                           text=shown["cylinder"], hoverinfo="text+x+y"),
                go.Scatter(x=new["measured"], y=new["predicted"], mode="markers+text",
                           text=new["cylinder"], textposition="top center",
                           marker=dict(size=18, color="rgba(0,0,0,0)",
                                       line=dict(color=RED, width=2)),
                           showlegend=False, hoverinfo="skip"),
            ]

        fig = go.Figure(data=frame_data(0))
        fig.frames = [go.Frame(data=frame_data(k), name=p4s.loc[k, "cylinder"])
                      for k in range(len(p4s))]
        fig.update_layout(xaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)],
                                     title="measured cycles to failure"),
                          yaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)],
                                     title="predicted cycles to failure"),
                          height=500, margin=dict(t=20, b=110),
                          legend=dict(orientation="h", y=1.06))
        play(fig, frame_ms=700, transition_ms=200, slider_prefix="added ")
        show(fig)
        resid = np.log10(p4["predicted"]) - np.log10(p4["measured"])
        st.caption(
            f"The {len(p4)} held-out cylinders added one at a time, shortest life first. "
            f"Every prediction is within a factor of {10 ** np.abs(resid).max():.1f} of "
            "its measured life.")
        with st.expander("The numbers behind the chart"):
            st.dataframe(
                p4s[["cylinder", "stress_level_MPa", "measured", "predicted"]]
                .assign(ratio=p4s["predicted"] / p4s["measured"])
                .rename(columns={"cylinder": "Cylinder", "stress_level_MPa": "σmax (MPa)",
                                 "measured": "Measured", "predicted": "Predicted",
                                 "ratio": "Ratio"})
                .style.format({"Measured": "{:,.0f}", "Predicted": "{:,.0f}",
                               "Ratio": "{:.2f}"}),
                width="stretch", hide_index=True)

    _render_monte_carlo()
    _render_bayesian()
    go_to("phase4", "Continue to Phase 4: validation")


def _render_growth(p4):
    st.markdown("## One crack, from the governing defect to collapse")
    cyl = list(p4["cylinder"])
    sp = st.selectbox(
        "Held-out cylinder", cyl, index=cyl.index("P017") if "P017" in cyl else 0,
        format_func=lambda c: f"{c}, {int(p4.set_index('cylinder').loc[c, 'stress_level_MPa'])} MPa",
        key="p3_growth_spec")
    row = p4.set_index("cylinder").loc[sp]
    a0, ac, Nf, a, N, dK, idx = _crack_growth(row)
    meas = float(row["measured"])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("crack depth against cycles",
                                        "driving force against crack depth"))

    def data_for(k):
        j = idx[k]
        return [
            go.Scatter(x=N[:j + 1], y=a[:j + 1], mode="lines", line=dict(color=NAVY, width=2.5),
                       showlegend=False),
            go.Scatter(x=[N[j]], y=[a[j]], mode="markers",
                       marker=dict(size=11, color=NAVY, line=dict(color="white", width=2)),
                       showlegend=False, hoverinfo="skip"),
            go.Scatter(x=a[:j + 1], y=dK[:j + 1], mode="lines", line=dict(color=RED, width=2.5),
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
    show(fig)

    half = float(np.interp(0.5, N / Nf, a))
    k2 = float(np.interp(2 * a0, a, N)) / Nf if 2 * a0 < ac else 1.0
    st.markdown(
        f"{sp}'s governing defect is {a0 * 1000:,.0f} µm. Predicted life "
        f"{Nf:,.0f} cycles against {meas:,.0f} measured. The crack hardly moves for most "
        f"of that life: at half-life it is still only {half * 1000:,.0f} µm deep, and "
        + (f"{k2:.0%} of the cycles pass before it doubles. " if 2 * a0 < ac else
           "it does not double before collapse. ") +
        "ΔK grows with the crack, so growth accelerates and the end comes quickly. This is "
        "why the starting defect size decides the life.")


# ============================================================== Monte Carlo
def _render_monte_carlo():
    st.markdown("## Monte Carlo: a distribution of lives")
    mc = _json(MCB_DATA, "monte_carlo_results.json")
    if mc is None:
        finding("The Monte Carlo result file is not present.")
        return
    dp, cp = mc["priors"]["defect"], mc["priors"]["C"]
    st.markdown(
        f"Parts printed the same way differ. Each virtual part draws a governing defect "
        f"from a lognormal fitted to the 14 measured sizes, "
        f"$\\ln a_0 \\sim \\mathcal{{N}}({dp['mu']:.3f}, {dp['sigma']:.3f}^2)$ with $a_0$ "
        f"in µm, and a growth rate from the scatter of the calibration, "
        f"$\\sigma(\\ln C) = {cp['sigma_C']:.3f}$, then goes through the life integral. "
        "A part still unbroken at 10 million cycles is counted as a run-out, as the real "
        "tests were.")

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
        return [go.Bar(x=centres, y=h, width=np.diff(bins) * 0.95, marker_color=NAVY)]

    fig = go.Figure(data=hist_frame(steps[0]))
    fig.frames = [go.Frame(data=hist_frame(n), name=f"{n:,} draws") for n in steps]
    for p, label in ((sn["log10_p5"], "5th percentile"), (sn["log10_p50"], "median")):
        fig.add_vline(x=p, line_dash="dash", line_color=GREY, annotation_text=label)
    fig.update_layout(xaxis_title="log₁₀ cycles to failure", yaxis_title="virtual parts",
                      yaxis_range=[0, ymax], height=400, margin=dict(t=20, b=110),
                      bargap=0)
    play(fig, frame_ms=600, transition_ms=300, slider_prefix="")
    show(fig)
    pct = 100 * sn["n_at_runout"] / sn["n_total"]
    st.caption(
        f"The distribution building up as draws arrive ({len(logN):,} of the "
        f"{sn['n_total']:,} draws are shipped with the app; percentages are counted on "
        f"all {sn['n_total']:,}). Median life {10 ** sn['log10_p50']:,.0f} cycles; "
        f"run-outs {pct:.1f}%.")
    if pct > 2:
        st.markdown(
            f"The bar at the right edge is the run-outs: {pct:.1f}% of virtual parts at "
            f"{smax} never grow a crack far enough to fail inside 10 million cycles. The "
            "model was given no fatigue limit; the pile-up comes out of the physics near "
            "the endurance limit.")

    n_tot = sn["n_total"]
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n_tot))
    m_need = int(np.ceil(np.log(2 / 0.05) / (2 * 0.05 ** 2)))
    st.markdown("**Convergence.** The Dvoretzky–Kiefer–Wolfowitz inequality bounds how far "
                "the sampled distribution can lie from the true one: "
                r"$P\big(\sup_N |\hat F_M(N)-F(N)| > \varepsilon\big) \le 2e^{-2M\varepsilon^2}$. "
                f"For ε = 0.05 at 95% confidence it needs M ≥ {m_need:,} draws. The run "
                f"used M = {n_tot:,}, which gives ε = {eps:.4f}.")
    v2 = pd.DataFrame(mc["verify_convergence"]["table"])
    figv2 = go.Figure(go.Scatter(x=v2["n_samples"], y=v2["rmse"], mode="lines+markers",
                                 line=dict(color=NAVY), name="observed"))
    ref = v2["rmse"].iloc[0] * np.sqrt(v2["n_samples"].iloc[0] / v2["n_samples"])
    figv2.add_trace(go.Scatter(x=v2["n_samples"], y=ref, mode="lines", name="M^(−1/2)",
                               line=dict(color=GREY, dash="dot")))
    figv2.update_layout(xaxis_type="log", yaxis_type="log", height=300,
                        xaxis_title="draws M", yaxis_title="RMSE of the median (decades)",
                        margin=dict(t=10), legend=dict(orientation="h", y=1.1))
    show(figv2)
    st.caption(f"Each sample size repeated {int(v2['n_repeats'].iloc[0])} times. The error "
               "of the median falls close to the M^(−1/2) rate of Monte Carlo.")

    with st.expander("Check of the defect-size fit"):
        v1 = pd.DataFrame(mc["verify_extremes"]).sort_values("percentile")
        figv1 = go.Figure(go.Bar(x=v1["specimen"], y=v1["percentile"], marker_color=NAVY))
        figv1.update_layout(yaxis_title="leave-one-out percentile (%)", height=300,
                            yaxis_range=[0, 100], margin=dict(t=10))
        show(figv1)
        st.caption(f"Each specimen's own governing defect placed in a fit made without it. "
                   f"A good fit spreads these evenly from 0 to 100%; the Kolmogorov–Smirnov "
                   f"statistic against a uniform spread is {mc['verify_extremes_ks']:.3f}. "
                   f"The growth-rate scatter uses {cp['n_used']} of {cp['n_total']} "
                   "calibration residuals, one near-threshold point trimmed.")


# ============================================================== Bayesian filter
def _render_bayesian():
    st.markdown("## Particle filter: one part's life as inspections arrive")
    bf = _json(MCB_DATA, "bayesian_results.json")
    if bf is None:
        finding("The Bayesian result file is not present.")
        return
    r = bf["verify_recovery"]
    step = next((i for i, h in enumerate(r["history"], 1)
                 if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
    ess = min(h["ess"] for h in r["history"])
    st.markdown(
        "Once a part is in service, each crack-length reading narrows its life. 4,000 "
        "particles, each a possible growth rate $C_i$, are reweighted by how well they "
        "predict the reading and then resampled. Crack size and rate separate in the "
        "Paris integral, so each particle's history is one shared function:")
    st.latex(r"S(a) = \int_{a_0}^{a} \frac{da'}{\big(\Delta K(a')-\Delta K_{th}\big)^{m}}"
             r"\quad\Rightarrow\quad a_i(N) = S^{-1}(C_i\,N)")
    met = step is not None and step <= 3 and ess >= 250
    verdict(f"Hypothesis 3 {'met' if met else 'partly met'}: within 20% of the true life "
            f"at inspection {step}, against 3 hypothesised; effective sample size never "
            f"below {ess:,.0f}, against 250.", "met" if met else "part")

    cases = {"Synthetic part with a known life": (r["history"], r["Nf_true"])}
    for sp, rs in bf["specimens"].items():
        cases[f"{sp}, measured {rs['Nf_measured']:,} cycles"] = (rs["history"],
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
                       fillcolor="rgba(31,78,121,0.15)", line=dict(width=0),
                       name="10th to 90th percentile"),
            go.Scatter(x=h["N_inspection"], y=h["Nf_mean"], mode="lines+markers",
                       line=dict(color=NAVY, width=2.5), name="posterior mean"),
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
                      height=420, margin=dict(t=20, b=110),
                      legend=dict(orientation="h", y=1.08))
    play(fig, frame_ms=1000, transition_ms=300, slider_prefix="")
    show(fig)
    last = hist[-1]
    st.caption(
        f"One inspection added per frame. After the last, the posterior mean is "
        f"{last['Nf_mean']:,.0f} cycles against {truth:,.0f}; the band narrowed "
        f"{(hist[0]['Nf_p90'] - hist[0]['Nf_p10']) / max(last['Nf_p90'] - last['Nf_p10'], 1):.0f}-fold "
        "from the first inspection to the last.")
    if "P045" in which:
        finding(
            "<b>P045 does not converge.</b> With m near 4 its crack grows late in life, so "
            "readings through 90% of its life see growth smaller than the measurement "
            "noise. Slow growth is then the likeliest explanation and the filter predicts "
            "a run-out. Any inspection programme meets this limit.")

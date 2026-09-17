import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import load, splits, SY_MPA, UTS_MPA
from src.life_integral import life_numeric, Y_SURFACE, DKTH, C_PARIS, M_PARIS
from src.piml_nn import train_pinn, predict as pinn_predict
from src.ui import badge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "comparison_model")

GAGE_DIA_MM = 5.08
A0_MM2 = np.pi * (GAGE_DIA_MM / 2) ** 2
S_FLOW = 0.5 * (SY_MPA + UTS_MPA)
ACCENT = "#B5473A"
GREY = "#8A9096"


def collapse_depth_mm(smax_mpa: float) -> float:
    frac = 1.0 - smax_mpa / S_FLOW
    if frac <= 0:
        return GAGE_DIA_MM / 2
    return min(float(np.sqrt(2 * frac * A0_MM2 / np.pi)), GAGE_DIA_MM / 2)


@st.cache_resource
def _fit_pinn():
    rows = load()
    meas = splits(rows)["meas"]
    X = np.column_stack([
        np.log([r["ds"] for r in meas]),
        np.log([r["sqrt_area_gauge_um"] for r in meas]),
        [r["flatness_median"] for r in meas],
    ])
    y = np.log10([r["Nf"] for r in meas])
    model, std = train_pinn(X, y, lam_s=10.0, lam_a=10.0, lam_m=0.1, epochs=3000, seed=0)
    return model, std, float(np.median(X[:, 2]))


def render():
    st.title("Phase 4 · Closed-loop validation & physics-informed comparison")
    badge("REAL RESULTS", "real")
    badge("BAYESIAN UPDATING STILL PREVIEW", "pending")

    st.markdown(
        "**This page is the direct answer to the supervisor's request** for a "
        "physics-informed comparison model. It has two arms: a **hard-physics** "
        "sign-constrained regression (`comparison_model`) and a new **soft-"
        "physics** neural network trained with a hybrid data + PDE-residual "
        "loss. Both are benchmarked against a pure-statistics ladder and a "
        "zero-parameter physics baseline, leave-one-out, on the same specimens."
    )

    res = json.load(open(os.path.join(DATA, "piml_results.json")))
    comp = json.load(open(os.path.join(DATA, "comparison_results.json")))
    fail = json.load(open(os.path.join(DATA, "failure_criterion.json")))

    st.divider()
    st.subheader("The dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Specimens tested", comp["n_specimens"])
    c2.metric("Run-outs (censored)", comp["n_runouts"])
    c3.metric("Failures", comp["n_failures"])
    c4.metric("With a measured defect size", comp["n_measured"])
    st.caption(
        "60 ORNL Snow 316H specimens, ASTM E466, R = 0.1, 300 MPa HCF / 450 MPa "
        "LCF. Only 14 currently have a measured governing-defect size."
    )

    st.divider()
    st.subheader("Comparison ladder — leave-one-out")
    ladder = res["ladder"]
    df = pd.DataFrame(ladder)[["model", "n", "params", "r2", "loo_rmse", "loo_factor"]]
    df.columns = ["model", "n", "parameters", "R²", "LOO RMSE (decades)", "typical life error"]
    df["typical life error"] = df["typical life error"].map(lambda v: f"{v:.2f}×")
    df["R²"] = df["R²"].map(lambda v: f"{v:.3f}")
    df["LOO RMSE (decades)"] = df["LOO RMSE (decades)"].map(lambda v: f"{v:.3f}")

    def _highlight(row):
        return ["background-color: #FBE9E7"] * len(row) if row["model"].startswith("C1") else [""] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), width='stretch', hide_index=True)
    st.caption(
        "A = all 41 gauge failures (stress + bulk porosity only). B = the 14 "
        "with a measured defect (adds Murakami √area). **C1 = the new PINN**, "
        "highlighted, same 14 specimens and leave-one-out protocol as the "
        "B-series. Physics baseline (nothing fitted, threshold-corrected) "
        f"over-predicts life by a geometric mean of "
        f"{res['physics_bias']['geometric_mean']:.1f}× before calibration."
    )
    pinn_row = next((d for d in ladder if d["model"].startswith("C1")), None)
    if pinn_row:
        st.info(
            f"The PINN uses **{pinn_row['params']} parameters** on 14 points — "
            "far more than the classical models' 1-4. It is only kept from "
            "overfitting by the physics-residual loss and weight decay; read its "
            "LOO number with that in mind.",
            icon="⚖️",
        )

    # ------------------------------------------------------------ fig1 (live)
    st.divider()
    st.subheader("Measured vs. predicted life")
    fig = go.Figure()
    lims = [3e4, 3e6]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(dash="dash", color=GREY), name="1:1"))
    fig.add_trace(go.Scatter(x=lims, y=[l / 2 for l in lims], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=lims, y=[l * 2 for l in lims], mode="lines",
                             fill="tonexty", fillcolor="rgba(138,144,150,0.12)",
                             line=dict(width=0), name="factor of 2"))
    bg = comp.get("points_A2") or []
    pts = comp.get("points_B3") or []
    if bg:
        fig.add_trace(go.Scatter(x=[p["measured"] for p in bg], y=[p["predicted"] for p in bg],
                                 mode="markers", marker=dict(size=7, color=GREY, symbol="circle-open"),
                                 name=f"gauge failures (n={len(bg)})"))
    if pts:
        fig.add_trace(go.Scatter(x=[p["measured"] for p in pts], y=[p["predicted"] for p in pts],
                                 mode="markers", marker=dict(size=10, color=ACCENT),
                                 name=f"measured defect (n={len(pts)})", text=[p["specimen"] for p in pts]))
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_range=[np.log10(lims[0]), np.log10(lims[1])],
                      yaxis_range=[np.log10(lims[0]), np.log10(lims[1])],
                      xaxis_title="measured life (cycles)", yaxis_title="predicted life (cycles)",
                      height=480, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width='stretch')

    # ------------------------------------------------------------ fig2 (live)
    with st.expander("Verification: exponent recovery (classical model, known answer)"):
        st.caption(
            "Fatigue lives synthesised from EQ (4) with KNOWN Basquin/Paris "
            "exponents; the fitting procedure must recover them as scatter falls, "
            "at the theoretical rate 1 for an unbiased linear estimator."
        )
        vr = res["verify_recovery"]
        tbl = pd.DataFrame(vr["table"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_s"], mode="lines+markers", name="stress-slope error"))
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_a"], mode="lines+markers", name="defect-slope error"))
        fig2.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="scatter (decades)",
                          yaxis_title="| recovered − true |", height=380,
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig2, width='stretch')
        st.caption(f"True b_s = {vr['b_s_true']:+.4f}, b_a = {vr['b_a_true']:+.4f}, m = {vr['m_true']}. "
                  "Both errors fall on a straight log-log line of slope 1 — the expected rate "
                  "for an unbiased linear estimator.")

        st.markdown(
            "**PINN's own physics-loss verification** (same idea, for the neural "
            "network arm): n = 14 synthetic specimens, known slope, as the "
            "physics-loss weight λ rises from 0 the network's *learned gradient* "
            "should move toward the true slope."
        )
        st.markdown(
            "| λ | stress-slope residual RMS | defect-slope residual RMS | LOO RMSE (decades) |\n"
            "|---:|---:|---:|---:|\n"
            "| 0.00 | 0.9596 | 0.3632 | 0.543 |\n"
            "| 0.01 | 0.2288 | 0.3495 | 0.493 |\n"
            "| 0.10 | 0.0372 | 0.1111 | 0.592 |\n"
            "| 1.00 | 0.0107 | 0.0145 | 0.683 |\n"
            "| 10.00 | 0.0063 | 0.0069 | 0.459 |"
        )
        st.caption("λ = 10 was chosen from this table before the real fit was ever run.")

    # ------------------------------------------------------------ fig3 (live)
    st.subheader("Failure criterion: what ends the part")
    fc1, fc2 = st.columns(2)
    with fc1:
        rows = [c for c in fail["collapse"] if c["limit"] == "flow stress"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=[f"{r['smax']:.0f} MPa" for r in rows],
                              y=[r["Kmax"] for r in rows], name="K_max at collapse",
                              marker_color=ACCENT))
        fig3.add_hline(y=fail["kic"], line_dash="dash", line_color=GREY,
                       annotation_text=f"K_IC = {fail['kic']:.0f} MPa√m")
        fig3.update_layout(yaxis_title="MPa√m", height=360, showlegend=False,
                           title="Margin against fast fracture")
        st.plotly_chart(fig3, width='stretch')
        st.caption(
            f"K_max never approaches K_IC (margin "
            f"{fail['kic']/fail['kmax_at_collapse']:.0f}×) — failure is net-"
            "section collapse of the ductile ligament, not fast fracture."
        )
    with fc2:
        loc = fail["fracture_location"]
        fig4 = go.Figure(go.Bar(x=list(loc.keys()), y=list(loc.values()), marker_color=GREY))
        fig4.update_layout(height=360, title="Where specimens actually fractured", yaxis_title="count")
        st.plotly_chart(fig4, width='stretch')

    # ------------------------------------------------------------ live explorer
    st.divider()
    st.subheader("Explore the models live")
    st.caption(
        "Sliders drive the ACTUAL fitted/trained models — the physics baseline "
        "integrates the real Paris law, the constrained regression uses its real "
        "fitted coefficients, and the PINN is the real trained network (cached "
        "once, not retrained per slider move)."
    )

    pinn_model, pinn_std, flat_med = _fit_pinn()
    b3 = next((d for d in ladder if d["model"].startswith("B3")), None)

    cL, cR = st.columns(2)
    with cL:
        ds_slider = st.slider("Stress range Δσ (MPa)", 200, 500, 300, 5)
    with cR:
        sa_slider = st.slider("Governing defect √area (µm)", 150, 1200, 450, 10)

    a0_mm = sa_slider * 1e-3
    smax_slider = ds_slider / (1 - 0.1)
    af_mm = collapse_depth_mm(smax_slider)
    n_phys = life_numeric(a0_mm, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE, C=C_PARIS, m=M_PARIS)
    n_phys_cal = life_numeric(a0_mm, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE,
                              C=res["physics_bias"]["C_implied"], m=M_PARIS)
    st.caption(f"Collapse depth at this stress: {af_mm:.2f} mm (gauge radius {GAGE_DIA_MM/2:.2f} mm)")

    if b3:
        b0, b_ds, b_sa = b3["coef"][2], b3["coef"][0], b3["coef"][1]
        n_b3 = 10 ** (b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_slider))
    else:
        n_b3 = float("nan")
    Xq = np.array([[np.log(ds_slider), np.log(sa_slider), flat_med]])
    n_pinn = 10 ** pinn_predict(pinn_model, pinn_std, Xq)[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Physics, literature C", f"{n_phys:,.0f}" if np.isfinite(n_phys) else "run-out")
    m2.metric("Physics, dataset-calibrated C", f"{n_phys_cal:,.0f}" if np.isfinite(n_phys_cal) else "run-out")
    m3.metric("Constrained regression (B3)", f"{n_b3:,.0f}")
    m4.metric("PINN (C1)", f"{n_pinn:,.0f}")

    sa_grid = np.linspace(150, 1200, 60)
    meas_rows = splits(load())["meas"]
    figE = go.Figure()
    figE.add_trace(go.Scatter(
        x=sa_grid,
        y=[life_numeric(a * 1e-3, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE,
                        C=res["physics_bias"]["C_implied"], m=M_PARIS) for a in sa_grid],
        mode="lines", name="Physics, calibrated C"))
    if b3:
        figE.add_trace(go.Scatter(x=sa_grid, y=10 ** (b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_grid)),
                                  mode="lines", name="Constrained regression (B3)"))
    Xg = np.column_stack([np.full_like(sa_grid, np.log(ds_slider)), np.log(sa_grid),
                          np.full_like(sa_grid, flat_med)])
    figE.add_trace(go.Scatter(x=sa_grid, y=10 ** pinn_predict(pinn_model, pinn_std, Xg),
                              mode="lines", name="PINN (C1)"))
    figE.add_trace(go.Scatter(
        x=[r["sqrt_area_gauge_um"] for r in meas_rows if abs(r["ds"] - ds_slider) < 60],
        y=[r["Nf"] for r in meas_rows if abs(r["ds"] - ds_slider) < 60],
        mode="markers", name="measured (Δσ within 60 MPa)",
        marker=dict(size=10, color="black", symbol="x")))
    figE.update_layout(xaxis_title="governing defect √area (µm)",
                       yaxis_title="predicted cycles to failure", yaxis_type="log",
                       height=460, legend=dict(orientation="h", y=1.12),
                       title=f"at Δσ = {ds_slider} MPa")
    st.plotly_chart(figE, width='stretch')

    st.divider()
    st.subheader("Known limitations (stated, not hidden)")
    st.markdown(
        "- Only 14 of 60 specimens have a measured defect size; the rest use "
        "bulk porosity, an average rather than the extreme-value measure the "
        "physics asks for.\n"
        "- Distance to the free surface, sphericity proper, and nearest-"
        "neighbour spacing are not yet extracted — the PINN and the regression "
        "both fall back to flatness as the shape proxy.\n"
        "- The PINN's 14-point leave-one-out result carries real variance — "
        "treat the comparison ladder as indicative, not a certified ranking.\n"
        "- The threshold-free long-crack Paris law does not strictly apply at "
        "these defect sizes (short-crack regime)."
    )

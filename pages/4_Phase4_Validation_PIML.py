import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import setup, badge
from src.data import load, splits, SY_MPA, UTS_MPA
from src.life_integral import life_numeric, Y_SURFACE, DKTH, C_PARIS, M_PARIS
from src.piml_nn import train_pinn, predict as pinn_predict

GAGE_DIA_MM = 5.08
A0_MM2 = np.pi * (GAGE_DIA_MM / 2) ** 2
S_FLOW = 0.5 * (SY_MPA + UTS_MPA)


def collapse_depth_mm(smax_mpa: float) -> float:
    """Depth of the equivalent semicircular surface crack at net-section collapse.
    Same criterion as comparison_model/piml_model.py's collapse_depth_mm() -
    capped at the bar radius, since a crack cannot exceed it."""
    frac = 1.0 - smax_mpa / S_FLOW
    if frac <= 0:
        return GAGE_DIA_MM / 2
    return min(float(np.sqrt(2 * frac * A0_MM2 / np.pi)), GAGE_DIA_MM / 2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
DATA = os.path.join(ROOT, "data", "comparison_model")

setup("Phase 4 - Validation & PIML", "\U0001F3AF")
st.title("Phase 4 · Closed-loop validation & physics-informed comparison")
badge("REAL RESULTS", "real")
badge("BAYESIAN UPDATING STILL PREVIEW", "pending")

st.markdown(
    "**This page is the direct answer to the supervisor's request** for a "
    "physics-informed comparison model. It has two arms: a **hard-physics** "
    "sign-constrained regression (already the project's `comparison_model`) and a "
    "new **soft-physics** neural network trained with a hybrid data + PDE-residual "
    "loss. Both are benchmarked against a pure-statistics ladder and a zero-"
    "parameter physics baseline, leave-one-out, on the same specimens."
)

res = json.load(open(os.path.join(DATA, "piml_results.json")))

st.divider()
st.subheader("The dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Specimens tested", "60")
c2.metric("Run-outs (censored)", "3")
c3.metric("Failures", "57")
c4.metric("With a measured defect size", "14")
st.caption(
    "60 ORNL Snow 316H specimens, ASTM E466, R = 0.1, 300 MPa HCF / 450 MPa LCF. "
    "Only 14 currently have a measured governing-defect size (Phase 2's Work "
    "remaining item 2: extend to all 60)."
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
    if row["model"].startswith("C1"):
        return ["background-color: #FBE9E7"] * len(row)
    return [""] * len(row)


st.dataframe(df.style.apply(_highlight, axis=1), width='stretch', hide_index=True)
st.caption(
    "A = all 41 gauge failures (stress + bulk porosity only). B = the 14 with a "
    "measured defect (adds Murakami √area). **C1 = the new PINN**, highlighted, "
    "same 14 specimens and same leave-one-out protocol as the B-series, so the row "
    "is directly comparable. Physics baseline (nothing fitted, threshold-corrected) "
    "over-predicts life by a geometric mean of "
    f"{res['physics_bias']['geometric_mean']:.1f}× before any calibration."
)

pinn_row = next((d for d in ladder if d["model"].startswith("C1")), None)
if pinn_row:
    st.info(
        f"The PINN uses **{pinn_row['params']} parameters** on 14 points — far more "
        "than the 1-4 parameter classical models. It is only kept from overfitting "
        "by the physics-residual loss term and weight decay; read its LOO number "
        "with that in mind, not as a free win over the constrained regression.",
        icon="⚖️",
    )

st.divider()
st.subheader("Measured vs. predicted life")
img1, img2 = st.columns(2)
with img1:
    st.image(os.path.join(ASSETS, "fig1_measured_vs_predicted.png"), width='stretch')
with img2:
    st.image(os.path.join(ASSETS, "fig3_failure_criterion.png"), width='stretch')

with st.expander("Verification: PINN physics-loss recovery (synthetic, known answer)"):
    st.markdown(
        "Before trusting the PINN on real data: fatigue lives synthesised from a "
        "**known** Basquin/Paris slope, n = 14 (matching the real specimen count), "
        "with noise. As the physics-loss weight λ rises from 0, the network's "
        "*learned gradient* should move toward the true slope."
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
    st.caption(
        "Both residual columns fall by two orders of magnitude as λ rises 0 → 10 "
        "— the physics term demonstrably pulls the network toward the true physics. "
        "LOO error is noisier (single-seed, stochastic training) but does not "
        "degrade materially. λ = 10 was chosen from this table, before the real fit "
        "was ever run — not tuned against the real data afterward."
    )

st.divider()
st.subheader("Explore the models live")
st.caption(
    "Sliders drive the ACTUAL fitted/trained models below — the physics baseline "
    "integrates the real Paris law, the constrained regression uses its real fitted "
    "coefficients, and the PINN is the real trained network (cached once at app "
    "start, not retrained per slider move).")


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


pinn_model, pinn_std, flat_med = _fit_pinn()
b3 = next((d for d in ladder if d["model"].startswith("B3")), None)

cL, cR = st.columns(2)
with cL:
    ds_slider = st.slider("Stress range Δσ (MPa)", 200, 500, 300, 5)
with cR:
    sa_slider = st.slider("Governing defect √area (µm)", 150, 1200, 450, 10)

a0_mm = sa_slider * 1e-3
smax_slider = ds_slider / (1 - 0.1)          # R = 0.1, same convention throughout
af_mm = collapse_depth_mm(smax_slider)
n_phys = life_numeric(a0_mm, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE, C=C_PARIS, m=M_PARIS)
n_phys_cal = life_numeric(a0_mm, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE,
                          C=res["physics_bias"]["C_implied"], m=M_PARIS)
st.caption(f"Collapse depth at this stress: {af_mm:.2f} mm (gauge radius {GAGE_DIA_MM/2:.2f} mm)")

if b3:
    b0, b_ds, b_sa = b3["coef"][2], b3["coef"][0], b3["coef"][1]
    log10_b3 = b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_slider)
    n_b3 = 10 ** log10_b3
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
rows_data = load()
meas_rows = splits(rows_data)["meas"]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sa_grid,
    y=[life_numeric(a * 1e-3, af_mm, ds_slider, dkth=DKTH, Y=Y_SURFACE,
                    C=res["physics_bias"]["C_implied"], m=M_PARIS) for a in sa_grid],
    mode="lines", name="Physics, calibrated C"))
if b3:
    fig.add_trace(go.Scatter(
        x=sa_grid, y=10 ** (b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_grid)),
        mode="lines", name="Constrained regression (B3)"))
Xg = np.column_stack([np.full_like(sa_grid, np.log(ds_slider)), np.log(sa_grid),
                      np.full_like(sa_grid, flat_med)])
fig.add_trace(go.Scatter(x=sa_grid, y=10 ** pinn_predict(pinn_model, pinn_std, Xg),
                         mode="lines", name="PINN (C1)"))
fig.add_trace(go.Scatter(
    x=[r["sqrt_area_gauge_um"] for r in meas_rows if abs(r["ds"] - ds_slider) < 60],
    y=[r["Nf"] for r in meas_rows if abs(r["ds"] - ds_slider) < 60],
    mode="markers", name="measured (Δσ within 60 MPa)",
    marker=dict(size=10, color="black", symbol="x")))
fig.update_layout(
    xaxis_title="governing defect √area (µm)", yaxis_title="predicted cycles to failure",
    yaxis_type="log", height=460, legend=dict(orientation="h", y=1.12),
    title=f"at Δσ = {ds_slider} MPa",
)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("Known limitations (stated, not hidden)")
st.markdown(
    "- Only 14 of 60 specimens have a measured defect size; the rest use bulk "
    "porosity, an average rather than the extreme-value measure the physics asks for.\n"
    "- Distance to the free surface, sphericity proper, and nearest-neighbour "
    "spacing are not yet extracted — the PINN and the regression both fall back to "
    "flatness as the shape proxy.\n"
    "- The PINN's 14-point leave-one-out result carries real variance (a single "
    "train/test protocol on a very small sample) — treat the comparison ladder as "
    "indicative, not a certified ranking.\n"
    "- The threshold-free long-crack Paris law does not strictly apply at these "
    "defect sizes (short-crack regime); see the threshold sensitivity sheet in "
    "`LPBF_PIML_Deliverables.xlsx`."
)

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import load, splits
from src.life_integral import life_numeric, Y_SURFACE, C_PARIS, M_PARIS
from src.piml_nn import train_pinn, predict as pinn_predict
from src.ui import GREY, NAVY, RED, RULE, finding, go_to, note, play, show, spec, verdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "comparison_model")
P4_DATA = os.path.join(ROOT, "data", "phase4_validation")

GAGE_DIA_MM = 5.08
SIGMA_UTS = 640.0          # as scripts/phase4_predict.py, used only for the collapse depth


def collapse_depth_mm(smax_mpa: float) -> float:
    """Path A's critical depth: the remaining ligament reaches the tensile strength,
    sigma_max / (1 - a^2 / 2r^2) = sigma_UTS, capped at the bar radius."""
    r = GAGE_DIA_MM / 2
    frac = 1.0 - smax_mpa / SIGMA_UTS
    if frac <= 0:
        return 0.05
    return float(np.clip(np.sqrt(2.0 * frac) * r, 0.05, r))


def _json(*parts):
    p = os.path.join(*parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


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
    st.title("Phase 4: validation")
    res = _json(DATA, "piml_results.json")
    comp = _json(DATA, "comparison_results.json")
    fail = _json(DATA, "failure_criterion.json")
    h4 = _json(P4_DATA, "h4_calibration.json")
    if not (res and comp):
        finding("The comparison result files in data/comparison_model/ are not present.")
        return
    lad = {r["model"].split()[0]: r for r in comp["ladder"]}
    pa = comp["phase4"]["log10_N_pathA"]["factor"]
    vr = res["verify_recovery"]
    rate = vr["table"][1].get("rate_s")
    spec(
        r"Theory fixes the form of a fatigue model. Neglecting the threshold, the life "
        r"integral is $N_f = a_0^{1-m/2}/\big[(m/2-1)\,C\,(Y\Delta\sigma\sqrt{\pi})^{m}\big]$, "
        r"so $\log_{10} N_f = b_0 + b_s\ln\Delta\sigma + b_a\ln\sqrt{\text{area}}$ with "
        r"$b_s=-m/\ln 10$ and $b_a=(1-m/2)/\ln 10$, both negative.",
        "A ladder of regressions from stress alone up to four parameters, the same "
        "regression with the signs of $b_s$ and $b_a$ bound (B3), and a physics-informed "
        "neural network (C1) whose loss penalises its own gradients for leaving the "
        "theoretical slopes. Every model is scored leave-one-out on the 14 cylinders "
        "with a measured defect.",
        (f"Fitted to lives synthesised with known exponents, the regression recovers them "
         f"with error falling at order {rate:.2f} as the scatter falls, the rate of an "
         "unbiased linear estimator." if rate else
         "The fitting procedure recovers known exponents from synthetic data."),
        (f"Path A, with nothing fitted, is within **{pa:.2f}×**. The best fit needs "
         f"{lad['B5']['params']} parameters for {lad['B5']['loo_factor']:.2f}×; the network "
         f"{lad['C1']['params']} for {lad['C1']['loo_factor']:.2f}×." +
         (f" Hypothesis 4 on B3: {h4['n_inside']} of {h4['n']} inside its interval, slope "
          f"{h4['slope']:.3f}." if h4 else "")))
    note("Leave-one-out: to score a model on one cylinder, fit it to the other thirteen, "
         "predict the one left out, and repeat for every cylinder. No prediction is made "
         "by a model that has seen its answer.")

    # ------------------------------------------------------------ dataset
    st.markdown("## The data")
    st.markdown(
        f"{comp['n_specimens']} cylinders of LPBF 316H were fatigue tested (ASTM E466, "
        f"R = 0.1, at 300 and 450 MPa): {comp['n_failures']} failed and "
        f"{comp['n_runouts']} ran out at 10 million cycles. Of the failures, "
        f"{comp['n_gauge']} broke in the gauge and {comp['n_radius']} at the fillet "
        f"radius. {comp['n_measured']} have a governing defect measured from their CT "
        "scan; they carry every model below that uses defect size.")

    # ------------------------------------------------------------ ladder
    st.markdown("## The comparison ladder")
    order = sorted(comp["ladder"], key=lambda r: (r["params"], r["model"]))
    names = [r["model"].split("  ")[0].strip() + "  " + r["model"].split("  ", 1)[-1].strip()
             for r in order]

    def rung(k):
        shown = order[:k + 1]
        cols = [RED if r["model"].startswith("C1") else NAVY if r["model"].startswith("B3")
                else GREY for r in shown]
        return [go.Bar(y=names[:k + 1], x=[r["loo_factor"] for r in shown], orientation="h",
                       marker_color=cols,
                       text=[f"{r['loo_factor']:.2f}×, {r['params']} parameters"
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
                                  range=[-0.5, len(names) - 0.5]),
                       height=470, margin=dict(t=30, b=110, l=10), showlegend=False)
    play(figl, frame_ms=650, transition_ms=250, slider_prefix="model ")
    show(figl)
    st.caption("The ladder climbed from the fewest parameters to the most. A: all gauge "
               "failures, without a defect size. B: the 14 with a measured defect. C1: "
               "the physics-informed network.")
    df = pd.DataFrame(res["ladder"])[["model", "n", "params", "r2", "loo_rmse", "loo_factor"]]
    df.columns = ["Model", "n", "Parameters", "R²", "LOO RMSE (decades)", "Typical error"]
    with st.expander("The ladder as a table"):
        st.dataframe(df.style.format({"R²": "{:.3f}", "LOO RMSE (decades)": "{:.3f}",
                                      "Typical error": "{:.2f}×"}),
                     width="stretch", hide_index=True)
    st.markdown(
        f"B3 and B4 score the same, {lad['B3']['loo_factor']:.3f}×: the unconstrained fit "
        "already has the physical signs, so the bound costs nothing. Adding defect size "
        f"takes stress-only from {lad['B1']['loo_factor']:.2f}× to "
        f"{lad['B3']['loo_factor']:.2f}× on the same cylinders. The network, with "
        f"{lad['C1']['params']} parameters on 14 points, does no better than three. "
        f"Path A matches the best fits without fitting anything to these lives.")

    # ------------------------------------------------------------ H4
    pts = comp.get("points_B3") or []
    if h4 and pts:
        st.markdown("## Hypothesis 4: are the predictions calibrated?")
        met = h4["slope_met"] and h4["coverage_met"]
        verdict(f"Criterion {'met' if met else 'not met'}: {h4['n_inside']} of {h4['n']} "
                f"({h4['coverage']:.1%}) inside the 90% interval, against 90%; slope "
                f"{h4['slope']:.3f}, against 0.90 to 1.10.", "met" if met else "no")
        s = h4["sigma_decades"]
        lims = [3e4, 3e6]
        k = 10 ** (1.645 * s)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lims, y=[l / k for l in lims], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=lims, y=[l * k for l in lims], mode="lines",
                                 fill="tonexty", fillcolor="rgba(31,78,121,0.10)",
                                 line=dict(width=0), name="90% interval"))
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                 line=dict(dash="dash", color=GREY), name="perfect agreement"))
        out = set(h4.get("specimens_outside", []))
        fig.add_trace(go.Scatter(
            x=[p["measured"] for p in pts], y=[p["predicted"] for p in pts],
            mode="markers+text", text=[p["specimen"] if p["specimen"] in out else ""
                                       for p in pts],
            textposition="top center",
            marker=dict(size=10, color=[RED if p["specimen"] in out else NAVY for p in pts]),
            customdata=[p["specimen"] for p in pts],
            hovertemplate="%{customdata}: measured %{x:,.0f}, predicted %{y:,.0f}",
            name="held-out cylinder"))
        fig.update_layout(xaxis_type="log", yaxis_type="log",
                          xaxis_range=np.log10(lims), yaxis_range=np.log10(lims),
                          xaxis_title="measured cycles to failure",
                          yaxis_title="B3 leave-one-out prediction", height=460,
                          legend=dict(orientation="h", y=1.1))
        show(fig)
        st.caption(f"The band is ±1.645 times the leave-one-out residual spread, "
                   f"σ = {s:.3f} decades. Red: the cylinder outside it.")
        if h4["slope"] - 0.90 < 0.01:
            st.markdown(f"The slope passes at the lower edge, {h4['slope']:.3f} against "
                        "0.90: B3 slightly compresses the range of lives, over-predicting "
                        "short ones and under-predicting long ones.")

    # ------------------------------------------------------------ PINN check
    with st.expander("Verification of the fitting methods against known answers"):
        tbl = pd.DataFrame(vr["table"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_s"], mode="lines+markers",
                                  name="stress slope", line=dict(color=NAVY)))
        fig2.add_trace(go.Scatter(x=tbl["scatter"], y=tbl["err_a"], mode="lines+markers",
                                  name="defect slope", line=dict(color=RED)))
        fig2.update_layout(xaxis_type="log", yaxis_type="log",
                           xaxis_title="scatter in synthetic lives (decades)",
                           yaxis_title="error in recovered slope", height=340,
                           legend=dict(orientation="h", y=1.15))
        show(fig2)
        st.caption(f"True slopes b_s = {vr['b_s_true']:+.4f}, b_a = {vr['b_a_true']:+.4f} "
                   f"(m = {vr['m_true']}). Both errors fall in proportion to the scatter.")
        pv = _json(DATA, "pinn_verification.json")
        if pv:
            st.markdown(
                f"For the network: {pv['n']} synthetic specimens with known slopes. As the "
                "physics weight λ rises from zero, the network's own learned gradient "
                "should move to the true slope.")
            st.dataframe(pd.DataFrame(pv["table"]).rename(columns={
                "lam": "λ", "b_s_resid_rms": "stress-slope residual",
                "b_a_resid_rms": "defect-slope residual",
                "loo_rmse": "LOO RMSE (decades)"}).style.format({
                    "λ": "{:.2f}", "stress-slope residual": "{:.4f}",
                    "defect-slope residual": "{:.4f}", "LOO RMSE (decades)": "{:.3f}"}),
                width="stretch", hide_index=True)
            st.caption("λ = 10 was chosen from this table before the real fit was run.")
        else:
            finding("The network's verification file, pinn_verification.json, has not been "
                    "generated: run python -m src.piml_nn.")

    # ------------------------------------------------------------ failure criterion
    if fail:
        st.markdown("## What ends the part")
        margin = fail["kic"] / fail["kmax_at_collapse"]
        st.markdown(
            f"At the collapse depth the stress intensity reaches at most "
            f"{fail['kmax_at_collapse']:.1f} MPa√m, against a fracture toughness of about "
            f"{fail['kic']:.0f} MPa√m, a margin of {margin:.0f}. 316 steel is too tough to "
            "fracture fast in a 5 mm bar: the part fails when the cracked section can no "
            "longer carry the load, which is how path A defines the final crack depth.")
        loc = fail["fracture_location"]
        st.dataframe(pd.DataFrame({"Fracture location": list(loc), "Cylinders": list(loc.values())}),
                     width="content", hide_index=True)
        st.markdown("The fillet-radius failures start at a stress concentration rather than "
                    "a defect and lie outside the defect-driven model.")

    # ------------------------------------------------------------ explorer
    st.markdown("## Try the models")
    st.markdown("The sliders drive the models themselves: path A integrates the Paris law "
                "with its literature constants, B3 uses its fitted coefficients, and the "
                "network is the trained network.")
    pinn_model, pinn_std, flat_med = _fit_pinn()
    b3 = next((d for d in res["ladder"] if d["model"].startswith("B3")), None)

    cL, cR = st.columns(2)
    ds_slider = cL.slider("Stress range Δσ (MPa)", 200, 500, 405, 5)
    sa_slider = cR.slider("Governing defect √area (µm)", 150, 1200, 450, 10)
    smax = ds_slider / (1 - 0.1)
    af_mm = collapse_depth_mm(smax)

    def n_path_a(sa_um):
        return life_numeric(sa_um * 1e-3, af_mm, ds_slider, dkth=0.0, Y=Y_SURFACE,
                            C=C_PARIS, m=M_PARIS)

    n_a = n_path_a(sa_slider)
    if b3:
        b_ds, b_sa, b0 = b3["coef"][0], b3["coef"][1], b3["coef"][2]
        n_b3 = 10 ** (b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_slider))
    Xq = np.array([[np.log(ds_slider), np.log(sa_slider), flat_med]])
    n_pinn = 10 ** pinn_predict(pinn_model, pinn_std, Xq)[0]
    rows = [("Path A, nothing fitted", n_a)]
    if b3:
        rows.append(("B3, constrained regression", n_b3))
    rows.append(("C1, physics-informed network", n_pinn))
    st.dataframe(pd.DataFrame([(a, f"{b:,.0f}") for a, b in rows],
                              columns=["Model", "Predicted cycles"]),
                 width="stretch", hide_index=True)
    st.caption(f"σmax = {smax:.0f} MPa; collapse depth {af_mm:.2f} mm (bar radius "
               f"{GAGE_DIA_MM / 2:.2f} mm).")

    sa_grid = np.linspace(150, 1200, 60)
    figE = go.Figure()
    figE.add_trace(go.Scatter(x=sa_grid, y=[n_path_a(a) for a in sa_grid], mode="lines",
                              name="path A", line=dict(color=NAVY, width=2.5)))
    if b3:
        figE.add_trace(go.Scatter(x=sa_grid,
                                  y=10 ** (b0 + b_ds * np.log(ds_slider) + b_sa * np.log(sa_grid)),
                                  mode="lines", name="B3", line=dict(color=GREY, dash="dash")))
    Xg = np.column_stack([np.full_like(sa_grid, np.log(ds_slider)), np.log(sa_grid),
                          np.full_like(sa_grid, flat_med)])
    figE.add_trace(go.Scatter(x=sa_grid, y=10 ** pinn_predict(pinn_model, pinn_std, Xg),
                              mode="lines", name="C1", line=dict(color=RED, dash="dot")))
    meas_rows = splits(load())["meas"]
    near = [r for r in meas_rows if abs(r["ds"] - ds_slider) < 60]
    figE.add_trace(go.Scatter(x=[r["sqrt_area_gauge_um"] for r in near],
                              y=[r["Nf"] for r in near], mode="markers",
                              name="measured, Δσ within 60 MPa",
                              marker=dict(size=9, color="#1E2A36", symbol="x")))
    figE.add_vline(x=sa_slider, line_color=RULE)
    figE.update_layout(xaxis_title="governing defect √area (µm)",
                       yaxis_title="cycles to failure", yaxis_type="log", height=440,
                       legend=dict(orientation="h", y=1.12))
    show(figE)

    st.markdown("## Limits")
    st.markdown(
        "- Only 14 of the 60 cylinders have a measured defect size; the ladder's A rungs "
        "use bulk porosity, an average where the physics asks for the extreme.\n"
        "- With 14 points the leave-one-out scores carry real variance; read the ladder "
        "as indicative, not as a certified ranking.\n"
        "- Path A's threshold of zero is a short-crack argument, supported by the "
        "threshold sweep in Phase 3, not a measured short-crack law.")
    go_to("conclusions", "Continue to the conclusions")

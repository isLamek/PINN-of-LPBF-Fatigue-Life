import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import badge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PD_DATA = os.path.join(ROOT, "data", "phase2_peridynamics")


def render():
    st.title("Phase 3 · Fatigue life prediction")
    badge("CALIBRATION DONE", "real")

    st.markdown(
        "Phase 2 grows a crack; this phase turns that crack path into an actual "
        "**number of cycles**. The peridynamic damage law (Phase 2) sets the "
        "crack's shape and direction correctly regardless of the rate "
        "coefficient $A$ — but $A$ alone sets how many simulated cycles one unit "
        "of bond damage corresponds to. Get $A$ wrong and the path is still "
        "right; the cycle count is meaningless. This is where $A$ is calibrated."
    )

    d = json.load(open(os.path.join(PD_DATA, "verify_04_results.json")))
    s5, s10 = d["5"], d["10"]

    st.divider()
    st.subheader("The governing equation, once A is known")
    st.markdown("Stress intensity factor range at crack depth $a$:")
    st.latex(r"\Delta K(a) = Y \cdot \Delta\sigma \cdot \sqrt{\pi a}")
    st.markdown("Paris crack-growth law:")
    st.latex(r"\frac{da}{dN} = C \cdot \big(\Delta K - \Delta K_{th}\big)^{m}")
    st.markdown("Cycles to grow the governing defect ($a_0 = \\sqrt{\\text{area}}$, Phase 2) to the collapse depth $a_f$:")
    st.latex(r"N = \int_{a_0}^{a_f} \frac{da}{C\,\big(\Delta K(a) - \Delta K_{th}\big)^{m}}")
    st.caption(
        "This integral has a closed form for $\\Delta K_{th}=0$ (verified "
        "against Simpson's rule to round-off precision); with the threshold "
        "retained it's evaluated numerically on a log grid, since the "
        "integrand is dominated by the lower limit when $m>2$ — almost every "
        "sample on a linear grid would land where it contributes nothing."
    )

    st.divider()
    st.subheader("Damage-coefficient calibration")
    st.caption(
        "The damage rate coefficient **A** sets how many simulated cycles "
        "correspond to one unit of bond damage — get it wrong and the crack "
        "path is still right, but the cycle count is meaningless."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("A, uncalibrated demo value", "5.00e6",
             help="Used only for early presentation animations.")
    c2.metric("A, calibrated vs. literature C (316L)", f"{s5['A_calibrated']:.3e}",
             help="Against Merot et al. 2022's 316L Paris coefficient.")
    c3.metric("A, calibrated vs. dataset C (316H)", f"{s10['A_calibrated_dataset']:.3e}",
             help="Against this project's own 316H-specific Paris coefficient (Phase 4).")

    st.markdown(
        f"The literature Paris coefficient (316L, C = {s10['C_316L']:.3g} "
        f"mm/cycle) under-predicts growth on the *actual* 316H Snow specimens by "
        f"a geometric mean of **{s10['ratio']:.1f}×** (Phase 4's physics-only "
        f"baseline, 13 held-out specimens). Correcting for that and rescaling A "
        f"by the same ratio — exact, no new simulation, since the damage rate "
        f"is linear in A — gives the calibrated value above. Net effect: the "
        f"original demo value was only **{s10['demo_over_calibrated']:.2f}×** "
        "off, once both corrections are combined."
    )

    st.markdown("**Why calibration needs no extra simulation.** The damage rate is linear in $A$, so one run at $A_{run}$ tells you the rate at any other $A$:")
    st.latex(r"A_{\text{target}} = A_{\text{run}} \cdot \frac{C_{\text{target}} \cdot \Delta K_{\text{ref}}^{\,\beta}}{(da/dN)_{\text{run at ref}}}")
    st.caption(
        "So rescaling the target Paris coefficient from literature 316L to this "
        "project's own dataset-specific 316H value is a single multiplication of "
        "the already-calibrated $A$ — not a new simulation."
    )

    with st.expander("Transfer verification (no fitting after the single calibration point)"):
        tf = pd.DataFrame(s10["transfer"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tf["dK"], y=tf["predicted"], mode="lines+markers",
                                 name="Paris law prediction"))
        fig.add_trace(go.Scatter(x=tf["dK"], y=tf["model"], mode="markers",
                                 name="Peridynamic model", marker=dict(size=10, symbol="x")))
        fig.update_layout(xaxis_title="ΔK (MPa√m)", yaxis_title="da/dN (mm/cycle)",
                          yaxis_type="log", height=400, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')
        errs = [t["err_pct"] for t in s10["transfer"] if t["dK"] != tf["dK"].iloc[0]]
        st.caption(f"Mean error on the non-calibration points: {np.mean(errs):.1f}%. "
                  "A is calibrated against ONE point; the rest are pure predictions.")

    st.divider()
    st.subheader("Per-specimen results")
    st.caption("See Phase 2 for the crack-growth animation on P017's real measured defect field.")
    spec = pd.read_csv(os.path.join(PD_DATA, "specimen_summary.csv"))
    has_life = "cycles_to_failure" in spec.columns and "is_life" in spec.columns
    n_life = int(spec["is_life"].astype(str).str.lower().eq("true").sum()) if has_life else 0

    if has_life and n_life > 0:
        badge("FATIGUE LIFE RESULTS", "real")
        live = spec[spec["is_life"].astype(str).str.lower() == "true"].copy()
        live["cycles_to_failure"] = pd.to_numeric(live["cycles_to_failure"])
        live["Nf_measured"] = pd.to_numeric(live["Nf_measured"])
        fig = go.Figure()
        lims = [live[["cycles_to_failure", "Nf_measured"]].min().min() / 2,
               live[["cycles_to_failure", "Nf_measured"]].max().max() * 2]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                 line=dict(dash="dash", color="#8A9096"), name="1:1"))
        fig.add_trace(go.Scatter(x=live["Nf_measured"], y=live["cycles_to_failure"],
                                 mode="markers+text", text=live["specimen"],
                                 textposition="top center", marker=dict(size=10, color="#B5473A"),
                                 name="peridynamic prediction"))
        fig.update_layout(xaxis_type="log", yaxis_type="log",
                          xaxis_title="measured cycles to failure",
                          yaxis_title="peridynamic-predicted cycles to failure",
                          height=460)
        st.plotly_chart(fig, width='stretch')
        resid = np.log10(live["cycles_to_failure"]) - np.log10(live["Nf_measured"])
        st.metric("Typical life error (this arm)", f"{10**np.sqrt((resid**2).mean()):.2f}×",
                  help=f"RMSE in decades over {len(live)} specimens that reached the "
                       "stiffness failure criterion, not the event budget.")
    else:
        badge("TO-FAILURE RE-RUN PENDING", "pending")

    st.dataframe(spec, width='stretch', hide_index=True)

    if has_life and 0 < n_life < len(spec):
        st.warning(
            f"{n_life} of {len(spec)} specimens reached the stiffness failure "
            "criterion; the rest are still mid-run (stopped by the event budget) "
            "— re-run with a larger --events budget to complete them.",
            icon="\U0001F6A7",
        )
    elif not has_life:
        st.error(
            "Every run on file stopped at a fixed event budget, not the 90% "
            "stiffness-loss failure criterion, and used the OLD uncalibrated "
            "A = 5.0e6. **None of these numbers is a fatigue-life prediction.** "
            "Re-run `scripts/run_all_specimens.py --redo` under the calibrated A "
            "to get real predicted lives.",
            icon="\U0001F6A7",
        )

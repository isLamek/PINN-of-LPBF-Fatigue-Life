import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import setup, badge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
DATA = os.path.join(ROOT, "data", "phase2_extraction")
PD_DATA = os.path.join(ROOT, "data", "phase2_peridynamics")

setup("Phase 2 - Defect Extraction", "\U0001F50E")
st.title("Phase 2 · Defect extraction and crack growth")
badge("REAL RESULTS", "real")
st.caption(
    "3-D connected-components labelling of the Phase 1 predicted flaw masks, cross-"
    "checked against the AOP24 pre-extracted cluster reference derived from the same "
    "Snow dataset."
)

cat = pd.read_csv(os.path.join(DATA, "phase2_defect_catalogue.csv"))
cyl = pd.read_csv(os.path.join(DATA, "phase2_cylinder_summary.csv"))

c1, c2, c3 = st.columns(3)
c1.metric("Cylinders catalogued", cyl["cylinder"].nunique())
c2.metric("Defects catalogued", f"{len(cat):,}")
c3.metric("Largest √area observed", f"{cat['sqrt_area_um'].max():,.0f} µm")

st.divider()
st.subheader("Cylinder-level summary")
st.dataframe(
    cyl.rename(columns={"largest_sqrt_area_um": "largest √area (µm)",
                        "aop24_porosity_pct": "porosity % (AOP24)"}),
    width='stretch', hide_index=True,
)

st.subheader("Segmentation vs. AOP24 cross-check")
p = os.path.join(ASSETS, "phase2_extraction_crosscheck.png")
if os.path.exists(p):
    st.image(p, width='stretch')

st.divider()
st.subheader("Defect catalogue — 1342 defects")
process = st.selectbox("Filter by cylinder", ["All"] + sorted(cat["cylinder"].unique()))
view = cat if process == "All" else cat[cat["cylinder"] == process]
fig = px.scatter(
    view, x="sqrt_area_um", y="flatness", color="cylinder" if process == "All" else None,
    hover_data=["defect_id", "volume_mm3", "equiv_diam_um"],
    labels={"sqrt_area_um": "√area (µm), Murakami parameter", "flatness": "flatness (build-axis / in-plane extent)"},
    title="Defect size vs. shape — flatness ≈ 0.1 is a lack-of-fusion pancake, ≈ 1 a spherical gas pore",
)
fig.update_layout(height=480)
st.plotly_chart(fig, width='stretch')
st.dataframe(view.sort_values("sqrt_area_um", ascending=False), width='stretch', hide_index=True, height=280)

st.divider()
st.subheader("Fatigue crack growth through a measured defect field (P017)")
st.caption(
    "Peridynamic bond-damage animation on P017's real measured defect field, 300 MPa "
    "at R = 0.1. Cyan = pre-existing lack-of-fusion defects. Orange = fatigue damage "
    "accumulated since cycle zero. Defect POSITIONS are a random realisation — the "
    "catalogue records size and shape, not coordinates."
)
gif = os.path.join(ASSETS, "phase2_crack_growth.gif")
stages = os.path.join(ASSETS, "phase2_crack_growth_stages_P017.png")
cc1, cc2 = st.columns(2)
with cc1:
    if os.path.exists(gif):
        st.image(gif, caption="Crack growth animation", width='stretch')
with cc2:
    if os.path.exists(stages):
        st.image(stages, caption="Growth stages, P017", width='stretch')

st.warning(
    "This animation's cycle count predates the damage-coefficient calibration below "
    "— it shows the crack **path**, not a life. The panel does not claim a number "
    "of cycles.",
    icon="⚠️",
)

st.divider()
st.subheader("Damage-coefficient calibration")
badge("CALIBRATION DONE", "real")
badge("TO-FAILURE RE-RUN PENDING", "pending")

d = json.load(open(os.path.join(PD_DATA, "verify_04_results.json")))
s5, s10 = d["5"], d["10"]

st.markdown(
    "Bond-based peridynamics (Silling & Askari 2014 cyclic-damage law) grows a crack "
    "from the governing defect under the specimen's own ASTM E466 loading. The damage "
    "rate coefficient **A** sets how many simulated cycles correspond to one unit of "
    "bond damage — get it wrong and the crack path is still right, but the cycle "
    "count is meaningless."
)

c1, c2, c3 = st.columns(3)
c1.metric("A, uncalibrated demo value", "5.00e6", help="Used only for presentation animations.")
c2.metric("A, calibrated vs. literature C (316L)", f"{s5['A_calibrated']:.3e}",
         help="verify_05_calibration.py, against Merot et al. 2022's 316L Paris coefficient.")
c3.metric("A, calibrated vs. dataset C (316H)", f"{s10['A_calibrated_dataset']:.3e}",
         help="verify_10_calibration_snow316H.py — this project's own contribution.")

st.markdown(
    f"The literature Paris coefficient (316L, C = {s10['C_316L']:.3g} mm/cycle) "
    f"under-predicts growth on the *actual* 316H Snow specimens by a geometric mean "
    f"of **{s10['ratio']:.1f}×** (from the Phase 4 physics-only baseline, 13 held-out "
    f"specimens — see the Phase 4 page). Correcting for that gives the dataset-"
    f"specific C = {s10['C_implied']:.3e} mm/cycle, and rescaling A by the same "
    "ratio — exact, no new simulation, since the damage rate is linear in A — gives "
    f"the calibrated value above. Net effect: the original demo value turns out to "
    f"be only **{s10['demo_over_calibrated']:.2f}×** off, once both corrections are "
    "combined, much less than either correction alone."
)

with st.expander("Transfer verification (no fitting after the single calibration point)"):
    st.caption(
        "A is calibrated against ONE point; the rest are pure predictions. The error "
        "percentages are algebraically invariant to which target C is used (both "
        "scale identically), confirmed here to floating-point precision as an "
        "independent cross-check."
    )
    tf = pd.DataFrame(s10["transfer"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tf["dK"], y=tf["predicted"], mode="lines+markers",
                             name="Paris law prediction"))
    fig.add_trace(go.Scatter(x=tf["dK"], y=tf["model"], mode="markers",
                             name="Peridynamic model", marker=dict(size=10, symbol="x")))
    fig.update_layout(xaxis_title="ΔK (MPa√m)", yaxis_title="da/dN (mm/cycle)",
                      yaxis_type="log", height=420, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

st.subheader("The 14 real specimen runs — honest status")
spec = pd.read_csv(os.path.join(PD_DATA, "specimen_summary.csv"))
st.dataframe(spec, width='stretch', hide_index=True)
st.error(
    "Every run above stopped at the fixed 200-event budget (`stopped_by = steps`), "
    "not the 90% stiffness-loss failure criterion, and all were run at the OLD "
    "uncalibrated A = 5.0e6. **None of these numbers is a fatigue-life prediction.** "
    "Re-running `run_all_specimens.py --redo` to the failure criterion under the "
    "calibrated A is the remaining step before Phase 2's peridynamics can report "
    "cycles to compare against Phase 4.",
    icon="🚧",
)

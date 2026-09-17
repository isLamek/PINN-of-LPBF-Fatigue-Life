import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ui import setup, badge

setup("Overview", "\U0001F52C")

st.title("Mathematical Modelling of Internal Properties of LPBF Components")
st.caption(
    "Lamek S. Indongo (221427503) · BSc Hons Mechanical Engineering, UNAM · "
    "Supervisors: Assoc. Prof. Albert Shikongo, Dr. Surendra Kumar Saini"
)

st.markdown(
    "A pipeline from LPBF X-ray CT volumes to a validated fatigue-life prediction: "
    "segment internal defects &rarr; extract 3-D defect geometry &rarr; grow a peridynamic "
    "crack from the governing defect &rarr; validate against measured ASTM E466 fatigue "
    "life on the same 316H specimens (Snow et al. 2025, ORNL, DOI 10.13139/ORNLNCCS/2524534)."
)

st.divider()
st.subheader("Pipeline status")

cols = st.columns(4)
stages = [
    ("Phase 1 · Segmentation", "real",
     "nnU-Net v2, 5-fold cross-validation, held-out test evaluation.", "1_Phase1_Segmentation"),
    ("Phase 2 · Defect extraction & peridynamics", "real",
     "3-D connected components, Murakami √area, crack-growth animation, and the "
     "damage-coefficient A calibration (to-failure re-run still owed).", "2_Phase2_Defect_Extraction"),
    ("Phase 3 · Monte Carlo", "preview",
     "Probabilistic S-N sampling over the full defect population. Not yet run.",
     "3_Phase3_Monte_Carlo"),
    ("Phase 4 · Validation & PIML", "real",
     "Closed-loop validation, statistical ladder, physics-constrained regression, "
     "and a physics-informed neural network.", "4_Phase4_Validation_PIML"),
]
kind_label = {"real": "REAL RESULTS", "preview": "PREVIEW", "pending": "RE-RUN PENDING"}
for c, (name, kind, desc, page) in zip(cols, stages):
    with c:
        st.markdown(f"**{name}**")
        badge(kind_label[kind], kind)
        st.caption(desc)

st.divider()
st.subheader("What this app is")
st.markdown(
    "- Every number and figure here comes from the verified pipeline in "
    "`Research files/` — nothing on these pages is fabricated for display.\n"
    "- Phase 4 includes the **physics-informed comparison model** requested by the "
    "supervisor: a Paris/Murakami crack-growth law fit with sign-constrained "
    "coefficients, benchmarked against a statistical ladder, plus a new small "
    "**physics-informed neural network (PINN)** trained with a hybrid data + "
    "physics-residual loss — see the Phase 4 page.\n"
    "- Where something is not yet complete (the peridynamic-to-failure re-run), "
    "it is labelled as such rather than shown as a finished result."
)

st.info(
    "Use the sidebar to move between phases. The Phase 4 page is the one built "
    "directly from the supervisor's request for a PIML comparison model.",
    icon="\U0001F4CC",
)

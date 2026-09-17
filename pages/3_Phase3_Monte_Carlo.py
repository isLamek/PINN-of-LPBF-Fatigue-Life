import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui import setup, badge

setup("Phase 3 - Monte Carlo", "\U0001F3B2")
st.title("Phase 3 · Monte Carlo fatigue sampling")
badge("NOT YET RUN", "preview")

st.warning(
    "This phase has not been computed. Nothing on this page is a result — it "
    "describes what Phase 3 will do and what it depends on, so the status is "
    "visible rather than papered over with an illustrative placeholder.",
    icon="\U0001F6A7",
)

st.markdown(
    "**Planned method.** Sample the Murakami √area / LEFM life model over the "
    "full extracted defect population (Phase 2's 1342-defect catalogue) to build a "
    "probabilistic S–N curve per process regime, rather than the single-governing-"
    "defect point estimate Phase 2 and Phase 4 currently use."
)

c1, c2, c3 = st.columns(3)
c1.metric("Life model", "Murakami √area · LEFM")
c2.metric("Planned samples / stress level", "10 000")
c3.metric("Defect input", "Phase 2 catalogue (1342 defects)")

st.markdown(
    "**Depends on:** the Phase 2 defect-extraction pipeline (done) and a decision "
    "on whether to draw defect populations from the network-predicted catalogue, "
    "the ground-truth labels, or both (see Phase 4's Path A / Path B split for why "
    "that choice matters)."
)
st.markdown(
    "**Not started because:** Phases 1, 2 and 4 covered the supervisor's immediate "
    "request — the PIML comparison model — first. This page will be replaced with "
    "real output once Phase 3 is run, in the same style as Phase 4: a clearly "
    "labelled REAL RESULTS badge and cited numbers, nothing implied."
)

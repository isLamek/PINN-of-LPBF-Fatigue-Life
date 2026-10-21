import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import ct_viewer, data, hotreload, life_integral, piml_nn, ui
from src.sections import conclusions, overview, phase1, phase2, phase3, phase4

# shared modules first, then the pages that import names from them
hotreload.refresh(ui, data, life_integral, piml_nn, ct_viewer,
                  overview, phase1, phase2, phase3, phase4, conclusions)
from src.ui import setup, top_nav  # noqa: E402  (after the refresh)

setup("LPBF Fatigue Life", "\U0001F52C")

page = top_nav()

RENDER = {"overview": overview.render, "phase1": phase1.render, "phase2": phase2.render,
          "phase3": phase3.render, "phase4": phase4.render,
          "conclusions": conclusions.render}
with st.container(key=f"content_{page}"):
    RENDER[page]()

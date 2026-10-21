import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ui import TITLE, setup, top_nav
from src.sections import conclusions, overview, phase1, phase2, phase3, phase4

setup("LPBF Fatigue Life", "\U0001F52C")

st.caption(
    f"{TITLE} · Lamek S. Indongo (221427503) · BSc Hons Mechanical Engineering, UNAM · "
    "Supervisor: Assoc. Prof. Albert Shikongo · Co-supervisor: Dr. Surendra Kumar Saini"
)

page = top_nav()

RENDER = {"overview": overview.render, "phase1": phase1.render, "phase2": phase2.render,
          "phase3": phase3.render, "phase4": phase4.render,
          "conclusions": conclusions.render}
with st.container(key=f"content_{page}"):
    RENDER[page]()

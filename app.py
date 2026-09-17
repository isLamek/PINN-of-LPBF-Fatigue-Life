import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ui import setup, top_nav
from src.sections import phase1, phase2, phase3, phase4

setup("LPBF Fatigue Life", "\U0001F52C")

st.title("Mathematical Modelling of Internal Properties of LPBF Components")
st.caption(
    "Lamek S. Indongo (221427503) · BSc Hons Mechanical Engineering, UNAM · "
    "Supervisors: Assoc. Prof. Albert Shikongo, Dr. Surendra Kumar Saini"
)

phase = top_nav()

RENDER = {"phase1": phase1.render, "phase2": phase2.render,
         "phase3": phase3.render, "phase4": phase4.render}
with st.container(key=f"content_{phase}"):
    RENDER[phase]()

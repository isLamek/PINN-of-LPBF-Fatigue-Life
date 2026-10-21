import base64
import csv
import io
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src import ct_viewer as cv
from src.data import FOLD_DICE, HELDOUT_SIZE_R2, STUDY_METRICS
from src.ui import ACCENT, GREY, badge, plain, play, thinking

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_CSV = os.path.join(ROOT, "data", "00_docs", "Appendix_A_train_test_split.csv")
SPEC_LABELS = {"SN_P017": "P017 — governing defect 1017 µm (largest of the 14)",
               "SN_P059": "P059 — most defects in gauge (286)"}


@st.cache_data(show_spinner="Preparing the CT stack…")
def _stack_frames(spec: str, size: int = 340):
    """Each slice as a small PNG (raw CT on the left, AI defects on the right), base64
    encoded: a 40-frame animation of raw arrays would be tens of megabytes."""
    out = []
    for z in cv.list_slices(spec):
        ct = cv.load_ct_slice(spec, z)
        ov = cv.composite_slice(spec, z)
        mask = cv.load_mask_slice(spec, z)
        if ct is None or ov is None or ct.mean() < 0.05:
            continue                     # blank scans at the ends of the stack
        left = Image.fromarray((ct * 255).astype("uint8")).convert("RGB").resize((size, size))
        right = Image.fromarray(ov).resize((size, size))
        both = Image.new("RGB", (2 * size + 8, size), (255, 255, 255))
        both.paste(left, (0, 0))
        both.paste(right, (size + 8, 0))
        buf = io.BytesIO()
        both.save(buf, format="PNG", optimize=True)
        n_px = int(mask.sum()) if mask is not None else 0
        out.append((z, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    n_px))
    return out


def _split_counts():
    rows = list(csv.DictReader(open(SPLIT_CSV, encoding="utf-8-sig")))
    c = {}
    for r in rows:
        c[r["split"]] = c.get(r["split"], 0) + 1
    return len(rows), c


def render():
    st.title("Phase 1 · Find the defects")
    badge("REAL RESULTS", "real")
    n_all, sp = _split_counts()
    thinking(
        "Where are the holes inside this part, and how big is each one?",
        "An nnU-Net v2 network labels every pixel of every CT slice as metal or defect.",
        f"5-fold cross-validation, and a size check on {sp.get('test', 0)} cylinders held "
        "out from all training.",
        f"Mean Dice {STUDY_METRICS['Dice']:.3f}, above the 0.80 criterion. "
        f"Held-out defect size R² = {HELDOUT_SIZE_R2:.3f}.",
        "The defect map that every later phase uses can be trusted. Hypothesis H1 is "
        "supported.")
    plain("<b>What is Dice?</b> It measures overlap between the AI's defect pixels and "
          "the reference labels: 1 is perfect, 0 is no overlap. Defects fill a tiny "
          "fraction of each image, which makes Dice harsh, so 0.80 already means close "
          "agreement.")

    # ---------------------------------------------------------------- animated stack
    st.subheader("Watch the AI work through a cylinder, slice by slice")
    specs = cv.available_specimens()
    if not specs:
        st.warning("No CT slice data found in data/ct_slices/.", icon="⚠️")
    else:
        spec = st.selectbox("Specimen", specs, format_func=lambda s: SPEC_LABELS.get(s, s),
                            key="p1_spec")
        frames = _stack_frames(spec)
        if frames:
            z0, src0, _ = frames[0]
            fig = go.Figure(go.Image(source=src0))
            fig.frames = [go.Frame(data=[go.Image(source=s)], name=f"z{z:04d}")
                          for z, s, _ in frames]
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=90),
                              xaxis=dict(visible=False), yaxis=dict(visible=False),
                              title=dict(text="raw CT  ·  defects found by the AI (red)",
                                         x=0.5, font=dict(size=13)))
            play(fig, frame_ms=250, transition_ms=0, slider_prefix="slice ")
            st.plotly_chart(fig, width="stretch")
            zs = [z for z, _, _ in frames]
            px_ = [n for _, _, n in frames]
            st.caption(
                f"{len(frames)} slices of {spec.replace('SN_', '')}, z{zs[0]:04d} to "
                f"z{zs[-1]:04d}. Press Play: defects appear and disappear as the slice "
                f"passes through them. The slice with the most defect pixels is "
                f"z{zs[int(np.argmax(px_))]:04d}.")

    # ---------------------------------------------------------------- the check
    st.divider()
    st.subheader("The check: does it hold on data it has never seen?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dice (5-fold mean)", f"{STUDY_METRICS['Dice']:.3f}")
    c2.metric("Precision", f"{STUDY_METRICS['Precision']:.3f}",
              help="Of the pixels the AI calls defect, the share that are defect.")
    c3.metric("Recall", f"{STUDY_METRICS['Recall']:.3f}",
              help="Of the true defect pixels, the share the AI finds.")
    c4.metric("IoU", f"{STUDY_METRICS['IoU']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        mean = sum(FOLD_DICE) / len(FOLD_DICE)
        folds = [f"fold {i + 1}" for i in range(len(FOLD_DICE))]
        fig = go.Figure(go.Bar(x=folds, y=FOLD_DICE, marker_color=GREY,
                               text=[f"{v:.3f}" for v in FOLD_DICE], textposition="outside"))
        fig.add_hline(y=0.80, line_dash="dot", line_color=ACCENT,
                      annotation_text="criterion 0.80", annotation_position="bottom right")
        fig.add_hline(y=mean, line_dash="dash", line_color="#1F4E79",
                      annotation_text=f"mean {mean:.3f}", annotation_position="top left")
        fig.update_layout(yaxis_range=[0.75, 0.85], yaxis_title="Dice", height=340,
                          margin=dict(t=20, b=10), title="Dice by cross-validation fold")
        st.plotly_chart(fig, width="stretch")
        st.caption("Every fold lands close to the mean: the result does not hang on one "
                   "lucky split.")
    with col2:
        st.markdown("**Why precision above recall is the safer error**")
        st.markdown(
            f"Precision ({STUDY_METRICS['Precision']:.3f}) is above recall "
            f"({STUDY_METRICS['Recall']:.3f}): the AI misses slightly more true defect "
            "pixels than it invents. The next phase only needs the **largest** defect in "
            "the gauge. Trimming the edge of a large defect barely changes its √area, "
            "while inventing a large one that is not there would change the predicted "
            "life completely.")
        st.metric("Held-out governing defect size, R² against the reference labels",
                  f"{HELDOUT_SIZE_R2:.3f}")

    st.divider()
    st.subheader("How the data were split")
    st.markdown(
        f"The {n_all} cylinders were split **before any training**: "
        f"**{sp.get('train', 0)} training**, **{sp.get('test', 0)} held out**, and "
        f"{sp.get('excluded_not_tested', 0)} not fatigue tested. The split is stratified "
        "by process regime (keyhole, spatter, clean) and by stress level (300 and "
        "450 MPa), so neither set is dominated by one condition. The held-out cylinders "
        "are used throughout Phases 2 to 4, and no model was ever fitted on them.")
    st.markdown(
        "**Labels.** The reference labels come from the originating team's own "
        "Laplacian-of-Gaussian, threshold and clustering pipeline, not from hand "
        "annotation, so the network is learning to reproduce a published method on "
        "unseen cylinders.")
    st.caption("nnU-Net v2, 2-D, single-channel 16-bit input, two classes; RTX 3050 4 GB, "
               "patch 768×768, batch 1, 200 epochs, 5-fold cross-validation.")

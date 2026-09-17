"""Shared chrome for the single-page app: page config, a top pipeline bar
(mirroring app_fatigue_framework/ui/pipeline_bar.py's node strip instead of
Streamlit's default sidebar), a little CSS, small helpers.
"""
import streamlit as st

ACCENT = "#B5473A"
PANEL = "#F4F1EC"

PHASES = [
    ("phase1", "Phase 1", "Segmentation", "real"),
    ("phase2", "Phase 2", "Defect extraction", "real"),
    ("phase3", "Phase 3", "Peridynamic crack growth", "real"),
    ("phase4", "Phase 4", "Validation & PIML", "real"),
]
STATUS_LABEL = {"real": "REAL RESULTS", "preview": "PREVIEW", "pending": "PENDING"}


def setup(title: str = "LPBF Fatigue Life", icon: str = "\U0001F52C"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown("""
    <style>
      [data-testid="stSidebar"] { display: none; }
      [data-testid="collapsedControl"] { display: none; }
      .block-container { padding-top: 1.5rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      .status-real { color: #2E7D32; font-weight: 600; }
      .status-preview { color: #B5473A; font-weight: 600; }
      .status-pending { color: #8A6D3B; font-weight: 600; }
      .lpbf-badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.72rem; font-weight:700; margin-right:6px; letter-spacing:.02em;
      }
      .b-real { background:#E4F3E6; color:#2E7D32; }
      .b-preview { background:#FBE9E7; color:#B5473A; }
      .b-pending { background:#FBF3DE; color:#8A6D3B; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      div[data-testid="stSegmentedControl"] button {
        font-weight: 600 !important;
      }
    </style>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str = "real"):
    st.markdown(f'<span class="lpbf-badge b-{kind}">{text}</span>', unsafe_allow_html=True)


def top_nav() -> str:
    """The pipeline bar: four phase nodes across the top, status badge under
    each. Returns the selected phase key ('phase1'..'phase4')."""
    if "phase" not in st.session_state:
        st.session_state["phase"] = "phase1"

    labels = [f"{num} · {name}" for _, num, name, _ in PHASES]
    keys = [k for k, *_ in PHASES]
    current_idx = keys.index(st.session_state["phase"])

    choice = st.segmented_control(
        "Pipeline", labels, default=labels[current_idx], label_visibility="collapsed",
        key="_nav_control",
    )
    if choice is not None:
        st.session_state["phase"] = keys[labels.index(choice)]

    cols = st.columns(len(PHASES))
    for c, (key, num, name, status) in zip(cols, PHASES):
        with c:
            badge(STATUS_LABEL[status], status)
    st.divider()
    return st.session_state["phase"]

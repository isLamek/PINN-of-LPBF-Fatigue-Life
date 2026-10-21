"""Shared chrome for the single-page app: page config, the top pipeline bar
(mirroring app_fatigue_framework/ui/pipeline_bar.py's node strip instead of
Streamlit's default sidebar), CSS, and the small building blocks every section
uses to tell its part of the story the same way:

    thinking()   the question -> how -> the check -> what we found -> what it means
    play()       a Play / Pause control for a Plotly animation
"""
import streamlit as st

ACCENT = "#B5473A"
NAVY = "#1F4E79"
GREEN = "#2E7D32"
GREY = "#8A9096"
PANEL = "#F4F1EC"

TITLE = "Mathematical Modelling of Fatigue Life of Laser Powder Bed Fusion Components"

PAGES = [
    ("overview", "Overview", "The question and the answer", "real"),
    ("phase1", "Phase 1", "Find the defects", "real"),
    ("phase2", "Phase 2", "Measure them · where cracks start", "real"),
    ("phase3", "Phase 3", "Predict the life", "real"),
    ("phase4", "Phase 4", "Test the prediction", "real"),
    ("conclusions", "Conclusions", "What was shown", "real"),
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
      .lpbf-badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.72rem; font-weight:700; margin-right:6px; letter-spacing:.02em;
      }
      .b-real { background:#E4F3E6; color:#2E7D32; }
      .b-preview { background:#FBE9E7; color:#B5473A; }
      .b-pending { background:#FBF3DE; color:#8A6D3B; }
      .b-no { background:#FBE9E7; color:#B5473A; }
      .b-part { background:#FBF3DE; color:#8A6D3B; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      div[data-testid="stSegmentedControl"] button { font-weight: 600 !important; }

      /* the thinking strip */
      .think { display:grid; grid-template-columns: repeat(5, 1fr); gap:10px;
               margin: 6px 0 18px 0; }
      .think .t { background:#FFFFFF; border:1px solid #E3E0DA; border-radius:10px;
                  padding:10px 12px; font-size:0.86rem; line-height:1.35;
                  opacity:0; animation: rise .5s ease forwards; }
      .think .t b { display:block; font-size:0.70rem; letter-spacing:.06em;
                    text-transform:uppercase; color:#8A9096; margin-bottom:4px; }
      .think .t:nth-child(1){animation-delay:.05s; border-top:3px solid #1F4E79}
      .think .t:nth-child(2){animation-delay:.20s; border-top:3px solid #1F4E79}
      .think .t:nth-child(3){animation-delay:.35s; border-top:3px solid #8A9096}
      .think .t:nth-child(4){animation-delay:.50s; border-top:3px solid #B5473A}
      .think .t:nth-child(5){animation-delay:.65s; border-top:3px solid #2E7D32}
      @keyframes rise { from {opacity:0; transform:translateY(8px)}
                        to {opacity:1; transform:none} }
      @media (max-width: 900px) { .think { grid-template-columns: 1fr; } }

      /* plain-language aside */
      .plain { background:#F4F1EC; border-left:4px solid #1F4E79; border-radius:6px;
               padding:10px 14px; margin:8px 0 14px 0; font-size:0.95rem; }
      .plain b { color:#1F4E79; }
    </style>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str = "real"):
    st.markdown(f'<span class="lpbf-badge b-{kind}">{text}</span>', unsafe_allow_html=True)


def thinking(question: str, how: str, check: str, found: str, meaning: str):
    """The same five-step reading of every phase, so a reader always knows what
    question is being asked, how it was answered, how the answer was checked, what
    came out and why it matters."""
    cells = [("The question", question), ("How", how), ("The check", check),
             ("What we found", found), ("What it means", meaning)]
    html = "".join(f'<div class="t"><b>{h}</b>{t}</div>' for h, t in cells)
    st.markdown(f'<div class="think">{html}</div>', unsafe_allow_html=True)


def plain(text: str):
    """A plain-language aside for a reader who is not a fatigue specialist."""
    st.markdown(f'<div class="plain">{text}</div>', unsafe_allow_html=True)


def play(fig, frame_ms: int = 400, transition_ms: int = 250, slider_prefix: str = ""):
    """Add a Play / Pause control and a frame slider to a Plotly figure whose
    frames are already set. Frames must be named; the slider labels use the names."""
    names = [f.name for f in fig.frames]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.18, xanchor="left",
            yanchor="top", pad=dict(r=10, t=0), showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=transition_ms),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.16, y=-0.12, len=0.84, xanchor="left", yanchor="top",
            currentvalue=dict(prefix=slider_prefix, font=dict(size=12)),
            pad=dict(t=0, b=0),
            steps=[dict(label=n, method="animate",
                        args=[[n], dict(frame=dict(duration=0, redraw=True),
                                        mode="immediate")]) for n in names])],
    )
    return fig


def _labels():
    return [f"{num} · {name}" if num.startswith("Phase") else num
            for _, num, name, _ in PAGES]


def top_nav() -> str:
    """The pipeline bar: Overview, the four phases, Conclusions. Returns the key."""
    if "page" not in st.session_state:
        st.session_state["page"] = "overview"
    labels = _labels()
    keys = [k for k, *_ in PAGES]
    current = keys.index(st.session_state["page"])
    choice = st.segmented_control(
        "Pipeline", labels, default=labels[current], label_visibility="collapsed",
        key="_nav_control")
    if choice is not None:
        st.session_state["page"] = keys[labels.index(choice)]
    st.divider()
    return st.session_state["page"]


def go_to(page: str, label: str):
    """A button that moves the reader to another page of the story."""
    def _go():
        # a callback runs before the next script pass, the one point at which a
        # widget's own state may be changed
        st.session_state["page"] = page
        st.session_state["_nav_control"] = _labels()[[k for k, *_ in PAGES].index(page)]
    st.button(label, key=f"goto_{page}_{label}", on_click=_go, type="primary")

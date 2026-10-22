"""Shared design for the app: one palette, one serif family, a white reading sheet on a
light grey page, and the few building blocks every page uses.

The app is a technical companion to the thesis for a supervisor and examiners. Each
phase page therefore opens with the same four facts about its link in the model chain,
in a plain definition list: the governing equation, the numerical scheme, the
verification against a known answer, and the result on real data.

    spec()      the four-row definition list
    note()      a short aside, for a reader who is not a fatigue specialist
    finding()   a reported negative or unexpected result
    verdict()   a hypothesis outcome in plain words
    show()      every chart; charts with frames autoplay on a loop with a Pause control
"""
import html as _html
import uuid

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# palette -- shared with the thesis figures (navy = life arm, dark red = crack-path arm)
INK = "#1E2A36"
MUTED = "#5B6670"
RULE = "#D3D8DD"
PAGE = "#EEF0F2"
SHEET = "#FFFFFF"
NAVY = "#1F4E79"
RED = "#8B1E1E"
GREEN = "#2E6B3A"
GREY = "#8A949C"
ACCENT = NAVY            # data series
FONT = 'Georgia, "Times New Roman", serif'

TITLE = "Mathematical Modelling of Fatigue Life of Laser Powder Bed Fusion Components"

PAGES = [
    ("overview", "Overview"),
    ("phase1", "1  Defects"),
    ("phase2", "2  Crack initiation"),
    ("phase3", "3  Fatigue life"),
    ("phase4", "4  Validation"),
    ("conclusions", "Conclusions"),
]

pio.templates["thesis"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=13, color=INK),
    paper_bgcolor=SHEET, plot_bgcolor=SHEET,
    colorway=[NAVY, RED, GREY, GREEN, "#C9A227"],
    xaxis=dict(gridcolor="#E6E9EC", linecolor=RULE, zerolinecolor=RULE, ticks="outside",
               tickcolor=RULE),
    yaxis=dict(gridcolor="#E6E9EC", linecolor=RULE, zerolinecolor=RULE, ticks="outside",
               tickcolor=RULE),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=60, r=20, t=30, b=50),
))
pio.templates.default = "thesis"


def setup(title: str = "LPBF Fatigue Life", icon: str = "\U0001F52C"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(f"""
    <style>
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
      /* the reading sheet */
      .block-container {{ max-width: 1060px; padding: 2.2rem 3rem 3rem 3rem;
        margin-top: 1.2rem; margin-bottom: 2rem; background: {SHEET};
        border: 1px solid {RULE}; }}
      h1 {{ font-size: 2.0rem; font-weight: 600; letter-spacing: 0; margin-bottom: .4rem; }}
      h2 {{ font-size: 1.45rem; font-weight: 600; margin-top: 1.6rem; }}
      h3 {{ font-size: 1.15rem; font-weight: 600; }}
      p, li {{ line-height: 1.6; max-width: 78ch; }}
      /* navigation: plain text, the current page underlined */
      div[data-testid="stSegmentedControl"] button {{
        background: transparent !important; border: none !important;
        border-bottom: 2px solid transparent !important; border-radius: 0 !important;
        color: {MUTED} !important; font-weight: 500 !important; padding: 4px 10px !important;
        box-shadow: none !important; }}
      div[data-testid="stSegmentedControl"] button[aria-checked="true"],
      div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {{
        color: {INK} !important; border-bottom-color: {NAVY} !important; }}
      div[data-testid="stSegmentedControl"] button:focus-visible {{
        outline: 2px solid {NAVY} !important; outline-offset: 2px; }}
      /* definition list */
      .spec-label {{ font-weight: 600; color: {INK}; padding-top: .15rem; }}
      .spec-rule {{ border-top: 1px solid {RULE}; margin: .35rem 0 .55rem 0; }}
      /* asides */
      .note {{ border-left: 3px solid {NAVY}; padding: .5rem .9rem; margin: .6rem 0 1rem 0;
               background: #F6F8FA; max-width: 78ch; }}
      .finding {{ border-left: 3px solid {RED}; padding: .5rem .9rem; margin: .6rem 0 1rem 0;
                  background: #FBF6F6; max-width: 78ch; }}
      .verdict-met {{ color: {GREEN}; font-weight: 600; }}
      .verdict-part {{ color: #7A5B12; font-weight: 600; }}
      .verdict-no {{ color: {RED}; font-weight: 600; }}
      .byline {{ color: {MUTED}; font-size: .92rem; margin-bottom: .2rem; }}
      @media (max-width: 800px) {{ .block-container {{ padding: 1.2rem 1rem; }} }}
    </style>
    """, unsafe_allow_html=True)


def spec(equation: str, scheme: str, verification: str, result: str):
    """The four facts about one link in the model chain."""
    for i, (label, body) in enumerate((("Governing equation", equation),
                                       ("Numerical scheme", scheme),
                                       ("Verification", verification),
                                       ("Result", result))):
        if i:
            st.markdown('<div class="spec-rule"></div>', unsafe_allow_html=True)
        a, b = st.columns([1, 4], gap="medium")
        a.markdown(f'<div class="spec-label">{label}</div>', unsafe_allow_html=True)
        with b:
            st.markdown(body)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def finding(text: str):
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


plain = note             # older name, kept so nothing breaks


def verdict(text: str, kind: str = "met"):
    """kind: 'met', 'part' or 'no'."""
    st.markdown(f'<span class="verdict-{kind}">{_html.escape(text)}</span>',
                unsafe_allow_html=True)


def badge(text: str, kind: str = "real"):
    """Retired: status pills read as template decoration. Kept as a no-op so an old
    call cannot break a page."""
    return None


def thinking(*_args, **_kw):
    """Retired in favour of spec()."""
    return None


def play(fig, frame_ms: int = 400, transition_ms: int = 250, slider_prefix: str = ""):
    """Play / Pause and a frame slider for a figure whose frames are set and named."""
    names = [f.name for f in fig.frames]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=-0.16, xanchor="left",
            yanchor="top", pad=dict(r=8, t=0), showactive=False,
            bgcolor=SHEET, bordercolor=RULE, font=dict(color=INK, size=12),
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=transition_ms),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.14, y=-0.11, len=0.86, xanchor="left", yanchor="top",
            currentvalue=dict(prefix=slider_prefix, font=dict(size=12, color=MUTED)),
            pad=dict(t=0, b=0), bgcolor=RULE, activebgcolor=NAVY, bordercolor=RULE,
            tickcolor=RULE, font=dict(color=MUTED, size=10),
            steps=[dict(label=n, method="animate",
                        args=[[n], dict(frame=dict(duration=0, redraw=True),
                                        mode="immediate")]) for n in names])],
    )
    fig._lpbf_timing = (frame_ms, transition_ms)
    return fig


def show(fig, height: int | None = None):
    """Every chart goes through here. A chart with frames is embedded directly so it can
    start by itself, loop, and stop when Pause is pressed; it stays still for a reader
    whose system asks for reduced motion. Other charts are ordinary Plotly charts."""
    if not fig.frames:
        st.plotly_chart(fig, theme=None, width="stretch")
        return
    frame_ms, trans_ms = getattr(fig, "_lpbf_timing", (500, 200))
    h = height or int(fig.layout.height or 450)
    div = f"anim_{uuid.uuid4().hex[:8]}"
    # plotly.js from cdnjs, falling back to jsDelivr: cdn.plot.ly did not serve the
    # version this plotly bundles, which left every animated chart blank
    from plotly.offline import get_plotlyjs_version
    ver = get_plotlyjs_version()
    loader = (f'<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{ver}/plotly.min.js">'
              f'</script><script>window.Plotly||document.write(\'<script src="https://cdn.'
              f'jsdelivr.net/npm/plotly.js-dist-min@{ver}/plotly.min.js"><\\/script>\')</script>')
    body = loader + fig.to_html(full_html=False, include_plotlyjs=False, div_id=div,
                                auto_play=False,
                                config={"displaylogo": False, "responsive": True},
                                default_width="100%", default_height=f"{h}px")
    script = f"""
    <script>
    (function() {{
      var gd = document.getElementById('{div}');
      var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      var opts = {{frame: {{duration: {frame_ms}, redraw: true}},
                  transition: {{duration: {trans_ms}}}, mode: 'immediate'}};
      var paused = false;
      function run(fromStart) {{
        if (still || paused) return;
        Plotly.animate(gd, null, Object.assign({{fromcurrent: !fromStart}}, opts));
      }}
      function ready() {{
        gd.on('plotly_animated', function() {{ setTimeout(function() {{ run(true); }}, 1400); }});
        gd.on('plotly_buttonclicked', function(e) {{
          paused = (e.button && e.button.label === 'Pause');
        }});
        run(true);
      }}
      if (gd && gd.on) ready(); else window.addEventListener('load', ready);
    }})();
    </script>"""
    components.html(f'<div style="font-family:{FONT}">{body}</div>{script}',
                    height=h + 20)


def top_nav() -> str:
    if "page" not in st.session_state:
        st.session_state["page"] = "overview"
    labels = [lab for _, lab in PAGES]
    keys = [k for k, _ in PAGES]
    current = keys.index(st.session_state["page"])
    choice = st.segmented_control("Pages", labels, default=labels[current],
                                  label_visibility="collapsed", key="_nav_control")
    if choice is not None:
        st.session_state["page"] = keys[labels.index(choice)]
    st.markdown(f'<div class="spec-rule" style="margin:.2rem 0 1.4rem 0"></div>',
                unsafe_allow_html=True)
    return st.session_state["page"]


def go_to(page: str, label: str):
    """A plain text link to another page."""
    labels = [lab for _, lab in PAGES]
    keys = [k for k, _ in PAGES]

    def _go():
        st.session_state["page"] = page
        st.session_state["_nav_control"] = labels[keys.index(page)]
    st.button(label, key=f"goto_{page}_{label}", on_click=_go, type="tertiary")

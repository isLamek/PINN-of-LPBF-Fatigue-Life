"""Shared Streamlit chrome: page config, a little CSS, small helpers."""
import streamlit as st

ACCENT = "#B5473A"
INK = "#1F1B16"
PAPER = "#FFFFFF"
PANEL = "#F4F1EC"


def setup(title: str, icon: str = "\U0001F527"):
    st.set_page_config(page_title=f"{title} - LPBF PIML", page_icon=icon, layout="wide")
    st.markdown(f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1150px; }}
      h1, h2, h3 {{ letter-spacing: -0.01em; }}
      .status-real {{ color: #2E7D32; font-weight: 600; }}
      .status-preview {{ color: #B5473A; font-weight: 600; }}
      .status-pending {{ color: #8A6D3B; font-weight: 600; }}
      .lpbf-badge {{
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.75rem; font-weight:600; margin-right:6px;
      }}
      .b-real {{ background:#E4F3E6; color:#2E7D32; }}
      .b-preview {{ background:#FBE9E7; color:#B5473A; }}
      .b-pending {{ background:#FBF3DE; color:#8A6D3B; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    </style>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str = "real"):
    st.markdown(f'<span class="lpbf-badge b-{kind}">{text}</span>', unsafe_allow_html=True)

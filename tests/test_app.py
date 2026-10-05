"""Render every phase headlessly and fail on any uncaught exception."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.mark.parametrize("phase", ["phase1", "phase2", "phase3", "phase4"])
def test_phase_renders(phase):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["phase"] = phase
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title


def test_phase4_sliders_drive_the_models():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["phase"] = "phase4"
    at.run()
    before = [m.value for m in at.metric][-4:]
    at.slider[0].set_value(400).run()
    assert not at.exception
    assert [m.value for m in at.metric][-4:] != before

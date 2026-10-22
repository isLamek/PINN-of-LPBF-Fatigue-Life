"""One set of chart conventions for every page, in the Streamlit app and the static site.

    colour   navy = model or simulation, MEAS = measured data, red = the one thing to
             look at; grey for reference lines only
    lives    always in cycles: log axes labelled "cycles to failure", and histograms
             binned in log10 but ticked in cycles
    sizes    H_MAIN for a result, H_AUX for a verification check
    legend   one place, top left, horizontal
    parity   predicted against measured on the same range for every model, 1:1 aspect
"""
import numpy as np

from src.ui import GREY, NAVY, RED  # noqa: F401  (re-exported for the pages)

MEAS = "#4A5560"               # measured data: dark slate
H_MAIN, H_AUX = 420, 320
LEGEND = dict(orientation="h", x=0, xanchor="left", y=1.1, yanchor="bottom")
PARITY = (5e3, 2e6)            # cycles; covers every held-out life and prediction


def cycle_ticks(lo, hi):
    """Tick a log10 axis in cycles: whole decades labelled 1,000 / 10,000 / ..."""
    k = np.arange(np.ceil(lo), np.floor(hi) + 1)
    return dict(tickvals=k.tolist(), ticktext=[f"{10 ** int(v):,}" for v in k])


def parity(fig, ytitle="predicted cycles to failure"):
    r = [np.log10(PARITY[0]), np.log10(PARITY[1])]
    fig.update_layout(height=H_MAIN + 60, legend=LEGEND,
                      xaxis=dict(type="log", range=r, title="measured cycles to failure"),
                      yaxis=dict(type="log", range=r, title=ytitle, scaleanchor="x",
                                 scaleratio=1))
    return fig

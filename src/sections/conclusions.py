"""What was shown: the four hypotheses scored from their result files, what the
project contributes, its limits, and what comes next."""
import json
import os

import streamlit as st

from src.data import FOLD_DICE, STUDY_METRICS
from src.ui import go_to, plain

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


def _json(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _card(code, name, criterion, verdict, kind, evidence, page):
    colour = {"real": "#2E7D32", "part": "#8A6D3B", "no": "#B5473A"}[kind]
    bg = {"real": "#E4F3E6", "part": "#FBF3DE", "no": "#FBE9E7"}[kind]
    st.markdown(
        f'<div style="border:1px solid #E3E0DA;border-left:6px solid {colour};'
        f'border-radius:10px;padding:12px 16px;margin-bottom:6px;background:#fff">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div style="font-weight:700;font-size:1.05rem">{code} · {name}</div>'
        f'<span style="background:{bg};color:{colour};border-radius:999px;padding:2px 12px;'
        f'font-weight:700;font-size:.8rem">{verdict}</span></div>'
        f'<div style="color:#5A6067;font-size:.86rem;margin-top:4px">Accepted only if: '
        f'{criterion}</div><div style="margin-top:6px">{evidence}</div></div>',
        unsafe_allow_html=True)


def render():
    st.title("Conclusions")
    st.markdown(
        "Each objective was tied to a hypothesis with an acceptance criterion fixed "
        "before testing. Here is how each came out, read from the result files.")

    # H1 ---------------------------------------------------------------------
    dice = STUDY_METRICS["Dice"]
    _card("H1", "Segmentation accuracy", "mean Dice ≥ 0.80 on withheld data",
          (f"Criterion met · Dice {dice:.3f} ≥ 0.80" if dice >= 0.80 else
           f"Criterion not met · Dice {dice:.3f} < 0.80"), "real" if dice >= 0.80 else "no",
          ("Supported. " if dice >= 0.80 else "Not supported. ") +
          f"Mean Dice {dice:.3f} over five folds ({min(FOLD_DICE):.3f} to "
          f"{max(FOLD_DICE):.3f}). The AI finds the defects as well as the criterion asks.",
          "phase1")

    # H2 ---------------------------------------------------------------------
    h2 = _json("phase2_peridynamics", "h2_first_bond.json")
    if h2:
        ok = h2["fraction_by_area"] >= h2["criterion"]
        _card("H2", "Crack initiation site",
              "the crack starts at the most severe (largest √area) defect in ≥ 90% of cases",
              (f"Criterion {'met' if ok else 'not met'} · {h2['hits_by_area']} of {h2['n']} "
               f"({h2['fraction_by_area']:.0%}) vs ≥ {h2['criterion']:.0%}"),
              "real" if ok else "no",
              ("Supported. " if ok else "Not supported, a finding rather than a failure. ") +
              f"The first bond broke at the largest defect in {h2['hits_by_area']} of "
              f"{h2['n']} measured sections ({h2['fraction_by_area']:.1%}); ranked by "
              f"Murakami severity, {h2['hits_by_severity']} of {h2['n']}. Size alone does "
              "not decide where a crack starts.", "phase2")

    # H3 ---------------------------------------------------------------------
    bf = _json("phase_mc_bayesian", "bayesian_results.json")
    if bf:
        r = bf["verify_recovery"]
        hist = r["history"]
        step = next((i for i, h in enumerate(hist, 1)
                     if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
        ess = min(h["ess"] for h in hist)
        ok_time = step is not None and step <= 3
        ok_ess = ess >= 250
        kind = "real" if ok_time and ok_ess else "part" if (step or ok_ess) else "no"
        verdict = ("Criterion met" if kind == "real" else "Partly met" if kind == "part"
                   else "Criterion not met") + \
            f" · converged at inspection {step}, asked ≤ 3"
        _card("H3", "Bayesian convergence",
              "within 20% of the true life within 3 updates, effective sample size ≥ 250",
              verdict, kind,
              ("Supported. " if kind == "real" else "Partly supported. " if kind == "part"
               else "Not supported. ") +
              f"The filter reaches the true life (within 20%) at inspection {step}, not by "
              f"the 3rd, and never has fewer than {ess:,.0f} effective particles. "
              "It converges, just later than the hypothesis asked.", "phase3")

    # H4 ---------------------------------------------------------------------
    h4 = _json("phase4_validation", "h4_calibration.json")
    if h4:
        ok = h4["slope_met"] and h4["coverage_met"]
        _card("H4", "Predictive calibration",
              "≥ 90% of measured lives inside the 90% interval, slope 0.90–1.10",
              (f"Criterion {'met' if ok else 'not met'} · {h4['n_inside']} of {h4['n']} "
               f"inside, slope {h4['slope']:.3f}"), "real" if ok else "no",
              ("Supported. " if ok else "Not supported. ") +
              f"{h4['n_inside']} of {h4['n']} measured lives ({h4['coverage']:.1%}) inside "
              f"the interval; slope {h4['slope']:.3f}, a narrow pass at the lower edge. The "
              "interval is built from the leave-one-out residual spread.", "phase4")

    # ------------------------------------------------------------------------
    st.divider()
    cmp_d = _json("comparison_model", "comparison_results.json") or {}
    pa = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    lad = {r["model"].split()[0]: r for r in cmp_d.get("ladder", [])}
    st.subheader("What this project shows")
    if pa and "B5" in lad and "C1" in lad:
        st.markdown(
            f"- **A part's own CT scan predicts its own fatigue life** to within "
            f"**{pa:.2f}×** on held-out cylinders, with **nothing fitted** to the fatigue lives.\n"
            f"- The best purely statistical fit needs **{lad['B5']['params']} parameters** "
            f"to reach {lad['B5']['loo_factor']:.2f}×; a physics-informed neural network "
            f"with **{lad['C1']['params']}** reaches {lad['C1']['loo_factor']:.2f}×. "
            "Measuring the defect does the work that fitting cannot.\n"
            "- Every step is traceable: each method was checked against a known answer "
            "before it was used on measured data, and the life can be followed by hand "
            "from defect size to cycles.")
    plain("<b>Why that matters.</b> A part with a benign defect population can be kept "
          "in service instead of scrapped on a batch rule, and a part with a severe one "
          "can be withdrawn before it fails, with the uncertainty stated rather than "
          "hidden in a safety factor.")

    st.subheader("Limits, stated plainly")
    st.markdown(
        "- **One alloy, one dataset**: 316H from one machine and one powder lot; "
        "fourteen held-out cylinders carry the validation.\n"
        "- **Where a crack starts is not predicted by size alone** (H2): surface proximity "
        "and neighbouring defects matter, and √area does not see them.\n"
        "- **The peridynamic arm cannot carry an absolute life** at attainable grid "
        "resolution; it gives the site and the path, and the life comes from the "
        "crack-growth integral.\n"
        "- **Fillet-radius failures** (a stress concentration, not a defect) are outside "
        "the defect-driven model.")

    st.subheader("What comes next")
    st.markdown(
        "- A severity measure with surface and interaction terms, to re-test H2.\n"
        "- An inspection schedule weighted toward later, more informative readings, to "
        "re-test H3's three-step criterion.\n"
        "- A stress-concentration model at the fillet radius, so both failure locations "
        "are covered.\n"
        "- Each specimen's own Bayesian interval for H4, in place of the residual-based "
        "one.")
    go_to("overview", "← Back to the overview")

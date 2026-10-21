"""What was shown: the four hypotheses scored from their result files, what the
project contributes, its limits, and what comes next."""
import json
import os

import streamlit as st

from src.data import FOLD_DICE, STUDY_METRICS
from src.ui import go_to, note, verdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


def _json(*parts):
    p = os.path.join(DATA, *parts)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _hypothesis(code, name, criterion, outcome, kind, evidence):
    st.markdown(f"### {code}. {name}")
    st.markdown(f"Accepted only if {criterion}.")
    verdict(outcome, kind)
    st.markdown(evidence)


def render():
    st.title("Conclusions")
    st.markdown(
        "Each objective was tied to a hypothesis with an acceptance criterion fixed "
        "before testing. Each outcome below is read from the result files.")

    dice = STUDY_METRICS["Dice"]
    ok = dice >= 0.80
    _hypothesis("H1", "Segmentation accuracy", "the mean Dice on withheld data is at least 0.80",
                f"Criterion {'met' if ok else 'not met'}: Dice {dice:.3f}.",
                "met" if ok else "no",
                f"Mean over five folds, {min(FOLD_DICE):.3f} to {max(FOLD_DICE):.3f} per fold. "
                "The network finds the defects well enough for their size to be trusted.")

    h2 = _json("phase2_peridynamics", "h2_first_bond.json")
    if h2:
        ok = h2["fraction_by_area"] >= h2["criterion"]
        _hypothesis("H2", "Crack initiation site",
                    f"the crack starts at the largest defect in at least "
                    f"{h2['criterion']:.0%} of sections",
                    f"Criterion {'met' if ok else 'not met'}: {h2['hits_by_area']} of "
                    f"{h2['n']} ({h2['fraction_by_area']:.1%}).", "met" if ok else "no",
                    f"Ranked by Murakami severity instead of size, {h2['hits_by_severity']} of "
                    f"{h2['n']}. Size alone does not decide where a crack starts: closeness to "
                    "the surface and neighbouring defects matter too.")

    bf = _json("phase_mc_bayesian", "bayesian_results.json")
    if bf:
        r = bf["verify_recovery"]
        step = next((i for i, h in enumerate(r["history"], 1)
                     if abs(h["Nf_mean"] - r["Nf_true"]) / r["Nf_true"] <= 0.20), None)
        ess = min(h["ess"] for h in r["history"])
        ok_t, ok_e = step is not None and step <= 3, ess >= 250
        kind = "met" if ok_t and ok_e else "part" if (step or ok_e) else "no"
        word = {"met": "Criterion met", "part": "Partly met", "no": "Criterion not met"}[kind]
        _hypothesis("H3", "Bayesian convergence",
                    "the posterior is within 20% of the true life by the third inspection "
                    "with an effective sample size of at least 250",
                    f"{word}: within 20% at inspection {step}; effective sample size at least "
                    f"{ess:,.0f}.", kind,
                    "With the true growth rate placed in the tails of the prior instead, the "
                    "filter still converges by the third inspection, but its effective "
                    "sample size can dip below 250 before resampling restores it. The "
                    "speed condition holds throughout; the sample-size condition fails "
                    "where the prior is badly wrong.")

    h4 = _json("phase4_validation", "h4_calibration.json")
    if h4:
        ok = h4["slope_met"] and h4["coverage_met"]
        _hypothesis("H4", "Predictive calibration",
                    "at least 90% of measured lives fall inside the 90% interval and the "
                    "calibration slope lies between 0.90 and 1.10",
                    f"Criterion {'met' if ok else 'not met'}: {h4['n_inside']} of {h4['n']} "
                    f"inside ({h4['coverage']:.1%}); slope {h4['slope']:.3f}.",
                    "met" if ok else "no",
                    f"Tested on {h4['model']}, with the interval built from its leave-one-out "
                    "residual spread. The slope passes at the lower edge.")

    cmp_d = _json("comparison_model", "comparison_results.json") or {}
    pa = cmp_d.get("phase4", {}).get("log10_N_pathA", {}).get("factor")
    lad = {r["model"].split()[0]: r for r in cmp_d.get("ladder", [])}
    st.markdown("## What this work shows")
    if pa and "B5" in lad and "C1" in lad:
        st.markdown(
            f"- A part's own CT scan predicts its own fatigue life to within **{pa:.2f}×** "
            "on held-out cylinders, with nothing fitted to the fatigue lives.\n"
            f"- The best statistical fit needs {lad['B5']['params']} parameters to reach "
            f"{lad['B5']['loo_factor']:.2f}×; a physics-informed neural network with "
            f"{lad['C1']['params']} reaches {lad['C1']['loo_factor']:.2f}×. Measuring the "
            "defect does the work that fitting cannot.\n"
            "- Propagated through the same model, the scatter in defect size and growth "
            "rate reproduces the measured life distribution of 46 further cylinders not "
            "used to build it.\n"
            "- Each numerical method was checked against a known answer before it was used "
            "on measured data, and a life can be followed by hand from defect size to "
            "cycles.")
    note("A part with a benign defect population can be kept in service instead of "
         "scrapped on a batch rule, and a part with a severe one withdrawn before it fails, "
         "with the uncertainty stated rather than hidden in a safety factor.")

    mc = _json("phase_mc_bayesian", "monte_carlo_results.json") or {}
    v300 = mc.get("validation_independent", {}).get("300")
    runout_line = (f"\n- With the threshold removed for short cracks, the model predicts no "
                   f"run-outs, while {v300['n_runout']} of the {v300['n_tested']} independent "
                   "cylinders at 300 MPa ran out." if v300 and v300["n_runout"] else "")
    st.markdown("## Limits")
    st.markdown(
        "- One alloy and one dataset: 316H from one machine and one powder lot, with "
        "fourteen held-out cylinders carrying the validation.\n"
        "- Where a crack starts is not predicted by size alone (H2).\n"
        "- The peridynamic model cannot carry an absolute life at attainable grid "
        "resolution; it gives the site and the path, and the life comes from the "
        "crack-growth integral.\n"
        "- Fillet-radius failures start at a stress concentration, not a defect, and lie "
        "outside the defect-driven model." + runout_line)

    st.markdown("## What comes next")
    st.markdown(
        "- A severity measure with surface and interaction terms, to re-test H2.\n"
        "- A threshold that grows from the short-crack to the long-crack value as the "
        "crack lengthens, so the population model reproduces the run-outs at 300 MPa.\n"
        "- A stress-concentration model at the fillet radius, so both failure locations "
        "are covered.\n"
        "- Each specimen's own Bayesian interval for H4, in place of the residual-based "
        "one.")
    go_to("overview", "Back to the overview")

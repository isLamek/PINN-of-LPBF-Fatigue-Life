"""
Loads and merges the ORNL Snow 316H fatigue record with the CT-derived defect
measurements. Adapted from Research files/comparison_model/piml_model.py `load()`,
line for line, only the file paths changed to this repo's flat data/ layout.
The merge logic, the hand-curated gauge-cropped defect sizes (`gauge_sa`,
`gauge_nd`), and the constants are copied verbatim -- they are not recomputed
here, so this stays consistent with the verified pipeline in Research files/.
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

E_GPA, UTS_MPA, SY_MPA = 195.0, 642.0, 484.0     # Merot et al. 2022, EFM 276 108883
C_PARIS, M_PARIS       = 6.25e-10, 3.94          # mm/cycle, literature (316L)
DKTH_MPA_SQRTM         = 5.0
Y_SURFACE, Y_INTERNAL  = 0.65, 0.50
RUNOUT_CYCLES          = 10_000_000

# largest sqrt(area) inside the 3-D machined gauge, from the Phase 1 defect maps
GAUGE_SA = {"P059": 692, "P056": 416, "P016": 431, "P007": 345, "P032": 349,
            "P045": 267, "P023": 200, "P051": 749, "P017": 1017, "P008": 385,
            "P046": 457, "P027": 367, "P037": 325, "P057": 283}
GAUGE_ND = {"P059": 286, "P056": 84, "P016": 60, "P007": 55, "P032": 19,
            "P045": 17, "P023": 16, "P051": 241, "P017": 158, "P008": 59,
            "P046": 29, "P027": 22, "P037": 22, "P057": 16}

# Phase 1 study metrics, copied verbatim from app_fatigue_framework/config.py
# (5-fold cross-validation result of the segmentation study, not per-specimen).
STUDY_METRICS = {"Dice": 0.805, "Precision": 0.862, "Recall": 0.841, "IoU": 0.709}
FOLD_DICE = [0.8043, 0.7936, 0.7956, 0.8254, 0.8078]
HELDOUT_SIZE_R2 = 0.986


def load():
    fat = {("P%03d" % int(d["specimen"][1:])): d
           for d in json.load(open(os.path.join(DATA, "comparison_model", "fatigue_record.json")))}
    twin = {r["id"]: r for r in csv.DictReader(open(os.path.join(DATA, "twin_summary.csv")))}
    spl = {r["part_id"]: r for r in csv.DictReader(
        open(os.path.join(DATA, "00_docs", "Appendix_A_train_test_split.csv")))}
    sec = {r["specimen"].replace("SN_", ""): r for r in csv.DictReader(
        open(os.path.join(DATA, "phase2_peridynamics", "specimen_summary.csv")))}

    shape = {}
    cat = os.path.join(DATA, "phase2_extraction", "phase2_defect_catalogue.csv")
    if os.path.exists(cat):
        acc = {}
        for d in csv.DictReader(open(cat)):
            acc.setdefault(d["cylinder"], []).append(float(d["flatness"]))
        shape = {k: float(np.median(v)) for k, v in acc.items()}

    rows = []
    for pid, f in sorted(fat.items()):
        loc = f["fracture_location"].replace(",", ", ").replace("  ", " ").strip()
        rows.append(dict(
            specimen=pid,
            smax=f["smax"], smin=f["smin"], salt=f["salt"], R=f["R"],
            ds=f["smax"] - f["smin"], Nf=f["Nf"],
            runout=f["Nf"] >= RUNOUT_CYCLES,
            fracture_location=loc,
            gauge_failure=loc.lower().startswith("gage"),
            porosity_pct=float(twin[pid]["porosity_pct"]),
            process_group=spl[pid]["process_group"] if pid in spl else "",
            split=spl[pid]["split"] if pid in spl else "",
            sqrt_area_gauge_um=float(GAUGE_SA[pid]) if pid in GAUGE_SA else np.nan,
            n_defects_gauge=float(GAUGE_ND[pid]) if pid in GAUGE_ND else np.nan,
            flatness_median=shape.get(pid, np.nan),
            sqrt_area_section_um=float(sec[pid]["largest_sqrt_area_um"]) if pid in sec else np.nan,
            pd_delta_a_um=float(sec[pid]["delta_a_um"]) if pid in sec else np.nan,
            pd_stiffness_final=float(sec[pid]["stiffness_final"]) if pid in sec else np.nan,
        ))
    return rows


def splits(rows):
    """The same three specimen subsets piml_model.py works with."""
    fail = [r for r in rows if not r["runout"]]
    gauge = [r for r in fail if r["gauge_failure"]]
    meas = [r for r in fail if np.isfinite(r["sqrt_area_gauge_um"])]
    return dict(all=rows, fail=fail, gauge=gauge, meas=meas)

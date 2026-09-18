import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ct_viewer as cv
from src.ui import badge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "phase2_extraction")

SPEC_LABELS = {"SN_P017": "P017 — governing defect 1017 µm (largest of the 14)",
              "SN_P059": "P059 — most defects in gauge (286)"}


def _region_grid(rgb, regions, max_labels=6):
    """Static image with size labels on the largest regions, matching the
    desktop app's annotation -- done here with PIL since Streamlit has no
    canvas-overlay primitive as cheap as a direct draw."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for r in regions[:max_labels]:
        cx, cy = r["cx"], r["cy"]
        rad = max((r["area_px"] / np.pi) ** 0.5 * 1.8, 8.0)
        colour = tuple(r["colour"])
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=colour, width=1)
        draw.text((cx + rad + 3, cy - rad - 3), f"{r['d_um']:.0f} µm", fill=colour)
    return im


def render():
    st.title("Phase 2 · Defect extraction & crack growth")
    badge("REAL RESULTS", "real")
    st.caption(
        "Two steps on real data: (1) isolate and size every defect from the "
        "Phase 1 predicted masks, (2) grow a crack from the governing defect "
        "with peridynamics. Phase 3 turns that crack path into an actual cycle "
        "count — this phase is about the geometry and the mechanics."
    )

    st.markdown("#### How a defect becomes a number")
    st.markdown(
        "Every defect is isolated by 3-D connected-components labelling of the "
        "predicted mask, then sized the way fracture mechanics actually uses "
        "defect size — not by volume, but by the **square root of its projected "
        "area** on the plane perpendicular to the load axis (the Murakami "
        "parameter):"
    )
    st.latex(r"\sqrt{\text{area}} \;=\; \sqrt{A_{\text{projected}}}")
    st.caption(
        "This one number is what both the classical Murakami/Paris model "
        "(Phase 4) and the peridynamic crack-growth simulation (below) treat as "
        "the initial crack size $a_0$. Flatness — the build-direction extent "
        "over the larger in-plane extent — separates lack-of-fusion pancakes "
        "(≈ 0.1) from spherical gas pores (≈ 1); it distinguishes shape without "
        "assuming which is worse until Phase 4 tests that assumption."
    )

    cat = pd.read_csv(os.path.join(DATA, "phase2_defect_catalogue.csv"))
    cyl = pd.read_csv(os.path.join(DATA, "phase2_cylinder_summary.csv"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Cylinders catalogued", cyl["cylinder"].nunique())
    c2.metric("Defects catalogued", f"{len(cat):,}")
    c3.metric("Largest √area observed", f"{cat['sqrt_area_um'].max():,.0f} µm")

    st.divider()
    st.subheader("Raw CT · segmentation overlay · extracted regions")
    st.caption(
        "The same layer shown three ways: the raw CT slice, the CT with the "
        "network's predicted defect mask overlaid in red, and what the pipeline "
        "actually extracted — labelled 2-D connected components, one colour per "
        "region, sized in µm. Real data, not a mockup: two specimens' slices are "
        "shipped with this app for the z-range where each has predicted defects."
    )

    specs = cv.available_specimens()
    if not specs:
        st.warning("No CT slice data found in data/ct_slices/.", icon="⚠️")
    else:
        spec = st.selectbox("Specimen", specs, format_func=lambda s: SPEC_LABELS.get(s, s))
        zs = cv.list_slices(spec)
        z_idx = st.slider("z slice", 0, len(zs) - 1, len(zs) // 2,
                          format=f"slice %d of {len(zs)}")
        z = zs[z_idx]
        z_mm = z * cv.LAYER_UM / 1000.0
        st.caption(f"z{z:04d} · build height {z_mm:.2f} mm")

        ct = cv.load_ct_slice(spec, z)
        overlay = cv.composite_slice(spec, z)
        abst = cv.slice_abstraction(spec, z)

        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("**Raw CT**")
            if ct is not None:
                st.image((ct * 255).astype("uint8"), width='stretch')
        with v2:
            st.markdown("**+ predicted mask**")
            if overlay is not None:
                st.image(overlay, width='stretch')
        with v3:
            st.markdown("**Extracted regions**")
            if abst is not None:
                n = abst["n"]
                img = _region_grid(abst["rgb"], abst["regions"]) if abst["regions"] else abst["rgb"]
                st.image(img, width='stretch')
                st.caption(f"{n} region(s) extracted on this slice"
                          if n else "no defect regions on this slice")

    st.divider()
    st.subheader("Cylinder-level summary")
    st.dataframe(
        cyl.rename(columns={"largest_sqrt_area_um": "largest √area (µm)",
                            "aop24_porosity_pct": "porosity % (AOP24)"}),
        width='stretch', hide_index=True,
    )

    st.subheader("Segmentation vs. AOP24 cross-check")
    fig = px.scatter(
        cyl, x="aop24_flaw_vox", y="pred_flaw_vox_scaled", color="process_group",
        hover_data=["cylinder"],
        labels={"aop24_flaw_vox": "AOP24 reference flaw voxels",
               "pred_flaw_vox_scaled": "network-predicted flaw voxels (scaled)"},
    )
    lims = [0, max(cyl["aop24_flaw_vox"].max(), cyl["pred_flaw_vox_scaled"].max()) * 1.05]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="#8A9096"),
                             name="1:1", showlegend=True))
    r = np.corrcoef(cyl["aop24_flaw_vox"], cyl["pred_flaw_vox_scaled"])[0, 1]
    fig.update_layout(height=420, title=f"Pearson r = {r:.3f}")
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Defect catalogue — 1342 defects")
    process = st.selectbox("Filter by cylinder", ["All"] + sorted(cat["cylinder"].unique()), key="cat_filter")
    view = cat if process == "All" else cat[cat["cylinder"] == process]
    fig2 = px.scatter(
        view, x="sqrt_area_um", y="flatness", color="cylinder" if process == "All" else None,
        hover_data=["defect_id", "volume_mm3", "equiv_diam_um"],
        labels={"sqrt_area_um": "√area (µm), Murakami parameter",
               "flatness": "flatness (build-axis / in-plane extent)"},
        title="Defect size vs. shape — flatness ≈ 0.1 is a lack-of-fusion pancake, ≈ 1 a spherical gas pore",
    )
    fig2.update_layout(height=460)
    st.plotly_chart(fig2, width='stretch')
    st.dataframe(view.sort_values("sqrt_area_um", ascending=False), width='stretch',
                hide_index=True, height=260)

    st.divider()
    st.subheader("Growing a crack from the governing defect")
    st.markdown(
        "A 2-D section is cut through the largest defect and modelled as a "
        "**bond-based peridynamic** mesh — material points connected by bonds "
        "that carry force proportional to how much they stretch, instead of "
        "solving a differential equation on a continuum. The advantage for a "
        "crack problem: a bond simply **breaks** when it stretches too far, so "
        "a crack is not a special boundary condition, it is bonds going to zero."
    )
    st.markdown("Each bond carries a remaining life $\\lambda$, starting at 1, that falls under cyclic load:")
    st.latex(r"\frac{d\lambda_i}{dN} = -A \cdot \varepsilon_i^{\,\beta} \quad (\varepsilon_i > \varepsilon_{th}), "
            r"\qquad \text{bond } i \text{ breaks when } \lambda_i \le 0")
    st.caption(
        "$\\varepsilon_i$ is bond $i$'s cyclic strain range, $A$ a rate "
        "coefficient, $\\beta$ a rate exponent. Near a crack tip the bond strain "
        "scales with the stress intensity factor ($\\varepsilon \\sim K$), so the "
        "crack advances at $da/dN \\sim (\\Delta K)^{\\beta}$ — the Paris law, with "
        "exponent $\\beta$. $\\beta$ is therefore **set** to the measured Paris "
        "exponent (3.94) and is not fitted; only the rate coefficient $A$ is "
        "calibrated — see Phase 3."
    )

    st.markdown(
        "**Real measured defect field, P017, 300 MPa at R = 0.1.** Cyan = "
        "pre-existing lack-of-fusion defects (real size/shape from the "
        "catalogue above; positions are a random realisation since the "
        "catalogue records size and shape, not coordinates). Orange = fatigue "
        "damage accumulated since cycle zero."
    )
    gif = os.path.join(ROOT, "assets", "phase2_crack_growth.gif")
    if os.path.exists(gif):
        st.image(gif, width='stretch')
    st.warning(
        "This animation's cycle count predates the Phase 3 damage-coefficient "
        "calibration — it shows the crack **path**, which does not depend on A, "
        "not a life. Phase 3 has the calibrated numbers.",
        icon="⚠️",
    )

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
    st.title("Phase 2 · Defect extraction")
    badge("REAL RESULTS", "real")
    st.caption(
        "3-D connected-components labelling of the Phase 1 predicted flaw masks, "
        "cross-checked against the AOP24 pre-extracted cluster reference derived "
        "from the same Snow dataset."
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

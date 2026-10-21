import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ct_viewer as cv
from src.ui import badge, plain, play, thinking

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
    st.title("Phase 2 · Measure the defects, and find where cracks start")
    badge("REAL RESULTS", "real")
    h2 = _h2()
    thinking(
        "How big is each defect, and does the crack really start at the biggest one?",
        "Every defect is isolated in 3-D and sized as √area. A peridynamic model of the "
        "measured section shows which defect cracks first and where the crack goes.",
        "The peridynamic model is checked against classical elasticity before use: its "
        "stiffness error falls from 13.8% to 0.29% as the horizon refines.",
        (f"The first bond breaks at the largest defect in {h2['hits_by_area']} of "
         f"{h2['n']} sections ({h2['fraction_by_area']:.0%}), short of the 90% H2 "
         "required." if h2 else "H2 result file not found."),
        "The defect sizes feed the life prediction (Phase 3). Where a crack starts "
        "depends on more than size: surface proximity and neighbours matter.")
    plain("<b>Two arms, two questions.</b> The <b>size</b> of the governing defect goes "
          "to Phase 3 and becomes the predicted life. The <b>peridynamic crack model</b> "
          "answers a different question, where the crack starts and which way it runs. "
          "Its cycle count is not used as a life.")

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
    st.latex(r"\frac{d\lambda_i}{dN} = -A \cdot (\varepsilon_i - \varepsilon_{th})^{\,\beta} "
            r"\quad (\varepsilon_i > \varepsilon_{th}), "
            r"\qquad \text{bond } i \text{ breaks when } \lambda_i \le 0")
    st.caption(
        "$\\varepsilon_i$ is bond $i$'s cyclic strain range, $A$ a rate "
        "coefficient, $\\beta$ a rate exponent set to the measured Paris exponent "
        "(3.94), $\\varepsilon_{th}$ the strain form of the fatigue threshold."
    )
    st.info(
        "**What this phase delivers — and what it does not.** It delivers the crack "
        "**initiation site** (which of the several defects in a section governs) and "
        "the **path** the crack then follows. That is what hypothesis H2 is tested on, "
        "and the test is robust to the calibration: the cycle step goes as $1/A$ and the "
        "damage increment cancels it, so the order in which bonds break is *independent "
        "of the calibration constant entirely*.\n\n"
        "It does **not** deliver fatigue life. Verification established four independent "
        "defects in the chain that would have to hold for it to: the coefficient was "
        "calibrated on soda-lime glass (E = 72 GPa) and applied to 316H (E = 195 GPa), "
        "a 50.7× error measured directly; it was transferred across a 12.5× mesh change "
        "and an 8–65× crack-size change in a model whose effective Paris exponent under "
        "geometric self-similarity is provably **exactly 2**, whatever $\\beta$ is set "
        "to; the damage law carried no threshold while its target Paris coefficient "
        "does; and a cycle-accounting defect (since fixed) made the recorded counts "
        "artefacts. Redoing the calibration correctly moved the prediction *further* "
        "from the fatigue record — which is what establishes the limit as belonging to "
        "the method at feasible mesh resolution, not to the tuning. "
        "**Fatigue life comes from Phase 3**, leave-one-out validated at 1.36×.",
        icon="ℹ️",
    )

    st.markdown(
        "**The measured defect section of P017, 450 MPa, R = 0.1.** Teal = the "
        "lack-of-fusion defects segmented from this cylinder's own CT, at their real "
        "sizes and real positions, modelled as voids. Orange = fatigue damage "
        "accumulated since cycle zero. Watch the crack leave both tips of the large "
        "central defect and link towards its neighbours.")
    mp4 = os.path.join(ROOT, "assets", "phase2_crack_growth.mp4")
    gif = os.path.join(ROOT, "assets", "phase2_crack_growth.gif")
    if os.path.exists(mp4):
        st.video(mp4, autoplay=True, loop=True, muted=True)
    elif os.path.exists(gif):
        st.image(gif, width='stretch')
    st.warning(
        "The cycle count on this animation paces the playback; it is **not** a fatigue "
        "life and should not be read as one. What the animation shows — where the crack "
        "starts and the path it takes — does not depend on the damage coefficient and "
        "is the result being demonstrated. Fatigue life is in Phase 3.",
        icon="⚠️",
    )

    _render_h2()


# ============================================================== hypothesis H2
def _h2():
    p = os.path.join(ROOT, "data", "phase2_peridynamics", "h2_first_bond.json")
    if not os.path.exists(p):
        return None
    import json
    return json.load(open(p, encoding="utf-8"))


@st.cache_data(show_spinner="Drawing the fourteen sections…")
def _section_frames():
    """One image per measured section: metal grey, defects teal, the dominant defect
    dark teal, final damage orange. Rotated so the load axis runs left to right."""
    import base64
    import io
    from PIL import Image
    from scipy import ndimage as ndi

    z = np.load(os.path.join(ROOT, "data", "phase2_peridynamics", "h2_sections.npz"))
    d = _h2()
    out = []
    for r in d["rows"]:
        spec = r["specimen"]
        void = z[f"{spec}__void"]
        dmg = np.clip(z[f"{spec}__damage_final"].astype(float), 0, 1)
        lab, _ = ndi.label(void)
        rgb = np.empty(void.shape + (3,), float)
        rgb[:] = (221, 224, 227)
        rgb[void] = (46, 125, 145)
        rgb[lab == r["dominant_label"]] = (16, 70, 86)
        w = np.where(void, 0.0, np.clip(dmg * 3.0, 0, 1))[..., None]
        rgb = rgb * (1 - w) + np.array((240, 120, 30)) * w
        img = np.flipud(rgb.astype(np.uint8))           # z upward
        img = np.rot90(img, k=-1)                        # load axis left to right
        im = Image.fromarray(np.ascontiguousarray(img))
        scale = 3
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        out.append(dict(spec=spec, src="data:image/png;base64," +
                        base64.b64encode(buf.getvalue()).decode(),
                        w=im.width, h=im.height, row=r, scale=scale,
                        nz=void.shape[0], nx=void.shape[1]))
    return out


def _star_xy(f):
    """The first bond's pixel position after the same flip and rotation as the image.
    Worked out from the transforms rather than assumed, and checked by drawing."""
    r, s = f["row"], f["scale"]
    row, col = r["first_bond_row"], r["first_bond_col"]
    fr = f["nz"] - 1 - row              # after flipud: row index from the top
    # rot90(k=-1) maps (i, j) of an (H, W) image to (j, H - 1 - i)
    ri, ci = col, f["nz"] - 1 - fr
    return (ci + 0.5) * s, (ri + 0.5) * s


def _render_h2():
    """The H2 result, read from the pipeline output rather than typed in here.

    Source: data/phase2_peridynamics/h2_first_bond.json and h2_sections.npz, written
    by scripts/h2_first_bond.py in the Research files tree."""
    d = _h2()
    st.divider()
    st.subheader("Hypothesis H2: does the crack start at the most severe defect?")
    if d is None:
        badge("H2 NOT YET RUN", "pending")
        st.caption("data/phase2_peridynamics/h2_first_bond.json is not present.")
        return
    n, crit = d["n"], d["criterion"]
    met = d["fraction_by_area"] >= crit
    # the badge says what happened against the criterion; "not supported" is the thesis's
    # formal verdict and is stated in the text, not shouted in red
    badge(f"Criterion {'met' if met else 'not met'} · {d['hits_by_area']} of {n} "
          f"({d['fraction_by_area']:.0%}) vs ≥ {crit:.0%}", "real" if met else "no")
    st.markdown(
        f"H2 required the crack to start at the most severe defect in at least "
        f"**{crit:.0%}** of cases. It is tested on the **{n} measured sections** "
        "themselves: real defect sizes and real positions. "
        + ("**Supported.**" if met else
           "In the thesis's terms the hypothesis is **not supported**: a finding about "
           "what decides where a crack starts, not a failure of the method."))
    plain(
        "<b>How the starting point is found, exactly.</b> Every bond in the model starts "
        "with the same life and loses it fastest where it is stretched most. So the first "
        "bond to break is simply the most-stretched bond in the first elastic solution. "
        "One solve per section finds it, and the answer does not depend on the damage "
        "coefficient or on how many bonds the simulation lets break per step.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Starts at the largest defect", f"{d['fraction_by_area']:.1%}",
              delta=f"{d['hits_by_area']} of {n}", delta_color="off",
              help="Severity as H2 is worded: √area.")
    c2.metric("Starts at the most severe by Murakami ΔK", f"{d['fraction_by_severity']:.1%}",
              delta=f"{d['hits_by_severity']} of {n}", delta_color="off",
              help="√area weighted 0.65 for a surface defect, 0.50 for an enclosed one.")
    c3.metric("Criterion", f"{crit:.0%}", delta="not met" if not met else "met",
              delta_color="off")

    frames = _section_frames()
    if frames:
        def _title(f):
            r = f["row"]
            verdict = "✔ largest defect" if r["initiated_at_dominant"] else "✘ not the largest"
            where = "surface" if r["first_bond_defect_is_surface"] else "internal"
            return (f"<b>{f['spec'].replace('SN_', '')}</b> · {r['sigma_max_MPa']} MPa · "
                    f"first bond at a {r['first_bond_defect_sqrt_area_um']:.0f} µm {where} "
                    f"defect; largest is {r['dominant_sqrt_area_um']:.0f} µm · {verdict}")

        def _data(f):
            x, y = _star_xy(f)
            return [go.Image(source=f["src"]),
                    go.Scatter(x=[x], y=[y], mode="markers",
                               marker=dict(symbol="star", size=20, color="#FFD54F",
                                           line=dict(color="#7A1F12", width=2)),
                               name="first bond to break", hoverinfo="name")]

        f0 = frames[0]
        fig = go.Figure(data=_data(f0))
        fig.frames = [go.Frame(data=_data(f), name=f["spec"].replace("SN_", ""),
                               layout=go.Layout(title_text=_title(f))) for f in frames]
        fig.update_layout(title=dict(text=_title(f0), x=0.0, font=dict(size=13)),
                          height=360, margin=dict(l=0, r=0, t=40, b=95),
                          xaxis=dict(visible=False, range=[0, max(f["w"] for f in frames)]),
                          yaxis=dict(visible=False, scaleanchor="x",
                                     range=[max(f["h"] for f in frames), 0]),
                          showlegend=False)
        play(fig, frame_ms=1800, transition_ms=0, slider_prefix="section ")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each section with the load axis running left to right. Grey: 316H. Teal: "
            "defects, with the largest in dark teal. Orange: damage at the end of the run. "
            "Gold star: the first bond to break. Press Play to step through all fourteen.")

    rows = d["rows"]
    df = pd.DataFrame([{
        "specimen": r["specimen"].replace("SN_", ""),
        "σmax (MPa)": r["sigma_max_MPa"],
        "defects in section": r["n_defects"],
        "largest (µm)": r["dominant_sqrt_area_um"],
        "first bond at (µm)": r["first_bond_defect_sqrt_area_um"],
        "surface?": "yes" if r["first_bond_defect_is_surface"] else "no",
        "at largest?": "yes" if r["initiated_at_dominant"] else "no",
        "at most severe (ΔK)?": "yes" if r["initiated_at_most_severe"] else "no",
    } for r in rows])
    st.dataframe(df, width='stretch', hide_index=True)

    n_surf = sum(1 for r in rows if r["first_bond_defect_is_surface"])
    st.warning(
        f"**Reported as found.** Size alone does not predict where the crack starts. "
        f"{n_surf} of {n} sections started at a **surface** defect, which Murakami's "
        f"geometry factors make {(0.65 / 0.50) ** 2:.2f}× as severe per unit area as an "
        "enclosed one: enough for a small surface flaw to beat a much larger buried one. "
        f"Weighting by that factor lifts the count only to {d['hits_by_severity']} of {n}, "
        "so neighbouring defects matter too. This is a finding about the **severity "
        "criterion**, not about the segmentation that measures the defects or the life "
        "model that uses their size.",
        icon="🔍")

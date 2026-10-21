import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ct_viewer as cv
from src.ui import NAVY, finding, go_to, note, play, show, spec, verdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "phase2_extraction")
PD = os.path.join(ROOT, "data", "phase2_peridynamics")

SPEC_LABELS = {"SN_P017": "P017: largest governing defect of the 14",
               "SN_P059": "P059: most defects in its gauge"}


def _region_grid(rgb, regions, max_labels=6):
    """Size labels on the largest regions, drawn with PIL."""
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


def _json(name):
    p = os.path.join(PD, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def _h2():
    return _json("h2_first_bond.json")


def render():
    st.title("Phase 2: where cracks start")
    h2 = _h2()
    v1 = _json("verify_01_results.json")
    m_err = (v1["1A"][0]["err_pct"], v1["1A"][-1]["err_pct"], v1["1A"][0]["m"],
             v1["1A"][-1]["m"]) if v1 else None
    spec(
        r"Bond-based peridynamics: each point $x$ is held in equilibrium by bonds to every "
        r"point within its horizon $\delta$, "
        r"$\int_{H_x} c\,s\,\chi\,\frac{\xi+\eta}{|\xi+\eta|}\,dV_{x'} + b(x) = 0$, with bond "
        r"stretch $s=(|\xi+\eta|-|\xi|)/|\xi|$. Under cyclic load each bond loses life as "
        r"$d\lambda/dN = -A(\varepsilon-\varepsilon_{th})^{\beta}$ and breaks ($\chi=0$) "
        r"when $\lambda$ reaches zero.",
        "The measured section through the governing defect on a uniform grid, "
        "Δx = 40 µm, horizon δ = 3Δx, with defects as voids. Equilibrium is solved "
        "as a linear system at each damage event; time jumps to the next bond failure.",
        (f"As the horizon is refined the model's elastic modulus converges to the "
         f"classical value: error {m_err[0]:.1f}% at m = {m_err[2]:g} falling to "
         f"{m_err[1]:.2f}% at m = {m_err[3]:g} (chart below)." if m_err else
         "Horizon-refinement results not found."),
        (f"The first bond breaks at the largest defect in **{h2['hits_by_area']} of "
         f"{h2['n']}** measured sections ({h2['fraction_by_area']:.1%}); H2 asked for 90%."
         if h2 else "H2 result file not found."))
    note("This phase has two jobs. It measures every defect, and the size of the largest "
         "goes to Phase 3 to become the life. Its crack model then shows where the crack "
         "starts and which way it runs. The crack model's cycle count is not used as a "
         "life: at attainable grid resolution it cannot carry one.")

    # ------------------------------------------------------------ defects to numbers
    st.markdown("## From a defect to a number")
    cat = pd.read_csv(os.path.join(DATA, "phase2_defect_catalogue.csv"))
    cyl = pd.read_csv(os.path.join(DATA, "phase2_cylinder_summary.csv"))
    st.markdown(
        f"Every defect is isolated in three dimensions and reduced to the size fracture "
        f"mechanics uses: the square root of its area projected on the plane across the "
        f"load, √area (Murakami). The catalogue holds **{len(cat):,} defects** from "
        f"{cyl['cylinder'].nunique()} cylinders, the largest {cat['sqrt_area_um'].max():,.0f} "
        f"µm. Flatness separates lack-of-fusion flaws (about 0.1) from round gas pores "
        f"(about 1).")
    specs = cv.available_specimens()
    if specs:
        sp = st.selectbox("Cylinder", specs, format_func=lambda s: SPEC_LABELS.get(s, s))
        zs = cv.list_slices(sp)
        z = zs[st.slider("Slice", 0, len(zs) - 1, len(zs) // 2)]
        v1c, v2c, v3c = st.columns(3)
        ct = cv.load_ct_slice(sp, z)
        overlay = cv.composite_slice(sp, z)
        abst = cv.slice_abstraction(sp, z)
        if ct is not None:
            v1c.image((ct * 255).astype("uint8"), caption="Raw CT", width="stretch")
        if overlay is not None:
            v2c.image(overlay, caption="Predicted defects", width="stretch")
        if abst is not None:
            img = _region_grid(abst["rgb"], abst["regions"]) if abst["regions"] else abst["rgb"]
            v3c.image(img, caption=f"{abst['n']} region(s) sized", width="stretch")
    else:
        finding("No CT slice data found in data/ct_slices/.")

    fig2 = go.Figure(go.Scatter(
        x=cat["sqrt_area_um"], y=cat["flatness"], mode="markers",
        marker=dict(size=5, color=NAVY, opacity=0.45),
        text=cat["cylinder"], hovertemplate="%{text}: √area %{x:.0f} µm, flatness %{y:.2f}"))
    fig2.update_layout(height=360, xaxis_title="√area (µm)", yaxis_title="flatness",
                       xaxis_type="log")
    show(fig2)
    st.caption("Every catalogued defect by size and shape.")
    with st.expander("Cylinder summary and the cross-check against the reference flaw count"):
        st.dataframe(cyl.rename(columns={"largest_sqrt_area_um": "largest √area (µm)",
                                         "aop24_porosity_pct": "porosity (%)"}),
                     width="stretch", hide_index=True)
        r = np.corrcoef(cyl["aop24_flaw_vox"], cyl["pred_flaw_vox_scaled"])[0, 1]
        st.markdown(f"Predicted and reference flaw volumes correlate at r = {r:.3f}.")

    # ------------------------------------------------------------ verification
    st.markdown("## Verification: the model converges to classical elasticity")
    if v1:
        rows = v1["1A"]
        fig = go.Figure(go.Scatter(
            x=[r["m"] for r in rows], y=[r["err_pct"] for r in rows], mode="lines+markers",
            line=dict(color=NAVY), marker=dict(size=7),
            text=[f"{r['bonds_per_node']} bonds per node" for r in rows],
            hovertemplate="m = %{x}: %{y:.2f}% (%{text})"))
        fig.update_layout(height=340, xaxis_title="horizon multiplier m = δ/Δx",
                          yaxis_title="error in elastic modulus (%)", yaxis_type="log")
        show(fig)
        st.markdown(
            f"Refining the horizon drives the error from {rows[0]['err_pct']:.1f}% to "
            f"{rows[-1]['err_pct']:.2f}% (fitted order {rows[0]['fitted_order']:.2f}). The "
            "decline is not smooth because the number of bonds per node changes in steps. "
            "The bond-based form fixes Poisson's ratio at 1/3 in plane stress, against about "
            "0.30 for 316H; that is a property of the formulation, which is why a finite "
            "element model runs alongside as a cross-check.")

    # ------------------------------------------------------------ the crack
    st.markdown("## The crack in a measured section")
    mp4 = os.path.join(ROOT, "assets", "phase2_crack_growth.mp4")
    if os.path.exists(mp4):
        st.video(mp4, autoplay=True, loop=True, muted=True)
    st.markdown(
        "The measured section of P017 at 450 MPa, R = 0.1: its 22 defects at their real "
        "sizes and positions, modelled as voids (teal), and fatigue damage accumulating "
        "(orange). The crack leaves both tips of the large central defect. The video "
        "plays on a loop; use its controls to pause. Its cycle counter paces the playback "
        "and is not a fatigue life.")

    _render_h2()
    go_to("phase3", "Continue to Phase 3: the fatigue life")


# ============================================================== hypothesis H2
@st.cache_data(show_spinner="Drawing the fourteen sections")
def _section_frames():
    """One image per measured section: metal grey, defects teal, the largest dark teal,
    final damage orange, with the load axis left to right."""
    import base64
    import io
    from PIL import Image
    from scipy import ndimage as ndi

    z = np.load(os.path.join(PD, "h2_sections.npz"))
    d = _h2()
    out = []
    for r in d["rows"]:
        sp = r["specimen"]
        void = z[f"{sp}__void"]
        dmg = np.clip(z[f"{sp}__damage_final"].astype(float), 0, 1)
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
        out.append(dict(spec=sp, src="data:image/png;base64," +
                        base64.b64encode(buf.getvalue()).decode(),
                        w=im.width, h=im.height, row=r, scale=scale,
                        nz=void.shape[0], nx=void.shape[1]))
    return out


def _star_xy(f):
    """The first bond's pixel position after the same flip and rotation as the image."""
    r, s = f["row"], f["scale"]
    row, col = r["first_bond_row"], r["first_bond_col"]
    fr = f["nz"] - 1 - row              # after flipud: row index from the top
    ri, ci = col, f["nz"] - 1 - fr      # rot90(k=-1): (i, j) -> (j, H - 1 - i)
    return (ci + 0.5) * s, (ri + 0.5) * s


def _render_h2():
    d = _h2()
    st.markdown("## Hypothesis 2: does the crack start at the largest defect?")
    if d is None:
        finding("The H2 result file, data/phase2_peridynamics/h2_first_bond.json, is not "
                "present.")
        return
    n, crit = d["n"], d["criterion"]
    met = d["fraction_by_area"] >= crit
    verdict(f"Criterion {'met' if met else 'not met'}: {d['hits_by_area']} of {n} "
            f"({d['fraction_by_area']:.0%}) against {crit:.0%}. By Murakami severity, "
            f"{d['hits_by_severity']} of {n}.", "met" if met else "no")
    st.markdown(
        "Every bond starts with the same life and loses it fastest where it is stretched "
        "most, so the first bond to break is the most strained bond of the initial elastic "
        "solution: one solve per section, independent of the damage coefficient and of how "
        "many bonds the model breaks per step. The animation steps through all fourteen "
        "measured sections.")

    frames = _section_frames()
    if frames:
        def _title(f):
            r = f["row"]
            where = "surface" if r["first_bond_defect_is_surface"] else "internal"
            res = "at the largest defect" if r["initiated_at_dominant"] else \
                "not at the largest defect"
            return (f"{f['spec'].replace('SN_', '')}, {r['sigma_max_MPa']} MPa: first bond "
                    f"{res} ({r['first_bond_defect_sqrt_area_um']:.0f} µm {where}; largest "
                    f"{r['dominant_sqrt_area_um']:.0f} µm)")

        def _data(f):
            x, y = _star_xy(f)
            return [go.Image(source=f["src"]),
                    go.Scatter(x=[x], y=[y], mode="markers",
                               marker=dict(symbol="star", size=20, color="#FFD54F",
                                           line=dict(color="#7A1F12", width=2)),
                               name="first bond", hoverinfo="name")]

        f0 = frames[0]
        fig = go.Figure(data=_data(f0))
        fig.frames = [go.Frame(data=_data(f), name=f["spec"].replace("SN_", ""),
                               layout=go.Layout(title_text=_title(f))) for f in frames]
        fig.update_layout(title=dict(text=_title(f0), x=0.0, font=dict(size=13)),
                          height=330, margin=dict(l=0, r=0, t=40, b=90),
                          xaxis=dict(visible=False, range=[0, max(f["w"] for f in frames)]),
                          yaxis=dict(visible=False, scaleanchor="x",
                                     range=[max(f["h"] for f in frames), 0]),
                          showlegend=False)
        play(fig, frame_ms=2200, transition_ms=0, slider_prefix="")
        show(fig, height=330)
        st.caption("Grey: 316H. Teal: defects, the largest darker. Orange: damage at the "
                   "end of the run. Star: the first bond to break. Load axis left to right.")

    rows = d["rows"]
    st.dataframe(pd.DataFrame([{
        "Cylinder": r["specimen"].replace("SN_", ""),
        "σmax (MPa)": r["sigma_max_MPa"],
        "Defects": r["n_defects"],
        "Largest √area (µm)": r["dominant_sqrt_area_um"],
        "First bond at (µm)": r["first_bond_defect_sqrt_area_um"],
        "Surface": "yes" if r["first_bond_defect_is_surface"] else "no",
        "At largest": "yes" if r["initiated_at_dominant"] else "no",
        "At most severe (ΔK)": "yes" if r["initiated_at_most_severe"] else "no",
    } for r in rows]), width="stretch", hide_index=True)
    n_surf = sum(1 for r in rows if r["first_bond_defect_is_surface"])
    finding(
        f"<b>Reported as found.</b> Size alone does not decide where a crack starts. "
        f"{n_surf} of {n} sections started at a surface defect, which Murakami's geometry "
        f"factors make {(0.65 / 0.50) ** 2:.2f} times as severe per unit area as a buried "
        f"one; weighting for that lifts the count only to {d['hits_by_severity']} of {n}, "
        "so neighbouring defects matter too. This concerns the severity criterion, not "
        "the segmentation or the life model, which uses defect size and is tested in "
        "Phase 4.")

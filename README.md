# PINN of LPBF Fatigue Life

A Streamlit dashboard of the physics-informed fatigue-life work for *Mathematical
Modelling of Internal Properties of LPBF Components* (Lamek S. Indongo, BSc Hons
Mechanical Engineering, UNAM). Built for the supervisor's request for a
**physics-informed machine learning (PIML) comparison model** alongside the
project's peridynamic framework.

Every number and figure in this app is read from the verified pipeline in the
private `Research files/` working tree, not recomputed or approximated for
display. Where a stage has not been run, the app says so rather than showing an
illustrative placeholder.

## Layout

A single-page app with a top pipeline bar (Phase 1-4, styled after the project's
own desktop demo tool), not a sidebar. Every chart is a live Plotly graph built
from the underlying data at render time — nothing is a pre-rendered PNG. The one
exception is the crack-growth GIF (Phase 3), which is real simulation footage,
not a statistical chart, so there is nothing to "graph" instead.

Phase numbering matches the thesis report exactly (not the exploratory "Monte
Carlo" phase that appeared in an earlier, superseded version of the demo app):

| Phase | Status |
|---|---|
| 1 · Segmentation | Real — nnU-Net v2, 5-fold cross-validated |
| 2 · Defect extraction | Real — 1342-defect catalogue, AOP24 cross-check, and a live raw CT / segmentation overlay / extracted-region viewer on real CT slices |
| 3 · Peridynamic crack growth | Real — damage-coefficient A calibrated against the dataset-specific Paris coefficient; per-specimen fatigue-life results appear automatically once `run_all_specimens.py --redo` finishes |
| 4 · Validation & PIML | Real — closed-loop validation, statistical ladder, sign-constrained physics regression, and a new physics-informed neural network (PINN) |

## The two PIML arms (Phase 4)

1. **Hard physics** — `comparison_model`'s existing sign-constrained least-squares
   regression: `log10(Nf) = b0 + b_s*ln(ds) + b_a*ln(sqrt_area)` with `b_s, b_a <= 0`
   enforced as solver bounds, so a physically inconsistent fit is not representable.
2. **Soft physics (new)** — `src/piml_nn.py`: a small neural network trained with a
   hybrid loss, data MSE plus a PDE-residual term that penalises the network's
   *learned gradient* (via autograd, at collocation points spanning the feature
   domain, not only the 14 data points) for deviating from the theoretical
   Basquin/Paris slope. Verified first against a known synthetic answer before
   being trusted on the real 14-specimen dataset — see the verification table on
   the Phase 4 page.

Both arms are benchmarked leave-one-out against the same statistical ladder and
zero-parameter physics baseline, on the same specimens, in one table.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
streamlit run app.py
```

## Deploying (Streamlit Community Cloud, free)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. "New app" → pick this repo → branch `main` → main file `app.py` → Deploy.
4. Streamlit Cloud installs `requirements.txt` (including the CPU build of
   PyTorch) automatically. First deploy takes a few minutes; the PINN itself
   trains once per app restart (cached with `st.cache_resource`), not per click.
5. You get a public URL (`https://<name>.streamlit.app`) to share — free-tier
   apps are visible to anyone with the link, so treat it like a public preprint,
   not a private document.

## Data included in this repo

Small derived artefacts (catalogues, verification JSON, the crack-growth GIF)
plus a curated set of real CT and predicted-mask slices for two specimens
(P017, P059 — the z-range where each actually has predicted defects), ~55 MB
total, for the live raw/overlay/extracted-region viewer on the Phase 2 page.
The raw 127 GB ORNL HDF5 archive and the full 436 MB held-out CT/mask folder
stay in the private `Research files/` working tree.

## Known limitations (see the Phase 4 page for the full list)

- Only 14 of 60 specimens have a measured governing-defect size; the rest use
  bulk porosity as a coarser proxy.
- Distance to the free surface, true sphericity, and nearest-neighbour spacing
  are not yet extracted — both PIML arms fall back to flatness as the shape term.
- The PINN's leave-one-out result on 14 points carries real variance; read the
  comparison ladder as indicative, not a certified ranking.

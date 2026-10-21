# PINN of LPBF Fatigue Life

A Streamlit app that explains, and shows the evidence for, *Mathematical Modelling of
Fatigue Life of Laser Powder Bed Fusion Components* (Lamek S. Indongo, BSc Hons
Mechanical Engineering, UNAM; supervisor Assoc. Prof. Albert Shikongo, co-supervisor
Dr. Surendra Kumar Saini). It was first built for the supervisor's request for a
**physics-informed machine learning (PIML) comparison model**, and now walks a reader
through the whole method.

Every number and figure is read from the verified pipeline's result files in `data/`
at render time, not recomputed or approximated for display. Where a stage has not
been run, the app says so rather than showing a placeholder.

## Layout

A single-page app with a top bar, read left to right as a story:

| Page | What it shows |
|---|---|
| Overview | The question in plain words, the measured 21× scatter in life, the two-arm method, and one held-out cylinder (P017) followed from CT slice to predicted life |
| Phase 1 · Find the defects | nnU-Net v2 segmentation, an animated CT stack, 5-fold cross-validation (Dice 0.805) |
| Phase 2 · Measure them · where cracks start | The 1342-defect catalogue, the raw CT / overlay / extracted-region viewer, the peridynamic crack animation, and H2 on all 14 measured sections, animated, with the exact first bond marked |
| Phase 3 · Predict the life | The Paris integral animated for any held-out cylinder, every held-out prediction revealed one by one, the Monte Carlo distribution building up, the Bayesian band narrowing per inspection |
| Phase 4 · Test the prediction | The comparison ladder climbed rung by rung, the sign-constrained regression and the PINN, with their verifications |
| Conclusions | The four hypotheses scored from their result files, contributions, limits, next steps |

Every phase opens with the same five boxes: the question, how it is answered, the
check against a known answer, what was found, and what it means.

### Which arm produces the fatigue life

The defect catalogue feeds two independent arms and only one ends in a life. The
**geometry arm** takes the governing defect size into the Paris integral: that is the
life. The validated prediction (path A) uses literature constants for 316L
(C = 6.25e-10, m = 3.94), Murakami's Y = 0.65 and the network's own gauge-cropped
defect size, with the long-crack threshold removed, because at the literature value
of 5 MPa sqrt(m) most of these specimens would be predicted never to fail although
every one failed (the short-crack regime). Nothing is fitted to the fatigue lives; the
typical error is 1.36x on the 14 held-out cylinders, where the best fitted statistical
model in the ladder needs four free parameters to reach 1.43x. The **mechanics arm** is
peridynamics, which establishes the crack initiation site and path.

Peridynamic cycle counts are **not** fatigue lives and the app does not present them as
such. The coefficient behind them was calibrated on soda-lime glass (E = 72 GPa) and
applied to 316H (E = 195 GPa) — 50.7× measured directly — then transferred across a 12.5×
mesh change and an 8–65× crack-size change in a model whose effective Paris exponent under
geometric self-similarity is exactly 2 whatever β is set to. Redoing that calibration
correctly moved the prediction further from the record, not closer, which is what makes
the limit a property of the method rather than of the tuning.

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

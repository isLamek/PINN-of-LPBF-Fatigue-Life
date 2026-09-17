"""Raw CT / segmentation overlay / extracted-region viewer.

Ported line-for-line from app_fatigue_framework/data/loader.py (the desktop app's
real reader), swapping Qt for plain numpy arrays a Streamlit page can render
directly. Same windowing, same overlay colour, same connected-components
labelling -- this is not a re-derivation, it is the same code on real CT slices.

Only two specimens' slices ship in this repo (data/ct_slices/), covering the
z-range where each one actually has predicted defects -- not the full 436 MB
held-out folder, which stays in the private working tree.
"""
import glob
import os
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CT_DIR = os.path.join(HERE, "..", "data", "ct_slices", "imagesTs")
MASK_DIR = os.path.join(HERE, "..", "data", "ct_slices", "predictions_pp")

VOXEL_XY_UM = 17.8       # in-plane voxel size
LAYER_UM = 100.0         # print layer thickness; slice index IS the layer number
OVERLAY_RED = np.array([176, 58, 46], dtype=np.float64)

_REGION_COLOURS = [
    (86, 156, 214), (222, 148, 86), (108, 190, 120), (214, 120, 154),
    (172, 142, 220), (222, 200, 96), (96, 200, 200), (232, 110, 96),
    (150, 170, 205), (180, 210, 130),
]
_ABS_BG = (20, 22, 26)
_ABS_METAL = (33, 36, 41)
_ABS_EDGE = (74, 80, 88)

_Z_RE = re.compile(r"_z(\d{4})(?:_0000)?\.png$")


def available_specimens() -> list[str]:
    files = glob.glob(os.path.join(CT_DIR, "SN_*_z*_0000.png"))
    specs = sorted({os.path.basename(f).split("_z")[0] for f in files})
    return specs


def list_slices(specimen: str) -> list[int]:
    files = sorted(glob.glob(os.path.join(CT_DIR, f"{specimen}_z*_0000.png")))
    zs = []
    for f in files:
        m = _Z_RE.search(os.path.basename(f))
        if m:
            zs.append(int(m.group(1)))
    return zs


def load_ct_slice(specimen: str, z: int) -> np.ndarray | None:
    """16-bit CT slice, display-windowed to [0, 1] float."""
    p = os.path.join(CT_DIR, f"{specimen}_z{z:04d}_0000.png")
    if not os.path.isfile(p):
        return None
    a = np.asarray(Image.open(p), dtype=np.float64)
    metal = a[a > 0]
    if metal.size:
        lo, hi = np.percentile(metal, [1.0, 99.5])
        if hi <= lo:
            lo, hi = float(metal.min()), float(metal.max() or 1.0)
    else:
        lo, hi = 0.0, 1.0
    out = np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    out[a == 0] = 0.0
    return out


def load_mask_slice(specimen: str, z: int) -> np.ndarray | None:
    p = os.path.join(MASK_DIR, f"{specimen}_z{z:04d}.png")
    if not os.path.isfile(p):
        return None
    return np.asarray(Image.open(p)) > 0


def composite_slice(specimen: str, z: int, alpha: float = 0.45) -> np.ndarray | None:
    """Grey CT with the predicted mask blended in translucent red. (H, W, 3) uint8."""
    ct = load_ct_slice(specimen, z)
    if ct is None:
        return None
    g = (ct * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float64)
    mask = load_mask_slice(specimen, z)
    if mask is not None and mask.shape == ct.shape and mask.any():
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * OVERLAY_RED
    return rgb.astype(np.uint8)


def slice_abstraction(specimen: str, z: int) -> dict | None:
    """Labelled 2-D connected components of the predicted mask on this slice:
    the abstraction the pipeline actually extracts, one colour per region."""
    mask = load_mask_slice(specimen, z)
    if mask is None:
        return None
    h, w = mask.shape
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = _ABS_BG

    ct = load_ct_slice(specimen, z)
    metal = ct > 0 if (ct is not None and ct.shape == mask.shape) else None
    if metal is not None and metal.any():
        rgb[metal] = _ABS_METAL

    from scipy import ndimage
    if metal is not None and metal.any():
        edge = metal & ~ndimage.binary_erosion(metal)
        rgb[edge] = _ABS_EDGE

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    regions = []
    if n:
        pal = np.asarray(_REGION_COLOURS, dtype=np.uint8)
        rgb[mask] = pal[(lbl[mask] - 1) % len(pal)]
        idx = np.arange(1, n + 1)
        areas = ndimage.sum_labels(mask, lbl, idx)
        cents = ndimage.center_of_mass(mask, lbl, idx)
        for i, (a, (cy, cx)) in enumerate(zip(areas, cents)):
            regions.append(dict(
                cy=float(cy), cx=float(cx), area_px=int(a),
                d_um=2.0 * VOXEL_XY_UM * float(np.sqrt(a / np.pi)),
                colour=_REGION_COLOURS[i % len(_REGION_COLOURS)],
            ))
        regions.sort(key=lambda r: r["area_px"], reverse=True)
    return dict(rgb=rgb, regions=regions, n=int(n))

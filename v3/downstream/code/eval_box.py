#!/usr/bin/env python3
"""eval_box.py BOX SURF_PATH [SURF_PATH ...] : before/after check of a re-blended box (read only).
For each surface (released ones by keyword: r1 = R1 _deseam, r2 = R2 _deseam, r2final = R2 _FINAL, or a file path):
 PEDR cells: the PEDR census shots of the box ($PC/shots/<BOX>.shots.npz, census orbit/segment masks), footprint-smoothed surface
   (Gaussian FWHM 168 m) sampled bilinearly; per 128 px cell with >= 3 shots, fail when |median(surface - shot)| > max(30,
   250 tan(MOLA slope)) (the census rule). Counts of scored / failing cells, and the shot MAE.
 step lines: one-pixel steps > 20 m and > 5x the median of the same step 3 px either side, along rows (N-S lines) and columns
   (E-W lines); pixels, and 128 px cells where one line covers >= 50 % of the cell.
 flips (2026-10-27, the v3 QA's churn-tolerant PEDR rule): with two or more surfaces, PEDR cells scored by both the first and
   the last surface that fail in only one of them ("PEDR flips ... only-last N, only-first N, both N, scored in both N").
Both inside the box's own 10x10 deg cell (+2 km) only (the part that reaches the global mosaic)."""
import os, sys, math, numpy as np, rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from pyproj import CRS, Transformer
os.environ.setdefault("PROJ_NETWORK", "OFF")
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; PC = f"{G}/global_run/fix_2026-10/r2_delivery/pedr_census"
T1, T2 = "ta_implicit_d1_implrel100s1_e088", "ta_implicit_d1_implrel100s1_e088r2"
box = sys.argv[1]; RD = f"{G}/global_run/boxes/{box}/box_run"
KEY = dict(r1=f"{RD}/box_dem_{T1}_deseam.tif", r2=f"{RD}/box_dem_{T2}_deseam.tif", r2final=f"{RD}/box_dem_{T2}_FINAL.tif",
           r2dlc2=f"{RD}/box_dem_{T2}_deseam_dlc2.tif")
LL = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs"); SIG = 168.0 / 2.3548 / 18.0; CELL = 128; BLK = 2048


def cellbounds(b):
    e, n = b.split("_"); lo = int(e[1:].replace("p", "+")); la = int(n[1:].replace("p", "+")); return lo, lo + 10, la, la + 10


def incell_mask(S, r0, c0, h, w):
    """pixels of the window inside the box's own cell + 2 km"""
    lo0, lo1, la0, la1 = cellbounds(box); tr = S.transform
    rr, cc = np.mgrid[r0:r0 + h:16, c0:c0 + w:16]
    x, y = tr * (cc + 0.5, rr + 0.5); lon, lat = Transformer.from_crs(S.crs, LL, always_xy=True).transform(x, y)
    lon = np.asarray(lon); lat = np.asarray(lat); kx = 59.27 * np.maximum(np.cos(np.radians(lat)), 0.02)
    dl = ((lon - lo0 + 180) % 360) - 180
    ok = (np.minimum(dl, 10 - dl) * kx > -2) & (np.minimum(lat - la0, la1 - lat) * 59.27 > -2)
    return np.kron(ok, np.ones((16, 16), bool))[:h, :w]


def steps(z, axis):
    d = np.abs(np.diff(z, axis=axis)); dd = np.where(np.isfinite(d), d, np.inf)       # no-data neighbours count as huge (conservative)
    ref = np.median(np.stack([np.roll(dd, k, axis=axis) for k in (-3, -2, -1, 1, 2, 3)]), 0)   # same step 1-3 px to either side (census rule)
    with np.errstate(invalid="ignore"): return (d > 20) & (d > 5 * ref)


def block(job):
    path, br, bc, h, w = job
    with rasterio.open(path) as S:
        nd = S.nodata
        z = S.read(1, window=Window(bc - 24, br - 24, w + 48, h + 48), boundless=True, fill_value=nd).astype(np.float64)
        m = np.isfinite(z) & (z > -1e4) & (z < 3e4)
        if nd is not None: m &= z != nd
        if not m.any(): return None
        cm = incell_mask(S, br, bc, h, w)
    zc = np.where(m, z, np.nan)[24:24 + h, 24:24 + w]
    sx, sy = steps(zc, 1), steps(zc, 0); sx &= cm[:, 1:]; sy &= cm[1:, :]
    st = dict(ns=int(sx.sum()), ew=int(sy.sum()), px=int((cm & np.isfinite(zc)).sum()), ns_cells=0, ew_cells=0)
    for a_ in range(0, h - CELL + 1, CELL):
        for b_ in range(0, w - CELL + 1, CELL):
            if sx[a_:a_ + CELL, b_:b_ + CELL - 1].sum(0).max(initial=0) >= CELL // 2: st["ns_cells"] += 1
            if sy[a_:a_ + CELL - 1, b_:b_ + CELL].sum(1).max(initial=0) >= CELL // 2: st["ew_cells"] += 1
    sel = np.flatnonzero((ROWS >= br) & (ROWS < br + h) & (COLS >= bc) & (COLS < bc + w))
    if not len(sel): return st, sel, np.zeros(0), np.zeros(0, bool)
    zs_ = gaussian_filter(np.where(m, z, 0.0), SIG); ws_ = gaussian_filter(m.astype(float), SIG)
    with np.errstate(all="ignore"): f = np.where(ws_ > 0.9, zs_ / ws_, np.nan)
    yy, xx = ROWS[sel] - (br - 24), COLS[sel] - (bc - 24); y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    fy, fx = yy - y0, xx - x0
    v = f[y0, x0] * (1 - fy) * (1 - fx) + f[y0, x0 + 1] * (1 - fy) * fx + f[y0 + 1, x0] * fy * (1 - fx) + f[y0 + 1, x0 + 1] * fy * fx
    ic = cm[np.clip(np.round(yy - 24).astype(int), 0, h - 1), np.clip(np.round(xx - 24).astype(int), 0, w - 1)]
    return st, sel, v, ic


Z = dict(np.load(f"{PC}/shots/{box}.shots.npz")); n = len(Z["elev"])
drop = np.zeros(n, bool)
if os.path.exists(f"{PC}/cells/{box}.drop.npz"):
    D = np.load(f"{PC}/cells/{box}.drop.npz"); drop = (np.unpackbits(D["orbit"])[:n] | np.unpackbits(D["seg"])[:n]).astype(bool)
C = np.load(f"{PC}/cells/{box}.cells.npz"); ms = {(int(a), int(c)): float(m) for a, c, m in zip(C["crow"], C["ccol"], C["mslope"])}
import multiprocessing as mp
ROWS = COLS = None
FAILS = []
for arg in sys.argv[2:]:
    path = KEY.get(arg, arg)
    with rasterio.open(path) as S:
        H, W = S.height, S.width
        x, y = Transformer.from_crs(LL, S.crs, always_xy=True).transform(Z["lon"].astype(float), Z["lat"].astype(float))
        COLS = (np.asarray(x) - S.transform.c) / S.transform.a - 0.5; ROWS = (np.asarray(y) - S.transform.f) / S.transform.e - 0.5
    zs = np.full(n, np.nan); inc = np.zeros(n, bool); stp = dict(ns=0, ew=0, ns_cells=0, ew_cells=0, px=0)
    jobs = [(path, br, bc, min(BLK, H - br), min(BLK, W - bc)) for br in range(0, H, BLK) for bc in range(0, W, BLK)]
    with mp.get_context("fork").Pool(int(os.environ.get("NP", "16"))) as pool:
        for r in pool.imap_unordered(block, jobs):
            if r is None: continue
            st, sel, v, ic = r
            for k in stp: stp[k] += st[k]
            zs[sel] = v; inc[sel] = ic
    res = zs - Z["elev"]; ok = np.isfinite(res) & ~drop & inc
    cr, cc = (Z["row"] // CELL)[ok], (Z["col"] // CELL)[ok]; rv = res[ok]
    key = cr.astype(np.int64) * 100000 + cc; order = np.argsort(key); key, rv = key[order], rv[order]
    u, st, cnt = np.unique(key, return_index=True, return_counts=True)
    scored = fail = 0; fl = {}
    for k, s0, c0 in zip(u, st, cnt):
        if c0 < 3: continue
        th = max(30.0, 250.0 * math.tan(math.radians(ms.get((int(k // 100000), int(k % 100000)), 0.0))))
        f_ = bool(abs(np.median(rv[s0:s0 + c0])) > th); scored += 1; fail += f_; fl[int(k)] = f_
    FAILS.append(fl)
    print(f"{box} {os.path.basename(path)[-28:]:28s} cell-core px {stp['px']:>11d} | step px N-S {stp['ns']:7d} E-W {stp['ew']:7d} | "
          f"line cells N-S {stp['ns_cells']:5d} E-W {stp['ew_cells']:5d} | PEDR cells scored {scored} fail {fail} | shot MAE {np.mean(np.abs(rv)):.2f} (n {len(rv)})",
          flush=True)
if len(FAILS) >= 2:
    a_, b_ = FAILS[0], FAILS[-1]; both = set(a_) & set(b_)
    print(f"{box} PEDR flips last vs first: only-last {sum(b_[k] and not a_[k] for k in both)}, only-first {sum(a_[k] and not b_[k] for k in both)}, "
          f"both {sum(a_[k] and b_[k] for k in both)}, scored in both {len(both)}", flush=True)

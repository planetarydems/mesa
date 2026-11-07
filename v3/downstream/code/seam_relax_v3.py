#!/usr/bin/env python3
"""seam_relax_v3.py BOX SURFACE OUT.tif [options] : PROTOTYPE (2026-10-27, scratch only) of a seam relaxation at abutting frames.

Where two full-weight frames meet with no overlap, the v3 feather blend hands over in one pixel and the surface steps by the two
frames' local disagreement. This pass removes that step smoothly:
 1. dominant frame + its share of the effective blend weight at every pixel, computed exactly as level_frames_box_v3.feather_blend
    (frames leveled_<T3>/*_leveled.tif + *_w.tif warped bilinearly to the box grid, factors from the box's weights CSV, feathered
    demotion over --demote-ramp px) -> cached in --cache (dom int16, share uint8 0..250);
 2. hard hand-overs: 4-neighbour pixel pairs (p, q) with different dominant frames, both with share >= --share (0.9);
 3. offset at each pair = right-hand surface extrapolated to the boundary minus left-hand surface extrapolated to it (each side's
    local slope from 4 px, so a terrain slope is not counted); smoothed along each seam (pairs of the same frame pair and
    orientation, ordered along the seam: running median then running mean over --along px);
 4. correction: the left pixel gets +o/2, the right pixel -o/2; inside each frame's own region (dominant = that frame) the
    correction of the nearest seam pixel of that frame, times max(0, 1 - d / --ramp) (d = Euclidean px). Pixels without a dominant
    frame (infill) within --ramp px of corrected ones take the nearest corrected value, ramped the same way.
SURFACE is any surface on the box grid (v3 FINAL / deseam / leveled); OUT = SURFACE + correction (float32, deflate). Read only on
products. Prints and writes OUT.json: seam pairs, offsets, changed pixels."""
import argparse, csv, glob, json, os, sys, time, numpy as np, rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt, median_filter, uniform_filter1d

G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3D = os.environ.get("V3D", f"{G}/global_run/v3_2026-10/downstream")
ap = argparse.ArgumentParser()
ap.add_argument("box"); ap.add_argument("surface"); ap.add_argument("out")
ap.add_argument("--tag", default=os.environ.get("T3", "ta_implicit_d1_spline100s1_e098v3"))
ap.add_argument("--ramp", type=float, default=256.0); ap.add_argument("--share", type=float, default=0.9)
ap.add_argument("--along", type=int, default=31, help="px along the seam for the running median and mean of the offset")
ap.add_argument("--demote-ramp", type=float, default=512.0); ap.add_argument("--cache", default=None)
ap.add_argument("--mode", default="blend", choices=["blend", "region"], help="blend (default): per-frame correction fields blended with the leveller's own weights; region: each frame's own region only (first prototype)")
ap.add_argument("--np", type=int, default=int(os.environ.get("NP", "24"))); ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args(); t0 = time.time()
RD = f"{os.environ.get('BOXES_ROOT', G + '/global_run/boxes')}/{a.box}/box_run"
with rasterio.open(f"{RD}/box_dem_{a.tag}_leveled.tif") as s: H, W, TR, CRS = s.height, s.width, s.transform, s.crs
FR = sorted(glob.glob(f"{RD}/leveled_{a.tag}/*_leveled.tif")); WT = [p.replace("_leveled.tif", "_w.tif") for p in FR]
fac = {}
wc = f"{V3D}/weights/{a.box}.csv"
if os.path.exists(wc):
    for r in csv.DictReader(open(wc)): fac[r["stem"]] = float(r["factor"])
FAC = [min(fac.get(os.path.basename(p).split("_pred")[0], 1.0), 1.0) for p in FR]
SPAN = []
for p in FR:
    with rasterio.open(p) as s: b = s.bounds
    SPAN.append((max(0, int((TR.f - b.top) / TR.a) - 1), min(H, int((TR.f - b.bottom) / TR.a) + 1),
                 max(0, int((b.left - TR.c) / TR.a) - 2), min(W, int(np.ceil((b.right - TR.c) / TR.a)) + 2)))


def dom_chunk(rs):
    re_ = min(rs + a.chunk, H); h = re_ - rs
    parts = []
    for k, (r0, r1, c0, c1) in enumerate(SPAN):
        if r1 <= rs or r0 >= re_: continue
        with rasterio.open(FR[k]) as ds, WarpedVRT(ds, crs=CRS, transform=TR, width=W, height=H, resampling=Resampling.bilinear) as v:
            dem = v.read(1, window=Window(c0, rs, c1 - c0, h)); nd = ds.nodata
        with rasterio.open(WT[k]) as ws, WarpedVRT(ws, crs=CRS, transform=TR, width=W, height=H, resampling=Resampling.bilinear) as v:
            wt = v.read(1, window=Window(c0, rs, c1 - c0, h)).astype(np.float64)
        ok = np.isfinite(dem) & (wt > 0)
        if nd is not None and np.isfinite(nd): ok &= dem != nd
        if ok.any(): parts.append((k, c0, c1, np.where(ok, wt, 0.0)))
    wfull = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        if FAC[k] >= 1.0: wfull[:, c0:c1] += wt
    best = np.zeros((h, W)); bi = np.full((h, W), -1, np.int16); tot = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        f = FAC[k]; e = 1.0 if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull[:, c0:c1] / a.demote_ramp, 0.0, 1.0)
        ww = wt * e; tot[:, c0:c1] += ww; sl = best[:, c0:c1]; m = ww > sl
        sl[m] = ww[m]; bi[:, c0:c1][m] = k
    with np.errstate(all="ignore"): sh = np.where(tot > 0, best / tot, 0.0)
    return rs, bi, np.round(sh * 250).astype(np.uint8)


cache = a.cache or os.path.dirname(os.path.abspath(a.out))
dp, sp = f"{cache}/{a.box}_dom.npy", f"{cache}/{a.box}_share.npy"
if os.path.exists(dp) and os.path.exists(sp):
    DOM = np.load(dp, mmap_mode="r"); SHR = np.load(sp, mmap_mode="r"); print(f"[dom] cached {dp}", flush=True)
else:
    import multiprocessing as mp
    DOM = np.lib.format.open_memmap(dp + ".partial.npy", "w+", np.int16, (H, W)); SHR = np.lib.format.open_memmap(sp + ".partial.npy", "w+", np.uint8, (H, W))
    with mp.get_context("fork").Pool(a.np) as pool:
        for rs, bi, sh in pool.imap_unordered(dom_chunk, range(0, H, a.chunk)):
            DOM[rs:rs + len(bi)] = bi; SHR[rs:rs + len(bi)] = sh
    DOM.flush(); SHR.flush(); del DOM, SHR
    os.replace(dp + ".partial.npy", dp); os.replace(sp + ".partial.npy", sp)
    DOM = np.load(dp, mmap_mode="r"); SHR = np.load(sp, mmap_mode="r")
    print(f"[dom] {len(FR)} frames ({sum(f < 1 for f in FAC)} demoted) -> {dp} ({time.time() - t0:.0f}s)", flush=True)

with rasterio.open(a.surface) as s:
    Z = s.read(1).astype(np.float32); nd = s.nodata; prof = s.profile.copy()
Zv = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4)
if nd is not None: Zv &= Z != nd
DOM = np.asarray(DOM); TH = int(round(a.share * 250))
# ---- hard hand-over pairs + offsets ----
P = []                                       # rows: r_p, c_p, r_q, c_q, frame_p, frame_q, orient, offset
for axis in (1, 0):
    for r0 in range(0, H, 2048):
        r1 = min(H, r0 + 2048 + (1 if axis == 0 else 0))
        d = DOM[r0:r1]; s_ = SHR[r0:r1]; z = Z[r0:r1]; zv = Zv[r0:r1]
        if axis == 1: A, B, sa, sb = d[:, :-1], d[:, 1:], s_[:, :-1], s_[:, 1:]
        else: A, B, sa, sb = d[:-1], d[1:], s_[:-1], s_[1:]
        m = (A != B) & (A >= 0) & (B >= 0) & (sa >= TH) & (sb >= TH)
        rr, cc = np.nonzero(m)
        if not len(rr): continue
        if axis == 1:
            ok = (cc >= 4) & (cc + 5 < W); rr, cc = rr[ok], cc[ok]
            g = lambda k: z[rr, cc + k]; gv = lambda k: zv[rr, cc + k]
        else:
            ok = (rr >= 4) & (rr + 5 < z.shape[0]); rr, cc = rr[ok], cc[ok]
            g = lambda k: z[rr + k, cc]; gv = lambda k: zv[rr + k, cc]
        vv = gv(-4) & gv(0) & gv(1) & gv(5)
        sl, sr = (g(0) - g(-4)) / 4.0, (g(5) - g(1)) / 4.0
        o = (g(1) - 0.5 * sr) - (g(0) + 0.5 * sl)
        rr, cc, o = rr[vv], cc[vv], o[vv]
        rp = rr + r0; cp = cc; rq = rp + (1 if axis == 0 else 0); cq = cp + (1 if axis == 1 else 0)
        P.append(np.c_[rp, cp, rq, cq, DOM[rp, cp], DOM[rq, cq], np.full(len(rr), axis), o])
P = np.concatenate(P) if P else np.zeros((0, 8))
print(f"[pairs] {len(P)} hard hand-over pairs (share >= {a.share}) ({time.time() - t0:.0f}s)", flush=True)
# ---- smooth the offset along each seam ----
os_ = np.zeros(len(P))
if len(P):
    key = (P[:, 4].astype(np.int64) * 100000 + P[:, 5].astype(np.int64)) * 2 + P[:, 6].astype(np.int64)
    for k in np.unique(key):
        ix = np.flatnonzero(key == k)
        along = P[ix, 0] if P[ix[0], 6] == 1 else P[ix, 1]          # N-S seam (column pairs): order by row; E-W: by column
        o_ = np.argsort(along, kind="stable"); v = P[ix[o_], 7]
        if len(v) >= 3:
            n = min(a.along, len(v) | 1); v = uniform_filter1d(median_filter(v, size=n, mode="nearest"), n, mode="nearest")
        os_[ix[o_]] = v
# ---- correction field ----
C = np.zeros((H, W), np.float32); R = int(a.ramp)
SEEDS = {}
if len(P):
    for (rp, cp, rq, cq, fp, fq), v in zip(P[:, :6].astype(np.int64), os_):
        SEEDS.setdefault(int(fp), []).append((rp, cp, +v / 2)); SEEDS.setdefault(int(fq), []).append((rq, cq, -v / 2))
    SEEDS = {k: np.array(L) for k, L in SEEDS.items()}


def field(k, y0, y1, x0, x1):
    """frame k's correction on rows y0..y1, cols x0..x1: its nearest seed's value x max(0, 1 - d / ramp)"""
    L = SEEDS[k]; m = (L[:, 0] >= y0 - R) & (L[:, 0] < y1 + R) & (L[:, 1] >= x0 - R) & (L[:, 1] < x1 + R)
    if not m.any(): return None
    L = L[m]; a0_, a1_, b0_, b1_ = max(0, y0 - R), min(H, y1 + R), max(0, x0 - R), min(W, x1 + R)
    r_, c_ = L[:, 0].astype(int) - a0_, L[:, 1].astype(int) - b0_
    sv = np.zeros((a1_ - a0_, b1_ - b0_)); cnt = np.zeros_like(sv)
    np.add.at(sv, (r_, c_), L[:, 2]); np.add.at(cnt, (r_, c_), 1); seed = cnt > 0; sv[seed] /= cnt[seed]
    d, (iy, ix_) = distance_transform_edt(~seed, return_indices=True)
    f = sv[iy, ix_] * np.clip(1.0 - d / R, 0.0, None)
    return f[y0 - a0_:y1 - a0_, x0 - b0_:x1 - b0_]


def corr_chunk(rs):
    """blend of the per-frame fields with the leveller's effective weights (feather_blend formula) on rows rs..rs+chunk"""
    re_ = min(rs + a.chunk, H); h = re_ - rs
    parts = []
    for k, (r0, r1, c0, c1) in enumerate(SPAN):
        if r1 <= rs or r0 >= re_: continue
        with rasterio.open(FR[k]) as ds, WarpedVRT(ds, crs=CRS, transform=TR, width=W, height=H, resampling=Resampling.bilinear) as v:
            dem = v.read(1, window=Window(c0, rs, c1 - c0, h)); nd = ds.nodata
        with rasterio.open(WT[k]) as ws, WarpedVRT(ws, crs=CRS, transform=TR, width=W, height=H, resampling=Resampling.bilinear) as v:
            wt = v.read(1, window=Window(c0, rs, c1 - c0, h)).astype(np.float64)
        ok = np.isfinite(dem) & (wt > 0)
        if nd is not None and np.isfinite(nd): ok &= dem != nd
        if ok.any(): parts.append((k, c0, c1, np.where(ok, wt, 0.0)))
    wfull = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        if FAC[k] >= 1.0: wfull[:, c0:c1] += wt
    num = np.zeros((h, W)); den = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        f = FAC[k]; e = 1.0 if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull[:, c0:c1] / a.demote_ramp, 0.0, 1.0)
        ww = wt * e; den[:, c0:c1] += ww
        if k in SEEDS:
            fk = field(k, rs, re_, c0, c1)
            if fk is not None: num[:, c0:c1] += ww * fk
    with np.errstate(all="ignore"): return rs, np.where(den > 0, num / den, 0.0).astype(np.float32)


if len(P) and a.mode == "blend":
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.np) as pool:
        for rs, cc in pool.imap_unordered(corr_chunk, range(0, H, a.chunk)): C[rs:rs + len(cc)] = cc
    print(f"[corr] blended per-frame fields of {len(SEEDS)} frames ({time.time() - t0:.0f}s)", flush=True)
elif len(P):
    for k, L in SEEDS.items():
        r_, c_ = L[:, 0].astype(int), L[:, 1].astype(int)
        y0, y1, x0, x1 = max(0, r_.min() - R), min(H, r_.max() + R + 1), max(0, c_.min() - R), min(W, c_.max() + R + 1)
        own = DOM[y0:y1, x0:x1] == k
        C[y0:y1, x0:x1][own] += field(k, y0, y1, x0, x1)[own].astype(np.float32)
if len(P):
    # infill pixels (no dominant frame) near corrected ones: nearest corrected value, ramped
    for r0 in range(0, H, 4096):
        a0, a1 = max(0, r0 - R), min(H, r0 + 4096 + R)
        cs = C[a0:a1]; hole = (DOM[a0:a1] < 0) & Zv[a0:a1]
        if not hole.any() or not np.any(cs): continue
        d, (iy, ix_) = distance_transform_edt(DOM[a0:a1] < 0, return_indices=True)
        fillv = cs[iy, ix_] * np.clip(1.0 - d / R, 0.0, None)
        sl = slice(r0 - a0, r0 - a0 + min(4096, H - r0))
        C[r0:r0 + min(4096, H - r0)][hole[sl]] = fillv[sl][hole[sl]]
C[~Zv] = 0.0
out = np.where(Zv, Z + C, Z).astype(np.float32)
prof.update(dtype="float32", compress="deflate", tiled=True, blockxsize=256, blockysize=256, bigtiff="yes")
with rasterio.open(a.out + ".partial.tif", "w", **prof) as d: d.write(out, 1)
os.replace(a.out + ".partial.tif", a.out)
ch = C != 0
st = dict(box=a.box, surface=a.surface, pairs=int(len(P)), seams=int(len(np.unique(key))) if len(P) else 0,
          offset_abs_p50=float(np.percentile(np.abs(os_), 50)) if len(P) else 0, offset_abs_p90=float(np.percentile(np.abs(os_), 90)) if len(P) else 0,
          offset_abs_p99=float(np.percentile(np.abs(os_), 99)) if len(P) else 0, pairs_gt20=int((np.abs(os_) > 20).sum()),
          changed_px=int(ch.sum()), changed_gt1m=int((np.abs(C) > 1).sum()), max_abs_corr=float(np.abs(C).max()), secs=round(time.time() - t0),
          params=dict(mode=a.mode, ramp=a.ramp, share=a.share, along=a.along, demote_ramp=a.demote_ramp))
json.dump(st, open(a.out + ".json", "w"), indent=1)
np.save(a.out + ".pairs.npy", np.c_[P[:, :7], P[:, 7], os_].astype(np.float32))
print("[relax] " + json.dumps(st), flush=True)

#!/usr/bin/env python3
"""seam_stage_v3.py BOX SURFACE OUT.tif [options] : the v3 seam stage (2026-10-30, user: "Deploy these fixes now"). R1's seam method, src/inpainting/deseam_splice.py: feather-tied
Poisson solve + min-cut splice, with both fixes of 08-18 (rim fix: every changed pixel is written; PEDR datum term, per shot, mu = authority x lam x n_pix / n_shots, authority 0.01 = the 08-18 per-shot weight at this lam, screened orbits).
Building blocks are R1's, unchanged: poisson_mosaic.grad/div, poisson_datum.build_shot_operator/solve_tile_datum/choose_mu, seamline_mosaic.solve_useA.

What differs from R1 and why:
  Seam locations : from frame geometry only (user rule 10-30): dominant frame of the leveller's effective weights (factors + feathered demotion, as level_frames_box_v3.feather_blend). A line = connected
                   hand-overs of one frame pair, hard (share >= 0.9 on both sides) or narrow (share >= 0.9 reached within 16 px on each side). R1's census (aF on a 72 m max-pooled map, cov >= 2) is not used:
                   it cannot see 0-px joints, where cov = 1 on both sides. Hard joints (share >= 0.9 on both sides) are listed and measured but not solved: they go to the relaxation (PS below). Lines with |offset| < --min-off (2 m) are left as they are; that saves compute, it is not how seams are found.
  Mean gradient  : each frame's gradient counts with its effective weight, 0 where a factor-0 frame is out-ramped, so an excluded frame feeds nothing into G. R1: equal weights for all.
  Edges no frame covers: between two frames (0-px joint) the mean of the neighbouring edges (terrain slope carried across, offset not); next to infill the surface's own gradient. R1: G = 0, which flattens P across
                   infill (the drift of up to -300 m found on 10-23).
  Band           : changes stop at --band-px + --feather + 1 px (Euclidean) from the line, by construction. Pixels next to infill are forced to the surface, so a cut cannot end at an infill edge at full size.
  Running surface: lines are solved in batches whose footprints do not overlap, each from the surface left by the batches before it, so the result does not depend on --np. R1: all sites from one base.
  Guard          : per chunk of --chunk-along px along the line (a line can run 60 km, its problems are local), all of: median |offset| falls by >= 20 % or stays under 2 m; away from the seam being solved (> 3 px; the correction
                   must jump at that seam, that is the step going away; another frame's seam gets no exemption) no 1-px jump in the change larger than max(3 m, 0.25 x offset); with >= 5 fitted shots on moved pixels their MAE does not worsen by > 0.5 m; if the band exceeds
                   48 px, the 99th percentile of |change| beyond 48 px stays under max(5 m, 0.5 x offset). A chunk that fails keeps its surface. On top of that, per pixel along the seam (31 px running median) the correction is kept only where the local offset falls by >= 20 % or stays under 2 m. Accepted and rejected parts meet through a Gaussian taper (sigma 8 px).

Also writes, with --seam-prefix P: P_domid.tif (int16 dominant-frame index in sorted *_w.tif order, -1 = no frame) and P_mask.tif (uint8: bit1 frame<->frame, bit2 frame<->inpaint, dilated 2 px), the two files
R1's stage left for the uncertainty layer and seam_qc, and P_offset.tif (int16 dm: the frames' disagreement at every hand-over pixel before the fix, for the uncertainty layer's seam flag). Output: OUT.tif, OUT.json (summary), OUT.lines.tsv (per-line report)."""
import argparse, csv, glob, json, os, sys, time
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.ndimage import median_filter, uniform_filter1d
from pyproj import CRS, Transformer
os.environ.setdefault("PROJ_NETWORK", "OFF")
S = "/home/gridsan/groups/planetary_dtm/mars_dem/src/inpainting"; sys.path.insert(0, S)
from poisson_mosaic import grad
from poisson_datum import build_shot_operator, solve_tile_datum, choose_mu
from seamline_mosaic import solve_useA
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; GR = os.environ.get("BOXES_ROOT", f"{G}/global_run/boxes"); SHAPE = f"{G}/global_run/shape_error"; V3D = os.environ.get("V3D", f"{G}/global_run/v3_2026-10/downstream")
ap = argparse.ArgumentParser(); ap.add_argument("box"); ap.add_argument("surface"); ap.add_argument("out"); ap.add_argument("--tag", default=os.environ.get("T3", "ta_implicit_d1_spline100s1_e098v3"))
ap.add_argument("--band-px", type=int, default=32); ap.add_argument("--feather", type=int, default=4); ap.add_argument("--lam", type=float, default=1e-2); ap.add_argument("--halo", type=int, default=64)
ap.add_argument("--iters", type=int, default=600); ap.add_argument("--tol", type=float, default=1e-6); ap.add_argument("--lam-cut", type=float, default=1.0); ap.add_argument("--w-offset", type=float, default=1.0)
ap.add_argument("--authority", type=float, default=0.01, help="PEDR datum weight: mu = authority x lam x n_pix / n_shots was calibrated on 08-18 at lam 1e-4; at this stage's lam 1e-2, 0.01 reproduces that per-shot weight. At 1 the term stamps shots into the surface (E-060_N-50: high-passed change within 4 px of a shot rms 1.88 m against 0.52 at 0 and 0.53 at 0.01) and removes no more seam offset (narrow km above 20 m 9.7 / 10.6 / 9.4 / 9.3 at 1 / 0 / 0.01 / 0.1)."); ap.add_argument("--orbit-parity", type=int, default=-1); ap.add_argument("--no-datum", action="store_true"); ap.add_argument("--min-off", type=float, default=2.0); ap.add_argument("--min-px", type=int, default=5)
ap.add_argument("--np", type=int, default=16); ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--demote-ramp", type=float, default=512.0); ap.add_argument("--share", type=float, default=0.9); ap.add_argument("--near", type=int, default=16)
ap.add_argument("--chunk-along", type=int, default=128); ap.add_argument("--cache", default=None, help="dir for the dominant-frame cache <BOX>_dom.npy, <BOX>_share.npy"); ap.add_argument("--seam-prefix", default=None); ap.add_argument("--no-guard", action="store_true", help="accept every solved line (to see what the guard does)"); ap.add_argument("--solve-hard", action="store_true", help="also solve hard joints (0-1 px overlap); off by default, they zigzag")
a = ap.parse_args(); t0 = time.time(); B = a.box; T = a.tag; RD = f"{GR}/{B}/box_run"; FP = a.band_px + a.feather + 1; ND = -9999.0; TH = int(round(a.share * 250)); NB = a.near; R_ = None
assert a.halo >= FP, "halo must reach the footprint"
with rasterio.open(a.surface) as s: H, W, TR, CRS_ = s.height, s.width, s.transform, s.crs; prof = s.profile.copy(); Z = s.read(1).astype(np.float32); ndv = s.nodata
Zvalid = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4)
if ndv is not None: Zvalid &= Z != ndv
Z[~Zvalid] = np.nan
FR = sorted(glob.glob(f"{RD}/leveled_{T}/*_leveled.tif")); WT = [p.replace("_leveled.tif", "_w.tif") for p in FR]; STEM = [os.path.basename(p).split("_pred")[0] for p in FR]; fac = {}
wc = f"{V3D}/weights/{B}.csv"
if os.path.exists(wc):
    for r in csv.DictReader(open(wc)): fac[r["stem"]] = float(r["factor"])
FAC = np.array([min(fac.get(s_, 1.0), 1.0) for s_ in STEM]); SPAN = []; BND = []
for p in FR:
    with rasterio.open(p) as s: b = s.bounds
    BND.append(b); SPAN.append((max(0, int((TR.f - b.top) / TR.a) - 1), min(H, int((TR.f - b.bottom) / TR.a) + 1), max(0, int((b.left - TR.c) / TR.a) - 2), min(W, int(np.ceil((b.right - TR.c) / TR.a)) + 2)))


def rdw(path, win, rs=Resampling.bilinear):
    with rasterio.open(path) as ds, WarpedVRT(ds, crs=CRS_, transform=TR, width=W, height=H, resampling=rs) as v: return v.read(1, window=win), ds.nodata


def dom_chunk(rs):
    re_ = min(rs + a.chunk, H); h = re_ - rs; parts = []
    for k, (r0, r1, c0, c1) in enumerate(SPAN):
        if r1 <= rs or r0 >= re_: continue
        win = Window(c0, rs, c1 - c0, h); dem, nd = rdw(FR[k], win); wt = rdw(WT[k], win)[0].astype(np.float64); ok = np.isfinite(dem) & (wt > 0)
        if nd is not None and np.isfinite(nd): ok &= dem != nd
        if ok.any(): parts.append((k, c0, c1, np.where(ok, wt, 0.0)))
    wfull = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        if FAC[k] >= 1.0: wfull[:, c0:c1] += wt
    best = np.zeros((h, W)); bi = np.full((h, W), -1, np.int16); tot = np.zeros((h, W))
    for k, c0, c1, wt in parts:
        f = FAC[k]; e = 1.0 if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull[:, c0:c1] / a.demote_ramp, 0.0, 1.0); ww = wt * e; tot[:, c0:c1] += ww; sl = best[:, c0:c1]; m = ww > sl; sl[m] = ww[m]; bi[:, c0:c1][m] = k
    with np.errstate(all="ignore"): sh = np.where(tot > 0, best / tot, 0.0)
    return rs, bi, np.round(sh * 250).astype(np.uint8)


import multiprocessing as mp
dp, sp_ = (f"{a.cache}/{B}_dom.npy", f"{a.cache}/{B}_share.npy") if a.cache else (None, None)   # same files as seam_relax_v3.py, which then reuses them
if dp and os.path.exists(dp) and os.path.exists(sp_): DOM, SHR = np.load(dp), np.load(sp_); print(f"[dom] cached {dp}", flush=True)
else:
    DOM = np.zeros((H, W), np.int16); SHR = np.zeros((H, W), np.uint8)
    with mp.get_context("fork").Pool(a.np, maxtasksperchild=40) as pool:
        for rs, bi, sh in pool.imap_unordered(dom_chunk, range(0, H, a.chunk)): DOM[rs:rs + len(bi)] = bi; SHR[rs:rs + len(bi)] = sh
    if dp: np.save(dp + ".partial.npy", DOM); np.save(sp_ + ".partial.npy", SHR); os.replace(dp + ".partial.npy", dp); os.replace(sp_ + ".partial.npy", sp_)
print(f"[{B}] dominant frames: {len(FR)} frames, {(FAC < 1).sum()} with a factor < 1 ({(FAC == 0).sum()} at 0), {time.time() - t0:.0f}s", flush=True)


def write_seam(prefix):
    pr = dict(driver="GTiff", height=H, width=W, count=1, crs=CRS_, transform=TR, compress="deflate", tiled=True, blockxsize=256, blockysize=256, bigtiff="yes")
    with rasterio.open(prefix + "_domid.partial.tif", "w", dtype="int16", nodata=-1, **pr) as d: d.write(DOM, 1)
    with rasterio.open(PROV) as p_: prov = p_.read(1) if (p_.height, p_.width) == (H, W) else None
    bit1 = np.zeros((H, W), bool); dd = (DOM[:, :-1] != DOM[:, 1:]) & (DOM[:, :-1] >= 0) & (DOM[:, 1:] >= 0); bit1[:, :-1] |= dd; bit1[:, 1:] |= dd
    dd = (DOM[:-1, :] != DOM[1:, :]) & (DOM[:-1, :] >= 0) & (DOM[1:, :] >= 0); bit1[:-1, :] |= dd; bit1[1:, :] |= dd
    m = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, 4096):
        r1 = min(H, r0 + 4096); a0, a1 = max(0, r0 - 4), min(H, r1 + 4); e = ndi.binary_dilation(bit1[a0:a1], iterations=2)[r0 - a0:r1 - a0]; m[r0:r1][e] |= 1
    if prov is not None:
        bit2 = np.zeros((H, W), bool); pb = prov == 1; dd = pb[:, :-1] != pb[:, 1:]; bit2[:, :-1] |= dd; bit2[:, 1:] |= dd; dd = pb[:-1, :] != pb[1:, :]; bit2[:-1, :] |= dd; bit2[1:, :] |= dd
        for r0 in range(0, H, 4096):
            r1 = min(H, r0 + 4096); a0, a1 = max(0, r0 - 4), min(H, r1 + 4); e = ndi.binary_dilation(bit2[a0:a1], iterations=2)[r0 - a0:r1 - a0]; m[r0:r1][e] |= 2
    with rasterio.open(prefix + "_mask.partial.tif", "w", dtype="uint8", nodata=None, **pr) as d: d.write(m, 1)
    os.replace(prefix + "_domid.partial.tif", prefix + "_domid.tif"); os.replace(prefix + "_mask.partial.tif", prefix + "_mask.tif")


PROV = f"{RD}/box_dem_{T}_filled.tif.prov.tif"
if a.seam_prefix: write_seam(a.seam_prefix); print(f"[seam] wrote {a.seam_prefix}_domid.tif, _mask.tif {time.time() - t0:.0f}s", flush=True)
# ---- pairs and lines
PL = []
for axis in (1, 0):
    for b0 in range(0, H if axis == 1 else W, 2048):
        if axis == 1: b1 = min(H, b0 + 2048); d = DOM[b0:b1]; s_ = SHR[b0:b1]
        else: b1 = min(W, b0 + 2048); d = DOM[:, b0:b1].T; s_ = SHR[:, b0:b1].T
        Lx = d.shape[1]; A_, B_, sa, sb = d[:, :-1], d[:, 1:], s_[:, :-1], s_[:, 1:]; sw = (A_ != B_) & (A_ >= 0) & (B_ >= 0); ab = (lambda r_, c_: (r_ + b0, c_)) if axis == 1 else (lambda r_, c_: (c_, r_ + b0))
        hard = sw & (sa >= TH) & (sb >= TH); rr, cc = np.nonzero(hard); ok = (cc >= 5) & (cc + 6 < Lx); rr, cc = rr[ok], cc[ok]
        if len(rr): R_, C_ = ab(rr, cc); PL.append(np.c_[np.full(len(rr), axis), R_, C_, d[rr, cc], d[rr, cc + 1], np.zeros(len(rr)), np.ones(len(rr)), np.ones(len(rr))])
        cand = sw & ~hard; rr, cc = np.nonzero(cand); ok = (cc >= NB + 5) & (cc + NB + 6 < Lx); rr, cc = rr[ok], cc[ok]
        if not len(rr): continue
        p, q = d[rr, cc], d[rr, cc + 1]; jp = np.full(len(rr), -1); jq = np.full(len(rr), -1)
        for j in range(NB, -1, -1): m = (d[rr, cc - j] == p) & (s_[rr, cc - j] >= TH); jp[m] = j
        for j in range(NB + 1, 0, -1): m = (d[rr, cc + j] == q) & (s_[rr, cc + j] >= TH); jq[m] = j
        ok = (jp >= 0) & (jq >= 1); rr, cc, p, q, jp, jq = rr[ok], cc[ok], p[ok], q[ok], jp[ok], jq[ok]
        if len(rr): R_, C_ = ab(rr, cc); PL.append(np.c_[np.full(len(rr), axis), R_, C_, p, q, jp, jq, np.zeros(len(rr))])
P = np.concatenate(PL) if PL else np.zeros((0, 8)); del PL
np.save(a.out + ".pairs.npy", P.astype(np.float32))


def gv(Z_, P_, k, r0=0, c0=0):
    ax = P_[:, 0].astype(int); r = P_[:, 1].astype(np.int64) - r0; c = P_[:, 2].astype(np.int64) - c0; k = np.broadcast_to(np.asarray(k, np.int64), r.shape)
    rr = r + k * (ax == 0); cc = c + k * (ax == 1); ok = (rr >= 0) & (rr < Z_.shape[0]) & (cc >= 0) & (cc < Z_.shape[1]); o = np.full(len(r), np.nan, np.float64); o[ok] = Z_[rr[ok], cc[ok]]; return o


def offset_at(Z_, P_, r0=0, c0=0):
    jp, jq = P_[:, 5].astype(np.int64), P_[:, 6].astype(np.int64); zp0, zp4, zq0, zq4 = gv(Z_, P_, -jp, r0, c0), gv(Z_, P_, -jp - 4, r0, c0), gv(Z_, P_, jq, r0, c0), gv(Z_, P_, jq + 4, r0, c0)
    return (zq0 - (zq4 - zq0) / 4.0 * (jq - 0.5)) - (zp0 + (zp0 - zp4) / 4.0 * (jp + 0.5))


o0 = offset_at(Z, P) if len(P) else np.zeros(0)
PS = P[P[:, 7] < 0.5] if (len(P) and not a.solve_hard) else P   # hard joints (share >= 0.9 on both sides: 0-1 px overlap on a pixel staircase) are left to the relaxation. Every Poisson variant tested on 10-30 (5 gradient targets, 3 splices, lam 1e-2 to 1e-4, bands 32-64) turns their dashed meeting line into a zigzag chain; narrow seams have no staircase. In the 142-box census hard seams above 20 m were 1 km against 95 km narrow.
LINES = []
if len(PS):
    key = (PS[:, 3].astype(np.int64) * 100000 + PS[:, 4].astype(np.int64)) * 2 + PS[:, 0].astype(np.int64); o_all = np.argsort(key, kind="stable"); ks = key[o_all]; st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]); en = np.r_[st[1:], len(ks)]; oS = offset_at(Z, PS)
    for s_, e_ in zip(st, en):
        ix = o_all[s_:e_]; al = PS[ix, 1] if PS[ix[0], 0] == 1 else PS[ix, 2]; o_ = np.argsort(al, kind="stable"); ix, al = ix[o_], al[o_]; g = np.r_[0, np.cumsum(np.diff(al) > 16)]
        for gi in range(int(g[-1]) + 1):
            jx = ix[g == gi]
            if len(jx) < a.min_px: continue
            m = np.nanmedian(np.abs(oS[jx])) if np.isfinite(oS[jx]).any() else np.nan
            r_, c_ = PS[jx, 1].astype(int), PS[jx, 2].astype(int); LINES.append(dict(ix=jx, r0=int(r_.min()), r1=int(r_.max()) + 1, c0=int(c_.min()), c1=int(c_.max()) + 1, off=float(m) if np.isfinite(m) else 0.0))
if a.seam_prefix and len(P):   # the measured disagreement between the two frames at every hand-over pixel, before any fix: int16 decimetres, q side minus p side, nodata -32768. For the uncertainty layer's seam flag.
    OFF = np.full((H, W), -32768, np.int16); ov = np.clip(np.nan_to_num(o0, nan=-3276.8) * 10.0, -32767, 32767).astype(np.int16); jp_, jq_ = P[:, 5].astype(np.int64), P[:, 6].astype(np.int64)
    for j in range(0, int((jp_ + jq_).max()) + 1):
        k = np.minimum(-jp_ + j, jq_); rr = P[:, 1].astype(np.int64) + k * (P[:, 0] == 0); cc = P[:, 2].astype(np.int64) + k * (P[:, 0] == 1); ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W); OFF[rr[ok], cc[ok]] = ov[ok]
    with rasterio.open(a.seam_prefix + "_offset.partial.tif", "w", dtype="int16", nodata=-32768, driver="GTiff", height=H, width=W, count=1, crs=CRS_, transform=TR, compress="deflate", tiled=True, blockxsize=256, blockysize=256, bigtiff="yes") as d: d.write(OFF, 1)
    os.replace(a.seam_prefix + "_offset.partial.tif", a.seam_prefix + "_offset.tif"); del OFF, ov; print(f"[seam] wrote {a.seam_prefix}_offset.tif {time.time() - t0:.0f}s", flush=True)
gated = [l for l in LINES if l["off"] >= a.min_off]; print(f"[{B}] pairs {len(P)} ({len(P) * 0.018:.0f} km; hard {int((P[:, 7] > 0.5).sum()) * 0.018:.0f} km left to the relaxation), lines {len(LINES)}, to solve (|offset| >= {a.min_off} m) {len(gated)} ({sum(len(l['ix']) for l in gated) * 0.018:.0f} km) {time.time() - t0:.0f}s", flush=True)
# ---- batches of lines with non-overlapping footprints, longest first
gated.sort(key=lambda l: -len(l["ix"])); batches = []; bbs = []
for li, l in enumerate(gated):
    bb = (l["r0"] - FP, l["r1"] + FP, l["c0"] - FP, l["c1"] + FP)
    for b_, bl in zip(batches, bbs):
        if all(bb[1] <= o[0] or bb[0] >= o[1] or bb[3] <= o[2] or bb[2] >= o[3] for o in bl): b_.append(li); bl.append(bb); break
    else: batches.append([li]); bbs.append([bb])
print(f"[{B}] batches {len(batches)} {time.time() - t0:.0f}s", flush=True)
# ---- PEDR (per-shot datum term; as level_frames_box: ELEV, physical bounds, orbit screen)
SHOTS = None; LL = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
if not a.no_datum:
    PC = f"{G}/global_run/fix_2026-10/r2_delivery/pedr_census"; scr = f"{SHAPE}/{B}/orbit_screen_{B}.txt"; bad = np.array(open(scr).read().split(), np.int64) if os.path.exists(scr) else np.zeros(0, np.int64); sp = f"{PC}/shots/{B}.shots.npz"
    if os.path.exists(sp):   # the PEDR census's shots of this box (sqlite extract with the box's bounds, ELEV within -8500..22000) and its drop masks (radially offset orbits, bad orbit segments)
        sz = np.load(sp); n = len(sz["elev"]); keep = np.ones(n, bool); ok_ = next((k for k in sz.files if k.lower().startswith("orb")), None); orb = sz[ok_].astype(np.int64) if ok_ else np.zeros(n, np.int64)
        if os.path.exists(f"{PC}/cells/{B}.drop.npz"): D_ = np.load(f"{PC}/cells/{B}.drop.npz"); keep = ~(np.unpackbits(D_["orbit"])[:n] | np.unpackbits(D_["seg"])[:n]).astype(bool)
        keep &= ~np.isin(orb, bad)
        if a.orbit_parity >= 0: keep &= (orb % 2) == a.orbit_parity
        x_, y_ = Transformer.from_crs(LL, CRS_, always_xy=True).transform(sz["lon"][keep].astype(float), sz["lat"][keep].astype(float)); x_, y_ = np.asarray(x_), np.asarray(y_); ze = sz["elev"][keep].astype(np.float64)
        rr_ = np.floor((y_ - TR.f) / TR.e).astype(np.int64); cc_ = np.floor((x_ - TR.c) / TR.a).astype(np.int64); ok = (rr_ >= 0) & (rr_ < H) & (cc_ >= 0) & (cc_ < W); SHOTS = (rr_[ok], cc_[ok], ze[ok])
        print(f"[{B}] PEDR census shots {int(ok.sum())} of {n} (parity {a.orbit_parity}, {len(bad)} orbits in the screen) {time.time() - t0:.0f}s", flush=True)
    else: print(f"[{B}] no PEDR census shots: plain solve", flush=True)
sig = 168.0 / 2.3548 / 18.0; yk, xk = np.mgrid[-12:13, -12:13]; KERN = np.exp(-(xk ** 2 + yk ** 2) / (2 * sig ** 2)); KERN /= KERN.sum()
with rasterio.open(PROV) as p_: PROV_SAME = (p_.height, p_.width) == (H, W)


def solve_line(li):
    l = gated[li]; r0, r1, c0, c1 = l["r0"], l["r1"], l["c0"], l["c1"]; hr0, hr1 = max(0, r0 - a.halo), min(H, r1 + a.halo); hc0, hc1 = max(0, c0 - a.halo), min(W, c1 + a.halo); th, tw = hr1 - hr0, hc1 - hc0
    win = Window(hc0, hr0, tw, th); F = Z[hr0:hr1, hc0:hc1].astype(np.float64); Fv = np.isfinite(F); Q = PS[l["ix"]]; rec = dict(line=li, p=STEM[int(Q[0, 3])], q=STEM[int(Q[0, 4])], axis=int(Q[0, 0]), km=round(len(Q) * 0.018, 3), row=int(Q[len(Q) // 2, 1]), col=int(Q[len(Q) // 2, 2]), off_before=round(l["off"], 2))
    try:
        with rasterio.open(PROV) as p_: infill = (p_.read(1, window=win) == 1) if PROV_SAME else np.zeros((th, tw), bool)
        px_ = TR.a; y1, y0 = TR.f - hr0 * px_, TR.f - hr1 * px_; x0, x1 = TR.c + hc0 * px_, TR.c + hc1 * px_; parts = []
        for k, b in enumerate(BND):
            if b.right <= x0 or b.left >= x1 or b.top <= y0 or b.bottom >= y1: continue
            A_, nd = rdw(FR[k], win); wt = rdw(WT[k], win)[0].astype(np.float64); ok = np.isfinite(A_) & (wt > 0)
            if nd is not None and np.isfinite(nd): ok &= A_ != nd
            if ok.any(): parts.append((k, A_.astype(np.float64), ok, wt))
        wfull = np.zeros((th, tw)); fc = np.zeros((th, tw), bool)
        for k, A_, ok, wt in parts:
            fc |= ok
            if FAC[k] >= 1.0: wfull += np.where(ok, wt, 0.0)
        Gx, Gy, Nx, Ny = np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw)), np.zeros((th, tw))
        for k, A_, ok, wt in parts:
            f = FAC[k]; e = np.ones((th, tw)) if f >= 1.0 else f + (1.0 - f) * np.clip(1.0 - wfull / a.demote_ramp, 0.0, 1.0); gx, gy = grad(np.where(ok, A_, 0.0)); vx = ok.copy(); vx[:, :-1] &= ok[:, 1:]; vx[:, -1] = False; vy = ok.copy(); vy[:-1, :] &= ok[1:, :]; vy[-1, :] = False
            Gx += np.where(vx, e * gx, 0.0); Nx += np.where(vx, e, 0.0); Gy += np.where(vy, e * gy, 0.0); Ny += np.where(vy, e, 0.0)
        Fg = np.where(Fv, F, np.nanmedian(F)); fx, fy = grad(Fg); Mx = np.where(Nx > 0, Gx / np.maximum(Nx, 1e-9), 0.0); My = np.where(Ny > 0, Gy / np.maximum(Ny, 1e-9), 0.0)
        bx_ = fc.copy(); bx_[:, :-1] &= fc[:, 1:]; bx_[:, -1] = False; by_ = fc.copy(); by_[:-1, :] &= fc[1:, :]; by_[-1, :] = False
        ex = 0.5 * (np.roll(Mx, 1, 1) + np.roll(Mx, -1, 1)); ey = 0.5 * (np.roll(My, 1, 0) + np.roll(My, -1, 0))
        Gxf = np.where(Nx > 0, Mx, np.where(bx_, ex, fx)); Gyf = np.where(Ny > 0, My, np.where(by_, ey, fy)); Pz = None; rec.update(shots_fit=0, cg=-1)
        if SHOTS is not None and a.authority > 0:
            rr_, cc_, zz = SHOTS; m = (rr_ >= hr0) & (rr_ < hr1) & (cc_ >= hc0) & (cc_ < hc1)
            if m.sum() >= 4:
                Aop, keep = build_shot_operator(F.shape, rr_[m] - hr0, cc_[m] - hc0, KERN)
                if Aop is not None and Aop.shape[0] >= 4:
                    mu = choose_mu(a.lam, Aop.shape[0], F.size, authority=a.authority); Pz, inf = solve_tile_datum(Gxf, Gyf, Fg, a.lam, a.iters, a.tol, A=Aop, z=zz[m][keep], mu=mu, mode="shots", valid=Fv); rec.update(shots_fit=int(Aop.shape[0]), cg=int(inf))
        if Pz is None: Pz, inf = solve_tile_datum(Gxf, Gyf, Fg, a.lam, a.iters, a.tol); rec["cg"] = int(inf)
        seam = np.zeros((th, tw), bool); lab = np.zeros((th, tw), np.int32); pix = np.zeros((th, tw), np.int32); jp, jq = Q[:, 5].astype(np.int64), Q[:, 6].astype(np.int64); al = Q[:, 1] if Q[0, 0] == 1 else Q[:, 2]; cid = ((al - al.min()) // a.chunk_along).astype(np.int64); pid = np.arange(len(Q)) + 1
        for j in range(0, int((jp + jq).max()) + 1):
            k = np.minimum(-jp + j, jq); rr = Q[:, 1].astype(np.int64) + k * (Q[:, 0] == 0) - hr0; cc = Q[:, 2].astype(np.int64) + k * (Q[:, 0] == 1) - hc0; ok = (rr >= 0) & (rr < th) & (cc >= 0) & (cc < tw); seam[rr[ok], cc[ok]] = True; lab[rr[ok], cc[ok]] = cid[ok] + 1; pix[rr[ok], cc[ok]] = pid[ok]
        dseam, (iy, ix) = ndi.distance_transform_edt(~seam, return_indices=True); LC = lab[iy, ix]; LP = pix[iy, ix]; valid = Fv & fc & ~infill; band = (dseam <= a.band_px) & valid; inner = band & (dseam <= 1.5); outside = ~band & (Fv | infill); outer = band & ndi.binary_dilation(outside | infill, iterations=1) & ~inner
        if not (inner.any() and outer.any()): rec.update(verdict="no_band", fails=""); return rec, None
        useP = solve_useA(Pz, Fg, band, inner, outer, a.lam_cut, a.w_offset); aHard = seam | (band & useP); sd = ndi.distance_transform_edt(aHard) - ndi.distance_transform_edt(~aHard); alpha = np.clip(0.5 + sd / (2.0 * max(a.feather, 1)), 0.0, 1.0)
        alpha = np.where((dseam <= a.band_px + a.feather + 1) & valid, alpha, 0.0); alpha = np.where(seam & valid, 1.0, alpha); out = np.where(valid, alpha * Pz + (1.0 - alpha) * Fg, F); C = np.nan_to_num(out - F)
        ob = offset_at(F, Q, hr0, hc0); on = offset_at(out, Q, hr0, hc0); mb = float(np.nanmedian(np.abs(ob))) if np.isfinite(ob).any() else l["off"]; mn = float(np.nanmedian(np.abs(on))) if np.isfinite(on).any() else mb
        # away from the seam being solved the correction must be continuous, also across other frames' seams: a correction that stops on another seam leaves a new step there
        away = ~ndi.binary_dilation(seam, iterations=3); jx_ = np.where(away[:, :-1] & away[:, 1:], np.abs(np.diff(C, axis=1)), 0.0); jy_ = np.where(away[:-1, :] & away[1:, :], np.abs(np.diff(C, axis=0)), 0.0); J_ = np.zeros((th, tw)); J_[:, :-1] = jx_; J_[:-1, :] = np.maximum(J_[:-1, :], jy_)
        nch = int(cid.max()) + 1; okc = np.ones(nch + 1, bool); okc[0] = False; fails = []; jmax = np.zeros(nch + 1); ndi_max = ndi.maximum(J_, LC, index=np.arange(nch + 1)); jmax[:] = np.nan_to_num(np.asarray(ndi_max, float)); far = (dseam > 48) & (np.abs(C) > 0)
        sv = None
        if SHOTS is not None:
            rr_, cc_, zz = SHOTS; m = (rr_ >= hr0 + 13) & (rr_ < hr1 - 13) & (cc_ >= hc0 + 13) & (cc_ < hc1 - 13)
            if m.any():
                r2, c2, z2 = rr_[m] - hr0, cc_[m] - hc0, zz[m]; mv = np.abs(C[r2, c2]) > 1; r2, c2, z2 = r2[mv], c2[mv], z2[mv]
                if len(r2):
                    pb = F[r2[:, None, None] + yk[None], c2[:, None, None] + xk[None]]; pn = out[r2[:, None, None] + yk[None], c2[:, None, None] + xk[None]]; w_ = np.isfinite(pb) * KERN[None]; ws = np.maximum(w_.sum((1, 2)), 1e-9)
                    sv = (LC[r2, c2], np.abs(np.nansum(np.nan_to_num(pb) * w_, (1, 2)) / ws - z2), np.abs(np.nansum(np.nan_to_num(pn) * w_, (1, 2)) / ws - z2))
        nf = dict(seam=0, cliff=0, reach=0, pedr=0); csz = np.bincount(cid, minlength=nch)
        # per pixel along the seam (31 px running median): the correction is kept only where the local offset falls by >= 20 % or stays under 2 m. A chunk median hides a part of the chunk where a 0 m seam becomes a 14 m one
        # (E-030_N-40 r3307 c26032) or a 1 m seam an 11 m one (E-060_N-50 r6623 c13221), and a correction kept there is a new step, not a fix
        nw = min(31, len(Q) | 1); obs = median_filter(np.nan_to_num(np.abs(ob), nan=0.0), size=nw, mode="nearest"); ons = median_filter(np.nan_to_num(np.abs(on), nan=0.0), size=nw, mode="nearest"); pok = ons <= np.maximum(0.8 * obs, 2.0)
        for c in range(nch):
            sel = cid == c; oc = float(np.nanmedian(np.abs(ob[sel]))) if np.isfinite(ob[sel]).any() else 0.0; nc = float(np.nanmedian(np.abs(on[sel]))) if np.isfinite(on[sel]).any() else oc; bad_ = []
            if nc > max(0.8 * oc, 2.0): bad_.append("seam")
            if jmax[c + 1] > max(3.0, 0.25 * oc): bad_.append("cliff")
            if a.band_px > 48:
                f_ = far & (LC == c + 1)
                if f_.any() and float(np.percentile(np.abs(C[f_]), 99)) > max(5.0, 0.5 * oc): bad_.append("reach")
            if sv is not None:
                s_ = sv[0] == c + 1
                if s_.sum() >= 5 and sv[2][s_].mean() > sv[1][s_].mean() + 0.5: bad_.append("pedr")
            if bad_ and not a.no_guard:
                okc[c + 1] = False
                for b__ in bad_: nf[b__] += 1
        if a.no_guard: pok[:] = True
        acc = okc[cid + 1] & pok; kmok = float(acc.sum()) * 0.018; pxrej = int((okc[cid + 1] & ~pok).sum()); Afl = (okc[LC] & np.r_[False, pok][LP]).astype(np.float64); allok = bool(acc.all()); noneok = not acc.any()
        if noneok: rec.update(off_after=round(mb, 2), km_accepted=0.0, chunks=nch, chunks_ok=0, verdict="rejected", fails="+".join(k for k, v in nf.items() if v), nf_seam=nf["seam"], nf_cliff=nf["cliff"], nf_reach=nf["reach"], nf_pedr=nf["pedr"]); return rec, None
        Af = Afl if allok else ndi.gaussian_filter(Afl, 8.0); Cf = Af * C; out = np.where(valid, F + Cf, F); on2 = offset_at(out, Q, hr0, hc0); mn2 = float(np.nanmedian(np.abs(on2))) if np.isfinite(on2).any() else mb
        cliff = float(max((np.where(away[:, :-1] & away[:, 1:], np.abs(np.diff(Cf, axis=1)), 0.0)).max(), (np.where(away[:-1, :] & away[1:, :], np.abs(np.diff(Cf, axis=0)), 0.0)).max())); sb = sn = None; nsh = 0
        if sv is not None:
            s_ = (Af[r2, c2] > 0.5); nsh = int(s_.sum())
            if nsh: sb, sn = float(sv[1][s_].mean()), float((np.abs(np.nansum(np.nan_to_num(out[r2[:, None, None] + yk[None], c2[:, None, None] + xk[None]]) * w_, (1, 2)) / ws - z2))[s_].mean())
        rec.update(off_before=round(mb, 2), off_after=round(mn2, 2), px_changed=int((np.abs(Cf) > 0.01).sum()), c_max=round(float(np.abs(Cf).max()), 2), cliff=round(cliff, 2), km_accepted=round(kmok, 3), chunks=nch, chunks_ok=int(okc[1:].sum()), shots_moved=nsh, shot_before=None if sb is None else round(sb, 2), shot_after=None if sn is None else round(sn, 2),
                   nf_seam=nf["seam"], nf_cliff=nf["cliff"], nf_reach=nf["reach"], nf_pedr=nf["pedr"], px_rej=pxrej, fails="+".join(k for k, v in nf.items() if v), verdict="accepted" if allok else "partial")
        ch = np.abs(Cf) > 1e-4; rr, cc = np.nonzero(ch); return rec, (rr + hr0, cc + hc0, out[rr, cc].astype(np.float32))
    except Exception as e:
        rec.update(verdict="error", fails=str(e)[:100]); return rec, None


RES = []
for bi, b_ in enumerate(batches):
    wrs = []   # applied after the batch: maxtasksperchild re-forks workers mid-batch, so a write during the batch would reach later lines' halos and the result would depend on --np and arrival order
    with mp.get_context("fork").Pool(min(a.np, max(1, len(b_))), maxtasksperchild=20) as pool:
        for rec, wr in pool.imap_unordered(solve_line, b_, chunksize=1):
            RES.append(rec)
            if wr is not None: wrs.append(wr)
    for wr in wrs: Z[wr[0], wr[1]] = wr[2]
    del wrs
    if bi % 10 == 0 or bi == len(batches) - 1: print(f"[{B}] batch {bi + 1}/{len(batches)}: {sum(r['verdict'] == 'accepted' for r in RES)} accepted, {sum(r['verdict'] == 'partial' for r in RES)} partial, {sum(r['verdict'] == 'rejected' for r in RES)} rejected, {sum(r.get('km_accepted', 0.0) for r in RES):.0f} km accepted, {time.time() - t0:.0f}s", flush=True)
out_ = np.where(np.isfinite(Z), Z, ndv if ndv is not None else ND).astype(np.float32); prof.update(dtype="float32", compress="deflate", tiled=True, blockxsize=256, blockysize=256, bigtiff="yes"); prof.pop("interleave", None)
with rasterio.open(a.out + ".partial.tif", "w", **prof) as d: d.write(out_, 1)
os.replace(a.out + ".partial.tif", a.out)
summ = dict(box=B, pairs=int(len(P)), seam_km=round(len(P) * 0.018, 1), lines=len(LINES), solved=len(RES), km_solved=round(sum(r["km"] for r in RES), 1), km_accepted=round(sum(r.get("km_accepted", 0.0) for r in RES), 1), lines_accepted=sum(r["verdict"] == "accepted" for r in RES), lines_partial=sum(r["verdict"] == "partial" for r in RES), lines_rejected=sum(r["verdict"] == "rejected" for r in RES),
            errors=sum(r["verdict"] == "error" for r in RES), no_band=sum(r["verdict"] == "no_band" for r in RES), chunks=sum(r.get("chunks", 0) for r in RES), chunks_ok=sum(r.get("chunks_ok", 0) for r in RES), chunk_fails={k: sum(r.get(f"nf_{k}", 0) for r in RES) for k in ("seam", "cliff", "reach", "pedr")}, batches=len(batches), secs=round(time.time() - t0), params={k: v for k, v in vars(a).items() if k not in ("box", "surface", "out")})
json.dump(summ, open(a.out + ".json", "w"), indent=1)
keys = ["line", "p", "q", "axis", "km", "km_accepted", "chunks", "chunks_ok", "px_rej", "row", "col", "off_before", "off_after", "px_changed", "c_max", "cliff", "shots_fit", "cg", "shots_moved", "shot_before", "shot_after", "verdict", "fails"]
with open(a.out + ".lines.tsv", "w") as f:
    f.write("\t".join(keys) + "\n")
    for r in sorted(RES, key=lambda r: r["line"]): f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
print("[done]", {k: v for k, v in summ.items() if k != "params"}, flush=True)

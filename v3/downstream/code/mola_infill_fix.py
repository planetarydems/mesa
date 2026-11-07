#!/usr/bin/env python3
"""mola_infill_fix.py BOX SURFACE OUT_DIR [--t-polar 30] [--t-other 150] [--polar-lat 60] -- infill-vs-MOLA rule, v2.1 (2026-10-27).

v2.2 (2026-10-30): each region writes only its own pixels (wf > 0) into the running output; v2.1 wrote its whole window from the input surface,
which reset earlier regions' corrections where windows overlap. Nothing else changes.
v2.1 replaces v1 (2026-10-24, kept in backup_<date>/), whose whole-cell replacement made straight-edged 2.3 km blocks (polar BA13,
BA01, BA03; census: 275 blocks in 44 of 54 boxes). Same thresholds (30 m poleward of 60 deg, 150 m elsewhere) and outputs. Changes:
 1. no 128 px cell blocks: the replacement weight is a smooth function of the infill's departure from MOLA. d_s = (surface - MOLA)
    over infill pixels, nan-aware Gaussian (--sigma 32 px) on the 8 px grid; w = clip((|d_s| - T/2) / T, 0, 1) where the smoothed
    infill share >= 0.25, then smoothed again (--sigma) so no rings or islands remain, times a --feather px ramp from non-infill pixels.
 2. robust edge matching: the residual r = surface - MOLA of the kept pixels is filled into the patch (pull-push) twice, with all
    kept pixels (r_full) and without kept pixels more than max(4 T, 120 m) off MOLA (r_robust); r = r_full next to kept pixels,
    handing over to r_robust over --res-ramp 64 px, so the patch meets every kept edge without a step while a frame edge
    hundreds of metres off cannot drag the whole patch.
 New surface on infill pixels = w (MOLA + r) + (1 - w) surface. Mask (OUT_mask.tif) = w >= 0.5.
Tested 2026-10-27 on 8 boxes (3 block sites + polar and mid-latitude controls): blocks removed; QA step px unchanged; PEDR failing
cells and shot MAE equal or better (E-020_Np80 31 -> 18 cells, 2.83 -> 2.62 m).
v1 description follows.

Inpainting-net infill (prov == 1) that departs from MOLA is replaced by an edge-matched MOLA surface. Per 128 px cell (2.3 km) with
>= 25 % infill pixels: d = median(surface - MOLA) over its infill pixels (8 px sampling). A cell is replaced when |d| > T, with
T = 30 m where |lat| >= 60 deg and 150 m elsewhere (tested against the PEDR census shots: polar infill cells beyond 30 m improve
in 88 % of cases, failing cells 1,946 -> 458; at lower latitudes MOLA beats the infill only beyond ~150 m). The replaced pixels
(the infill pixels of the selected cells) get MOLA + r (ramped in over --feather 24 px from the hole edge), where r = surface - MOLA outside them, filled into each hole by pull-push
(masked pyramid averaging), so the result meets the kept surface without a step. MOLA: 463 m mosaic, point registered,
bilinear (as gm_common.Mola), evaluated every 8 px and upsampled bilinearly.
Writes OUT_DIR/<BOX>/box_run/<basename of SURFACE without .tif>_mif.tif and ..._mif_mask.tif (1 = replaced), + .json stats."""
import argparse, json, math, os, sys, time
import numpy as np, rasterio
from rasterio.windows import Window
from scipy.ndimage import label, find_objects, zoom, binary_dilation, distance_transform_edt, gaussian_filter
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; sys.path.insert(0, f"{G}/src")
from proj_helpers import parse_crs, xy_to_lonlat
TAG = "ta_implicit_d1_implrel100s1_e088r2"
MOLA = f"{G}/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff"; R = 3396190.0; DEG = math.pi * R / 180.0; CELL = 128; D = 8


class Mola:
    def __init__(self):
        self.S = rasterio.open(MOLA); t = self.S.transform; self.t = t; self.W, self.H = self.S.width, self.S.height; self.nd = self.S.nodata
        self.cache = {}

    def at(self, lon, lat):
        out = np.full(lon.shape, np.nan)
        for la0 in np.arange(np.floor(np.nanmin(lat)), np.nanmax(lat) + 1, 1.0):
            m = (lat >= la0) & (lat < la0 + 1)
            if not m.any(): continue
            sx = 2 * math.pi * R / self.W
            fx = np.mod(lon[m] * DEG - self.t.c, 2 * math.pi * R) / sx; fy = np.clip((self.t.f - lat[m] * DEG) / self.t.a, 0, self.H - 1.001)
            y0 = np.floor(fy).astype(int); r0, r1 = y0.min(), min(y0.max() + 2, self.H)
            a = self.S.read(1, window=Window(0, r0, self.W, r1 - r0)).astype(np.float64)
            if a.shape[0] < y0.max() + 2 - r0: a = np.vstack([a, a[-1:]])
            a[a == self.nd] = np.nan
            x0 = np.floor(fx).astype(int) % self.W; x1 = (x0 + 1) % self.W; ax = fx - np.floor(fx); ay = fy - y0; yy = y0 - r0
            out[m] = (a[yy, x0] * (1 - ax) + a[yy, x1] * ax) * (1 - ay) + (a[yy + 1, x0] * (1 - ax) + a[yy + 1, x1] * ax) * ay
        return out


def pullpush(v, m):
    """fill v where m is False from v where m is True (masked pyramid average, then upsampling into the gaps)"""
    if m.all() or not m.any(): return v
    levels = [(np.where(m, v, 0.0), m.astype(float))]
    while levels[-1][1].shape[0] > 1 and levels[-1][1].shape[1] > 1 and not (levels[-1][1] > 0).all():
        s, w = levels[-1]; h2, w2 = (s.shape[0] + 1) // 2 * 2, (s.shape[1] + 1) // 2 * 2
        sp = np.zeros((h2, w2)); wp = np.zeros((h2, w2)); sp[:s.shape[0], :s.shape[1]] = s * w; wp[:s.shape[0], :s.shape[1]] = w
        S_ = sp.reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3)); W_ = wp.reshape(h2 // 2, 2, w2 // 2, 2).sum((1, 3))
        levels.append((np.where(W_ > 0, S_ / np.maximum(W_, 1e-12), 0.0), np.minimum(W_, 1.0)))
    est = levels[-1][0]
    for s, w in reversed(levels[:-1]):
        up = zoom(est, 2, order=1)[:s.shape[0], :s.shape[1]] if est.shape != s.shape else est
        if up.shape != s.shape:
            up2 = np.zeros(s.shape); up2[:up.shape[0], :up.shape[1]] = up; up2[up.shape[0]:, :] = up2[up.shape[0] - 1:up.shape[0], :]; up = up2
        est = w * s + (1 - w) * up
    return np.where(m, v, est)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("box"); ap.add_argument("surface"); ap.add_argument("out_dir")
    ap.add_argument("--t-polar", type=float, default=30.0); ap.add_argument("--t-other", type=float, default=150.0)
    ap.add_argument("--polar-lat", type=float, default=60.0); ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--prov", default=None, help="provenance raster (default: box_run/box_dem_<TAG env or R2>_filled.tif.prov.tif)")
    ap.add_argument("--out", default=None, help="output surface path (default OUT_DIR/<BOX>/box_run/<surface>_mif.tif)")
    ap.add_argument("--feather", type=float, default=24.0, help="px over which the replacement ramps in from the hole edge")
    ap.add_argument("--smooth", type=float, default=8.0, help="Gaussian sigma (px) of the filled residual inside the hole")
    ap.add_argument("--sigma", type=float, default=32.0, help="v2: Gaussian sigma (px) of the infill departure from MOLA and of the weight")
    ap.add_argument("--res-ramp", type=float, default=64.0, help="v2: px from kept pixels over which the full edge residual hands over to the robust one")
    a = ap.parse_args(); t0 = time.time()
    RD = f"{G}/global_run/boxes/{a.box}/box_run"; OD = f"{a.out_dir}/{a.box}/box_run"; os.makedirs(OD, exist_ok=True)
    S = rasterio.open(a.surface); P = rasterio.open(a.prov or f"{RD}/box_dem_{os.environ.get('TAG', TAG)}_filled.tif.prov.tif")
    assert S.transform == P.transform and S.shape == P.shape, "surface and provenance grids differ"
    kind, _R, lon0, lat_ts = parse_crs(S.crs); tr = S.transform; H, W = S.shape; nd = S.nodata; M = Mola()
    # 1. coarse (8 px) departure of the infill from MOLA over the whole box, smoothed; weight w_c
    Hc, Wc = (H + D - 1) // D, (W + D - 1) // D
    dev = np.full((Hc, Wc), np.nan); latc = np.zeros((Hc, Wc)); inf = np.zeros((Hc, Wc), bool); stats = dict(coarse_infill=0)
    B = 4096
    for br in range(0, H, B):
        for bc in range(0, W, B):
            h, w = min(B, H - br), min(B, W - bc)
            pv = P.read(1, window=Window(bc, br, w, h))[::D, ::D]
            if not (pv == 1).any(): continue
            z = S.read(1, window=Window(bc, br, w, h))[::D, ::D].astype(np.float64)
            rr, cc = np.mgrid[br:br + h:D, bc:bc + w:D] + D / 2
            x, y = tr * (cc, rr); lo, la = xy_to_lonlat(np.asarray(x).ravel(), np.asarray(y).ravel(), kind, lon0, lat_ts)
            lo = np.asarray(lo).reshape(z.shape); la = np.asarray(la).reshape(z.shape)
            m = (pv == 1) & (z > -1e4) & (z != nd)
            mo = np.full(z.shape, np.nan); mo[m] = M.at(lo[m], la[m])
            i0, j0 = br // D, bc // D
            dev[i0:i0 + z.shape[0], j0:j0 + z.shape[1]] = np.where(m & np.isfinite(mo), z - mo, np.nan)
            latc[i0:i0 + z.shape[0], j0:j0 + z.shape[1]] = la; inf[i0:i0 + z.shape[0], j0:j0 + z.shape[1]] = m
    ok = np.isfinite(dev); sg = a.sigma / D
    num = gaussian_filter(np.where(ok, dev, 0.0), sg); den = gaussian_filter(ok.astype(float), sg)
    with np.errstate(all="ignore"): ds = np.where(den > 0.05, num / den, np.nan)
    Tc = np.where(np.abs(latc) >= a.polar_lat, a.t_polar, a.t_other)
    wc = np.where(inf & (den >= 0.25) & np.isfinite(ds), np.clip((np.abs(np.nan_to_num(ds)) - 0.5 * Tc) / Tc, 0.0, 1.0), 0.0)
    wc = np.where(inf, np.clip(gaussian_filter(wc, sg), 0.0, 1.0), 0.0); wc[wc < 0.02] = 0.0     # v2.1: no rings / islands
    stats.update(coarse_infill=int(inf.sum()), coarse_w_pos=int((wc > 0).sum()), coarse_w_full=int((wc >= 1).sum()))
    print(f"[{a.box}] coarse infill px {stats['coarse_infill']}, weight > 0 {stats['coarse_w_pos']}, = 1 {stats['coarse_w_full']} ({time.time() - t0:.0f}s)", flush=True)
    base = os.path.basename(a.surface)[:-4]; outp = a.out or f"{OD}/{base}_mif.tif"
    outm = outp[:-4] + "_mask.tif" if a.out else f"{OD}/{base}_mif_mask.tif"
    prof = S.profile.copy(); prof.update(compress="deflate", tiled=True, blockxsize=256, blockysize=256, bigtiff="yes")
    mprof = prof.copy(); mprof.update(dtype="uint8", nodata=None)
    CC = CELL // D
    lab, n = label(binary_dilation(wc > 0, iterations=CC)); objs = find_objects(lab)
    with rasterio.open(outp + ".partial.tif", "w+", **prof) as dst, rasterio.open(outm, "w+", **mprof) as dm:
        for br in range(0, H, B):                                       # copy the surface
            h = min(B, H - br); dst.write(S.read(1, window=Window(0, br, W, h)), 1, window=Window(0, br, W, h))
        replaced = 0; changed = 0
        for k_, sl in enumerate(objs, 1):
            if sl is None: continue
            r0 = max(0, sl[0].start * D - a.margin); r1 = min(H, sl[0].stop * D + a.margin)
            c0 = max(0, sl[1].start * D - a.margin); c1 = min(W, sl[1].stop * D + a.margin)
            win = Window(c0, r0, c1 - c0, r1 - r0)
            z = S.read(1, window=win).astype(np.float64); pv = P.read(1, window=win)
            valid = (z > -1e4) & (z != nd); infl = (pv == 1) & valid
            # weight on the full grid: bilinear upsampling of the coarse weight (this region only), infill only, ramp from non-infill
            yc0, xc0 = r0 // D, c0 // D; yc1, xc1 = min(Hc, (r1 + D - 1) // D + 1), min(Wc, (c1 + D - 1) // D + 1)
            wsub = np.where(lab[yc0:yc1, xc0:xc1] == k_, wc[yc0:yc1, xc0:xc1], 0.0)
            wf = zoom(wsub, D, order=1); oy, ox = r0 - yc0 * D, c0 - xc0 * D
            wf = wf[oy:oy + z.shape[0], ox:ox + z.shape[1]]
            if wf.shape != z.shape:
                tmp = np.zeros(z.shape); tmp[:wf.shape[0], :wf.shape[1]] = wf; wf = tmp
            wf = wf * infl * np.clip(distance_transform_edt(infl) / a.feather, 0.0, 1.0)
            if not (wf > 0).any(): continue
            rr, cc = np.mgrid[r0:r1 + D:D, c0:c1 + D:D] + 0.5
            x, y = tr * (cc, rr); lo, la = xy_to_lonlat(np.asarray(x).ravel(), np.asarray(y).ravel(), kind, lon0, lat_ts)
            mo_c = M.at(np.asarray(lo), np.asarray(la)).reshape(rr.shape)
            mo = zoom(mo_c, D, order=1)[:z.shape[0], :z.shape[1]]
            la_w = float(np.nanmedian(np.asarray(la))); T = a.t_polar if abs(la_w) >= a.polar_lat else a.t_other
            cap = max(4 * T, 120.0)
            kall = valid & (wf < 0.01) & np.isfinite(mo); known = kall & (np.abs(z - mo) <= cap)
            rfull = pullpush(np.where(kall, z - mo, 0.0), kall)                       # matches every kept edge
            rrob = pullpush(np.where(known, z - mo, 0.0), known) if known.any() else rfull   # ignores kept pixels > cap off MOLA
            beta = np.clip(1.0 - distance_transform_edt(~kall) / a.res_ramp, 0.0, 1.0)  # v2.1: full residual next to kept pixels
            res = beta * rfull + (1.0 - beta) * rrob
            sm = gaussian_filter(res, a.smooth); res = np.where(kall, res, sm)
            # v2.2 (2026-10-30): only this region's own pixels go into the running output. Whole-window writes from the input surface reset the
            # corrections of earlier regions whose pixels lie in this window, leaving straight step lines at window edges (E-110_N-80, E-050_Np80).
            sel = (wf > 0) & np.isfinite(mo); cur = dst.read(1, window=win)
            cur[sel] = (wf * (mo + res) + (1 - wf) * z)[sel].astype(cur.dtype)
            dst.write(cur, 1, window=win)
            mk = dm.read(1, window=win); mk[wf >= 0.5] = 1; dm.write(mk, 1, window=win)
            replaced += int((wf >= 0.5).sum()); changed += int((wf > 0).sum())
    os.replace(outp + ".partial.tif", outp)
    stats.update(box=a.box, surface=a.surface, replaced_px=replaced, changed_px=changed, version="v2.2", sigma=a.sigma, res_ramp=a.res_ramp, t_polar=a.t_polar, t_other=a.t_other, secs=round(time.time() - t0))
    json.dump(stats, open(outp[:-4] + ".json", "w"), indent=1)
    print(f"[{a.box}] replaced {replaced} px -> {outp} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

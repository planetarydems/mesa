#!/usr/bin/env python3
"""join_check.py LEVELED MODEL FINAL [--ref FINAL_REF] [--label NAME] : acceptance metrics of an infill join (read only), with the
definitions of revision_work/infill_join_census/code/join_census.py (md5 91b6f73f), applied to one FINAL surface:
  eligible gap   8-connected component of leveled nodata >= 30 px; small gap: < 30 px
  infill         MODEL clamped as the merge does (every 37th valid leveled px, p0.01/p99.99 -+ 500 m)
  ring           valid pixel 4-adjacent to an eligible gap
  RESET          ring pixel with FINAL == leveled (bitwise) and |infill - leveled| > 1 cm (infill present)
  PARTIAL        ring pixel with FINAL != leveled, |infill - leveled| >= 0.5 m, implied alpha (F - L)/(M - L) in [0, 0.55)
  edge jump      |F(gap px) - F(valid px)| over 4-adjacent pairs; control = the matched outward pair |F(valid) - F(valid - d)|
                 (both valid, not in any gap)
  outside        pixels outside every gap and outside the write reach of every gap (eligible: L1 41 px; small: the gap itself)
                 with FINAL != leveled (target 0), and, with --ref, FINAL != FINAL_REF there (target 0).
Prints one summary line per surface (and JSON with --json)."""
import argparse, json, os, sys, time
os.environ.setdefault("GDAL_CACHEMAX", "512")
import numpy as np, rasterio
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("leveled"); ap.add_argument("model"); ap.add_argument("final"); ap.add_argument("--ref", default=None)
ap.add_argument("--label", default=""); ap.add_argument("--json", default=None)
a = ap.parse_args(); t0 = time.time()
with rasterio.open(a.leveled) as s: L = s.read(1).astype(np.float32); nd = s.nodata
with rasterio.open(a.model) as s: M = s.read(1).astype(np.float32); mnd = s.nodata
with rasterio.open(a.final) as s: F = s.read(1).astype(np.float32)
hole = ~np.isfinite(L) | (L == nd); valid = ~hole
cov = np.isfinite(M) & (M != mnd)
vv = L.reshape(-1)[::37]; vv = vv[np.isfinite(vv) & (vv != nd)]
lo, hi = float(np.percentile(vv, 0.01)) - 500.0, float(np.percentile(vv, 99.99)) + 500.0; del vv
np.clip(M, lo, hi, out=M)
lab, n = ndi.label(hole, structure=np.ones((3, 3)))
sizes = np.zeros(n + 1, np.int64)
for i in range(0, lab.shape[0], 4096): sizes += np.bincount(lab[i:i + 4096].ravel(), minlength=n + 1)
elig_lab = sizes >= 30; elig_lab[0] = False
elig = elig_lab[lab]; small = hole & ~elig
cross = ndi.generate_binary_structure(2, 1)
ring = valid & ndi.binary_dilation(elig, structure=cross)
dML = M - L
reset = ring & cov & (F == L) & (np.abs(dML) > 0.01)
with np.errstate(all="ignore"): al = (F - L) / dML
partial = ring & cov & (F != L) & (np.abs(dML) >= 0.5) & (al >= 0) & (al < 0.55)
fin = np.isfinite(F) & (F != nd)


def pairs(gapmask):
    """edge jumps (gap px vs 4-adjacent valid px) and matched outward controls"""
    J, C = [], []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        g = np.roll(gapmask, (-dy, -dx), (0, 1))                 # p such that p + d is a gap pixel
        p = valid & g & fin
        q = np.roll(fin, (-dy, -dx), (0, 1))
        m = p & q
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False            # no wrap-around pairs
        Fp = F[m]; Fg = np.roll(F, (-dy, -dx), (0, 1))[m]; J.append(np.abs(Fg - Fp))
        out_ = np.roll(valid & fin & ~hole, (dy, dx), (0, 1))     # p - d valid
        mc = m & out_
        mc[:2, :] = mc[-2:, :] = mc[:, :2] = mc[:, -2:] = False
        C.append(np.abs(F[mc] - np.roll(F, (dy, dx), (0, 1))[mc]))
    J = np.concatenate(J); C = np.concatenate(C)
    return J, C


Je, Ce = pairs(elig & cov & fin); Js, Cs = pairs(small & cov & fin)
reach = ndi.binary_dilation(elig, structure=cross, iterations=41) | small
outside = valid & ~reach
chg_L = int(np.count_nonzero(outside & (F != L) & fin))
res = dict(label=a.label, final=a.final, gaps=int(n), eligible=int(elig_lab.sum()), small=int(n - elig_lab.sum()),
           ring_px=int(ring.sum()), reset_px=int(reset.sum()), partial_px=int(partial.sum()),
           elig_jump_median=float(np.median(Je)) if len(Je) else None, elig_jump_p99=float(np.percentile(Je, 99)) if len(Je) else None,
           elig_jump_gt5=int((Je > 5).sum()), elig_control_median=float(np.median(Ce)) if len(Ce) else None,
           small_pairs=int(len(Js)), small_jump_median=float(np.median(Js)) if len(Js) else None,
           small_jump_p99=float(np.percentile(Js, 99)) if len(Js) else None, small_jump_gt5=int((Js > 5).sum()),
           small_control_median=float(np.median(Cs)) if len(Cs) else None, changed_outside_vs_leveled=chg_L)
if a.ref:
    with rasterio.open(a.ref) as s: R = s.read(1).astype(np.float32)
    res["changed_outside_vs_ref"] = int(np.count_nonzero(outside & (F != R)))
    res["changed_total_vs_ref"] = int(np.count_nonzero((F != R) & fin))
    res["max_abs_diff_vs_ref"] = float(np.nanmax(np.abs(np.where(fin, F - R, 0))))
res["secs"] = round(time.time() - t0)
f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else f"{v:,}")
print(f"[{a.label}] gaps {f(res['gaps'])} (eligible {f(res['eligible'])}) | RESET {f(res['reset_px'])} PARTIAL {f(res['partial_px'])} of ring "
      f"{f(res['ring_px'])} | eligible-edge jump median {f(res['elig_jump_median'])} p99 {f(res['elig_jump_p99'])} >5m {f(res['elig_jump_gt5'])} "
      f"(control {f(res['elig_control_median'])}) | small-gap pairs {f(res['small_pairs'])} jump median {f(res['small_jump_median'])} "
      f"p99 {f(res['small_jump_p99'])} >5m {f(res['small_jump_gt5'])} (control {f(res['small_control_median'])}) | changed outside "
      f"vs leveled {f(res['changed_outside_vs_leveled'])}" + (f", vs ref {f(res['changed_outside_vs_ref'])} (total changed "
      f"{f(res['changed_total_vs_ref'])}, max {f(res['max_abs_diff_vs_ref'])} m)" if a.ref else ""), flush=True)
if a.json: json.dump(res, open(a.json, "w"), indent=1)

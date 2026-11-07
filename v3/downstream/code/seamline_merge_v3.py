#!/usr/bin/env python3
"""
seamline_merge_v3.py -- v3 COPY (2026-10-26) of src/inpainting/seamline_merge.py (md5 01796232, left untouched: R1/R2 provenance).
Same cut, feather, clamp and gap fill; two join defects of the original fixed (infill_join_census 10-17):
  1. ORDER-INDEPENDENT JOIN. The original wrote alpha * infill + (1 - alpha) * DEM over each component's whole write region in
     label order, so a later component's region reset an earlier neighbour's join band to the raw DEM (where the later alpha
     is 0): 103k reset edge px in 101 R1 tiles. Now every eligible component's cut and alpha are computed independently against
     the ORIGINAL leveled DEM and accumulated into one alpha map by the per-pixel MAXIMUM over overlapping write regions; the
     blend is composited once. Where no other component reaches, the result is bit-identical to the original's.
  2. SMALL GAPS (< --min-hole-px) were hard-filled with the infill (no join): median edge jump 2.45 m vs 0.26 m control. Now
     each small component (with infill) gets the infill SHIFTED by the edge residual: r = surface - infill on the valid pixels
     4-adjacent to it (the composited surface, so a small gap inside a large gap's band matches the band), interpolated into
     the gap by inverse squared distance; the gap meets its edge without a step and no pixel outside the gap changes.
Log lines as the original ([clamp], [merge], [gapfill], [done]) plus [merge-v3] (alpha overlaps, small-gap counts).

Original docstring:
seamline_merge.py -- App 1: merge the inpaint surface into the CTX DEM along a
MINIMUM-SLOPE-MISMATCH seamline instead of hard-compositing at the hole edge.

For each hole component, open a buffer of width w INTO the valid CTX (both surfaces
defined there), and route the merge boundary through the locus where the two agree
best -- specifically where Delta = model - dem changes least between adjacent pixels
(|grad Delta| small => slopes match). Solved as a binary graph min-cut per component
(scipy maximum_flow): hole side forced to inpaint, far-CTX side forced to CTX, cut
cost between neighbours = lam*|Delta(p)-Delta(q)|. Then feather +-k px across the cut.

Everything outside (hole ∪ buffer ∪ feather) is the untouched CTX DEM.

  python seamline_merge.py --dem box_dem_trackaware_conf.tif \
     --crop 49485 30042 9975 11920 \
     --model box_dem_coarsefill_testbox.tif.model.tif \
     --out box_dem_seammerge_w32.tif --buffer 32 --feather 8
"""
import argparse, time
import numpy as np, rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
import sys, os
sys.path.insert(0, "/home/gridsan/groups/planetary_dtm/mars_dem/src/inpainting")   # v3: the production solve_useA, unchanged
from seamline_mosaic import solve_useA

SCALE = 10.0        # metres -> int cost
BIG   = 1 << 24     # hard-constraint capacity
BASE  = 1           # per-edge floor so the cut prefers to stay short


def comp_cut(dem, model, comp, valid, w, k, lam, w_off=1.0, cut_target=300000):
    """Return (alpha, region) for one hole component in a local array.
    alpha in [0,1]: 1=use model, 0=use dem. region=where alpha applies (else keep dem)."""
    band = ndi.binary_dilation(comp, iterations=w) & valid          # buffer into CTX
    if not band.any():
        return None                                                  # pure void, no CTX to merge
    outside = valid & ~band & ~comp
    inner = band & ndi.binary_dilation(comp, iterations=1)           # touches the hole
    outer = band & ndi.binary_dilation(outside, iterations=1)        # touches far CTX
    if not inner.any() or not outer.any():
        return None

    # adaptive min-cut (decimates huge bands: a seamline needs no 18 m precision).
    # A=model (inpaint), B=dem (CTX); forceA=inner (hole side), forceB=outer (far CTX).
    useA = solve_useA(model, dem, band, inner, outer, lam, w_off, target=cut_target)

    alpha_hard = np.maximum(comp.astype(np.float32),
                            (useA & band).astype(np.float32))        # 1 on hole + inpaint-side band

    # feather +-k across the label boundary via signed distance
    a01 = alpha_hard > 0.5
    sd = ndi.distance_transform_edt(a01) - ndi.distance_transform_edt(~a01)
    alpha = np.clip(0.5 + sd / (2.0 * max(k, 1)), 0.0, 1.0).astype(np.float32)
    region = ndi.binary_dilation(comp | band, iterations=k + 1)
    alpha = np.where(region, alpha, 0.0).astype(np.float32)
    alpha = np.where(comp, 1.0, alpha)                               # hole always model
    return alpha, region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("R0", "C0", "H", "W"))
    ap.add_argument("--buffer", type=int, default=32)
    ap.add_argument("--feather", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0, help="weight on slope-mismatch (kink)")
    ap.add_argument("--w-offset", type=float, default=1.0, help="weight on |Delta| (step)")
    ap.add_argument("--min-hole-px", type=int, default=30)
    ap.add_argument("--cut-target", type=int, default=300000,
                    help="target node count for the min-cut; larger bands are decimated")
    ap.add_argument("--clamp-margin", type=float, default=500.0,
                    help="margin (m) beyond the valid DEM p0.01/p99.99 for the inpaint clamp")
    ap.add_argument("--gap-fill-px", type=int, default=64,
                    help="nearest-fill residual nodata within this many px of valid data")
    ap.add_argument("--pad", type=int, default=8, help="extra bbox pad beyond buffer+feather")
    args = ap.parse_args()
    t0 = time.time()

    md = rasterio.open(args.model)
    model = md.read(1).astype(np.float32); mnd = md.nodata
    dd = rasterio.open(args.dem)
    if args.crop:
        r0, c0, H, W = args.crop
        dem = dd.read(1, window=Window(c0, r0, W, H)).astype(np.float32)
        out_tr = rasterio.windows.transform(Window(c0, r0, W, H), dd.transform)
    else:
        dem = dd.read(1).astype(np.float32); out_tr = dd.transform
    nd = dd.nodata
    assert dem.shape == model.shape, f"dem {dem.shape} != model {model.shape}"

    hole = (dem == nd)
    valid = np.isfinite(dem) & ~hole
    covered = (model != mnd) & np.isfinite(model)
    valid &= covered                                                 # buffer needs model too

    # Blunder guard: the coarse net can emit wild extrapolations in context-starved
    # windows (observed: -4250 m against a box range of -1150..1450). Clamp the inpaint
    # surface to a robust envelope of the SURROUNDING real DEM before it is composited.
    # subsample for the envelope: dem[valid] would copy ~13 GB and percentile would sort
    # another. Every 37th valid pixel gives identical p0.01/p99.99 to <0.1 m here.
    vv = dem.reshape(-1)[::37]
    vv = vv[np.isfinite(vv) & (vv != nd)]
    lo = float(np.percentile(vv, 0.01)) - args.clamp_margin
    hi = float(np.percentile(vv, 99.99)) + args.clamp_margin
    del vv
    n_bad = int(np.count_nonzero(covered & ((model < lo) | (model > hi))))
    np.clip(model, lo, hi, out=model)                                # IN-PLACE (was a 16 GB copy)
    print(f"[clamp] inpaint envelope [{lo:.0f}, {hi:.0f}] m (valid p0.01/p99.99 +- "
          f"{args.clamp_margin:.0f})  clamped={n_bad:,} px", flush=True)

    out = dem.copy()
    out[hole & covered] = model[hole & covered]                      # baseline = plain fill (holes)
    alpha_g = np.zeros(dem.shape, np.float32)                        # v3: max over components of the safe alpha
    nover = 0

    lab, ncomp = ndi.label(hole, structure=np.ones((3, 3)))
    sizes_all = np.zeros(ncomp + 1, np.int64)                        # block-wise bincount:
    for i in range(0, lab.shape[0], 4096):                           # ones_like(lab) was 16 GB
        sizes_all += np.bincount(lab[i:i + 4096].ravel(), minlength=ncomp + 1)
    sizes = sizes_all[1:]
    objs = ndi.find_objects(lab)
    pad = args.buffer + args.feather + args.pad
    done = merged = skipped = 0
    for ci in range(1, ncomp + 1):
        if sizes[ci - 1] < args.min_hole_px:
            skipped += 1; continue
        sl = objs[ci - 1]
        y0 = max(0, sl[0].start - pad); y1 = min(dem.shape[0], sl[0].stop + pad)
        x0 = max(0, sl[1].start - pad); x1 = min(dem.shape[1], sl[1].stop + pad)
        ldem = dem[y0:y1, x0:x1]; lmodel = model[y0:y1, x0:x1]
        lcomp = lab[y0:y1, x0:x1] == ci
        lvalid = valid[y0:y1, x0:x1]
        r = comp_cut(ldem, lmodel, lcomp, lvalid, args.buffer, args.feather, args.lam,
                     args.w_offset, args.cut_target)
        done += 1
        if r is None:
            skipped += 1; continue
        alpha, region = r
        # NEVER blend against nodata. Build a safe alpha: model unusable -> 0, dem
        # unusable -> 1, neither -> NaN (left as nodata for the bounded gapfill).
        # v3: accumulate this component's alpha (blend weight of the infill on valid DEM pixels that have infill) into the
        # global map by the maximum; nothing is written to `out` here, so no component can undo another's join.
        lcov = covered[y0:y1, x0:x1]
        dvalid = np.isfinite(ldem) & (ldem != nd)
        a = np.where(region & lcov & dvalid, alpha, 0.0).astype(np.float32)
        cur = alpha_g[y0:y1, x0:x1]
        nover += int(np.count_nonzero((a > 0) & (cur > 0)))
        np.maximum(cur, a, out=cur)
        merged += 1
        if done % 500 == 0:
            print(f"[merge] comp {done} (merged={merged} skipped={skipped})  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[merge] components={ncomp} processed={done} merged={merged} skipped={skipped}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    # v3: one composite of all joins, against the ORIGINAL leveled DEM (holes already hold the infill)
    for i in range(0, dem.shape[0], 4096):
        sl_ = slice(i, i + 4096)
        ag = alpha_g[sl_]; m_ = ag > 0
        if m_.any():
            o_ = out[sl_]; o_[m_] = (ag[m_] * model[sl_][m_] + (1.0 - ag[m_]) * dem[sl_][m_]).astype(np.float32)
    n_alpha = int(np.count_nonzero(alpha_g)); del alpha_g

    # v3: small components (< min_hole_px) with infill: infill + edge residual (inverse squared distance), no step at the edge
    n_small = n_small_fix = n_small_px = 0
    for ci in range(1, ncomp + 1):
        if sizes[ci - 1] >= args.min_hole_px:
            continue
        n_small += 1
        sl = objs[ci - 1]
        y0 = max(0, sl[0].start - 1); y1 = min(dem.shape[0], sl[0].stop + 1)
        x0 = max(0, sl[1].start - 1); x1 = min(dem.shape[1], sl[1].stop + 1)
        comp = lab[y0:y1, x0:x1] == ci
        lcov = covered[y0:y1, x0:x1]; lo_ = out[y0:y1, x0:x1]; lm = model[y0:y1, x0:x1]
        gap = comp & lcov
        if not gap.any():
            continue
        ring = ndi.binary_dilation(comp, structure=ndi.generate_binary_structure(2, 1)) & ~comp
        ring &= lcov & np.isfinite(lo_) & (lo_ != nd) & ~hole[y0:y1, x0:x1]
        if not ring.any():
            continue                                                 # no valid edge: keep the plain infill
        gy, gx = np.nonzero(gap); ry, rx = np.nonzero(ring)
        res = (lo_[ry, rx] - lm[ry, rx]).astype(np.float64)
        d2 = (gy[:, None] - ry[None, :]) ** 2 + (gx[:, None] - rx[None, :]) ** 2
        w = 1.0 / np.maximum(d2, 1).astype(np.float64)
        lo_[gy, gx] = (lm[gy, gx] + (w @ res) / w.sum(1)).astype(np.float32)
        n_small_fix += 1; n_small_px += len(gy)
    print(f"[merge-v3] joint composite: alpha>0 on {n_alpha:,} px, {nover:,} px where component bands overlap (max taken); "
          f"small components {n_small:,}, edge-matched {n_small_fix:,} ({n_small_px:,} px)", flush=True)

    # bounded nodata cleanup: the coarse fill cannot cover pixels no 3072-window centre
    # reaches (box borders). Nearest-fill ONLY gaps within --gap-fill-px of valid data, so
    # genuine large voids are never invented into. Left as -9999 otherwise.
    rem = (~np.isfinite(out)) | (out == nd)
    n_rem = int(np.count_nonzero(rem))
    n_fill = 0
    if n_rem:
        g = args.gap_fill_px
        rlab, rn = ndi.label(rem)
        for sl in ndi.find_objects(rlab):
            y0 = max(0, sl[0].start - g - 1); y1 = min(out.shape[0], sl[0].stop + g + 1)
            x0 = max(0, sl[1].start - g - 1); x1 = min(out.shape[1], sl[1].stop + g + 1)
            sub = out[y0:y1, x0:x1]
            r = (~np.isfinite(sub)) | (sub == nd)
            if not r.any() or r.all():
                continue
            dist, ind = ndi.distance_transform_edt(r, return_distances=True, return_indices=True)
            close = r & (dist <= g)
            if close.any():
                sub[close] = sub[tuple(i[close] for i in ind)]
                n_fill += int(np.count_nonzero(close))
            out[y0:y1, x0:x1] = sub
        del rlab
        still = int(np.count_nonzero((~np.isfinite(out)) | (out == nd)))
        print(f"[gapfill] residual nodata={n_rem:,} -> filled={n_fill:,} "
              f"(<={g}px from valid, per-component), left={still:,}", flush=True)
    prof = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1], count=1,
                dtype="float32", crs=dd.crs, transform=out_tr, nodata=nd,
                compress="deflate", tiled=True, bigtiff="yes")
    with rasterio.open(args.out, "w", **prof) as o:
        o.write(out, 1)
    print(f"[done] -> {args.out}  buffer={args.buffer} feather={args.feather}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
level_frames_box_v3.py -- drop-in replacement of fix_2026-10/blend/code/level_frames_box_v2.py (2026-10-24, prediction-failure
investigation). Same solve, same arguments, same outputs; two changes to the blend weights and a faster blend:
  * the frame's raster border counts as a data edge (validity padded by one invalid pixel before the distance transform);
    v2 measured the distance only to invalid pixels inside the array, so data touching the raster border (CTX strip ends, frames
    cropped to a straight line) kept full weight there and made hard steps (--no-edge-pad restores v2);
  * --frame-weights factors are applied as a feathered demotion instead of a plain multiplier:
        w = base * (f + (1 - f) * clip(1 - W_full / RAMP, 0, 1)),  W_full = summed base weight of the full-weight frames,
    so a demoted frame keeps its full weight where full-weight coverage begins and reaches f RAMP px inside it (--demote-ramp,
    default 512; 0 restores v2's multiplier, which handed over ~5 px inside the full frame's own edge);
  * *_w.tif now holds the base weight (factor NOT multiplied in); the factors are read from --frame-weights at blend time.
    Downstream, box_seam_<TAG>_domid.tif (argmax of *_w.tif) therefore shows the unweighted geometry;
  * each frame is read only over its own column span in the blend (identical values; ~20x faster).
Use with the de-seam splice switched off (de-seam surface = FINAL): the splice writes only each site's bounding box.

Original docstring (v2):
level_frames_box.py -- joint per-frame DATUM leveling, BEFORE the feather blend.

Replaces the post-feather smooth-field anchoring (shot_anchor_box.py). That ran on
the already-blended mosaic with a 3x3-median + bilinear field, which is smooth by
construction and therefore *cannot* cancel a step at a frame boundary -- it only
ramps it. This runs one stage earlier, on the individual frame DEMs, and removes the
step at its source: one additive offset per frame, solved jointly so overlaps AND
PEDR shots agree, then feather the already-consistent frames.

Solve (block adjustment of DC offsets; unknown = delta_f per frame, a frozen at 1):
  overlap (i,j):  delta_i - delta_j = -median(DEM_i - DEM_j)   weight sqrt(n_overlap)
  shot     f   :  delta_f          =  median(shot - footprint_avg(DEM_f))
                                                                weight lambda_shot*sqrt(n_shot)
  mola     f   :  delta_f          =  mola_mean_f - frame_mean_f   (ONLY if f has no shots;
                                                                small weight -- gauge fallback)
  ridge        :  delta_f -> 0    tiny    (keeps isolated frames solvable)
Huber-IRLS reweights the overlap rows. Shot rows set the absolute gauge.

Why offset-only (your prior result): scale 'a' is witnessed only weakly by shots and
differently by seams; floating it hides residual *in the elevations*. Offset is over-
determined and consistent -- every shot and every seam witnesses the same datum. The
leftover overlap disagreement after leveling is pure amplitude/shape, which we LOG per
seam (a free, spatially-resolved sigma map) and feather cosmetically.

  python level_frames_box.py \
      --frames-glob '/scratch/hamish/mars_dem/box_run/preds/*/pred_shot_anchor.tif' \
      --mola data/.../Mars_..._463m.tiff --pedr-db .../mola \
      --out /scratch/hamish/mars_dem/box_run/box_dem_leveled.tif --jobs 64
"""
import argparse, glob, math, multiprocessing as mp, os, re, sqlite3, sys, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

WORK = Path("/home/gridsan/groups/planetary_dtm/mars_dem")   # repo root (fix_2026-10 copy lives outside inference/)
sys.path.insert(0, str(WORK / "src/testing"))
try:
    from fit_patch_affine_to_shots import make_footprint_kernel, footprint_average
except ImportError as e:
    raise RuntimeError(f"cannot import footprint helpers from {WORK/'src/testing'}: {e}")

R_MARS = 3396190.0
FOOTPRINT_M = 168.0
_SHOTS = None          # (xs, ys, elev) in frame-CRS metres (eqc or sinu); set before fork, COW-inherited


# ----------------------------------------------------------------- PEDR (yours)
def load_shots(db, lat0, lat1, lon0, lon1, table=None, cols=None, exclude_orbits=None):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    if table is None:
        tabs = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
        found = None
        for t in tabs:
            cl = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            up = {c.upper(): c for c in cl}
            la = next((up[k] for k in up if k.startswith("LAT")), None)
            lo = next((up[k] for k in up if k.startswith("LON")), None)
            el = next((up[k] for k in up if "ELEV" in k), None)
            if la and lo and el:
                found = (t, la, lo, el); break
        if found is None:
            raise RuntimeError(f"no LAT/LON/ELEV table in {db}; tables={tabs}")
        table, la, lo, el = found
    else:
        la, lo, el = cols
    orbit_sql = ""
    if exclude_orbits:
        up = {r[1].upper(): r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        ob = next((up[k] for k in up if k.startswith("ORB")), None)
        if ob is None:
            raise RuntimeError(f"--exclude-orbits given but {db}:{table} has no ORBIT column")
        ids = ",".join(str(int(o)) for o in sorted(exclude_orbits))
        orbit_sql = f" AND {ob} NOT IN ({ids})"
        print(f"[pedr] excluding {len(exclude_orbits)} orbits", flush=True)
    l0, l1 = lon0 % 360.0, lon1 % 360.0
    # corrupt-record guard: one PEDR row with ELEV=710km was found surviving the orbit
    # screen (E-120_Np10, orbit 1647); physical Mars areoid range is ~-8.2..+21.3 km
    orbit_sql += f" AND {el} BETWEEN -10000 AND 25000"
    sel = f"SELECT {la},{lo},{el} FROM {table} WHERE {la} BETWEEN ? AND ? AND "
    if l0 <= l1:
        rows = con.execute(sel + f"{lo} BETWEEN ? AND ?" + orbit_sql,
                           (lat0, lat1, l0, l1)).fetchall()
    else:
        rows = con.execute(sel + f"({lo} >= ? OR {lo} <= ?)" + orbit_sql,
                           (lat0, lat1, l0, l1)).fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"PEDR query 0 shots lat[{lat0},{lat1}] lon[{lon0},{lon1}]")
    a = np.asarray(rows, dtype=np.float64)
    print(f"[pedr] {len(a)} shots from {table} ({la},{lo},{el})", flush=True)
    return a[:, 0], a[:, 1], a[:, 2]


# CRS maths: ONE implementation shared with the PEDR-side tools (pedr_shape_error.py,
# mk_shift_vrts.py, pedr_orbit_screen.py, mola_reblend*.py). proj_helpers lives in src/ (its
# production home) or, while staged, in global_run/polar_prep/pedr_sinu/. Its parse_crs returns
# (kind, R, lon0, lat_ts); bounds_to_lonlat_box applies the query margin itself and encodes
# "every longitude" as (0, 359.999999) so load_shots' `% 360` cannot collapse it to a sliver
# (the previous inline helper returned (lon0-180, lon0+180) for a pole-reaching bbox, which
# after `% 360` is `LON BETWEEN l AND l`: a 0.1-degree sliver, i.e. ~0 shots).
for _p in (str(WORK / "src"), str(WORK / "global_run/polar_prep/pedr_sinu")):
    if os.path.isfile(os.path.join(_p, "proj_helpers.py")) and _p not in sys.path:
        sys.path.insert(0, _p); break
from proj_helpers import parse_crs as _parse_crs4, bounds_to_lonlat_box, shots_to_xy


def parse_crs(crs):
    """(kind, lon0, lat_ts) -- thin adapter over proj_helpers.parse_crs (which also returns R)."""
    kind, _R, lon0, lat_ts = _parse_crs4(crs)
    return kind, lon0, lat_ts


def vmask(a, nd):
    m = np.isfinite(a)
    if nd is not None and np.isfinite(nd):
        m &= (a != nd)
    return m


def frame_stem(path):
    """Unique per-frame id. The predictions are all named pred_shot_anchor.tif under
    preds/<STEM>/, so the filename stem collides across frames -- the identity is the
    parent dir. Use <parent>_<filestem>, sanitized, so leveled outputs never overwrite."""
    p = Path(path)
    name = f"{p.parent.name}_{p.stem}" if p.parent.name not in ("", ".") else p.stem
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# ----------------------------------------------------------------- parallel workers
def _tie_worker(arg):
    """offset tie for one frame: median(shot - footprint_avg(pred)). a=1, no spread gate."""
    idx, path, footprint_m, min_shots, min_support = arg
    xs, ys, se = _SHOTS
    with rasterio.open(path) as ds:
        T, nd = ds.transform, ds.nodata
        pred = ds.read(1).astype(np.float64)
    inv = ~T
    col = inv.a * xs + inv.b * ys + inv.c
    row = inv.d * xs + inv.e * ys + inv.f
    H, W = pred.shape
    ib = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    if ib.sum() < min_shots:
        return idx, None, int(ib.sum())
    kernel = make_footprint_kernel(footprint_m, abs(T.a))
    avg, support = footprint_average(pred, kernel, nd if nd is not None else -9999.0)
    rr = np.clip(np.rint(row[ib]).astype(int), 0, H - 1)
    cc = np.clip(np.rint(col[ib]).astype(int), 0, W - 1)
    p = avg[rr, cc]; sup = support[rr, cc]; e = se[ib]
    good = np.isfinite(p) & np.isfinite(e) & (sup >= min_support)
    n = int(good.sum())
    if n < min_shots:
        return idx, None, n
    return idx, float(np.median(e[good] - p[good])), n


def _ovl_worker(arg):
    """robust median(DEM_i - DEM_j) over the decimated overlap, on a common grid."""
    i, j, pi, pj, inter, res, out_crs, min_px = arg
    left, bottom, right, top = inter
    w = max(1, int(round((right - left) / res)))
    h = max(1, int(round((top - bottom) / res)))
    tr = from_origin(left, top, res, res)

    def rd(p):
        with rasterio.open(p) as ds:
            nd = ds.nodata
            with WarpedVRT(ds, crs=out_crs, transform=tr, width=w, height=h,
                           resampling=Resampling.average) as v:
                a = v.read(1).astype(np.float64)
        return a, vmask(a, nd)

    ai, mi = rd(pi); aj, mj = rd(pj)
    both = mi & mj
    n = int(both.sum())
    if n < min_px:
        return i, j, None, n
    return i, j, float(np.median(ai[both] - aj[both])), n


# ----------------------------------------------------------------- global solve
def solve_offsets(N, overlaps, ties, mola_ties, lam_shot, lam_mola,
                  huber_m, ridge, n_iter=6):
    """Weighted LS for per-frame delta; Huber-IRLS on overlap rows. Returns delta[N]."""
    eqs = []  # (i, j_or_None, rhs, w0, is_overlap)
    for (i, j), (med, cnt) in overlaps.items():
        eqs.append((i, j, -med, math.sqrt(cnt), True))
    for f, (s, n) in ties.items():
        if s is not None:
            eqs.append((f, None, s, lam_shot * math.sqrt(n), False))
    for f, tie in mola_ties.items():
        eqs.append((f, None, tie, lam_mola, False))
    for f in range(N):                       # ridge -> solvable even if isolated
        eqs.append((f, None, 0.0, ridge, False))

    M = len(eqs)
    delta = np.zeros(N)
    for _ in range(n_iter):
        A = np.zeros((M, N)); rhs = np.zeros(M); wv = np.zeros(M)
        for r, (a_, b_, val, w0, isov) in enumerate(eqs):
            A[r, a_] = 1.0
            if b_ is not None:
                A[r, b_] = -1.0
            rhs[r] = val
            if isov:
                resid = float(A[r] @ delta - val)
                hw = 1.0 if abs(resid) <= huber_m else huber_m / max(abs(resid), 1e-6)
            else:
                hw = 1.0
            wv[r] = w0 * hw
        sw = np.sqrt(wv)
        delta = np.linalg.lstsq(A * sw[:, None], rhs * sw, rcond=None)[0]
    return delta


# ----------------------------------------------------------------- streaming feather
def blend_grid(bounds, crs, res, coverage_reference=None):
    """Union canvas; reference contributes grid geometry only, never elevations."""
    if not bounds or crs is None or not math.isfinite(res) or res <= 0:
        raise ValueError("invalid blend bounds/CRS/resolution")
    left = min(b.left for b in bounds); right = max(b.right for b in bounds)
    bottom = min(b.bottom for b in bounds); top = max(b.top for b in bounds)
    if coverage_reference is None:
        return (int(math.ceil((right-left)/res)), int(math.ceil((top-bottom)/res)),
                from_origin(left, top, res, res))
    # Open metadata only: no read(), sample(), or reference elevations enter blend.
    with rasterio.open(coverage_reference) as ref:
        tr, rb = ref.transform, ref.bounds
        if (ref.crs != crs or tr.b != 0 or tr.d != 0 or
                not math.isclose(tr.a, res, rel_tol=0, abs_tol=1e-8) or
                not math.isclose(tr.e, -res, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("coverage reference must match CRS and north-up pixel resolution")
        ax, ay = tr.c, tr.f
    # Snap outward on the existing lattice; tolerate floating roundoff at exact edges.
    eps = 1e-8
    c0 = math.floor((min(left, rb.left)-ax)/res + eps)
    c1 = math.ceil((max(right, rb.right)-ax)/res - eps)
    r0 = math.floor((ay-max(top, rb.top))/res + eps)
    r1 = math.ceil((ay-min(bottom, rb.bottom))/res - eps)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("empty coverage union")
    return c1-c0, r1-r0, from_origin(ax+c0*res, ay-r0*res, res, res)


def feather_blend(leveled, weights, nd, out_path, base_res, block_rows, taper_px, coverage_reference=None,
                  factors=None, ramp=512.0):
    """weighted streaming blend of leveled frames -> single mosaic. OOM-safe: one output block + the frames touching it.
    factors[k] < 1: feathered demotion (see module docstring); ramp <= 0: plain multiplier (v2)."""
    bounds = []
    crs = None
    for path in leveled:
        with rasterio.open(path) as ds:
            if crs is not None and ds.crs != crs:
                raise ValueError("leveled frame CRS mismatch")
            crs = ds.crs
            bounds.append(ds.bounds)
    Wo, Ho, out_tr = blend_grid(bounds, crs, base_res, coverage_reference)
    res = base_res
    top, left = out_tr.f, out_tr.c
    factors = list(factors) if factors is not None else [1.0] * len(leveled)
    print(f"[blend] out {Wo}x{Ho} @ {res} m/px; {sum(f < 1 for f in factors)} demoted frames, ramp {ramp:g} px", flush=True)

    dem_vrt, w_vrt, rspan, cspan = [], [], [], []
    for dp, wp in zip(leveled, weights):
        ds = rasterio.open(dp)
        dem_vrt.append(WarpedVRT(ds, crs=crs, transform=out_tr, width=Wo, height=Ho,
                                 resampling=Resampling.bilinear))
        w_vrt.append(WarpedVRT(rasterio.open(wp), crs=crs, transform=out_tr,
                               width=Wo, height=Ho, resampling=Resampling.bilinear))
        b = ds.bounds
        rspan.append((max(0, int((top - b.top) / res) - 1), min(Ho, int((top - b.bottom) / res) + 1)))
        cspan.append((max(0, int((b.left - left) / res) - 2), min(Wo, int(math.ceil((b.right - left) / res)) + 2)))

    prof = dict(driver="GTiff", height=Ho, width=Wo, count=1, dtype="float32",
                crs=crs, transform=out_tr, nodata=nd, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256, bigtiff="yes")
    with rasterio.open(out_path, "w", **prof) as dst:
        for rs in range(0, Ho, block_rows):
            re_ = min(rs + block_rows, Ho); h = re_ - rs
            parts = []
            for k, (r0, r1) in enumerate(rspan):
                if r1 <= rs or r0 >= re_:
                    continue
                c0, c1 = cspan[k]
                win = Window(c0, rs, c1 - c0, h)
                dem = dem_vrt[k].read(1, window=win)
                wt = w_vrt[k].read(1, window=win).astype(np.float64)
                ok = vmask(dem, nd) & (wt > 0)
                if ok.any():
                    parts.append((c0, c1, np.where(ok, dem, 0.0), np.where(ok, wt, 0.0), factors[k]))
            wfull = np.zeros((h, Wo))
            for c0, c1, dem, wt, f in parts:
                if f >= 1.0:
                    wfull[:, c0:c1] += wt
            wsum = np.zeros((h, Wo)); wpix = np.zeros((h, Wo))
            for c0, c1, dem, wt, f in parts:
                if f >= 1.0:
                    e = 1.0
                elif ramp > 0:
                    e = f + (1.0 - f) * np.clip(1.0 - wfull[:, c0:c1] / ramp, 0.0, 1.0)
                else:
                    e = f
                ww = wt * e
                wsum[:, c0:c1] += ww * dem
                wpix[:, c0:c1] += ww
            out = np.where(wpix > 0, wsum / np.maximum(wpix, 1e-9), nd).astype(np.float32)
            dst.write(out, 1, window=Window(0, rs, Wo, h))
            if (rs // block_rows) % 20 == 0:
                print(f"[blend] row {rs}/{Ho}", flush=True)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-glob", required=True, help="glob of per-frame DEM tifs")
    ap.add_argument("--mola", required=True)
    ap.add_argument("--pedr-db", required=True)
    ap.add_argument("--out", required=True, help="final leveled+feathered mosaic")
    ap.add_argument("--work-dir", default=None, help="leveled/weight scratch (default: out dir)")
    ap.add_argument("--footprint-m", type=float, default=FOOTPRINT_M)
    ap.add_argument("--min-shots", type=int, default=8)
    ap.add_argument("--min-support", type=float, default=0.5)
    ap.add_argument("--overlap-res", type=float, default=90.0, help="m/px for overlap diff")
    ap.add_argument("--min-overlap-px", type=int, default=200)
    ap.add_argument("--lambda-shot", type=float, default=1.0)
    ap.add_argument("--lambda-mola", type=float, default=0.05)
    ap.add_argument("--huber-m", type=float, default=3.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--taper-px", type=float, default=64.0, help="feather ramp width")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--coverage-reference", default=None, help="metadata-only old grid: preserve its extent and pixel lattice while allowing new coverage")
    ap.add_argument("--no-blend", action="store_true", help="stop after leveling (run your feather)")
    ap.add_argument("--table", default=None)
    ap.add_argument("--cols", nargs=3, default=None, metavar=("LAT", "LON", "ELEV"))
    ap.add_argument("--exclude-orbits", default=None,
                    help="file of MOLA orbit ids to drop, one per line (from "
                         "pedr_orbit_screen.py) -- bad passes disagreeing with MEGDR by "
                         "150-210 m; keeps the per-frame datum gauge clean")
    ap.add_argument("--jobs", type=int, default=0)
    # fix_2026-10 (release 2): per-frame blend-weight factors, CSV "stem,factor" (stem = the frame's preds/level_frames
    # directory name). The feather weight of a listed frame is multiplied by its factor, so a hazy frame (CTX image QC
    # 2026-10-16, tier A/B) gives way wherever another frame covers the ground and still fills ground nothing else
    # covers (the blend normalises by the summed weight). The leveling solve is unchanged.
    ap.add_argument("--frame-weights", default=None, help="CSV stem,factor of blend-weight factors (feathered demotion)")
    ap.add_argument("--demote-ramp", type=float, default=512.0, help="px of full-weight coverage over which a factor ramps in "
                    "(0 = plain multiplier as v2)")
    ap.add_argument("--no-edge-pad", action="store_true", help="v2 weights: raster border not counted as a data edge")
    args = ap.parse_args()
    t0 = time.time()
    global _SHOTS

    frames = sorted(glob.glob(args.frames_glob))
    if not frames:
        raise RuntimeError(f"no frames match {args.frames_glob}")
    N = len(frames)
    njobs = args.jobs if args.jobs > 0 else len(os.sched_getaffinity(0))
    work = Path(args.work_dir) if args.work_dir else Path(args.out).parent / "leveled"
    (work).mkdir(parents=True, exist_ok=True)
    print(f"[frames] {N} frames  jobs={njobs}  work={work}", flush=True)

    # frame geometry (shared CRS/res from Robbins grid)
    bnds, crs, base_res, nd = [], None, None, None
    for p in frames:
        with rasterio.open(p) as ds:
            bnds.append(ds.bounds)
            if crs is None:
                crs, base_res, nd = ds.crs, abs(ds.transform.a), ds.nodata
    kind, lon0, lat_ts = parse_crs(crs)
    print(f"[crs] {kind} lon0={lon0} lat_ts={lat_ts} res={base_res} m/px", flush=True)

    # one PEDR pull over the union (+margin); shots -> projected metres of the frame CRS
    L = min(b.left for b in bnds); Rg = max(b.right for b in bnds)
    B = min(b.bottom for b in bnds); Tp = max(b.top for b in bnds)
    # margin is applied INSIDE the helper (a pad added afterwards re-breaks the full-range case)
    lon_lo, lon_hi, lat_lo, lat_hi = bounds_to_lonlat_box(L, Rg, B, Tp, kind, lon0, lat_ts,
                                                          pad_lon=.05, pad_lat=.05)
    excl = None
    if args.exclude_orbits:
        excl = {int(t) for t in Path(args.exclude_orbits).read_text().split() if t.strip()}
    slat, slon, selev = load_shots(args.pedr_db, lat_lo, lat_hi, lon_lo, lon_hi,
                                   args.table, tuple(args.cols) if args.cols else None,
                                   exclude_orbits=excl)
    xs, ys = shots_to_xy(slon, slat, kind, lon0, lat_ts)
    _SHOTS = (xs, ys, selev)

    ctx = mp.get_context("fork")

    # ---- shot ties (per frame, parallel) ----
    tie_args = [(i, frames[i], args.footprint_m, args.min_shots, args.min_support)
                for i in range(N)]
    ties = {}
    with ctx.Pool(njobs) as pool:
        for idx, s, n in pool.imap_unordered(_tie_worker, tie_args, chunksize=1):
            ties[idx] = (s, n)
    n_tied = sum(1 for v in ties.values() if v[0] is not None)
    print(f"[ties] {n_tied}/{N} frames shot-tied  ({time.time()-t0:.0f}s)", flush=True)

    # ---- candidate overlap pairs by bbox intersection ----
    def inter(bi, bj):
        l, b = max(bi.left, bj.left), max(bi.bottom, bj.bottom)
        r, t = min(bi.right, bj.right), min(bi.top, bj.top)
        return (l, b, r, t) if (r > l and t > b) else None

    ovl_args = []
    for i in range(N):
        for j in range(i + 1, N):
            it = inter(bnds[i], bnds[j])
            if it is not None:
                ovl_args.append((i, j, frames[i], frames[j], it,
                                 args.overlap_res, crs, args.min_overlap_px))
    overlaps = {}
    with ctx.Pool(njobs) as pool:
        for i, j, med, n in pool.imap_unordered(_ovl_worker, ovl_args, chunksize=4):
            if med is not None:
                overlaps[(i, j)] = (med, n)
    print(f"[overlap] {len(overlaps)}/{len(ovl_args)} pairs usable "
          f"(>= {args.min_overlap_px}px)  ({time.time()-t0:.0f}s)", flush=True)

    # ---- MOLA gauge fallback for shot-less frames (weak) ----
    mola_ties = {}
    shotless = [f for f in range(N) if ties[f][0] is None]
    if shotless:
        with rasterio.open(args.mola) as mds:
            for f in shotless:
                with rasterio.open(frames[f]) as ds:
                    pred = ds.read(1).astype(np.float64); fnd = ds.nodata
                    with WarpedVRT(mds, crs=ds.crs, transform=ds.transform,
                                   width=ds.width, height=ds.height,
                                   resampling=Resampling.bilinear) as v:
                        mola = v.read(1).astype(np.float64)
                vm = vmask(pred, fnd) & np.isfinite(mola)
                if vm.sum() > 100:
                    mola_ties[f] = float(mola[vm].mean() - pred[vm].mean())
        print(f"[mola] {len(mola_ties)} shot-less frames given weak MOLA gauge", flush=True)

    # ---- solve ----
    delta = solve_offsets(N, overlaps, ties, mola_ties,
                          args.lambda_shot, args.lambda_mola, args.huber_m, args.ridge)

    # ---- diagnostics: overlap residual before/after; shot residual before/after ----
    ov_pre = np.array([abs(m) for (m, _) in overlaps.values()])
    ov_post = np.array([abs(m + delta[i] - delta[j]) for (i, j), (m, _) in overlaps.items()])
    sh_pre = np.array([s for (s, _) in ties.values() if s is not None])
    sh_post = np.array([s - delta[f] for f, (s, _) in ties.items() if s is not None])
    # A box can legitimately have NO usable overlap pair (E-060_Np80, 2026-09-23: 3 cap frames that do not
    # touch) or no shot-tied frame. The solve handles that; only these diagnostics did not, and numpy's
    # percentile on an empty array killed the level AFTER the solve had succeeded.
    if ov_pre.size:
        print(f"[resid] overlap |median| : before med={np.median(ov_pre):.2f} "
              f"p95={np.percentile(ov_pre,95):.2f} | after med={np.median(ov_post):.2f} "
              f"p95={np.percentile(ov_post,95):.2f} m", flush=True)
    else:
        print("[resid] overlap: no usable overlap pair (frames do not touch); offsets rest on the shot/MOLA gauge", flush=True)
    if sh_pre.size:
        print(f"[resid] shot   |median| : before med={np.median(np.abs(sh_pre)):.2f} | "
              f"after med={np.median(np.abs(sh_post)):.2f} m", flush=True)
    else:
        print("[resid] shot: no shot-tied frame", flush=True)
    print(f"[delta] med={np.median(delta):+.2f} p5={np.percentile(delta,5):+.2f} "
          f"p95={np.percentile(delta,95):+.2f} m", flush=True)

    csv = os.path.splitext(args.out)[0] + "_offsets.csv"
    with open(csv, "w") as fo:
        fo.write("frame,n_shots,shot_tie_m,delta_m,n_overlaps\n")
        nov = {f: 0 for f in range(N)}
        for (i, j) in overlaps:
            nov[i] += 1; nov[j] += 1
        for f in range(N):
            s, n = ties[f]
            fo.write(f"{frame_stem(frames[f])},{n if s is not None else 0},"
                     f"{'' if s is None else f'{s:.3f}'},{delta[f]:.3f},{nov[f]}\n")
    print(f"[csv] {csv}", flush=True)

    # ---- apply delta -> leveled frames + feather weights ----
    leveled, weights, factors = [], [], []
    fw = {}
    if args.frame_weights:
        import csv as _csv
        with open(args.frame_weights) as _fh:
            for _r in _csv.DictReader(_fh):
                fw[_r["stem"]] = float(_r["factor"])
        print(f"[weights] {len(fw)} frame factors from {args.frame_weights}", flush=True)
    n_fw = 0
    for f in range(N):
        stem = frame_stem(frames[f])
        lp = work / f"{stem}_leveled.tif"; wp = work / f"{stem}_w.tif"
        with rasterio.open(frames[f]) as ds:
            pred = ds.read(1).astype(np.float64); prof = ds.profile.copy(); fnd = ds.nodata
        vm = vmask(pred, fnd)
        out = np.where(vm, pred + delta[f], fnd if fnd is not None else -9999.0).astype(np.float32)
        prof.update(driver="GTiff", dtype="float32", compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256)
        with rasterio.open(lp, "w", **prof) as d:
            d.write(out, 1)
        # feather weight: distance to the nearest data edge (raster border included), capped at taper
        d = distance_transform_edt(vm) if args.no_edge_pad else distance_transform_edt(np.pad(vm, 1))[1:-1, 1:-1]
        w = np.minimum(d, args.taper_px).astype(np.float32)
        w[~vm] = 0.0; w[vm] = np.maximum(w[vm], 1e-3)
        fac = fw.get(Path(frames[f]).parent.name, 1.0)
        if fac != 1.0:
            n_fw += 1
            print(f"[weights] {Path(frames[f]).parent.name} x {fac:g} (feathered over {args.demote_ramp:g} px)", flush=True)
        factors.append(min(float(fac), 1.0) if fac < 1.0 else 1.0)
        wprof = prof.copy(); wprof.update(nodata=0.0)
        with rasterio.open(wp, "w", **wprof) as d:
            d.write(w, 1)
        leveled.append(str(lp)); weights.append(str(wp))
    print(f"[apply] {N} leveled frames + weights -> {work}", flush=True)
    if args.frame_weights:
        print(f"[weights] applied to {n_fw} of {N} frames (listed {len(fw)})", flush=True)

    if args.no_blend:
        print(f"[done] leveling only; feather {work}/*_leveled.tif yourself "
              f"({time.time()-t0:.0f}s)", flush=True)
        return

    out_nd = nd if nd is not None else -9999.0
    feather_blend(leveled, weights, out_nd, args.out, base_res, args.block, args.taper_px, args.coverage_reference,
                  factors=factors, ramp=args.demote_ramp)
    print(f"[done] {args.out}  (+ _offsets.csv, leveled/)  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
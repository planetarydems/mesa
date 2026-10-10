"""Levelling and blending of the predicted DEMs of one product tile (Section
2.5, steps 4 and 5 of the paper).

Each predicted DEM f receives one additive offset delta_f. The offsets are
solved jointly by weighted least squares from

    overlaps   delta_i - delta_j = -median(DEM_i - DEM_j)    weight sqrt(n_overlap px)
    shots      delta_f = median(shot - footprint average of DEM_f)
                                                            weight sqrt(n_shots)
    MOLA       delta_f = mean(MOLA - DEM_f), only for a DEM without shots, weight 0.05
    ridge      delta_f = 0, weight 1e-3, so an isolated DEM stays solvable

with Huber reweighting (3 m) of the overlap rows. The gain fitted to the shots
at prediction time is kept. The levelled DEMs are then blended with weights
equal to the distance from each DEM's edge, capped at 512 pixels.

Tiles within 30 S to 30 N are on the global equirectangular grid; images
registered in a sinusoidal projection are warped onto it first. Tiles poleward
of that use a sinusoidal projection centred on the tile.

    python -m mader.level --tile E-030_N-10 --frames 'preds/*/pred_shot_anchor.tif' \
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite \
        --exclude-orbits orbit_screen/E-030_N-10.txt --out tiles/E-030_N-10
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from .anchor import footprint_average, footprint_kernel
from .geo import FULL_LON, R_MARS, lonlat_to_xy, parse_crs
from .pedr import PedrDatabase

NODATA = -9999.0
PIXEL = 18.0
EQC = CRS.from_proj4('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs')
FOOTPRINT_M = 168.0
MIN_SHOTS = 8
OVERLAP_RES = 90.0
MIN_OVERLAP_PX = 200
LAMBDA_SHOT, LAMBDA_MOLA, HUBER_M, RIDGE = 1.0, 0.05, 3.0, 1e-3
TAPER_PX = 512
ELEV_RANGE = (-10000.0, 25000.0)

_SHOTS = None          # (x, y, elev) of the tile's shots, shared with worker processes


def tile_bounds(name):
    """(lon_min, lat_min) of a tile name such as E-030_N-10 or Ep170_Np20."""
    m = re.match(r'E([p+-])(\d{3})_N([p+-])(\d{2})$', name)
    sign = lambda s: -1 if s == '-' else 1
    return sign(m.group(1)) * int(m.group(2)), sign(m.group(3)) * int(m.group(4))


def is_equatorial(name):
    return -30 <= tile_bounds(name)[1] <= 20


def valid_mask(a, nodata):
    m = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        m &= a != nodata
    return m


# ---------------------------------------------------------------------------
# Equatorial tiles: sinusoidal DEMs -> global equirectangular grid
# ---------------------------------------------------------------------------

def warp_to_eqc(src_path, dst_path):
    """Bilinear warp of a sinusoidal DEM onto the global equirectangular grid
    (18 m). A DEM crossing the antimeridian is warped in an equirectangular
    projection centred on it and then relabelled (eqc is linear in longitude)."""
    with rasterio.open(src_path) as src:
        tr, w, h = calculate_default_transform(src.crs, EQC, src.width, src.height, *src.bounds, resolution=PIXEL)
        expected_w = src.width * src.res[0] / PIXEL * 1.5 + 64
        profile = src.profile.copy()
        profile.update(driver='GTiff', crs=EQC, nodata=NODATA, compress='deflate', tiled=True, bigtiff='if_safer')
        src_nd = src.nodata if src.nodata is not None else NODATA
        if w <= 2 * expected_w:
            profile.update(transform=tr, width=w, height=h)
            with rasterio.open(dst_path, 'w', **profile) as dst:
                reproject(rasterio.band(src, 1), rasterio.band(dst, 1), src_nodata=src_nd, dst_nodata=NODATA,
                          resampling=Resampling.bilinear, num_threads=2)
            return
        lon0 = parse_crs(src.crs)[2]
        centred = CRS.from_proj4(f'+proj=eqc +lat_ts=0 +lat_0=0 +lon_0={lon0} +x_0=0 +y_0=0 +R=3396190 '
                                 '+units=m +no_defs')
        tr_c, w, h = calculate_default_transform(src.crs, centred, src.width, src.height, *src.bounds,
                                                 resolution=PIXEL)
        if w > 2 * expected_w:
            raise ValueError(f'{src_path}: cannot place on the equirectangular grid')
        arr = np.full((h, w), NODATA, np.float32)
        reproject(rasterio.band(src, 1), arr, src_nodata=src_nd, dst_transform=tr_c, dst_crs=centred,
                  dst_nodata=NODATA, resampling=Resampling.bilinear, num_threads=2)
        deg = R_MARS * np.pi / 180.0
        tr = Affine(tr_c.a, tr_c.b, tr_c.c + lon0 * deg, tr_c.d, tr_c.e, tr_c.f)
        profile.update(transform=tr, width=w, height=h)
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(arr, 1)


def frames_on_tile_grid(tile, frames, work):
    """{image name: DEM path} with every DEM in the tile's projection. The
    image name is the DEM's directory name; sinusoidal DEMs of an equatorial
    tile are replaced by warped copies in work/."""
    out = {}
    for f in frames:
        name = Path(f).parent.name
        if is_equatorial(tile):
            with rasterio.open(f) as ds:
                sinusoidal = parse_crs(ds.crs)[0] == 'sinu'
            if sinusoidal:
                dst = Path(work) / f'{name}_eqc.tif'
                if not dst.exists():
                    warp_to_eqc(f, dst)
                f = str(dst)
        out[name] = f
    return out


# ---------------------------------------------------------------------------
# PEDR shots of the tile
# ---------------------------------------------------------------------------

def lonlat_box(left, right, bottom, top, proj, pad=0.05):
    """(lon_lo, lon_hi, lat_lo, lat_hi) covering a projected rectangle, padded.
    A sinusoidal rectangle is widest in longitude at its pole-nearest corners,
    so all four combinations of x edge and latitude are evaluated."""
    kind, R, lon0, lat_ts = proj
    lat_lo = math.degrees(min(bottom, top) / R)
    lat_hi = math.degrees(max(bottom, top) / R)
    L, Rg = min(left, right), max(left, right)
    if kind == 'eqc':
        cs = math.cos(math.radians(lat_ts))
        a, b = math.degrees(L / (R * cs)) + lon0, math.degrees(Rg / (R * cs)) + lon0
        lon_lo, lon_hi = min(a, b), max(a, b)
    else:
        cs = [math.cos(math.radians(la)) for la in (lat_lo, lat_hi)]
        if min(cs) <= 1e-9:
            lon_lo, lon_hi = FULL_LON
        else:
            lons = [math.degrees(x / (R * c)) + lon0 for x in (L, Rg) for c in cs]
            lon_lo, lon_hi = min(lons), max(lons)
    lon_lo, lon_hi = lon_lo - pad, lon_hi + pad
    lat_lo, lat_hi = max(lat_lo - pad, -90.0), min(lat_hi + pad, 90.0)
    if lon_hi - lon_lo >= 360.0:
        lon_lo, lon_hi = FULL_LON
    return lon_lo, lon_hi, lat_lo, lat_hi


# ---------------------------------------------------------------------------
# Observations (run in worker processes)
# ---------------------------------------------------------------------------

def shot_tie(path):
    """Median of (shot - footprint-averaged DEM) over the shots on one DEM."""
    xs, ys, elev = _SHOTS
    with rasterio.open(path) as ds:
        T, nd = ds.transform, ds.nodata
        dem = ds.read(1).astype(np.float64)
    inv = ~T
    col = inv.a * xs + inv.b * ys + inv.c
    row = inv.d * xs + inv.e * ys + inv.f
    H, W = dem.shape
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    if inside.sum() < MIN_SHOTS:
        return None, int(inside.sum())
    avg, support = footprint_average(dem, footprint_kernel(FOOTPRINT_M, abs(T.a)), nd if nd is not None else NODATA)
    rr = np.clip(np.rint(row[inside]).astype(int), 0, H - 1)
    cc = np.clip(np.rint(col[inside]).astype(int), 0, W - 1)
    p, s, e = avg[rr, cc], support[rr, cc], elev[inside]
    good = np.isfinite(p) & np.isfinite(e) & (s >= 0.5)
    n = int(good.sum())
    if n < MIN_SHOTS:
        return None, n
    return float(np.median(e[good] - p[good])), n


def overlap_difference(args):
    """Median of DEM_i - DEM_j on a common 90 m grid over their overlap."""
    path_i, path_j, (left, bottom, right, top), crs = args
    w = max(1, int(round((right - left) / OVERLAP_RES)))
    h = max(1, int(round((top - bottom) / OVERLAP_RES)))
    tr = from_origin(left, top, OVERLAP_RES, OVERLAP_RES)

    def read(path):
        with rasterio.open(path) as ds:
            nd = ds.nodata
            with WarpedVRT(ds, crs=crs, transform=tr, width=w, height=h, resampling=Resampling.average) as v:
                a = v.read(1).astype(np.float64)
        return a, valid_mask(a, nd)

    ai, mi = read(path_i)
    aj, mj = read(path_j)
    both = mi & mj
    n = int(both.sum())
    return (float(np.median(ai[both] - aj[both])), n) if n >= MIN_OVERLAP_PX else (None, n)


def mola_offset(path, mola_path):
    """mean(MOLA - DEM) over the valid pixels of a DEM without shots."""
    with rasterio.open(mola_path) as mds, rasterio.open(path) as ds:
        dem = ds.read(1).astype(np.float64)
        with WarpedVRT(mds, crs=ds.crs, transform=ds.transform, width=ds.width, height=ds.height,
                       resampling=Resampling.bilinear) as v:
            mola = v.read(1).astype(np.float64)
        ok = valid_mask(dem, ds.nodata) & np.isfinite(mola)
    return float(mola[ok].mean() - dem[ok].mean()) if ok.sum() > 100 else None


def solve_offsets(n_frames, overlaps, ties, mola_ties, n_iter=6):
    """Weighted least squares for the per-DEM offsets, Huber IRLS on overlaps."""
    eqs = [(i, j, -med, math.sqrt(cnt), True) for (i, j), (med, cnt) in overlaps]
    eqs += [(f, None, s, LAMBDA_SHOT * math.sqrt(n), False) for f, (s, n) in ties if s is not None]
    eqs += [(f, None, t, LAMBDA_MOLA, False) for f, t in mola_ties]
    eqs += [(f, None, 0.0, RIDGE, False) for f in range(n_frames)]
    A = np.zeros((len(eqs), n_frames))
    rhs = np.array([e[2] for e in eqs])
    for r, (i, j, _, _, _) in enumerate(eqs):
        A[r, i] = 1.0
        if j is not None:
            A[r, j] = -1.0
    w0 = np.array([e[3] for e in eqs])
    is_overlap = np.array([e[4] for e in eqs])
    delta = np.zeros(n_frames)
    for _ in range(n_iter):
        resid = np.abs(A @ delta - rhs)
        huber = np.where(is_overlap & (resid > HUBER_M), HUBER_M / np.maximum(resid, 1e-6), 1.0)
        sw = np.sqrt(w0 * huber)
        delta = np.linalg.lstsq(A * sw[:, None], rhs * sw, rcond=None)[0]
    return delta


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def feather_blend(levelled, weights, out_path, nodata=NODATA, res=PIXEL, block_rows=256):
    """Weighted mean of the levelled DEMs on the union of their extents,
    computed in blocks of rows so that only one block is in memory."""
    bounds = [rasterio.open(p).bounds for p in levelled]
    crs = rasterio.open(levelled[0]).crs
    left, right = min(b.left for b in bounds), max(b.right for b in bounds)
    bottom, top = min(b.bottom for b in bounds), max(b.top for b in bounds)
    W, H = int(math.ceil((right - left) / res)), int(math.ceil((top - bottom) / res))
    out_tr = from_origin(left, top, res, res)
    dems, wts, spans = [], [], []
    for dp, wp, b in zip(levelled, weights, bounds):
        dems.append(WarpedVRT(rasterio.open(dp), crs=crs, transform=out_tr, width=W, height=H,
                              resampling=Resampling.bilinear))
        wts.append(WarpedVRT(rasterio.open(wp), crs=crs, transform=out_tr, width=W, height=H,
                             resampling=Resampling.bilinear))
        spans.append((max(0, int((top - b.top) / res) - 1), min(H, int((top - b.bottom) / res) + 1)))
    profile = dict(driver='GTiff', height=H, width=W, count=1, dtype='float32', crs=crs, transform=out_tr,
                   nodata=nodata, compress='deflate', tiled=True, blockxsize=256, blockysize=256, bigtiff='yes')
    with rasterio.open(out_path, 'w', **profile) as dst:
        for r0 in range(0, H, block_rows):
            r1 = min(r0 + block_rows, H)
            win = Window(0, r0, W, r1 - r0)
            num = np.zeros((r1 - r0, W), np.float64)
            den = np.zeros((r1 - r0, W), np.float64)
            for k, (a, b) in enumerate(spans):
                if b <= r0 or a >= r1:
                    continue
                dem = dems[k].read(1, window=win)
                wt = wts[k].read(1, window=win).astype(np.float64)
                ok = valid_mask(dem, nodata) & (wt > 0)
                num += np.where(ok, wt * dem, 0.0)
                den += np.where(ok, wt, 0.0)
            dst.write(np.where(den > 0, num / np.maximum(den, 1e-9), nodata).astype(np.float32), 1, window=win)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def level_tile(frames, mola_path, pedr_path, out_path, work, exclude_orbits=None, jobs=None):
    """frames: {image name: predicted DEM}, all in the tile's projection.
    Writes the levelled mosaic, <out>_offsets.csv, and the levelled DEMs and
    blend weights (<image>_leveled.tif, <image>_w.tif) in work/."""
    global _SHOTS
    names = sorted(frames)
    frames = [frames[k] for k in names]
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    jobs = jobs or len(os.sched_getaffinity(0))
    n = len(frames)

    bounds = [rasterio.open(p).bounds for p in frames]
    with rasterio.open(frames[0]) as ds:
        crs, nodata, res = ds.crs, ds.nodata, abs(ds.transform.a)
    proj = parse_crs(crs)
    lon_lo, lon_hi, lat_lo, lat_hi = lonlat_box(min(b.left for b in bounds), max(b.right for b in bounds),
                                                min(b.bottom for b in bounds), max(b.top for b in bounds), proj)
    lat, lon, elev = PedrDatabase(pedr_path).query(lat_lo, lat_hi, lon_lo, lon_hi, elev_range=ELEV_RANGE,
                                                   exclude_orbits=exclude_orbits)
    if lat.size == 0:
        raise ValueError('no PEDR shots over the tile')
    xs, ys = lonlat_to_xy(lon, lat, proj)
    _SHOTS = (xs, ys, elev)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            l, b = max(bounds[i].left, bounds[j].left), max(bounds[i].bottom, bounds[j].bottom)
            r, t = min(bounds[i].right, bounds[j].right), min(bounds[i].top, bounds[j].top)
            if r > l and t > b:
                pairs.append((i, j, (l, b, r, t)))
    with mp.get_context('fork').Pool(jobs) as pool:
        ties = list(enumerate(pool.map(shot_tie, frames, chunksize=1)))
        diffs = pool.map(overlap_difference, [(frames[i], frames[j], box, crs) for i, j, box in pairs], chunksize=4)
    overlaps = [((i, j), d) for (i, j, _), d in zip(pairs, diffs) if d[0] is not None]
    mola_ties = []
    for f, (s, _) in ties:
        if s is None:
            t = mola_offset(frames[f], mola_path)
            if t is not None:
                mola_ties.append((f, t))
    delta = solve_offsets(n, overlaps, ties, mola_ties)
    print(f'{n} DEMs, {sum(s is not None for _, (s, _) in ties)} tied to shots, {len(overlaps)} overlaps, '
          f'{len(mola_ties)} on MOLA; offsets median {np.median(delta):+.2f} m')

    n_overlaps = np.zeros(n, int)
    for (i, j), _ in overlaps:
        n_overlaps[i] += 1
        n_overlaps[j] += 1
    with open(str(out_path).removesuffix('.tif') + '_offsets.csv', 'w') as fo:
        fo.write('frame,n_shots,shot_tie_m,delta_m,n_overlaps\n')
        for f, (s, k) in ties:
            fo.write(f"{names[f]},{k if s is not None else 0},{'' if s is None else f'{s:.3f}'},"
                     f"{delta[f]:.3f},{n_overlaps[f]}\n")

    levelled, weights = [], []
    for f in range(n):
        with rasterio.open(frames[f]) as ds:
            dem = ds.read(1).astype(np.float64)
            profile = ds.profile.copy()
            nd = ds.nodata
        ok = valid_mask(dem, nd)
        profile.update(driver='GTiff', dtype='float32', compress='deflate', tiled=True, blockxsize=256,
                       blockysize=256)
        lp, wp = work / f'{names[f]}_leveled.tif', work / f'{names[f]}_w.tif'
        with rasterio.open(lp, 'w', **profile) as d:
            d.write(np.where(ok, dem + delta[f], nd if nd is not None else NODATA).astype(np.float32), 1)
        w = np.minimum(distance_transform_edt(ok), TAPER_PX).astype(np.float32)
        w[~ok] = 0.0
        w[ok] = np.maximum(w[ok], 1e-3)
        profile.update(nodata=0.0)
        with rasterio.open(wp, 'w', **profile) as d:
            d.write(w, 1)
        levelled.append(str(lp))
        weights.append(str(wp))
    feather_blend(levelled, weights, out_path, nodata if nodata is not None else NODATA, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tile', required=True, help='tile name, e.g. E-030_N-10')
    ap.add_argument('--frames', required=True, help="glob of predicted DEMs, e.g. 'preds/*/pred_shot_anchor.tif'")
    ap.add_argument('--mola', required=True)
    ap.add_argument('--pedr', required=True)
    ap.add_argument('--exclude-orbits', default=None, help='file of MOLA orbit numbers to leave out')
    ap.add_argument('--out', required=True, help='output directory of the tile')
    ap.add_argument('--jobs', type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    work = out / 'leveled'
    work.mkdir(parents=True, exist_ok=True)
    frames = frames_on_tile_grid(args.tile, glob.glob(args.frames), work)
    excl = [int(t) for t in open(args.exclude_orbits).read().split()] if args.exclude_orbits else None
    level_tile(frames, args.mola, args.pedr, out / 'leveled.tif', work, excl, args.jobs)


if __name__ == '__main__':
    main()

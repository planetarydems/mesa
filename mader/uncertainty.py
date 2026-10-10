"""Per-pixel uncertainty layer (Section 4.3 of the paper; Text S12).

The expected error of a predicted pixel is the median absolute error of test
pixels with the same predicted slope, local incidence and distance to the
nearest shot, all three known at inference. Infilled pixels are calibrated by
their distance to the nearest predicted pixel.

calibrate   per test pair: histograms of |prediction - stereo DEM| (0.25 m bins
            to 300 m) by cell, from the co-registered pair grids
calib-infill  per test pair: histograms of |product - stereo DEM| for infilled
            pixels, by distance to the nearest predicted pixel
lut         pools the histograms into the look-up table. A cell keeps its own
            median when it holds 10,000 pixels from 20 pairs; otherwise it falls
            back to the slope x local-incidence marginal, then the slope
            marginal, then all pixels.
layer       the expected-error raster and class raster of a product tile:
            0 reliable (< 10 m), 1 caution (10-25 m), 2 unreliable (>= 25 m),
            3 infilled, 255 no data.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from .geo import parse_crs
from .pedr import PedrDatabase, shot_confidence, shot_distance

SLOPE_EDGES = [0, 1, 2, 5, 10, 15, 20, 30, np.inf]
LINC_EDGES = [0, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]          # last bin: self-shadowed
D1_EDGES = [0, 200, 500, 1000, 2000, 5000, np.inf]
INFILL_EDGES = [0, 500, 2000, np.inf]
ERR_BIN, ERR_MAX = 0.25, 300.0
NERR = int(ERR_MAX / ERR_BIN) + 1                                 # last bin collects >= 300 m
NS, NL, ND = len(SLOPE_EDGES) - 1, len(LINC_EDGES) - 1, len(D1_EDGES) - 1
MIN_PX, MIN_PAIRS = 10000, 20
CLASS_T1, CLASS_T2 = 10.0, 25.0


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def sun_vector(incidence_deg, azimuth_deg):
    """Unit vector towards the Sun (east, north, up) in double precision."""
    i, a = math.radians(incidence_deg), math.radians(azimuth_deg)
    return np.array([math.sin(i) * math.sin(a), math.sin(i) * math.cos(a), math.cos(i)], np.float64)


def ground_spacing(proj, transform, H, W):
    """(pixel width per row [H, 1], pixel height) in ground metres."""
    kind, R, lon0, lat_ts = proj
    px, py = abs(transform.a), abs(transform.e)
    y = transform.f + (np.arange(H, dtype=np.float64) + 0.5) * transform.e
    if kind == 'eqc':
        return (px * np.cos(y / R) / math.cos(math.radians(lat_ts)))[:, None], py
    if kind == 'sinu':
        return np.full((H, 1), px), py
    x0 = transform.c + (W / 2.0) * transform.a                    # stereographic: scale at the column centre
    lat = np.pi / 2 - 2 * np.arctan(np.sqrt(x0 ** 2 + y ** 2) / (2 * R))
    return (px / (2.0 / (1.0 + np.sin(lat))))[:, None], py


def slope_and_incidence(z, px_x, px_y, sun):
    """Slope (deg) and local incidence (deg, > 90 = self-shadowed) of a DEM
    (NaN = no data), with sun [3] or [H, W, 3] = (east, north, up)."""
    zz = z.astype(np.float64, copy=True)
    v = np.isfinite(zz)
    zz[~v] = 0.0
    gy, gx = np.gradient(zz)
    dzdx, dzdy = gx / px_x, -gy / px_y
    ok = v.copy()
    ok[1:, :] &= v[:-1, :]
    ok[:-1, :] &= v[1:, :]
    ok[:, 1:] &= v[:, :-1]
    ok[:, :-1] &= v[:, 1:]
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    nrm = np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
    if np.ndim(sun) == 1:
        cosi = (-dzdx * sun[0] - dzdy * sun[1] + sun[2]) / nrm
    else:
        cosi = (-dzdx * sun[..., 0] - dzdy * sun[..., 1] + sun[..., 2]) / nrm
    linc = np.degrees(np.arccos(np.clip(cosi, -1.0, 1.0)))
    slope[~ok] = np.nan
    linc[~ok] = np.nan
    return slope.astype(np.float32), linc.astype(np.float32)


def shot_distance_layer(pedr, proj, transform, W, H, pixel_size=18.0):
    """d1 (m) recovered from the network's own input layer d_shot = exp(-d1 / 2000)."""
    d1 = shot_distance(pedr, proj, transform, W, H, pixel_size)
    conf = np.zeros((H, W), np.float32) if d1 is None else shot_confidence(d1)
    with np.errstate(divide='ignore'):
        d = -2000.0 * np.log(conf.astype(np.float64))
    d[~np.isfinite(d)] = 60000.0
    return np.clip(d, 0.0, 60000.0).astype(np.float32)


def bin_index(x, edges):
    i = np.searchsorted(np.asarray(edges[1:-1], np.float64), x, side='right').astype(np.int16)
    i[~np.isfinite(x)] = -1
    return i


def cell_index(slope, linc, d1):
    s, l, d = bin_index(slope, SLOPE_EDGES), bin_index(linc, LINC_EDGES), bin_index(d1, D1_EDGES)
    c = (s.astype(np.int32) * NL + l) * ND + d
    c[(s < 0) | (l < 0) | (d < 0)] = -1
    return c


def error_histogram(index, err, n_index):
    """[n_index, NERR] counts of |error| per index value (NaN skipped)."""
    m = (index >= 0) & np.isfinite(err)
    e = np.minimum((err[m] / ERR_BIN).astype(np.int64), NERR - 1)
    return np.bincount(index[m].astype(np.int64) * NERR + e, minlength=n_index * NERR).reshape(n_index, NERR)


def hist_quantile(h, q):
    n = h.sum()
    if n == 0:
        return float('nan')
    c = np.cumsum(h)
    k = int(np.searchsorted(c, q * n))
    prev = c[k - 1] if k > 0 else 0
    return float((k + (q * n - prev) / max(h[k], 1)) * ERR_BIN)


def hist_mean(h):
    n = h.sum()
    return float('nan') if n == 0 else float((h * (np.arange(len(h)) + 0.5) * ERR_BIN).sum() / n)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _read(path):
    with rasterio.open(path) as d:
        a = d.read(1).astype(np.float64)
        a[(a == d.nodata) | (a < -1e5)] = np.nan
        return a, d.transform, d.crs


def calibrate_pair(pred_path, stereo_path, inc, az, pedr):
    """Histograms for one test pair from its co-registered grids."""
    pred, tr, crs = _read(pred_path)
    ster, _, _ = _read(stereo_path)
    proj = parse_crs(crs)
    H, W = pred.shape
    px_x, px_y = ground_spacing(proj, tr, H, W)
    slope, linc = slope_and_incidence(pred, px_x, px_y, sun_vector(inc, az))
    d1 = shot_distance_layer(pedr, proj, tr, W, H, pixel_size=abs(tr.a))
    err = np.abs(pred - ster)
    err[~(np.isfinite(pred) & np.isfinite(ster))] = np.nan
    cell = cell_index(slope, linc, d1)
    ok = (cell >= 0) & np.isfinite(err)
    s = (slice(3, None, 8), slice(3, None, 8))                     # 1-in-64 pixel sample
    sample = np.column_stack([slope[s][ok[s]], linc[s][ok[s]], d1[s][ok[s]], err[s][ok[s]]]).astype(np.float32)
    return dict(hist=error_histogram(cell, err, NS * NL * ND).astype(np.uint32), sample=sample, n=int(ok.sum()))


def covering_tiles(label_path, tiles_root, dem='deseam.tif', prov='infill.prov.tif'):
    """Product tiles that can cover a stereo pair, longitude-major, as (dem, prov) paths."""
    with rasterio.open(label_path) as s:
        lo0, la0, lo1, la1 = transform_bounds(s.crs, '+proj=longlat +R=3396190 +no_defs', *s.bounds, densify_pts=21)
    lons = sorted({int(math.floor((((x + 180) % 360) - 180) / 10) * 10) for x in (lo0, (lo0 + lo1) / 2, lo1)})
    lats = sorted({max(-90, min(80, int(math.floor(y / 10) * 10))) for y in (la0, (la0 + la1) / 2, la1)})
    name = lambda lo, la: 'E%s%03d_N%s%02d' % ('p' if lo >= 0 else '-', abs(lo), 'p' if la >= 0 else '-', abs(la))
    out = []
    for lo in lons:
        for la in lats:
            d = os.path.join(tiles_root, name(lo, la))
            if os.path.exists(os.path.join(d, dem)):
                out.append((os.path.join(d, dem), os.path.join(d, prov)))
    return out


def calibrate_infill_pair(label_path, tile_paths):
    """Histograms of |product - stereo DEM| for one test pair, in place.
    tile_paths: [(deseam.tif, infill.prov.tif)] of the product tiles that can
    cover the pair. Infilled pixels are binned by the distance to the nearest
    predicted pixel (the provenance mask is read 300 px beyond the pair so the
    distance is not truncated)."""
    with rasterio.open(label_path) as s:
        lab = s.read(1).astype(np.float32)
        crs, tf, (H, W) = s.crs, s.transform, s.shape
        if s.nodata is not None:
            lab[lab == s.nodata] = np.nan
    lab[(lab < -1e4) | (lab > 3e4)] = np.nan
    pad = 300
    tfp = tf * Affine.translation(-pad, -pad)
    Hp, Wp = H + 2 * pad, W + 2 * pad
    fin = np.full((H, W), np.nan, np.float32)
    prov = np.full((Hp, Wp), 255, np.uint8)
    for dem_path, prov_path in tile_paths:
        with rasterio.open(dem_path) as src, WarpedVRT(src, crs=crs, transform=tf, width=W, height=H,
                                                       resampling=Resampling.bilinear, src_nodata=-9999.0,
                                                       nodata=-9999.0) as v:
            f = v.read(1).astype(np.float32)
        f[f <= -9000] = np.nan
        with rasterio.open(prov_path) as src, WarpedVRT(src, crs=crs, transform=tfp, width=Wp, height=Hp,
                                                        resampling=Resampling.nearest, src_nodata=255,
                                                        nodata=255) as v:
            p = v.read(1)
        k = np.isnan(fin) & np.isfinite(f)
        fin[k] = f[k]
        k = (prov == 255) & (p != 255)
        prov[k] = p[k]
    dist = (distance_transform_edt(prov != 0) * 18.0)[pad:pad + H, pad:pad + W].astype(np.float32)
    provc = prov[pad:pad + H, pad:pad + W]
    err = np.abs(fin - lab)
    ok = np.isfinite(err)
    di = bin_index(dist, INFILL_EDGES)
    hist = error_histogram(np.where(ok & (provc == 1), di, -1), err, len(INFILL_EDGES) - 1)
    hmodel = error_histogram(np.where(ok & (provc == 0), 0, -1), err, 1)[0]
    return dict(hist=hist.astype(np.uint32), hmodel=hmodel.astype(np.uint32))


def build_lut(pair_npz, infill_npz=()):
    """Look-up table from the per-pair calibration files."""
    Hc = np.zeros((NS * NL * ND, NERR), np.int64)
    pairs_per_cell = np.zeros(NS * NL * ND, np.int64)
    for f in pair_npz:
        h = np.load(f)['hist'].astype(np.int64)
        Hc += h
        pairs_per_cell += h.sum(1) >= 100
    H3 = Hc.reshape(NS, NL, ND, NERR)
    P3 = pairs_per_cell.reshape(NS, NL, ND)
    h_sl, h_s, h_g, h_sd = H3.sum(2), H3.sum((1, 2)), H3.sum((0, 1, 2)), H3.sum(1)
    npx = H3.sum(3)
    med = np.full((NS, NL, ND), np.nan)
    p90, level = med.copy(), np.zeros((NS, NL, ND), np.int8)
    for s in range(NS):
        for li in range(NL):
            for d in range(ND):
                if npx[s, li, d] >= MIN_PX and P3[s, li, d] >= MIN_PAIRS:
                    h, lv = H3[s, li, d], 0
                elif h_sl[s, li].sum() >= MIN_PX:
                    h, lv = h_sl[s, li], 1
                elif h_s[s].sum() >= MIN_PX:
                    h, lv = h_s[s], 2
                else:
                    h, lv = h_g, 3
                med[s, li, d], p90[s, li, d], level[s, li, d] = hist_quantile(h, 0.5), hist_quantile(h, 0.9), lv
    # pixels without a frame (no local incidence): slope x d1 table
    nolinc = [h_sd[s, d] if h_sd[s, d].sum() >= MIN_PX else h_s[s] for s in range(NS) for d in range(ND)]
    lut = dict(cell_median=med.ravel().tolist(), cell_p90=p90.ravel().tolist(), cell_level=level.ravel().tolist(),
               nolinc_median=[hist_quantile(h, 0.5) for h in nolinc],
               nolinc_p90=[hist_quantile(h, 0.9) for h in nolinc],
               global_median=hist_quantile(h_g, 0.5), n_pairs=len(pair_npz), n_pixels=int(Hc.sum()))
    if infill_npz:
        HI = sum(np.load(f)['hist'].astype(np.int64) for f in infill_npz)
        lut['infill_median'] = [hist_quantile(HI[c], 0.5) for c in range(HI.shape[0])]
        lut['infill_p90'] = [hist_quantile(HI[c], 0.9) for c in range(HI.shape[0])]
    return lut


class LUT:

    def __init__(self, path):
        j = json.load(open(path))
        self.med = np.asarray(j['cell_median'], np.float32).reshape(NS, NL, ND)
        self.p90 = np.asarray(j['cell_p90'], np.float32).reshape(NS, NL, ND)
        self.nl_med = np.asarray(j['nolinc_median'], np.float32).reshape(NS, ND)
        self.nl_p90 = np.asarray(j['nolinc_p90'], np.float32).reshape(NS, ND)
        self.infill_med = np.asarray(j['infill_median'], np.float32)
        self.infill_p90 = np.asarray(j['infill_p90'], np.float32)

    def lookup(self, slope, linc, d1):
        s, l, d = bin_index(slope, SLOPE_EDGES), bin_index(linc, LINC_EDGES), bin_index(d1, D1_EDGES)
        ok = (s >= 0) & (l >= 0) & (d >= 0)
        s0, l0, d0 = np.where(ok, s, 0), np.where(ok, l, 0), np.where(ok, d, 0)
        med, p90 = self.med[s0, l0, d0], self.p90[s0, l0, d0]
        med[~ok] = np.nan
        p90[~ok] = np.nan
        nl = (s >= 0) & (d >= 0) & (l < 0)
        if nl.any():
            med[nl], p90[nl] = self.nl_med[s[nl], d[nl]], self.nl_p90[s[nl], d[nl]]
        return med, p90

    def lookup_infill(self, dist):
        i = bin_index(dist, INFILL_EDGES)
        ok = i >= 0
        i = np.where(ok, i, 0)
        med, p90 = self.infill_med[i], self.infill_p90[i]
        med[~ok] = np.nan
        p90[~ok] = np.nan
        return med, p90


# ---------------------------------------------------------------------------
# Layer of a product tile
# ---------------------------------------------------------------------------

def tile_layer(lut, dem_path, prov_path, domid_path, image_suns, pedr, out_prefix, rows=2048, dec=8):
    """image_suns: [(incidence, azimuth)] in the order of the domid indices
    (the sorted levelled DEMs of the tile); None where unknown.
    Writes <prefix>_expected_error_m.tif and <prefix>_uncertainty_class.tif."""
    suns = np.array([sun_vector(*s) if s is not None else [np.nan] * 3 for s in image_suns], np.float64)
    with rasterio.open(dem_path) as D, rasterio.open(prov_path) as PV, rasterio.open(domid_path) as DM:
        assert PV.shape == D.shape == DM.shape
        profile, (H, W), tr, nod = D.profile, D.shape, D.transform, D.nodata
        proj = parse_crs(D.crs)
        px = float(abs(tr.a))
        d1 = shot_distance_layer(pedr, proj, tr, W, H, pixel_size=px)
        # distance to the nearest predicted pixel, at dec x 18 m
        Hd, Wd = -(-H // dec), -(-W // dec)
        model_dec = np.zeros((Hd, Wd), bool)
        for r0 in range(0, H, rows):
            pv = PV.read(1, window=Window(0, r0, W, min(rows, H - r0)))
            h = pv.shape[0]
            hp, wp = -(-h // dec) * dec, Wd * dec
            blk = np.zeros((hp, wp), bool)
            blk[:h, :W] = pv == 0
            model_dec[r0 // dec:r0 // dec + hp // dec] |= blk.reshape(hp // dec, dec, wp // dec, dec).any((1, 3))
        dist_dec = ((distance_transform_edt(~model_dec) * dec * px).astype(np.float32) if model_dec.any()
                    else np.full((Hd, Wd), 60000.0, np.float32))
        pe = dict(profile, driver='GTiff', dtype='float32', nodata=-9999.0, compress='deflate', predictor=3,
                  tiled=True, blockxsize=512, blockysize=512, bigtiff='IF_SAFER', count=1)
        pc = dict(pe, dtype='uint8', nodata=255, predictor=2)
        tmp_e, tmp_c = f'{out_prefix}_tmp_expected.tif', f'{out_prefix}_tmp_class.tif'
        px_x_all, px_y = ground_spacing(proj, tr, H, W)
        area = np.zeros(4)
        with rasterio.open(tmp_e, 'w', **pe) as OE, rasterio.open(tmp_c, 'w', **pc) as OC:
            for r0 in range(0, H, rows):
                r1 = min(H, r0 + rows)
                a0, a1 = max(0, r0 - 1), min(H, r1 + 1)                # one row of halo for the gradient
                win = Window(0, a0, W, a1 - a0)
                z = D.read(1, window=win).astype(np.float64)
                z[(z == nod) | (z < -1e5) | ~np.isfinite(z)] = np.nan
                pv, dm = PV.read(1, window=win), DM.read(1, window=win)
                sun = np.full(z.shape + (3,), np.nan)
                ok = dm >= 0
                sun[ok] = suns[np.clip(dm[ok], 0, len(suns) - 1)]
                slope, linc = slope_and_incidence(z, px_x_all[a0:a1], px_y, sun)
                s = slice(r0 - a0, r0 - a0 + (r1 - r0))
                slope, linc, z, pv = slope[s], linc[s], z[s], pv[s]
                med, _ = lut.lookup(slope, linc, d1[r0:r1])
                infill = pv == 1
                model = (pv == 0) & np.isfinite(z)
                dd = np.repeat(np.repeat(dist_dec[r0 // dec:-(-r1 // dec)], dec, 0), dec, 1)[:r1 - r0, :W]
                imed, _ = lut.lookup_infill(dd)
                exp_err = np.where(model, med, np.where(infill, imed, np.nan)).astype(np.float32)
                cls = np.full(z.shape, 255, np.uint8)
                cls[model & (exp_err < CLASS_T1)] = 0
                cls[model & (exp_err >= CLASS_T1) & (exp_err < CLASS_T2)] = 1
                cls[model & (exp_err >= CLASS_T2)] = 2
                cls[infill] = 3
                cls[model & ~np.isfinite(exp_err)] = 2
                row_area = (px_x_all[r0:r1, 0] * px_y)[:, None]
                for k in range(4):
                    area[k] += float(((cls == k) * row_area).sum())
                exp_err[~np.isfinite(exp_err)] = -9999.0
                OE.write(exp_err, 1, window=Window(0, r0, W, r1 - r0))
                OC.write(cls, 1, window=Window(0, r0, W, r1 - r0))
    with rasterio.open(tmp_e, 'r+') as f:
        f.build_overviews([2, 4, 8, 16, 32, 64], Resampling.average)
    with rasterio.open(tmp_c, 'r+') as f:
        f.build_overviews([2, 4, 8, 16, 32, 64], Resampling.mode)
    for tmp, name, rs in ((tmp_e, 'expected_error_m', 'AVERAGE'), (tmp_c, 'uncertainty_class', 'MODE')):
        dst = f'{out_prefix}_{name}.tif'
        gdal_translate = shutil.which('gdal_translate')
        if gdal_translate and subprocess.run([gdal_translate, '-q', '-of', 'COG', '-co', 'COMPRESS=DEFLATE',
                                              '-co', 'PREDICTOR=YES', '-co', 'BIGTIFF=IF_SAFER',
                                              '-co', f'OVERVIEW_RESAMPLING={rs}', '-co', 'NUM_THREADS=ALL_CPUS',
                                              tmp, dst]).returncode == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, dst)
    names = ('reliable', 'caution', 'unreliable', 'infilled')
    fractions = {name: float(a / area.sum()) for name, a in zip(names, area)}
    json.dump(dict(class_area_fraction=fractions, class_area_m2=area.tolist()), open(f'{out_prefix}_summary.json', 'w'),
              indent=1)
    return fractions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('calibrate', help='one test pair')
    c.add_argument('--pred', required=True)
    c.add_argument('--stereo', required=True)
    c.add_argument('--inc', type=float, required=True)
    c.add_argument('--az', type=float, required=True)
    c.add_argument('--pedr', required=True)
    c.add_argument('--out', required=True, help='.npz')
    i = sub.add_parser('calib-infill', help='one test pair, infilled pixels in place')
    i.add_argument('--label', required=True, help='stereo DEM of the pair')
    i.add_argument('--tiles', nargs='+', required=True, help='deseam.tif:infill.prov.tif pairs')
    i.add_argument('--out', required=True)
    l_ = sub.add_parser('lut')
    l_.add_argument('--pairs', required=True, help="glob of calibrate outputs")
    l_.add_argument('--infill', required=True, help="glob of calib-infill outputs")
    l_.add_argument('--out', required=True)
    t = sub.add_parser('layer', help='one product tile')
    t.add_argument('--lut', required=True)
    t.add_argument('--dem', required=True)
    t.add_argument('--prov', required=True)
    t.add_argument('--domid', required=True)
    t.add_argument('--suns', required=True, help='CSV image,incidence,azimuth for the tile, any order')
    t.add_argument('--levelled', required=True, help="glob of the tile's levelled DEMs (defines the index order)")
    t.add_argument('--pedr', required=True)
    t.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()
    if args.cmd == 'calibrate':
        np.savez_compressed(args.out, **calibrate_pair(args.pred, args.stereo, args.inc, args.az,
                                                       PedrDatabase(args.pedr)))
    elif args.cmd == 'calib-infill':
        np.savez_compressed(args.out, **calibrate_infill_pair(args.label, [p.split(':') for p in args.tiles]))
    elif args.cmd == 'lut':
        lut = build_lut(sorted(glob.glob(args.pairs)), sorted(glob.glob(args.infill)))
        json.dump(lut, open(args.out, 'w'), indent=1,
                  default=lambda x: None if isinstance(x, float) and math.isnan(x) else float(x))
    else:
        import csv
        table = {r['image']: (float(r['incidence']), float(r['azimuth'])) for r in csv.DictReader(open(args.suns))}
        names = [os.path.basename(p).split('_leveled')[0][:15] for p in sorted(glob.glob(args.levelled))]
        tile_layer(LUT(args.lut), args.dem, args.prov, args.domid, [table.get(n) for n in names],
                   PedrDatabase(args.pedr), args.out)


if __name__ == '__main__':
    main()

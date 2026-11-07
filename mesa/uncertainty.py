"""Per-pixel uncertainty layer (Section 4.3 of the paper; Text S13).

The released layer is built in two stages.

Stage 1, risk score. The median absolute error of single-image test
predictions after image co-registration, in bins of predicted slope, local
incidence and distance to the nearest shot (all three known at inference).
Infilled pixels are binned by their distance to the nearest predicted pixel.
The table is applied to the assembled mosaic with the solar geometry of the
image of largest blend weight. The result is a score in meters.

calibrate   per test pair: histograms of |prediction - stereo DEM| (0.25 m bins
            to 300 m) by cell, from the co-registered pair grids
calib-infill  per test pair: histograms of |product - stereo DEM| for infilled
            pixels, by distance to the nearest predicted pixel
lut         pools the histograms into the look-up table. A cell keeps its own
            median when it holds 10,000 pixels from 20 pairs; otherwise it falls
            back to the slope x local-incidence marginal, then the slope
            marginal, then all pixels.
layer       the score raster and class raster of a product tile.

Stage 2, mosaic calibration (the released values). Each pixel is grouped by
its score (bins 0, 5, 10, 15, 25, 40, 60 m), by the number of images averaged
at it (overlap depth of the footprint layer, N = 1 or N >= 2) and, for
infilled pixels, by the distance class of stage 1. The released values are
the median and the 90th percentile of |mosaic - stereo DEM| of the group over
the test pairs, after the image co-registration of the mosaic itself (level
kept, nothing fitted to the reference).

calibrate-mosaic  per test pair: histograms of |mosaic - stereo DEM| by group,
            co-registered (rows of data/uncertainty_mosaic_coreg.csv) and in
            place
mosaic-lut  pools them. A group keeps its own quantiles when it holds 10,000
            pixels from 20 test sites (a site counts with 100 pixels).
            Otherwise the score bin with N pooled, then score bins pooled
            outward toward the middle of the score range, then all predicted
            pixels. An infill class falls back to all infilled pixels.
remap       the released rasters of a tile by streamed look-up:
            <prefix>_median_error_m.tif, <prefix>_p90_error_m.tif (float32,
            nodata -9999) and <prefix>_uncertainty_class.tif (uint8: 0 median
            below 10 m, 1 10-25 m, 2 25 m or more, 3 infilled, 255 nodata).
mean-baseline  validation baseline, not part of the product: at each pixel the
            arithmetic mean of the stage-1 estimates of the images with a blend
            weight there, each image counted once.

The values are empirical quantiles of the absolute elevation error against
reference DEMs, pooled over groups of pixels. They are not standard deviations
and do not give the uncertainty of a depth, slope or profile measured from the
DEM, which also depends on the spatial correlation of the error.
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


# ---------------------------------------------------------------------------
# Stage 2: calibration against the assembled mosaic
# ---------------------------------------------------------------------------

SCORE_EDGES = [0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, np.inf]
NSB = len(SCORE_EDGES) - 1
COV_NAMES = ('N1', 'N2plus', 'Nunknown')                          # overlap depth 1, >= 2, 0 on a predicted pixel
NCC = len(COV_NAMES)
CODE_PRED0 = 1                                                     # predicted: 1 + 3 * score bin + coverage class
CODE_PRED_NOSCORE = CODE_PRED0 + NSB * NCC                         # predicted pixel without a score
CODE_INF0 = CODE_PRED_NOSCORE + 1                                  # infilled: + distance class (0, 1, 2, 3 = unknown)
INFILL_NAMES = ('lt0.5km', '0.5-2km', 'gt2km', 'unknown')
NCODE = CODE_INF0 + len(INFILL_NAMES)
PBIN, NPB = 0.25, 401                                              # score bins of the joint histograms (to 100 m)
MIN_SITES, MIN_SITE_PX = 20, 100
PAD_M = 2300.0


def coverage_class(n):
    n = np.asarray(n)
    return np.where(n == 1, 0, np.where(n >= 2, 1, 2)).astype(np.int16)


def infill_class_from_score(score, infill_values):
    """Distance class of an infilled pixel from its stage-1 value (the stage-1
    infill medians are distinct, one per class); 3 where none matches."""
    k = np.full(np.shape(score), 3, np.int16)
    for i, v in enumerate(np.asarray(infill_values, np.float32)):
        k[score == v] = i
    return k


def group_codes(prov, score, depth, infill_values):
    """int16 group of every pixel (0 = not scored) from the provenance mask
    (0 predicted, 1 infilled), the stage-1 score and the overlap depth."""
    code = np.zeros(prov.shape, np.int16)
    sb = bin_index(score, SCORE_EDGES)
    c = (CODE_PRED0 + sb.astype(np.int32) * NCC + coverage_class(depth)).astype(np.int16)
    pred, inf = prov == 0, prov == 1
    code[pred] = np.where(sb[pred] >= 0, c[pred], CODE_PRED_NOSCORE)
    code[inf] = CODE_INF0 + infill_class_from_score(score[inf], infill_values)
    return code


def _window(ds, crs, transform, H, W, margin_px=4):
    left, bottom, right, top = rasterio.transform.array_bounds(H, W, transform)
    bb = transform_bounds(crs, ds.crs, left, bottom, right, top, densify_pts=41)
    inv = ~ds.transform
    xs, ys = zip(*[inv * (x, y) for x in (bb[0], bb[2]) for y in (bb[1], bb[3])])
    c0, c1 = max(int(math.floor(min(xs))) - margin_px, 0), min(int(math.ceil(max(xs))) + margin_px, ds.width)
    r0, r1 = max(int(math.floor(min(ys))) - margin_px, 0), min(int(math.ceil(max(ys))) + margin_px, ds.height)
    return None if c1 <= c0 or r1 <= r0 else Window(c0, r0, c1 - c0, r1 - r0)


def read_on_grid(path, crs, transform, H, W, resampling, dtype, nodata, fill):
    """A raster read through a window and resampled onto a grid (rasterio
    reproject), or None where the grid lies outside it."""
    from rasterio.warp import reproject
    with rasterio.open(path) as s:
        win = _window(s, crs, transform, H, W)
        if win is None:
            return None
        a = s.read(1, window=win)
        wtr, scrs = s.window_transform(win), s.crs
    if np.issubdtype(np.dtype(dtype), np.floating):
        a = a.astype(np.float32)
        if nodata is not None:
            a[a == nodata] = np.nan
        a[(a < -1e5) | (a > 1e5)] = np.nan
        nodata = np.nan
    out = np.full((H, W), fill, dtype)
    reproject(a, out, src_transform=wtr, src_crs=scrs, src_nodata=nodata, dst_transform=transform, dst_crs=crs,
              dst_nodata=fill, resampling=resampling)
    return out


def nominal_tiles(crs, transform, H, W, step=4):
    """Index of the 10 x 10 degree tile each pixel belongs to and the tile names.
    lon / lat are computed every `step` pixels (tiles overlap by >= 6 km)."""
    from pyproj import CRS as PCRS, Transformer
    t = Transformer.from_crs(PCRS.from_user_input(crs), PCRS.from_proj4('+proj=longlat +R=3396190 +no_defs'),
                             always_xy=True)
    rr = np.minimum(np.arange(0, H, step) + step / 2.0, H - 0.5)
    cc = np.minimum(np.arange(0, W, step) + step / 2.0, W - 0.5)
    C, R = np.meshgrid(cc, rr)
    lon, lat = (np.asarray(v) for v in t.transform(transform.c + C * transform.a, transform.f + R * transform.e))
    ok = np.isfinite(lon) & np.isfinite(lat)
    lo = np.floor((((np.where(ok, lon, 0.0) + 180.0) % 360.0) - 180.0) / 10.0) * 10.0
    la = np.clip(np.floor(np.where(ok, lat, 0.0) / 10.0) * 10.0, -90, 80)
    key = np.where(ok, (lo + 180).astype(np.int32) * 1000 + (la + 90).astype(np.int32), -1)
    names, idx = [], np.full(key.shape, -1, np.int16)
    for k in np.unique(key[key >= 0]):
        idx[key == k] = len(names)
        lo_, la_ = int(k // 1000) - 180, int(k % 1000) - 90
        names.append('E%s%03d_N%s%02d' % ('p' if lo_ >= 0 else '-', abs(lo_), 'p' if la_ >= 0 else '-', abs(la_)))
    return np.repeat(np.repeat(idx, step, 0), step, 1)[:H, :W], names


def mosaic_on_grid(crs, transform, H, W, paths):
    """The released layers of the covering tiles on a grid, each pixel from the
    tile whose own 10 degree cell holds it, else from any tile with data.
    paths(tile) -> dict(dem=, prov=, score=, depth=) or None. Returns dict of
    dem (bilinear, float32 NaN), prov (uint8, 255 nodata), score (float32 NaN),
    depth (int16, 0 nodata), all others nearest."""
    idx, names = nominal_tiles(crs, transform, H, W)
    out = dict(dem=np.full((H, W), np.nan, np.float32), prov=np.full((H, W), 255, np.uint8),
               score=np.full((H, W), np.nan, np.float32), depth=np.zeros((H, W), np.int16))
    layers = {}
    for i, t in enumerate(names):
        p = paths(t)
        if p is None or not os.path.exists(p['dem']):
            continue
        prov = read_on_grid(p['prov'], crs, transform, H, W, Resampling.nearest, np.uint8, 255, 255)
        if prov is None:
            continue
        L = dict(prov=prov, dem=read_on_grid(p['dem'], crs, transform, H, W, Resampling.bilinear, np.float32, -9999.0, np.nan),
                 score=read_on_grid(p['score'], crs, transform, H, W, Resampling.nearest, np.float32, -9999.0, np.nan),
                 depth=read_on_grid(p['depth'], crs, transform, H, W, Resampling.nearest, np.int16, 0, 0))
        layers[i] = L
    for first in (True, False):
        for i, L in layers.items():
            ok = (L['prov'] != 255) & np.isfinite(L['dem'])
            ok &= (idx == i) if first else (out['prov'] == 255)
            for k in out:
                out[k][ok] = L[k][ok]
    out['dem'][out['prov'] == 255] = np.nan
    return out


def _shift_any(a, dr, dc, fill):
    """out[r, c] = a[r + dr, c + dc], fill outside (any dtype)."""
    h, w = a.shape
    out = np.full_like(a, fill)
    y0, y1, x0, x1 = max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = a[y0 + dr:y1 + dr, x0 + dc:x1 + dc]
    return out


def _rotate_shift_nearest(a, transform, T, crop, spacing, left, top):
    """evaluate_images.rotate_shift with nearest-neighbour sampling (group maps), 0 outside."""
    from scipy.ndimage import map_coordinates
    H, W = a.shape
    r0, c0 = crop
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    x = (transform.c + cols * transform.a - left) / spacing - c0 - 0.5
    y = (top - (transform.f + rows * transform.e)) / spacing - r0 - 0.5
    x2, y2 = T[0, 0] * x + T[0, 1] * y + T[0, 2], T[1, 0] * x + T[1, 1] * y + T[1, 2]
    X2, Y2 = left + (c0 + x2 + 0.5) * spacing, top - (r0 + y2 + 0.5) * spacing
    return map_coordinates(a, [(Y2 - transform.f) / transform.e - 0.5, (X2 - transform.c) / transform.a - 0.5],
                           order=0, mode='constant', cval=0)


def _group_hist(code, err, valid):
    m = valid & (code > 0) & np.isfinite(err)
    e = np.minimum((err[m] / ERR_BIN).astype(np.int64), NERR - 1)
    return np.bincount(code[m].astype(np.int64) * NERR + e, minlength=NCODE * NERR).reshape(NCODE, NERR)


def _joint(code, score, err, valid):
    m = valid & (code >= CODE_PRED0) & (code < CODE_PRED_NOSCORE) & np.isfinite(err) & np.isfinite(score)
    cc = (code[m] - CODE_PRED0) % NCC
    pb = np.minimum((score[m] / PBIN).astype(np.int64), NPB - 1)
    eb = np.minimum((err[m] / ERR_BIN).astype(np.int64), NERR - 1)
    return np.bincount((cc.astype(np.int64) * NPB + pb) * NERR + eb, minlength=NCC * NPB * NERR).reshape(NCC, NPB, NERR)


def calibrate_mosaic_pair(label_path, paths, row, infill_values, mola_path=None):
    """Histograms of |mosaic - stereo DEM| by group for one test pair.
    row: the pair's row of data/uncertainty_mosaic_coreg.csv (the co-registration
    of the mosaic's own CTX image composite with the pair, measured from images
    only). Co-registered: the mosaic moved by the row's transform (cubic spline,
    evaluate_images.coregister), its group map by the same transform (nearest).
    In place: nothing moved. Pixels: stereo DEM, mosaic and (if given) MOLA
    valid, group > 0."""
    from .evaluate_images import coregister, warp, MOLA_NODATA
    with rasterio.open(label_path) as s:
        lab = s.read(1).astype(np.float32)
        tr, crs, nd = s.transform, s.crs, s.nodata
    if nd is not None:
        lab[lab == nd] = np.nan
    lab[(lab < -1e4) | (lab > 3e4)] = np.nan
    H, W = lab.shape
    pad = int(math.ceil(PAD_M / abs(tr.a)))
    trp = tr * Affine.translation(-pad, -pad)
    g = mosaic_on_grid(crs, trp, H + 2 * pad, W + 2 * pad, paths)
    core = (slice(pad, pad + H), slice(pad, pad + W))
    mos, prov, score, depth = (g[k][core].copy() for k in ('dem', 'prov', 'score', 'depth'))
    code = group_codes(prov, score, depth, infill_values)
    valid = np.isfinite(lab)
    if mola_path:
        valid &= np.isfinite(warp(mola_path, lab.shape, tr, crs, MOLA_NODATA))
    out = dict(h_in=_group_hist(code, np.abs(mos - lab), valid & np.isfinite(mos)),
               j_in=_joint(code, score, np.abs(mos - lab), valid & np.isfinite(mos)))
    aligned = coregister(mos.astype(np.float64), tr, row)
    if aligned is None:
        out.update(h_al=np.zeros_like(out['h_in']), j_al=np.zeros_like(out['j_in']))
    else:
        if row['method'] == 'rotation+shift':
            T = np.array([[float(row[k]) for k in ('t00', 't01', 't02')], [float(row[k]) for k in ('t10', 't11', 't12')]])
            args = (tr, T, (int(row['ref_row0']), int(row['ref_col0'])), float(row['ref_spacing_m']),
                    float(row['ref_left']), float(row['ref_top']))
            code_al, score_al = _rotate_shift_nearest(code, *args), _rotate_shift_nearest(np.nan_to_num(score), *args)
        else:
            dr, dc = int(row['shift_rows']), int(row['shift_cols'])
            code_al, score_al = _shift_any(code, dr, dc, 0), _shift_any(score, dr, dc, np.nan)
        score_al = np.where(score_al > 0, score_al, np.nan)
        err = np.abs(aligned - lab)
        ok = valid & np.isfinite(aligned)
        out.update(h_al=_group_hist(code_al, err, ok), j_al=_joint(code_al, score_al, err, ok))
    return {k: v.astype(np.uint32) for k, v in out.items()}


def build_mosaic_lut(pair_npz, site_of_pair, infill_values, key='h_al'):
    """Stage-2 look-up table from the per-pair files of calibrate-mosaic.
    site_of_pair: pair -> test site id. Only pairs with co-registered pixels
    enter (key h_al)."""
    sites, H = {}, []
    for f in pair_npz:
        z = np.load(f)
        h = z[key].astype(np.int64)
        if h.sum() == 0:
            continue
        s = site_of_pair[os.path.basename(f)[:-4]]
        sites.setdefault(s, np.zeros_like(h))
        sites[s] += h
    HS = np.stack(list(sites.values()))                               # [sites, NCODE, NERR]

    def stats(codes):
        hs = HS[:, codes, :].sum(1)
        return hs.sum(0), int(hs.sum()), int((hs.sum(1) >= MIN_SITE_PX).sum())

    def adequate(codes):
        _, n, ns = stats(codes)
        return n >= MIN_PX and ns >= MIN_SITES

    pc = lambda s, c: CODE_PRED0 + s * NCC + c
    all_pred = [pc(s, c) for s in range(NSB) for c in range(NCC)] + [CODE_PRED_NOSCORE]
    all_inf = list(range(CODE_INF0, NCODE))

    def from_l1(s):
        codes = [pc(s, c) for c in range(NCC)]
        if adequate(codes):
            return codes, 1
        lo = hi = s
        while lo > 0 or hi < NSB - 1:
            if s >= 3 and lo > 0:
                lo -= 1
            elif s <= 2 and hi < NSB - 1:
                hi += 1
            elif lo > 0:
                lo -= 1
            else:
                hi += 1
            codes = [pc(t, c) for t in range(lo, hi + 1) for c in range(NCC)]
            if adequate(codes):
                return codes, 2
        return all_pred, 3

    med, p90, lev = [], [], []
    for s in range(NSB):
        for c in range(NCC):
            codes, lv = ([pc(s, c)], 0) if c < 2 and adequate([pc(s, c)]) else from_l1(s)
            h = stats(codes)[0]
            med.append(hist_quantile(h, 0.5)); p90.append(hist_quantile(h, 0.9)); lev.append(lv)
    hp = stats(all_pred)[0]
    imed, ip90, ilev = [], [], []
    for k in range(4):
        if k < 3 and adequate([CODE_INF0 + k]):
            codes, lv = [CODE_INF0 + k], 0
        elif adequate(all_inf):
            codes, lv = all_inf, 1
        else:
            codes, lv = all_pred + all_inf, 2
        h = stats(codes)[0]
        imed.append(hist_quantile(h, 0.5)); ip90.append(hist_quantile(h, 0.9)); ilev.append(lv)
    return dict(score_edges=[*SCORE_EDGES[:-1], 'inf'], cov_names=list(COV_NAMES), infill_names=list(INFILL_NAMES),
                pred_median=med, pred_p90=p90, pred_level=lev, pred_noscore_median=hist_quantile(hp, 0.5),
                pred_noscore_p90=hist_quantile(hp, 0.9), infill_median=imed, infill_p90=ip90, infill_level=ilev,
                old_infill_values=[float(v) for v in infill_values], n_sites=len(sites), n_px=int(HS.sum()))


class MosaicLUT:
    """Stage-2 look-up: group -> median and 90th percentile (meters)."""

    def __init__(self, path):
        j = json.load(open(path))
        self.j = j
        tm = np.full(NCODE, np.nan, np.float32)
        tp = tm.copy()
        tm[CODE_PRED0:CODE_PRED_NOSCORE] = j['pred_median']
        tp[CODE_PRED0:CODE_PRED_NOSCORE] = j['pred_p90']
        tm[CODE_PRED_NOSCORE], tp[CODE_PRED_NOSCORE] = j['pred_noscore_median'], j['pred_noscore_p90']
        tm[CODE_INF0:], tp[CODE_INF0:] = j['infill_median'], j['infill_p90']
        self.table_median, self.table_p90 = tm, tp
        self.infill_values = np.asarray(j['old_infill_values'], np.float32)

    def lookup(self, prov, score, depth):
        code = group_codes(prov, score, depth, self.infill_values)
        med = np.full(code.shape, np.nan, np.float32)
        p90 = med.copy()
        ok = code > 0
        med[ok], p90[ok] = self.table_median[code[ok]], self.table_p90[code[ok]]
        return med, p90


def mosaic_classes(med, prov):
    c = np.full(med.shape, 255, np.uint8)
    pred = (prov == 0) & np.isfinite(med)
    c[pred & (med < CLASS_T1)] = 0
    c[pred & (med >= CLASS_T1) & (med < CLASS_T2)] = 1
    c[pred & (med >= CLASS_T2)] = 2
    c[(prov == 1) & np.isfinite(med)] = 3
    return c


def remap_tile(lut, dem_path, prov_path, score_path, depth_path, out_prefix, p90=True, rows=2048):
    """The released stage-2 rasters of one tile by streamed look-up (all inputs
    on the tile's grid). Nodata where the DEM is nodata or the provenance mask
    is neither 0 nor 1. Returns the class area on the sphere (m2) over the tile file."""
    names = ['median_error_m'] + (['p90_error_m'] if p90 else [])
    with rasterio.open(dem_path) as D, rasterio.open(prov_path) as PV, rasterio.open(score_path) as SC, \
            rasterio.open(depth_path) as DP:
        for d in (PV, SC, DP):
            assert d.shape == D.shape and d.transform.almost_equals(D.transform) and d.crs == D.crs, d.name
        H, W, tr, nod = D.height, D.width, D.transform, D.nodata
        proj = parse_crs(D.crs)
        pe = dict(D.profile, driver='GTiff', dtype='float32', nodata=-9999.0, compress='deflate', predictor=3,
                  tiled=True, blockxsize=512, blockysize=512, bigtiff='IF_SAFER', count=1)
        pc = dict(pe, dtype='uint8', nodata=255, predictor=2)
        tmp = {n: f'{out_prefix}_tmp_{n}.tif' for n in names + ['uncertainty_class']}
        outs = {n: rasterio.open(tmp[n], 'w', **(pc if n == 'uncertainty_class' else pe)) for n in tmp}
        px_x, px_y = ground_spacing(proj, tr, H, W)
        area = np.zeros(4)
        for r0 in range(0, H, rows):
            h = min(rows, H - r0)
            win = Window(0, r0, W, h)
            z = D.read(1, window=win)
            pv = PV.read(1, window=win)
            sc = SC.read(1, window=win).astype(np.float32)
            sc[sc == -9999.0] = np.nan
            dp = DP.read(1, window=win)
            sup = np.isfinite(z) & (z != nod) & (z > -1e5) & (z < 1e5) & ((pv == 0) | (pv == 1))
            pv = np.where(sup, pv, 255).astype(np.uint8)
            med, q90 = lut.lookup(pv, sc, dp)
            cls = mosaic_classes(med, pv)
            row_area = (px_x[r0:r0 + h, 0] * px_y)[:, None]
            for k in range(4):
                area[k] += float(((cls == k) * row_area).sum())
            outs['median_error_m'].write(np.where(np.isfinite(med), med, -9999.0).astype(np.float32), 1, window=win)
            if p90:
                outs['p90_error_m'].write(np.where(np.isfinite(q90), q90, -9999.0).astype(np.float32), 1, window=win)
            outs['uncertainty_class'].write(cls, 1, window=win)
        for o in outs.values():
            o.close()
    for n, path in tmp.items():
        rs = Resampling.mode if n == 'uncertainty_class' else Resampling.average
        with rasterio.open(path, 'r+') as f:
            f.build_overviews([2, 4, 8, 16, 32, 64], rs)
        dst = f'{out_prefix}_{n}.tif'
        gdal_translate = shutil.which('gdal_translate')
        predictor = 'YES' if n == 'uncertainty_class' else 'NO'      # the float layers hold ~20 distinct values
        if gdal_translate and subprocess.run([gdal_translate, '-q', '-of', 'COG', '-co', 'COMPRESS=DEFLATE',
                                              '-co', f'PREDICTOR={predictor}', '-co', 'BIGTIFF=IF_SAFER',
                                              '-co', f'OVERVIEW_RESAMPLING={rs.name.upper()}', '-co',
                                              'NUM_THREADS=ALL_CPUS', path, dst]).returncode == 0:
            os.remove(path)
        else:
            os.replace(path, dst)
    return area.tolist()


def mean_baseline(images, crs, transform, H, W, lut1, pedr):
    """Validation baseline: at each pixel of the grid the arithmetic mean of the
    stage-1 estimates (median and 90th percentile) of the images whose leveled
    DEM has a blend weight there. images: [(product_id, prediction path, weight
    path, incidence, azimuth)] of one tile; a product id is counted once. Each
    image's estimate comes from its own prediction on its own grid (slope, local
    incidence with its own solar geometry, shot distance), resampled to the grid
    by nearest neighbour. Sum of the valid estimates divided by their number,
    without blend weights. Returns (median, p90, number of valid images,
    number of images with a weight), the means NaN where no image has an estimate."""
    sm, sp = np.zeros((H, W)), np.zeros((H, W))
    nv, nc = np.zeros((H, W), np.int16), np.zeros((H, W), np.int16)
    seen = set()
    for pid, pred_path, weight_path, inc, az in images:
        w = read_on_grid(weight_path, crs, transform, H, W, Resampling.nearest, np.float32, 0.0, np.nan)
        if w is None:
            continue
        contrib = np.nan_to_num(w) > 0
        if not contrib.any() or pid in seen:
            continue
        seen.add(pid)
        nc += contrib
        if inc is None or not os.path.exists(pred_path):
            continue
        with rasterio.open(pred_path) as s:
            win = _window(s, crs, transform, H, W)
            if win is None:
                continue
            z = s.read(1, window=win).astype(np.float64)
            ztr, zcrs, znd = s.window_transform(win), s.crs, s.nodata
        z[(z == znd) | (z < -1e5) | ~np.isfinite(z)] = np.nan
        proj = parse_crs(zcrs)
        h, w_ = z.shape
        px_x, px_y = ground_spacing(proj, ztr, h, w_)
        slope, linc = slope_and_incidence(z, px_x, px_y, sun_vector(inc, az))
        d1 = shot_distance_layer(pedr, proj, ztr, w_, h, pixel_size=abs(ztr.a))
        em, e9 = lut1.lookup(slope, linc, d1)
        from rasterio.warp import reproject
        on = []
        for e in (em, e9):
            o = np.full((H, W), np.nan, np.float32)
            reproject(e.astype(np.float32), o, src_transform=ztr, src_crs=zcrs, src_nodata=np.nan, dst_transform=transform,
                      dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.nearest)
            on.append(o)
        v = contrib & np.isfinite(on[0]) & np.isfinite(on[1])
        sm[v] += on[0][v]
        sp[v] += on[1][v]
        nv += v
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.where(nv > 0, sm / nv, np.nan).astype(np.float32), np.where(nv > 0, sp / nv, np.nan).astype(np.float32),
                nv, nc)


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
    tiles_help = "template of the tile's file, with {tile}, e.g. 'tiles/{tile}/deseam.tif'"
    cm = sub.add_parser('calibrate-mosaic', help='stage 2, one test pair')
    cm.add_argument('--label', required=True, help='stereo DEM of the pair')
    cm.add_argument('--pair', required=True, help='pair name (row of --coreg)')
    cm.add_argument('--coreg', default='data/uncertainty_mosaic_coreg.csv')
    cm.add_argument('--dem', required=True, help=tiles_help)
    cm.add_argument('--prov', required=True, help=tiles_help)
    cm.add_argument('--score', required=True, help=tiles_help + ' (stage-1 layer)')
    cm.add_argument('--depth', required=True, help=tiles_help + ' (overlap depth of the footprint layer)')
    cm.add_argument('--lut1', required=True, help='stage-1 look-up table (its infill medians name the infill classes)')
    cm.add_argument('--mola', default=None, help='MOLA gridded DEM as distributed (validity mask of the scorer)')
    cm.add_argument('--out', required=True, help='.npz')
    ml = sub.add_parser('mosaic-lut', help='stage 2, pool the pairs')
    ml.add_argument('--pairs', required=True, help='glob of calibrate-mosaic outputs (<pair>.npz)')
    ml.add_argument('--coreg', default='data/uncertainty_mosaic_coreg.csv', help='gives the site of each pair')
    ml.add_argument('--lut1', required=True)
    ml.add_argument('--out', required=True)
    rm = sub.add_parser('remap', help='stage 2, the released rasters of one tile')
    rm.add_argument('--lut', required=True, help='stage-2 look-up table')
    rm.add_argument('--dem', required=True)
    rm.add_argument('--prov', required=True)
    rm.add_argument('--score', required=True, help='stage-1 layer of the tile')
    rm.add_argument('--depth', required=True, help='overlap depth of the tile')
    rm.add_argument('--no-p90', action='store_true')
    rm.add_argument('--out', required=True, help='output prefix')
    args = ap.parse_args()
    if args.cmd in ('calibrate-mosaic', 'remap', 'mosaic-lut'):
        import csv
        if args.cmd == 'remap':
            area = remap_tile(MosaicLUT(args.lut), args.dem, args.prov, args.score, args.depth, args.out,
                              p90=not args.no_p90)
            print(json.dumps(dict(class_area_m2=area)))
            return
        rows = {r['pair']: r for r in csv.DictReader(open(args.coreg))}
        infill_values = json.load(open(args.lut1))['infill_median']
        if args.cmd == 'mosaic-lut':
            lut = build_mosaic_lut(sorted(glob.glob(args.pairs)), {p: r['site_id'] for p, r in rows.items()}, infill_values)
            json.dump(lut, open(args.out, 'w'), indent=1)
            return
        paths = lambda t: dict(dem=args.dem.format(tile=t), prov=args.prov.format(tile=t),
                               score=args.score.format(tile=t), depth=args.depth.format(tile=t))
        np.savez_compressed(args.out, **calibrate_mosaic_pair(args.label, paths, rows[args.pair], infill_values,
                                                              args.mola))
        return
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

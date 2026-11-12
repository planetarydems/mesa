"""Accuracy against HiRISE stereo DEMs (Section 3.2, Tables 3 and 4 of the paper;
Text S11).

Each HiRISE DEM is compared with every DEM predicted from a single CTX image
that covers it (the output of mesa.predict, before levelling), one window per
CTX image. A window is the predicted DEM over the bounds of the HiRISE DEM
plus 1.2 km (66 px). On its 18 m grid:

  HiRISE      block-averaged onto the grid (GDAL 'average'); the prediction is
              never resampled
  MOLA        the distributed 463 m mosaic, warped bilinearly with its declared
              georeference: no shift, no half-cell correction, no level
  in place    the pixels where HiRISE, the prediction (with and without the
              fit to the shots) and MOLA are all valid, eroded by 3 px; a
              window is scored when it has at least 2,000 of them

Co-registration, from the images alone. The CTX image the prediction was made
from and the HiRISE orthoimage (read at its coarsest JP2 overview of at most
2 m, georeferenced by its PDS label) are block-averaged onto a 6 m grid in the
window's projection, over the orthoimage's extent plus 1.6 km within the
window (at least 3 km across, and 20,000 pixels valid in both). Both are
log-transformed and high-passed (minus a masked Gaussian low-pass at sigma =
8 and 3 px, 48 and 18 m), and each pair of high-passed images gives a
translation by masked normalised cross-correlation (Padfield, 2012) over
+-1.5 km, refined to sub-pixel by a parabola through the peak. The fine
estimate is accepted when the two translations agree within 2 px (12 m), the
better peak reaches 0.3 and the shift is not within 1 px of the search limit;
it is the translation of the better peak. Otherwise the images are averaged
to 12 m and band-passed (log, differences of Gaussians at 24-96, 48-192 and
96-384 m), and a coarse estimate is accepted when two adjacent scales agree
within 2 px (24 m) with a peak of at least 0.3, away from the limit. The
image shift d is the position of a feature in HiRISE minus its position in
the CTX image. HiRISE is block-averaged once more onto the window grid
translated by d, h_d(q) = h(q + d). A stereo DEM of the test set (bilinear
onto the window grid) is co-registered in the same way through its own
orthoimage: the first two of the pair's tied orthoimages are tried and the
better kept (accepted first, then by the fine peak).

Measures of a surface z against HiRISE on a set of pixels:

  deployMAE     mean |z - h|, HiRISE in place
  alignedMAE    mean |z - h_d|
  labelMAE      mean |z' - h_d|, z' being z rescaled to the mean and standard
                deviation of h_d over the pixels
  offset        median of z - h_d (the vertical offset; unsigned in Table 3)
  slope error   RMS difference of the slopes (central differences, 18 m) of z
                and h_d, on the pixels eroded by a further 2 px
  relief        2nd to 98th percentile range of h_d; error / relief = 100
                deployMAE / relief
  scale         the gain a of a robust (Huber) fit h_d ~ a z + b
  bands         the difference-of-Gaussian bands of z and h_d (sigma = 1.5,
                3, 6, 12, 24, 48 and 96 px, normalised low-passes over the
                pixels with edges replicated, in single precision) and the
                remaining low-pass: per band the pixels, covariance and
                variances (each band centred) and the mean squared difference

MOLA is always scored in place (h_d = h). The pixel sets are, for every
window: in place (above: model and MOLA); co-registered, when the CTX image
co-registers, the in-place pixels where h_d is valid, eroded by 3 px (model,
model without the fit to the shots, MOLA); and stereo, for each co-registered
stereo DEM, the co-registered pixels where the stereo DEM and HiRISE moved by
the stereo DEM's shift are valid, eroded by 3 px (model, stereo DEM and MOLA
on identical pixels).

A HiRISE DEM's value is the mean over its windows. Table 3 gives medians over
DEMs. Table 4 pools the band sums over the DEMs, each DEM contributing the
mean over its windows of pixels x statistic: amplitude sqrt(sum var_z / sum
var_h), r = sum cov / sqrt(sum var_z sum var_h), error sqrt(sum squared
difference / sum pixels). Intervals are the 2.5 and 97.5 percentiles of the
median over 2,000 resamplings of the DEMs (seed 0). Image shifts are given in
ground metres (the window's projection scaled at its centre).

    python -m mesa.evaluate_hirise score --dems data/hirise_dems.csv --windows data/hirise_windows.csv \\
        --stereo data/hirise_stereo_pairs.csv --test-pairs data/test_pairs.txt \\
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out hirise_scores.csv --jobs 24
    python -m mesa.evaluate_hirise tables --dems data/hirise_dems.csv --scores hirise_scores.csv \\
        --out hirise_tables.json

The tables of data/ give the 619 HiRISE DEMs (618 of them have a window with
enough pixels), their 1,172 windows and the 149 test pairs over them (132 of
them are scored, at 98 DEMs), with paths relative to the working directory:

hirise_dems.csv           one row per HiRISE DEM: dem, dem_path (the PDS
                          DTM, hirise_dtm/), ortho_path (its orthoimage, JP2
                          with the .LBL beside it, hirise_ortho/), lat, dist_km
                          (distance from the nearest training or validation
                          tile; blank beyond the search radius), pairs (test
                          stereo pairs over it, ';'-separated)
hirise_windows.csv        one row per (HiRISE DEM, CTX image): dem, image,
                          tile, prediction (directory written by mesa.predict,
                          preds/<tile>/<image>), registered_image (the image it
                          was predicted from, aligned_pedr/<tile>/), and
                          registered_r1 (1 or 0, optional; see below)
hirise_stereo_pairs.csv   one row per stereo pair: pair, dem_path (its full
                          stereo DEM, stereo_dems_full/, as scored by
                          mesa.evaluate_images), orthoimages (its tied
                          orthoimages, ';'-separated, in the order tried)

The paper (version 1) keeps the HiRISE windows of pre-release R1 and their outcome of the
image co-registration: a window whose co-registration was not accepted in
pre-release R1 (registered_r1 = 0) is not co-registered, even where its image now
passes the gate (5 windows, whose version-1 prediction covers a slightly different
extent). The image shifts of the accepted windows are recomputed and equal
those of pre-release R1. Without the column the gate alone decides.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
from multiprocessing import Pool

import numpy as np
import rasterio
import scipy.fft
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.stats import spearmanr

from .evaluate import relief
from .geo import R_MARS, parse_crs, xy_to_lonlat

WINDOW_MARGIN_M = 1200.0
ERODE_PX = 3
MIN_PX = 2000
GRID_M = 6.0
ORTHO_MARGIN_M = 1600.0
MIN_GRID_M = 3000.0
MIN_OVERLAP_PX = 20000
SEARCH_M = 1500.0
PEAK_MIN = 0.3
AGREE_PX = 2
FINE_SIGMAS = (8.0, 3.0)
COARSE_SCALES = ((2, 8), (4, 16), (8, 32))      # 12 m px: 24-96, 48-192, 96-384 m
BAND_SIGMAS = (1.5, 3, 6, 12, 24, 48, 96)
BAND_NAMES = ['<144 m', '144-288 m', '288-577 m', '577 m-1.2 km', '1.2-2.3 km', '2.3-4.6 km', '4.6-9.2 km', '>9.2 km']
BAND_STATS = ('px', 'cov', 'var_z', 'var_h', 'sq_err')
MEASURES = ('deploy_mae', 'aligned_mae', 'label_mae', 'offset', 'abs_offset', 'slope_rmse', 'relief', 'deploy_pct',
            'scale')


# ---------------------------------------------------------------------------
# rasters
# ---------------------------------------------------------------------------

def read_window(path, bounds, margin_px):
    """The part of a predicted DEM covering bounds (in its CRS) plus margin_px,
    as float32 with NaN where invalid. Returns (array, transform), or (None,
    None) without overlap."""
    with rasterio.open(path) as src:
        w = from_bounds(*bounds, transform=src.transform)
        r0 = max(0, int(math.floor(w.row_off)) - margin_px)
        c0 = max(0, int(math.floor(w.col_off)) - margin_px)
        r1 = min(src.height, int(math.ceil(w.row_off + w.height)) + margin_px)
        c1 = min(src.width, int(math.ceil(w.col_off + w.width)) + margin_px)
        if r1 <= r0 or c1 <= c0:
            return None, None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        a = src.read(1, window=win).astype(np.float32)
        nodata, transform = src.nodata, src.window_transform(win)
    if nodata is not None:
        a[a == nodata] = np.nan
    a[~np.isfinite(a) | (a < -20000) | (a > 30000)] = np.nan
    return a, transform


def read_raster(path):
    """(float32 array with NaN where invalid, transform, crs)."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nodata, transform, crs = src.nodata, src.transform, src.crs
    bad = ~np.isfinite(a)
    if nodata is not None:
        bad |= a == nodata
    a[bad] = np.nan
    return a, transform, crs


def onto_grid(raster, transform, crs, shape, resampling=Resampling.average):
    """A raster from read_raster resampled onto a grid (float32, NaN where
    invalid)."""
    a, src_transform, src_crs = raster
    out = np.full(shape, np.nan, np.float32)
    reproject(a, out, src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan, dst_transform=transform,
              dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    out[out < -1e6] = np.nan
    return out


def translated(transform, dx, dy):
    return Affine(transform.a, transform.b, transform.c + dx, transform.d, transform.e, transform.f + dy)


def mola_on_grid(path, transform, crs, shape):
    """The MOLA mosaic as distributed, bilinear onto a grid (float64)."""
    out = np.full(shape, np.nan, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata if src.nodata is not None else -32768,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out.astype(np.float64)


def image_on_grid(source, transform, crs, shape):
    """An image block-averaged onto a grid (float64, NaN where not positive).
    source is a path or (array, transform, crs)."""
    out = np.full(shape, np.nan, np.float32)
    kw = dict(dst_transform=transform, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    if isinstance(source, tuple):
        reproject(source[0], out, src_transform=source[1], src_crs=source[2], src_nodata=np.nan, **kw)
    else:
        with rasterio.open(source) as src:
            reproject(rasterio.band(src, 1), out, src_nodata=src.nodata if src.nodata is not None else 0, **kw)
    out = out.astype(np.float64)
    out[~(out > 0) | (out > 1e30)] = np.nan
    return out


def read_orthoimage(path):
    """HiRISE orthoimage: the pixels of its coarsest JP2 overview of at most a
    third of the 6 m grid (the full resolution if there is none), the
    georeference of the PDS label (.LBL) beside it. Returns ((array, transform,
    crs), bounds)."""
    with rasterio.open(os.path.splitext(path)[0] + '.LBL') as src:
        crs, transform, width, height = src.crs, src.transform, src.width, src.height
    with rasterio.open(path) as src:
        factors = src.overviews(1)
    levels = [i for i, k in enumerate(factors) if abs(transform.a) * k <= GRID_M / 3]
    with (rasterio.open(path, overview_level=levels[-1]) if levels else rasterio.open(path)) as src:
        a = src.read(1).astype(np.float32)
    a[a <= 0] = np.nan
    t = transform * Affine.scale(width / a.shape[1], height / a.shape[0])
    return (a, t, crs), array_bounds(height, width, transform)


def registration_grid(ortho_bounds, ortho_crs, crs, transform, shape):
    """6 m grid in the window's projection over the orthoimage plus 1.6 km,
    within the window; None if narrower than 3 km."""
    ob = transform_bounds(ortho_crs, crs, *ortho_bounds, densify_pts=41)
    H, W = shape
    left, top = transform.c, transform.f
    right, bottom = left + W * transform.a, top + H * transform.e
    x0, y0 = max(ob[0] - ORTHO_MARGIN_M, left), max(ob[1] - ORTHO_MARGIN_M, bottom)
    x1, y1 = min(ob[2] + ORTHO_MARGIN_M, right), min(ob[3] + ORTHO_MARGIN_M, top)
    if x1 - x0 < MIN_GRID_M or y1 - y0 < MIN_GRID_M:
        return None
    return Affine(GRID_M, 0, x0, 0, -GRID_M, y1), (int((y1 - y0) / GRID_M), int((x1 - x0) / GRID_M))


# ---------------------------------------------------------------------------
# co-registration of the images
# ---------------------------------------------------------------------------

def masked_ncc(a, b, ma, mb, overlap_ratio=0.3):
    """Masked normalised cross-correlation of a and b (Padfield, 2012) at every
    relative shift ('full' output; zero shift at index (b rows - 1, b cols -
    1)), computed with FFTs. Shifts where the masks overlap on fewer than
    overlap_ratio of the largest overlap are set to 0."""
    eps = np.finfo(np.float64).eps
    a = np.where(ma, a, 0.0)
    b = np.where(mb, b, 0.0)[::-1, ::-1]
    mb = mb[::-1, ::-1]
    final = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    fast = tuple(scipy.fft.next_fast_len(n) for n in final)

    def fft(x):
        return scipy.fft.fftn(x, s=fast, axes=(0, 1))

    def ifft(x):
        return scipy.fft.ifftn(x, s=fast, axes=(0, 1)).real

    fa, fb = fft(a), fft(b)
    fma, fmb = fft(ma.astype(np.float64)), fft(mb.astype(np.float64))
    n = ifft(fmb * fma)
    n[:] = np.round(n)
    n[:] = np.fmax(n, eps)
    sa, sb = ifft(fmb * fa), ifft(fma * fb)
    num = ifft(fb * fa)
    num -= sa * sb / n
    va = ifft(fmb * fft(np.square(a)))
    va -= np.square(sa) / n
    va[:] = np.fmax(va, 0.0)
    vb = ifft(fma * fft(np.square(b)))
    vb -= np.square(sb) / n
    vb[:] = np.fmax(vb, 0.0)
    den = np.sqrt(va * vb)
    num, den, n = num[:final[0], :final[1]], den[:final[0], :final[1]], n[:final[0], :final[1]]
    ok = den > 1e3 * eps * np.max(np.abs(den))
    out = np.zeros_like(den)
    out[ok] = num[ok] / den[ok]
    np.clip(out, -1, 1, out=out)
    out[n < overlap_ratio * np.max(n)] = 0.0
    return out


def ncc_shift(ref, mov, ref_mask, mov_mask, radius):
    """Shift (columns, rows) taking a feature at p in ref to p + shift in mov:
    the masked NCC peak within +-radius px, refined along each axis by a
    parabola through the peak and its neighbours (at most half a pixel).
    Returns (shift, integer shift, peak)."""
    c = masked_ncc(mov, ref, mov_mask, ref_mask)
    r = int(radius)
    cy, cx = ref.shape[0] - 1, ref.shape[1] - 1
    win = c[cy - r:cy + r + 1, cx - r:cx + r + 1]
    win = np.where(np.isfinite(win), win, -2)
    iy, ix = np.unravel_index(np.argmax(win), win.shape)

    def vertex(fm, f0, fp):
        den = fm - 2 * f0 + fp
        return 0.0 if den >= -1e-12 else float(np.clip(0.5 * (fm - fp) / den, -0.5, 0.5))

    sy = vertex(win[iy - 1, ix], win[iy, ix], win[iy + 1, ix]) if 0 < iy < win.shape[0] - 1 else 0.0
    sx = vertex(win[iy, ix - 1], win[iy, ix], win[iy, ix + 1]) if 0 < ix < win.shape[1] - 1 else 0.0
    return [ix - r + sx, iy - r + sy], [int(ix - r), int(iy - r)], float(win[iy, ix])


def log_highpass(a, sigma):
    """log(a) minus its masked Gaussian low-pass, clipped at 5 standard
    deviations; zero outside the pixels with more than half their low-pass
    weight valid. Returns (image, mask)."""
    m = np.isfinite(a) & (a > 0)
    L = np.where(m, np.log(np.where(m, a, 1.0)), 0.0)
    ws = gaussian_filter(m.astype(np.float64), sigma)
    m2 = m & (ws > 0.5)
    h = np.where(m2, L - gaussian_filter(L, sigma) / np.maximum(ws, 1e-6), 0.0)
    if m2.sum() > 100:
        s = float(h[m2].std())
        h = np.clip(h, -5 * s, 5 * s)
    return np.where(m2, h, 0.0), m2


def log_bandpass(a, s1, s2):
    """Masked difference of Gaussians (s1, s2) of log(a), clipped at 5 standard
    deviations, over the pixels with more than 60% of the s2 weight valid."""
    m = np.isfinite(a) & (a > 0)
    L = np.where(m, np.log(np.where(m, a, 1.0)), 0.0)
    w = m.astype(np.float64)

    def lowpass(s):
        return gaussian_filter(L, s) / np.maximum(gaussian_filter(w, s), 1e-6)

    m2 = m & (gaussian_filter(w, s2) > 0.6)
    b = np.where(m2, lowpass(s1) - lowpass(s2), 0.0)
    if m2.sum() > 100:
        s = float(b[m2].std())
        b = np.clip(b, -5 * s, 5 * s)
    return np.where(m2, b, 0.0), m2


def block_mean(a):
    """2 x 2 block mean, NaN where fewer than 3 of the 4 pixels are valid."""
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    b = a[:h, :w].reshape(h // 2, 2, w // 2, 2)
    with np.errstate(invalid='ignore'):
        n = np.isfinite(b).sum((1, 3))
        s = np.nansum(b, (1, 3))
    return np.where(n >= 3, s / np.maximum(n, 1), np.nan)


def coarse_shift(ref, mov):
    """Coarse estimate on the 12 m grid: (dx, dy) in metres, or None."""
    ref, mov = block_mean(ref), block_mean(mov)
    radius = int(round(SEARCH_M / (2 * GRID_M)))
    est = []
    for s1, s2 in COARSE_SCALES:
        a, ma = log_bandpass(ref, s1, s2)
        b, mb = log_bandpass(mov, s1, s2)
        if ma.sum() < MIN_PX or mb.sum() < MIN_PX:
            return None
        est.append(ncc_shift(a, b, ma, mb, radius))
    best = None
    for e1, e2 in zip(est[:-1], est[1:]):
        agree = max(abs(e1[1][0] - e2[1][0]), abs(e1[1][1] - e2[1][1]))
        at_limit = max(abs(v) for v in e1[1] + e2[1]) >= radius - 1
        e = e1 if e1[2] >= e2[2] else e2
        if agree <= AGREE_PX and e[2] >= PEAK_MIN and not at_limit and (best is None or e[2] > best[2]):
            best = e
    return None if best is None else (best[0][0] * 2 * GRID_M, -best[0][1] * 2 * GRID_M)


def register_images(ref, mov):
    """Translation of mov (the HiRISE orthoimage) relative to ref (a CTX
    image) on the 6 m grid: dict with 'accepted', 'method' (fine or coarse),
    'dx', 'dy' (projected metres east and north) and the fine estimate's
    'peak'; None if the images have too few valid pixels."""
    radius = SEARCH_M / GRID_M
    est = []
    for sigma in FINE_SIGMAS:
        a, ma = log_highpass(ref, sigma)
        b, mb = log_highpass(mov, sigma)
        if ma.sum() < MIN_PX or mb.sum() < MIN_PX:
            return None
        est.append(ncc_shift(a, b, ma, mb, radius))
    (_, i8, p8), (_, i3, p3) = est
    agree = max(abs(i8[0] - i3[0]), abs(i8[1] - i3[1]))
    d, di, peak = est[0] if p8 >= p3 else est[1]
    fine = agree <= AGREE_PX and peak >= PEAK_MIN and max(abs(di[0]), abs(di[1])) < radius - 1
    out = dict(accepted=fine, method='fine' if fine else None, dx=d[0] * GRID_M, dy=-d[1] * GRID_M, peak=peak)
    if not fine:
        coarse = coarse_shift(ref, mov)
        if coarse is not None:
            out.update(accepted=True, method='coarse', dx=coarse[0], dy=coarse[1])
    return out


def ground_shift(dx, dy, transform, crs, shape):
    """A shift in projected metres (east, north) as ground metres, with the
    scale of the projection at the window centre."""
    proj = parse_crs(crs)
    H, W = shape
    px = abs(transform.a)

    def lonlat(c, r):
        x, y = transform * (c, r)
        lon, lat = xy_to_lonlat(x, y, proj)
        return float(lon), float(lat)

    lon0, lat0 = lonlat(W / 2, H / 2)
    k = math.cos(math.radians(lat0))

    def step(c, r):
        lon, lat = lonlat(c, r)
        return R_MARS * k * math.radians(((lon - lon0 + 180.0) % 360.0) - 180.0), R_MARS * math.radians(lat - lat0)

    (ec, nc), (er, nr) = step(W / 2 + 1, H / 2), step(W / 2, H / 2 + 1)
    pc, pr = dx / px, -dy / px
    return pc * ec + pr * er, pc * nc + pr * nr


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------

def erode(m, r):
    return binary_erosion(m, np.ones((2 * r + 1, 2 * r + 1), bool), border_value=1)


def slope_deg(h, px):
    gy, gx = np.gradient(h, px)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def robust_scale(z, h):
    """Gain a of h ~ a z + b fitted by iteratively reweighted least squares
    with Huber weights (threshold 1.345 x 1.4826 MAD of the residuals)."""
    A = np.stack([z, np.ones_like(z)], 1)
    a, b = np.linalg.lstsq(A, h, rcond=None)[0]
    for _ in range(8):
        r = h - (a * z + b)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
        w = np.sqrt(np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9)))
        a, b = np.linalg.lstsq(A * w[:, None], h * w, rcond=None)[0]
    return float(a)


def lowpass_f32(x, m, sigma):
    """Gaussian low-pass of x over the pixels m (normalised convolution,
    kernel truncated at 3 sigma, edges replicated), in single precision as in
    the paper's computation; NaN outside m."""
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
    k /= k.sum()

    def conv(a):
        t = torch.tensor(a.astype(np.float32))[None, None]
        t = F.conv2d(F.pad(t, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
        return F.conv2d(F.pad(t, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))[0, 0].numpy()

    num, den = conv(np.nan_to_num(x) * m), conv(m)
    return np.where(m > 0, num / np.where(den > 1e-6, den, np.nan), np.nan)


def dog_bands(x, m):
    """Difference-of-Gaussian bands of x (finest first) and the last low-pass."""
    lp = [x] + [lowpass_f32(x, m, s) for s in BAND_SIGMAS]
    return [lp[i] - lp[i + 1] for i in range(len(BAND_SIGMAS))] + [lp[-1]]


def band_stats(z, h, m):
    """Per band: pixels, covariance and variances of the centred bands, and
    the mean squared difference of the bands (not centred)."""
    mf = m.astype(np.float32)
    out = []
    for bz, bh in zip(dog_bands(np.where(m, z, np.nan), mf), dog_bands(np.where(m, h, np.nan), mf)):
        use = m & np.isfinite(bz) & np.isfinite(bh)
        p, t = bz[use].astype(np.float64), bh[use].astype(np.float64)
        pc, tc = p - p.mean(), t - t.mean()
        out.append(dict(px=int(use.sum()), cov=float((pc * tc).mean()), var_z=float((pc * pc).mean()),
                        var_h=float((tc * tc).mean()), sq_err=float(((p - t) ** 2).mean())))
    return out


def measures(z, h, h_d, m, px, full=True):
    """Measures of surface z on pixels m against HiRISE in place (h) and moved
    by the image shift (h_d; h_d = h in place). full adds labelMAE, scale,
    slope error and the bands."""
    d0 = (z - h)[m].astype(np.float64)
    d = (z - h_d)[m].astype(np.float64)
    out = dict(px=int(m.sum()), deploy_mae=float(np.abs(d0).mean()), aligned_mae=float(np.abs(d).mean()),
               offset=float(np.median(d)), relief=relief(h_d, m))
    out['deploy_pct'] = 100 * out['deploy_mae'] / out['relief']
    if not full:
        return out
    zz, hh = z[m].astype(np.float64), h_d[m].astype(np.float64)
    sz = float(zz.std())
    out['label_mae'] = (float(np.abs((zz - zz.mean()) / sz * float(hh.std()) + float(hh.mean()) - hh).mean())
                        if sz > 1e-9 else float('nan'))
    out['scale'] = robust_scale(zz, hh)
    inner = erode(m, 2) & m
    sz_, sh_ = slope_deg(np.where(m, z, np.nan), px), slope_deg(np.where(m, h_d, np.nan), px)
    use = inner & np.isfinite(sz_) & np.isfinite(sh_)
    out['slope_rmse'] = float(np.sqrt(np.mean((sz_[use] - sh_[use]) ** 2)))
    out['bands'] = band_stats(z, np.where(m, h_d, np.nan), m)
    return out


# ---------------------------------------------------------------------------
# one HiRISE DEM
# ---------------------------------------------------------------------------

def score_window(win, hirise, dem_bounds, dem_crs, ortho, stereo, mola_path):
    """Record of one window (a CTX image over a HiRISE DEM): status, in-place
    measures, the image co-registration, and the co-registered and stereo
    measures where they exist."""
    rec = dict(image=win['image'])
    files = {'model': os.path.join(win['prediction'], 'pred_shot_anchor.tif'),
             'model_no_shots': os.path.join(win['prediction'], 'pred_feather.tif')}
    with rasterio.open(files['model']) as src:
        crs, px, grid = src.crs, abs(src.transform.a), (src.transform, src.shape)
    if not os.path.exists(files['model_no_shots']):
        files.pop('model_no_shots')
    else:
        with rasterio.open(files['model_no_shots']) as src:
            same_grid = src.crs == crs and src.shape == grid[1] and np.allclose(tuple(src.transform)[:6],
                                                                                tuple(grid[0])[:6])
        if not same_grid:
            files.pop('model_no_shots')
    bounds = transform_bounds(dem_crs, crs, *dem_bounds, densify_pts=21)
    surf, tr = {}, None
    for k, path in files.items():
        a, t = read_window(path, bounds, int(WINDOW_MARGIN_M / px))
        if a is None:
            break
        surf[k], tr = a, t
    if 'model' not in surf:
        rec['status'] = 'no_overlap'
        return rec
    shape = surf['model'].shape
    hi = onto_grid(hirise, tr, crs, shape)
    mola = mola_on_grid(mola_path, tr, crs, shape)
    valid = np.logical_and.reduce([np.isfinite(v) for v in surf.values()]) & np.isfinite(mola)
    m0 = erode(np.isfinite(hi) & valid, ERODE_PX)
    if m0.sum() < MIN_PX:
        rec['status'] = 'too_few_px'
        return rec
    rec['status'] = 'ok'
    rec['in_place'] = {'model': measures(surf['model'], hi, hi, m0, px, False),
                       'mola': measures(mola, hi, hi, m0, px, False)}

    grid6 = registration_grid(ortho[1], ortho[0][2], crs, tr, shape)
    if grid6 is None:
        rec['registration'] = dict(accepted=False, why='no_grid')
        return rec
    ctx6 = image_on_grid(win['registered_image'], grid6[0], crs, grid6[1])
    ortho6 = image_on_grid(ortho[0], grid6[0], crs, grid6[1])
    if (np.isfinite(ctx6) & np.isfinite(ortho6)).sum() < MIN_OVERLAP_PX:
        rec['registration'] = dict(accepted=False, why='too_little_overlap')
        return rec
    reg = register_images(ctx6, ortho6) or dict(accepted=False, why='too_few_px')
    if reg.get('accepted') and win.get('registered_r1', '') == '0':
        reg = dict(reg, accepted=False, why='not co-registered in release 1')
    rec['registration'] = reg
    if not reg['accepted']:
        return rec
    e, n = ground_shift(reg['dx'], reg['dy'], tr, crs, shape)
    rec.update(shift_east_m=e, shift_north_m=n, shift_m=math.hypot(e, n))
    h_d = onto_grid(hirise, translated(tr, reg['dx'], reg['dy']), crs, shape)
    m = erode(np.isfinite(hi) & np.isfinite(h_d) & valid, ERODE_PX)
    if m.sum() >= MIN_PX:
        rec['registered'] = {k: measures(z, hi, h_d, m, px) for k, z in surf.items()}
        rec['registered']['mola'] = measures(mola, hi, hi, m, px)

    rec['stereo'] = {}
    for pair, dem_path, orthos in stereo:
        st = onto_grid(read_raster(dem_path), tr, crs, shape, Resampling.bilinear)
        if np.isfinite(st).sum() < MIN_PX:
            continue
        best = None
        for path in orthos[:2]:
            s6 = image_on_grid(path, grid6[0], crs, grid6[1])
            if (np.isfinite(s6) & np.isfinite(ortho6)).sum() < MIN_OVERLAP_PX:
                continue
            g = register_images(s6, ortho6)
            if g is not None and (best is None or (g['accepted'], g['peak']) > (best['accepted'], best['peak'])):
                best = g
        if best is None or not best['accepted']:
            rec['stereo'][pair] = dict(registration=best)
            continue
        se, sn = ground_shift(best['dx'], best['dy'], tr, crs, shape)
        h_s = onto_grid(hirise, translated(tr, best['dx'], best['dy']), crs, shape)
        mb = erode(np.isfinite(hi) & np.isfinite(h_d) & np.isfinite(h_s) & np.isfinite(st) & valid, ERODE_PX)
        q = dict(registration=best, shift_m=math.hypot(se, sn))
        if mb.sum() >= MIN_PX:
            q.update(model=measures(surf['model'], hi, h_d, mb, px), stereo=measures(st, hi, h_s, mb, px),
                     mola=measures(mola, hi, hi, mb, px))
        rec['stereo'][pair] = q
    return rec


def score_dem(dem, windows, stereo, mola_path):
    """Records of every window of one HiRISE DEM. dem: dict with dem_path and
    ortho_path; windows: dicts with image, prediction and registered_image;
    stereo: (pair, stereo DEM path, orthoimage paths) of its test pairs."""
    with rasterio.open(dem['dem_path']) as src:
        dem_bounds, dem_crs = src.bounds, src.crs
    hirise = read_raster(dem['dem_path'])
    ortho = read_orthoimage(dem['ortho_path'])
    return [score_window(w, hirise, dem_bounds, dem_crs, ortho, stereo, mola_path) for w in windows]


def dem_rows(dem, records):
    """Per-DEM values: for each set and surface, the mean over the windows of
    every measure and of pixels x band statistic. The co-registered model row
    carries the mean ground image shift of the accepted windows, the stereo
    rows the shifts of the model and of the stereo DEM."""
    groups, shifts = {}, {}
    for r in records:
        for k, v in (r.get('in_place') or {}).items():
            groups.setdefault(('in_place', k), []).append(v)
        if (r.get('registration') or {}).get('accepted'):
            shifts.setdefault(('registered', 'model'), []).append(r['shift_m'])
        for k, v in (r.get('registered') or {}).items():
            groups.setdefault(('registered', k), []).append(v)
        for q in (r.get('stereo') or {}).values():
            if 'model' in q:
                for k in ('model', 'stereo', 'mola'):
                    groups.setdefault(('stereo', k), []).append(q[k])
                shifts.setdefault(('stereo', 'model'), []).append(r['shift_m'])
                shifts.setdefault(('stereo', 'stereo'), []).append(q['shift_m'])
    rows = []
    for (s, k), v in groups.items():
        row = dict(dem=dem, set=s, surface=k, windows=len(v))
        for q in MEASURES:
            vals = [abs(x['offset']) if q == 'abs_offset' else x.get(q, np.nan) for x in v]
            row[q] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')
        row['shift_m'] = float(np.mean(shifts[(s, k)])) if (s, k) in shifts else float('nan')
        for i in range(len(BAND_NAMES)):
            for q in BAND_STATS:
                bands = [x['bands'][i] for x in v if 'bands' in x]
                row[f'b{i + 1}_{q}'] = (sum(b[q] * b['px'] for b in bands) / len(v) if q != 'px' else
                                        sum(b['px'] for b in bands) / len(v)) if bands else float('nan')
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def band_of(lat):
    a = abs(lat)
    return 'eq' if a < 30 else ('mid' if a < 60 else 'polar')


def bootstrap(values, reps=2000, seed=0):
    """95% interval of the median from resampling the values with replacement."""
    rng = random.Random(seed)
    v = list(values)
    meds = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps)]


def pooled_bands(rows):
    """Table 4 cells from the per-DEM band sums of one surface."""
    out = []
    for i in range(len(BAND_NAMES)):
        s = {q: sum(r[f'b{i + 1}_{q}'] for r in rows) for q in BAND_STATS}
        out.append(dict(amp=math.sqrt(s['var_z'] / max(s['var_h'], 1e-9)),
                        r=s['cov'] / math.sqrt(max(s['var_z'] * s['var_h'], 1e-18)),
                        error=math.sqrt(s['sq_err'] / max(s['px'], 1)),
                        hirise_rms=math.sqrt(s['var_h'] / max(s['px'], 1))))
    return out


def tables(dems, scores):
    """Table 3, Table 4 and the HiRISE numbers of Section 3.2."""
    D = {}
    for r in scores:
        D.setdefault((r['set'], r['surface']), {})[r['dem']] = r
    lat = {d['dem']: float(d['lat']) for d in dems}
    dist = {d['dem']: float(d['dist_km']) if d.get('dist_km') not in (None, '', 'nan') else float('inf') for d in dems}
    in_place = sorted(set(D[('in_place', 'model')]) & set(D[('in_place', 'mola')]))
    registered = sorted(set(D[('registered', 'model')]) & set(D[('registered', 'mola')]))
    stereo = sorted(set(D.get(('stereo', 'model'), {})) & set(D.get(('stereo', 'stereo'), {}))
                    & set(D.get(('stereo', 'mola'), {})))

    def col(s, k, q, ds):
        return np.array([D[(s, k)][d][q] for d in ds], float)

    def med(s, k, q, ds):
        return float(np.nanmedian(col(s, k, q, ds)))

    def cells(s, k, ds, shape=True):
        c = {q: med(s, k, q, ds) for q in ('deploy_mae', 'abs_offset', 'deploy_pct')}
        if shape:
            c.update(label_mae=med(s, k, 'label_mae', ds), aligned_mae=med(s, k, 'aligned_mae', ds))
        return c

    t3 = {'a': dict(dems=len(in_place), model=cells('in_place', 'model', in_place, False),
                    mola=cells('in_place', 'mola', in_place, False)),
          'b': dict(dems=len(registered), model=cells('registered', 'model', registered),
                    mola=cells('registered', 'mola', registered, False)),
          'c': dict(dems=len(stereo), model=cells('stereo', 'model', stereo), stereo=cells('stereo', 'stereo', stereo),
                    mola=cells('stereo', 'mola', stereo, False))}

    def bands(s, k, ds):
        return pooled_bands([D[(s, k)][d] for d in ds])[:-1]          # the last band is the low-pass remainder

    t4 = {'a': dict(dems=len(registered), bands=BAND_NAMES[:-1],
                    **{k: bands('registered', k, registered) for k in ('model', 'mola')}),
          'b': dict(dems=len(stereo), bands=BAND_NAMES[:-1],
                    **{k: bands('stereo', k, stereo) for k in ('model', 'stereo', 'mola')})}

    # Section 3.2
    def pair(ds, s='in_place'):
        return dict(dems=len(ds), model=med(s, 'model', 'deploy_mae', ds), mola=med(s, 'mola', 'deploy_mae', ds))

    not_registered = [d for d in in_place if d not in set(registered)]
    polar = [d for d in in_place if band_of(lat[d]) == 'polar']
    pole80 = [d for d in in_place if abs(lat[d]) >= 80]
    shift = col('registered', 'model', 'shift_m', registered)
    dep, ali, lab = (col('registered', 'model', q, registered) for q in ('deploy_mae', 'aligned_mae', 'label_mae'))
    ns = [d for d in registered if d in D.get(('registered', 'model_no_shots'), {})]
    scale = col('registered', 'model', 'scale', registered)
    text = dict(
        in_place=dict(pair(in_place), relief=med('in_place', 'model', 'relief', in_place),
                      model_pct=med('in_place', 'model', 'deploy_pct', in_place),
                      mola_pct=med('in_place', 'mola', 'deploy_pct', in_place)),
        distance_9_25_km=pair([d for d in in_place if 9 <= dist[d] < 25]),
        distance_over_25_km=pair([d for d in in_place if dist[d] >= 25]),
        equatorial=pair([d for d in in_place if band_of(lat[d]) == 'eq']),
        mid_latitude=pair([d for d in in_place if band_of(lat[d]) == 'mid']),
        registered=dict(pair(registered, 'registered'), aligned=float(np.median(ali)), label=float(np.median(lab)),
                        model_slope=med('registered', 'model', 'slope_rmse', registered),
                        mola_slope=med('registered', 'mola', 'slope_rmse', registered),
                        model_offset=med('registered', 'model', 'abs_offset', registered),
                        mola_offset=med('registered', 'mola', 'abs_offset', registered)),
        not_registered=dict(pair(not_registered), polar=sum(1 for d in not_registered if band_of(lat[d]) == 'polar')),
        stereo=dict(dems=len(stereo), model_label=med('stereo', 'model', 'label_mae', stereo),
                    stereo_label=med('stereo', 'stereo', 'label_mae', stereo),
                    model_aligned=med('stereo', 'model', 'aligned_mae', stereo),
                    stereo_aligned=med('stereo', 'stereo', 'aligned_mae', stereo),
                    model_shift_m=med('stereo', 'model', 'shift_m', stereo),
                    stereo_shift_m=med('stereo', 'stereo', 'shift_m', stereo)),
        image_shift=dict(median_m=float(np.median(shift)), ci_m=bootstrap(shift),
                         p90_m=float(np.percentile(shift, 90)), over_300_m=int((shift > 300).sum())),
        change_with_alignment=dict(median=float(np.median(dep - ali)), ci=bootstrap(dep - ali)),
        change_with_rescaling=float(np.median(ali - lab)),
        change_with_rescaling_no_shots=float(np.median(col('registered', 'model_no_shots', 'aligned_mae', ns)
                                                       - col('registered', 'model_no_shots', 'label_mae', ns))),
        scale=dict(median=float(np.median(scale)), ci=bootstrap(scale),
                   no_shots=med('registered', 'model_no_shots', 'scale', ns)),
        poleward_60=dict(pair(polar), registered=sum(1 for d in polar if d in set(registered)), over_80=len(pole80)),
        poleward_80=dict(pair(pole80), registered=sum(1 for d in pole80 if d in set(registered))),
        spearman_error_relief=float(spearmanr(col('in_place', 'model', 'deploy_mae', in_place),
                                              col('in_place', 'model', 'relief', in_place)).correlation))
    return dict(table3=t3, table4=t4, text=text)


def print_tables(t):
    t3, t4 = t['table3'], t['table4']
    names = dict(model='Our Model (MESA)', stereo='CTX stereo DEM (test set)', mola='MOLA gridded DEM')
    print('Table 3        labelMAE  alignedMAE  deployMAE  offset  error/relief')
    for part in 'abc':
        print(f'({part}) {t3[part]["dems"]} HiRISE DEMs')
        for k in ('model', 'stereo', 'mola'):
            if k in t3[part]:
                c = t3[part][k]
                shape = ''.join(f'{c[q]:9.1f}' if q in c else '      ---' for q in ('label_mae', 'aligned_mae'))
                print(f'  {names[k]:27s}{shape}{c["deploy_mae"]:11.1f}{c["abs_offset"]:8.1f}{c["deploy_pct"]:8.1f}')
    for part in 'ab':
        ks = [k for k in ('model', 'stereo', 'mola') if k in t4[part]]
        print(f'Table 4({part}), {t4[part]["dems"]} HiRISE DEMs: HiRISE RMS | '
              + ' | '.join(f'{k} amp r error' for k in ks))
        for i, b in enumerate(t4[part]['bands']):
            cells = [t4[part][k][i] for k in ks]
            print(f'  {b:14s}{t4[part]["model"][i]["hirise_rms"]:7.2f} | '
                  + ' | '.join(f'{c["amp"]:.2f} {c["r"]:.2f} {c["error"]:.2f}' for c in cells))
    print(json.dumps(t['text'], indent=1))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_inputs(dems_csv, windows_csv, stereo_csv, test_pairs):
    """Jobs (dem, windows, stereo pairs) for score_dem, in the order of the DEM table."""
    keep = set(open(test_pairs).read().split()) if test_pairs else None
    pairs = {r['pair']: (r['pair'], r['dem_path'], [p for p in r['orthoimages'].split(';') if p])
             for r in read_csv(stereo_csv)} if stereo_csv else {}
    windows = {}
    for w in read_csv(windows_csv):
        windows.setdefault(w['dem'], []).append(w)
    jobs = []
    for d in read_csv(dems_csv):
        ps = [p for p in d.get('pairs', '').split(';') if p and p in pairs and (keep is None or p in keep)]
        jobs.append((d, windows.get(d['dem'], []), [pairs[p] for p in ps]))
    return jobs


def _score_job(args):
    d, windows, stereo, mola = args
    return dem_rows(d['dem'], score_dem(d, windows, stereo, mola))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score', help='per-DEM values of every HiRISE DEM')
    s.add_argument('--dems', required=True)
    s.add_argument('--windows', required=True)
    s.add_argument('--stereo', default=None)
    s.add_argument('--test-pairs', default=None, help='stereo pairs to score (the 359 test pairs)')
    s.add_argument('--mola', required=True, help='Mars_MGS_MOLA_DEM_mosaic_global_463m.tif')
    s.add_argument('--out', required=True)
    s.add_argument('--jobs', type=int, default=8)
    t = sub.add_parser('tables', help='Tables 3 and 4 and the numbers of Section 3.2')
    t.add_argument('--dems', required=True)
    t.add_argument('--scores', required=True)
    t.add_argument('--out', default=None, help='JSON with every value')
    args = ap.parse_args()

    if args.cmd == 'score':
        jobs = [j + (args.mola,) for j in load_inputs(args.dems, args.windows, args.stereo, args.test_pairs)]
        scored, w = 0, None
        with Pool(args.jobs) as pool, open(args.out, 'w', newline='') as f:
            for rows in pool.imap(_score_job, jobs):
                for r in rows:
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(r))
                        w.writeheader()
                    w.writerow(r)
                scored += bool(rows)
        print(f'{len(jobs)} HiRISE DEMs, {scored} scored')
    else:
        scores = [{k: (v if k in ('dem', 'set', 'surface') else float(v) if v != '' else float('nan'))
                   for k, v in r.items()} for r in read_csv(args.scores)]
        result = tables(read_csv(args.dems), scores)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(result, f, indent=1)
        print_tables(result)


if __name__ == '__main__':
    main()

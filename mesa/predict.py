"""Predict a DEM from one registered CTX image (Section 2.5 of the paper).

The image (a GeoTIFF or VRT whose geotransform carries the registration) is
resampled onto a north-up 18 m grid covering its footprint, together with the
MOLA gridded DEM and the shot-distance layer. The grid is processed in 384 px
windows at a stride of 192 px. In each window the network output is converted
to metres with MOLA's mean and standard deviation over the window, then fitted
to the PEDR shots (gain and offset, see anchor.py). The windows are blended
with a Hann weight.

    python -m mesa.predict --image aligned_pedr/B01_009956_2169_aligned.vrt \
        --inc 45.45 --az 262.89 --model mesa_e098.pt \
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --pedr pedr.sqlite \
        --out preds/B01_009956_2169

Two rules of the prediction of version 1 apply to any model. PEDR shots are queried
over all longitudes when the frame spans 360 deg of longitude or reaches
within 2 deg of a pole (pedr.PedrDatabase, full_lon_near_poles). A frame
whose shot-distance input has a mean below 0.05 (as printed to three
decimals) is refused, because it means the shot query failed
(--min-conf-mean; 0 turns the check off).

Writes pred_shot_anchor.tif (the product), pred_feather.tif (MOLA statistics
only, no shots), ctx_on_grid.tif and mola_on_grid.tif.

The shots are compared with the prediction through a 168 m Gaussian footprint
(anchor.FOOTPRINT_M), matching the evaluation and leveling configuration. The
frames of version 1 of the product were anchored through a 150 m footprint
(--footprint 150 reproduces them; verification/README.md).
"""
import argparse
import math
import os

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter

from . import anchor
from .data import sun_vector
from .geo import MolaSampler, lonlat_to_xy, parse_crs, pixel_centres, query_box
from .model import load_model, relief_rho
from .pedr import PedrDatabase, shot_confidence, shot_distance

PIXEL = 18.0
TILE, STRIDE = 384, 192
NODATA = -9999.0
STD_FLOOR_M = 5.0
MIN_VALID = 0.25          # windows with less valid image are skipped
MIN_CONF_MEAN = 0.05      # version 1: a frame whose shot-distance input averages less is refused


def northup_grid(transform, width, height, px=PIXEL):
    """North-up grid covering the (possibly rotated) image footprint."""
    cols = np.array([0, width, 0, width], float)
    rows = np.array([0, 0, height, height], float)
    X = transform.a * cols + transform.b * rows + transform.c
    Y = transform.d * cols + transform.e * rows + transform.f
    xmin, xmax, ymin, ymax = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())
    W = int(math.ceil((xmax - xmin) / px))
    H = int(math.ceil((ymax - ymin) / px))
    return Affine(px, 0.0, xmin, 0.0, -px, ymax), H, W


def resample_image(img, img_tf, nodata, grid_tf, shape):
    """Image (about 6 m) -> 18 m grid. Anti-aliased with a normalised Gaussian
    (sigma = half the size ratio) so no-data does not bleed in, then bilinear at
    the grid's pixel centres. A grid pixel is valid where more than half of its
    smoothed support is valid image."""
    H, W = shape
    valid = np.isfinite(img) & (img > 0.0)
    if nodata is not None:
        valid &= img != nodata
    valid = valid.astype(np.float32)
    src = np.where(valid > 0, img.astype(np.float32), np.float32(0.0))
    ratio = abs(grid_tf.a / img_tf.a)
    if ratio > 1.5:
        sigma = 0.5 * ratio
        num = gaussian_filter(src, sigma=sigma)
        den = gaussian_filter(valid, sigma=sigma)
        smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
        frac = den
    else:
        smooth, frac = src, valid
    X, Y = pixel_centres(grid_tf, H, W)
    inv = ~img_tf
    fc = inv.a * X + inv.b * Y + inv.c
    fr = inv.d * X + inv.e * Y + inv.f
    Hs, Ws = smooth.shape
    c0, r0 = np.floor(fc).astype(int), np.floor(fr).astype(int)
    dc, dr = fc - c0, fr - r0
    c0, r0 = np.clip(c0, 0, Ws - 2), np.clip(r0, 0, Hs - 2)

    def bilinear(a):
        top = a[r0, c0] * (1 - dc) + a[r0, c0 + 1] * dc
        bot = a[r0 + 1, c0] * (1 - dc) + a[r0 + 1, c0 + 1] * dc
        return top * (1 - dr) + bot * dr

    out = bilinear(smooth).astype(np.float32)
    inside = (fc >= 0) & (fc <= Ws - 1) & (fr >= 0) & (fr <= Hs - 1)
    ok = (bilinear(frac) > 0.5) & inside
    out[~ok] = 0.0
    return out, ok


def window_origins(n):
    if n <= TILE:
        return [0]
    out = list(range(0, n - TILE + 1, STRIDE))
    if out[-1] != n - TILE:
        out.append(n - TILE)
    return out


def hann_weight():
    w = np.clip(np.hanning(TILE), 1e-3, None)
    return np.outer(w, w).astype(np.float32)


def predict_window(model, ctx, mola, valid, conf, sun, device):
    """One 384 px window -> prediction in metres (MOLA statistics), MOLA mean
    and floored standard deviation."""
    lo, hi = np.percentile(ctx[valid], [1.0, 99.0])
    if hi - lo < 1e-6:
        raise ValueError('image window has no contrast')
    ctx_n = np.clip((ctx - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    ctx_n[~valid] = 0.5
    mmean = float(mola.mean())
    mstd = max(float(mola.std()), STD_FLOOR_M)
    x = np.stack([((mola - mmean) / mstd).astype(np.float32), conf.astype(np.float32)])
    cond = np.concatenate([sun[:3], np.array([relief_rho(mstd)], np.float32)]).astype(np.float32)
    with torch.no_grad():
        z, _ = model(torch.from_numpy(ctx_n)[None, None].to(device), torch.from_numpy(x)[None].to(device),
                     torch.from_numpy(cond)[None].to(device))
    z = z.cpu().numpy().astype(np.float32).reshape(-1, TILE, TILE)[0]
    return z * mstd + mmean, mmean, mstd


def fit_window(pred, r, c, pedr, grid_tf, proj, footprint_m=anchor.FOOTPRINT_M):
    """Gain/offset fit of one window to the shots inside it (None if unusable)."""
    rr = np.array([r, r, r + TILE, r + TILE], float)
    cc = np.array([c, c + TILE, c, c + TILE], float)
    X = grid_tf.a * cc + grid_tf.b * rr + grid_tf.c
    Y = grid_tf.d * cc + grid_tf.e * rr + grid_tf.f
    lat, lon, elev = pedr.query(*query_box(X.min(), X.max(), Y.min(), Y.max(), proj))
    if lat.size < anchor.MIN_SHOTS:
        return None
    xs, ys = lonlat_to_xy(lon, lat, proj)
    inv = ~grid_tf
    col = inv.a * xs + inv.b * ys + inv.c - c
    row = inv.d * xs + inv.e * ys + inv.f - r
    if ((col >= 0) & (col < TILE) & (row >= 0) & (row < TILE)).sum() < anchor.MIN_SHOTS:
        return None
    try:
        fit = anchor.fit_gain_offset(pred, row, col, elev, footprint_m=footprint_m)
    except ValueError:
        return None
    if fit['spread'] < anchor.MIN_SPREAD_M:
        fit['a'], fit['b'] = 1.0, fit['offset_only']
    return fit


def blend(windows, shape, anchored):
    """Hann-weighted mean of the windows. With anchored=True each window is
    first fitted to the shots, or shifted to the MOLA mean where the fit is
    missing or implausible (see anchor.py)."""
    H, W = shape
    acc = np.zeros((H, W), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for t in windows:
        pred = t['pred']
        if anchored:
            pm = float(np.average(pred, weights=t['w']))
            b_mola = t['mmean'] - pm
            fit = t['fit']
            if fit is not None:
                sigma = max(float(t['mstd']), anchor.SIGMA_FLOOR_M)
                b_shot = fit['a'] * pm + fit['b'] - pm
                if (abs(b_shot - b_mola) > anchor.OFFSET_REL * sigma
                        or fit['resid_rms'] > anchor.RESID_REL * sigma):
                    fit = None
            pred = fit['a'] * pred + fit['b'] if fit is not None else pred + b_mola
        r, c = t['r'], t['c']
        acc[r:r + TILE, c:c + TILE] += pred * t['w']
        wsum[r:r + TILE, c:c + TILE] += t['w']
    out = np.full((H, W), NODATA, np.float32)
    good = wsum > 1e-6
    out[good] = (acc[good] / wsum[good]).astype(np.float32)
    return out


def write(path, arr, transform, crs):
    profile = dict(driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1, dtype='float32',
                   crs=crs, transform=transform, nodata=NODATA, compress='deflate', tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def predict_image(image, inc, az, model, mola_sampler, pedr, out_dir, device='cpu',
                  footprint_m=anchor.FOOTPRINT_M, min_conf_mean=MIN_CONF_MEAN):
    with rasterio.open(image) as src:
        img = src.read(1).astype(np.float32)
        img_tf, crs, nodata = src.transform, src.crs, src.nodata
    proj = parse_crs(crs)
    grid_tf, H, W = northup_grid(img_tf, img.shape[1], img.shape[0])
    if H < TILE or W < TILE:
        raise ValueError(f'grid {W}x{H} is smaller than one window')
    ctx, valid = resample_image(img, img_tf, nodata, grid_tf, (H, W))
    mola = mola_sampler.on_grid(grid_tf, H, W, proj)
    d1 = shot_distance(pedr, proj, grid_tf, W, H)
    conf = np.zeros((H, W), np.float32) if d1 is None else shot_confidence(d1)
    if min_conf_mean > 0 and float(f'{conf.mean():.3f}') < min_conf_mean:
        raise ValueError(f'shot-distance input mean {conf.mean():.3f} < {min_conf_mean}: the PEDR query failed')
    sun = sun_vector(inc, az)

    hann = hann_weight()
    windows = []
    for r in window_origins(H):
        for c in window_origins(W):
            v = valid[r:r + TILE, c:c + TILE]
            if v.mean() < MIN_VALID:
                continue
            pred, mmean, mstd = predict_window(model, ctx[r:r + TILE, c:c + TILE], mola[r:r + TILE, c:c + TILE],
                                               v, conf[r:r + TILE, c:c + TILE], sun, device)
            windows.append({'r': r, 'c': c, 'pred': pred, 'w': (hann * v).astype(np.float32),
                            'mmean': mmean, 'mstd': mstd,
                            'fit': fit_window(pred, r, c, pedr, grid_tf, proj, footprint_m)})
    if not windows:
        raise ValueError('no window has enough valid image')

    os.makedirs(out_dir, exist_ok=True)
    write(os.path.join(out_dir, 'ctx_on_grid.tif'), np.where(valid, ctx, NODATA), grid_tf, crs)
    write(os.path.join(out_dir, 'mola_on_grid.tif'), mola, grid_tf, crs)
    write(os.path.join(out_dir, 'pred_shot_anchor.tif'), blend(windows, (H, W), True), grid_tf, crs)
    write(os.path.join(out_dir, 'pred_feather.tif'), blend(windows, (H, W), False), grid_tf, crs)
    n_fit = sum(t['fit'] is not None for t in windows)
    print(f'{image}: {W}x{H} px, {len(windows)} windows, {n_fit} fitted to shots')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='registered CTX image (GeoTIFF or VRT)')
    ap.add_argument('--inc', type=float, required=True, help='solar incidence angle (deg)')
    ap.add_argument('--az', type=float, required=True, help='solar azimuth, clockwise from north (deg)')
    ap.add_argument('--model', required=True)
    ap.add_argument('--mola', required=True, help='MOLA gridded DEM, 463 m')
    ap.add_argument('--pedr', required=True, help='PEDR SQLite database')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=None,
                    help='model configuration (default: the one stored with or named after the weights)')
    ap.add_argument('--footprint', type=float, default=anchor.FOOTPRINT_M,
                    help='full width at half maximum of the shot footprint in metres (default %(default)s)')
    ap.add_argument('--min-conf-mean', type=float, default=MIN_CONF_MEAN,
                    help='refuse a frame whose shot-distance input averages less (default %(default)s)')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_image(args.image, args.inc, args.az, load_model(args.model, device, args.config), MolaSampler(args.mola),
                  PedrDatabase(args.pedr, full_lon_near_poles=True), args.out, device, args.footprint,
                  args.min_conf_mean)


if __name__ == '__main__':
    main()

"""Accuracy against the CTX stereo DEMs of the test set (Section 2.8; Tables 1
and 2 of the paper).

Tiles (Table 1, tile rows). Each test sample, one CTX image over one tile of a
test stereo DEM, is scored on the central 384 px window of the tile:

  labelMAE        the prediction rescaled with the stereo DEM's own mean and
                  standard deviation over the window: the error of shape
  alignedMAE      the delivered surface: the prediction rescaled with MOLA's
                  statistics and anchored to the PEDR shots in the window
                  (anchor.anchor_window). A test tile was registered to its
                  stereo DEM when the archive was built, so no further
                  alignment is needed.
  <1 km r, amp.   Pearson correlation and RMS ratio of the fine fields of the
                  delivered surface and the stereo DEM, the fine field being a
                  surface minus its Gaussian low-pass at sigma = 1000 / (5.34 x
                  18 m) = 10.4 px, which passes half the amplitude at 1 km
  error / relief  alignedMAE as a percentage of the stereo DEM's relief over
                  the window (2nd to 98th percentile)

MOLA is scored on the same windows as it is distributed: the 463 m mosaic is
warped bilinearly onto the tile grid with its own georeferencing, unshifted.

Bins (Table 2). Each sample is cropped as in validation (384 px at a random
position, seed 0, batch 16, eight loader workers, over the 375 test pairs of
data/test_manifest_*.csv), and the labelMAE and alignedMAE errors of every
pixel of the test pairs are binned by the disagreement between MOLA and the
stereo DEM, the distance from the nearest shot, and the slope, aspect relative
to the Sun and local incidence angle of the predicted surface. Except for the
disagreement, a 4 px border of the crop is left out. A cell of the table is
the median over samples of each sample's MAE within the bin, each sample
weighted by its pixels in the bin. The disagreement is measured with the MOLA
input of the network (the half-cell corrected grid); the MOLA row of the
shot-distance block scores MOLA as distributed on the same crops.

    python -m mesa.evaluate tiles --model mesa_e088.pt --root mars_ctx_dem_v4_pedr \\
        --manifest data/test_manifest_eq.csv,data/test_manifest_mid.csv,data/test_manifest_polar.csv \\
        --pairs data/test_pairs.txt --out tiles.csv
    python -m mesa.evaluate mola-tiles --root mars_ctx_dem_v4_pedr --samples tiles.csv \\
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out tiles_mola.csv
    python -m mesa.evaluate bins --model mesa_e088.pt --root mars_ctx_dem_v4_pedr \\
        --manifest data/test_manifest_eq.csv --pairs data/test_pairs.txt \\
        --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out bins_eq.npz          (also _mid, _polar)
    python -m mesa.evaluate tables --samples tiles.csv --mola-tiles tiles_mola.csv --pairs data/test_pairs.txt \\
        --bins bins_eq.npz bins_mid.npz bins_polar.npz
"""
import argparse
import csv
import json
import math
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_erosion, convolve1d
from torch.utils.data import DataLoader, Dataset

from .anchor import FOOTPRINT_M, anchor_window
from .data import TileDataset, load_incidence
from .model import load_model, relief_rho

NODATA = -9999.0
STD_FLOOR_M = 5.0
WINDOW = 384
CONF_D0 = 2000.0
MAX_DIST_M = 60000.0
DEFAULT_SUN = (45.0, 225.0)
FINE_SIGMA_PX = 1000.0 / (5.34 * 18.0)

# Table 2: bin edges, and whether only pixels lit by the Sun (on the predicted
# surface) are binned. The disagreement bins use every valid pixel of the crop,
# the other axes the crop without its 4 px border.
DISAGREEMENT_EDGES = (20, 50, 100)
AXES = {
    'shot_distance': ((200, 500, 1000, 2000, 5000, 12000), False),
    'slope': ((1, 2, 5, 10, 15, 20, 30), False),
    'aspect': ((-135, -90, -45, 0, 45, 90, 135), True),
    'local_incidence': ((20, 30, 40, 50, 60, 70, 80, 90), True),
}
BORDER_PX = 4


# ---------------------------------------------------------------------------
# shared measures
# ---------------------------------------------------------------------------

def read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), src.transform


def centre(shape, size=WINDOW):
    """Origin and slices of the central size x size window."""
    y0, x0 = (shape[0] - size) // 2, (shape[1] - size) // 2
    return y0, x0, (slice(y0, y0 + size), slice(x0, x0 + size))


def sun_vector(incidence_deg, azimuth_deg):
    """Unit vector towards the Sun (east, north, up) in single precision."""
    i, a = math.radians(incidence_deg), math.radians(azimuth_deg)
    return np.array([math.sin(i) * math.sin(a), math.sin(i) * math.cos(a), math.cos(i)], np.float32)


def solar_geometry(row, table):
    """(incidence, azimuth) of a sample's image: from the image list, else the
    manifest, else the default."""
    if row.get('ctx_product_id', '') in table:
        return table[row['ctx_product_id']]
    try:
        return float(row['sun_incidence']), float(row['sun_azimuth'])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_SUN


def standardise(ctx, dem, mola):
    """Network inputs of one window, as in training: the image stretched
    between its 1st and 99th percentiles and MOLA standardised with its own
    mean and standard deviation. Returns them with the valid stereo-DEM pixels
    and (MOLA mean, MOLA std, DEM mean, DEM std), standard deviations floored
    at 5 m."""
    finite = ctx[np.isfinite(ctx)]
    ctx_n = np.zeros_like(ctx)
    if finite.size and finite.max() > finite.min():
        p1, p99 = np.percentile(finite, [1, 99])
        if p99 - p1 > 1e-8:
            ctx_n = np.clip((ctx - p1) / (p99 - p1), 0, 1)
    mvalid = (mola != NODATA) & np.isfinite(mola)
    mv = mola[mvalid]
    mmean = float(mv.mean()) if mv.size else 0.0
    mstd = max(float(mv.std()), STD_FLOOR_M) if mv.size else STD_FLOOR_M
    mola_n = np.zeros_like(mola)
    mola_n[mvalid] = (mola[mvalid] - mmean) / mstd
    dvalid = (dem != NODATA) & np.isfinite(dem)
    dv = dem[dvalid]
    dmean = float(dv.mean()) if dv.size else mmean
    dstd = max(float(dv.std()), STD_FLOOR_M) if dv.size else mstd
    return ctx_n, mola_n, dvalid, (mmean, mstd, dmean, dstd)


def mae(surface, ref, valid):
    m = valid & np.isfinite(surface) & np.isfinite(ref)
    if m.sum() < 16:
        return float('nan')
    return float(np.mean(np.abs(surface[m] - ref[m])))


def lowpass(a, sigma):
    """Gaussian low-pass over the finite pixels of a (normalised convolution,
    kernel truncated at 3 sigma); NaN where a is NaN."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ok = np.isfinite(a)

    def conv(x):
        return convolve1d(convolve1d(x, k, axis=1, mode='constant'), k, axis=0, mode='constant')

    num, den = conv(np.where(ok, a, 0.0)), conv(ok.astype(np.float64))
    return np.where(ok, num / np.where(den > 1e-6, den, np.nan), np.nan)


def fine_structure(surface, ref, valid, sigma=FINE_SIGMA_PX, min_px=1000):
    """Pearson r, amplitude ratio (RMS of the surface's fine field over the
    reference's) and RMS difference of the fine fields, a fine field being a
    surface minus its low-pass. Statistics use the common valid pixels eroded
    by 2 px."""
    m = valid & np.isfinite(surface) & np.isfinite(ref)
    out = dict(r=float('nan'), amp=float('nan'), rmse=float('nan'))
    if m.sum() < min_px:
        return out
    inner = binary_erosion(m, np.ones((5, 5), bool), border_value=1)
    s = np.where(m, surface, np.nan).astype(np.float64)
    t = np.where(m, ref, np.nan).astype(np.float64)
    s, t = s - lowpass(s, sigma), t - lowpass(t, sigma)
    use = inner & np.isfinite(s) & np.isfinite(t)
    if use.sum() < min_px:
        return out
    s, t = s[use], t[use]
    t_rms = float(np.sqrt(np.mean(t ** 2)))
    return dict(r=float(np.corrcoef(s, t)[0, 1]), amp=float(np.sqrt(np.mean(s ** 2)) / max(1e-6, t_rms)),
                rmse=float(np.sqrt(np.mean((s - t) ** 2))))


def relief(dem, valid):
    v = dem[valid]
    return float(np.percentile(v, 98) - np.percentile(v, 2))


# ---------------------------------------------------------------------------
# Table 1: test tiles
# ---------------------------------------------------------------------------

def tile_shots(path):
    """Rows, columns and elevations of the shots rasterised on a tile, or None."""
    if not Path(path).exists():
        return None
    a, _ = read(path)
    m = np.isfinite(a) & (a != NODATA)
    if not m.any():
        return None
    rr, cc = np.nonzero(m)
    return rr.astype(np.float64), cc.astype(np.float64), a[m]


@torch.no_grad()
def score_sample(model, row, root, incidence, device, footprint_m=FOOTPRINT_M):
    """Table 1 measures of one sample on the central window of its tile, or None
    if the tile is incomplete or less than half covered by the stereo DEM."""
    root = Path(root)
    stem = Path(row['dem_tile']).stem
    paths = [root / row['dem_tile'], root / row['ctx_tile'], root / 'mola_bilinear' / f'{stem}.tif',
             root / 'd1_shot' / f'{stem}.tif']
    if not all(p.exists() for p in paths):
        return None
    (dem, transform), (ctx, _), (mola, _), (dist, _) = (read(p) for p in paths)
    if not dem.shape == ctx.shape == mola.shape == dist.shape or min(dem.shape) < WINDOW:
        return None
    y0, x0, win = centre(dem.shape)
    dem, ctx, mola, dist = dem[win], ctx[win], mola[win], dist[win]
    if ((dem != NODATA) & np.isfinite(dem)).mean() < 0.5:
        return None

    ctx_n, mola_n, valid, (mmean, mstd, dmean, dstd) = standardise(ctx, dem, mola)
    conf = np.exp(-np.clip(np.where(np.isfinite(dist), dist, MAX_DIST_M), 0, MAX_DIST_M) / CONF_D0)
    inc, az = solar_geometry(row, incidence)
    cond = np.concatenate([sun_vector(inc, az), np.array([relief_rho(mstd)], np.float32)])

    def tensor(a, shape):
        return torch.from_numpy(a.astype(np.float32)).view(*shape).to(device)

    out = model(tensor(ctx_n, (1, 1, WINDOW, WINDOW)), tensor(np.stack([mola_n, conf]), (1, 2, WINDOW, WINDOW)),
                tensor(cond, (1, 4)))
    z = out[0].float().cpu().numpy()[0, 0]
    pred_mola = z * mstd + mmean
    pred_dem = z * dstd + dmean
    ref = np.where(valid, dem, np.nan)

    rows = cols = elev = np.zeros(0)
    shots = tile_shots(root / 'shot' / f'{stem}.tif')
    if shots is not None:
        r, c, e = shots
        r, c = r - y0, c - x0
        inside = (c >= 0) & (c < WINDOW) & (r >= 0) & (r < WINDOW)
        rows, cols, elev = r[inside], c[inside], e[inside]
    surface, how = anchor_window(pred_mola, valid, mmean, mstd, rows, cols, elev, abs(transform.a), footprint_m)
    fine = fine_structure(surface, ref, valid)
    return dict(pair_short=row['pair_short'], dem_tile=row['dem_tile'], ctx_product_id=row.get('ctx_product_id', ''),
                incidence_deg=inc, label_mae=mae(pred_dem, ref, valid), aligned_mae=mae(surface, ref, valid),
                fine_r=fine['r'], fine_amp=fine['amp'], relief_m=relief(dem, valid), anchoring=how,
                shots=len(elev), mola_std_m=mstd, dem_std_m=dstd)


def score_mola_tile(args):
    """MOLA as distributed on the central window of one stereo-DEM tile."""
    root, dem_tile, mola_path = args
    with rasterio.open(Path(root) / dem_tile) as src:
        dem = src.read(1).astype(np.float64)
        transform, crs = src.transform, src.crs
    mola = np.full(dem.shape, np.nan, np.float32)
    with rasterio.open(mola_path) as src:
        reproject(rasterio.band(src, 1), mola, src_nodata=-32768, dst_transform=transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    if min(dem.shape) < WINDOW:
        return None
    _, _, win = centre(dem.shape)
    dem, mola = dem[win], mola[win].astype(np.float64)
    valid = (dem != NODATA) & np.isfinite(dem)
    if valid.mean() < 0.5:
        return None
    both = valid & (mola != NODATA) & np.isfinite(mola)
    ref = np.where(valid, dem, np.nan)
    fine = fine_structure(mola, ref, both)
    return dict(dem_tile=dem_tile, mola_mae=mae(mola, ref, both), fine_r=fine['r'], fine_amp=fine['amp'],
                relief_m=relief(dem, valid))


def read_rows(manifests, pairs=None):
    rows = []
    for m in manifests:
        with open(m, newline='') as f:
            rows += [r for r in csv.DictReader(f) if pairs is None or r['pair_short'] in pairs]
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Table 2: binned errors
# ---------------------------------------------------------------------------

def surface_normals(z, pixel_size=18.0):
    """Unit normals (east, north, up) and slope (degrees) of [B, 1, H, W]
    surfaces by central differences; rows run north to south."""
    dz_drow = (z[:, :, 2:, :] - z[:, :, :-2, :]) * 0.5 / pixel_size
    dz_dcol = (z[:, :, :, 2:] - z[:, :, :, :-2]) * 0.5 / pixel_size
    dz_drow = F.pad(dz_drow, (0, 0, 1, 1), mode='replicate')
    dz_dcol = F.pad(dz_dcol, (1, 1, 0, 0), mode='replicate')
    dz_dx, dz_dy = dz_dcol, -dz_drow
    g2 = dz_dx * dz_dx + dz_dy * dz_dy
    inv = 1.0 / torch.sqrt(g2 + 1.0 + 1e-6)
    return -dz_dx * inv, -dz_dy * inv, inv, torch.rad2deg(torch.atan(torch.sqrt(g2 + 1e-12)))


class WithDistributedMola(Dataset):
    """Samples of a TileDataset with the distributed MOLA mosaic, warped
    bilinearly onto the tile grid, cut to the same crop ('mola_dist')."""

    def __init__(self, base, mola_path):
        self.base, self.mola_path = base, mola_path

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        with rasterio.open(self.base.root / self.base.rows[idx]['dem_tile']) as src:
            transform, crs, shape = src.transform, src.crs, src.shape
        mola = np.full(shape, np.nan, np.float32)
        with rasterio.open(self.mola_path) as src:
            reproject(rasterio.band(src, 1), mola, src_nodata=-32768, dst_transform=transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        cs = s['dem'].shape[-1]
        s['mola_dist'] = torch.from_numpy(mola[s['y0']:s['y0'] + cs, s['x0']:s['x0'] + cs].copy())[None]
        return s


@torch.no_grad()
def bin_errors(model, dataset, keep_pairs, mola_path, device, batch_size=16, num_workers=8, seed=0,
               footprint_m=FOOTPRINT_M):
    """Per sample and bin of each Table 2 axis, the sums of the labelMAE and
    alignedMAE errors and the number of pixels ([samples, bins, 3]); for
    'mola_shot_distance', the sum of the error of MOLA as distributed and the
    number of pixels ([samples, bins, 2]). Pixels of pairs outside keep_pairs
    are not counted, but their samples are drawn, so that every crop is the same
    as in the evaluation of all 375 pairs."""
    random.seed(seed)
    np.random.seed(seed)
    loader = DataLoader(WithDistributedMola(dataset, mola_path), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, worker_init_fn=lambda wid: random.seed(seed * 1000 + wid))
    edges = {k: torch.tensor(e, device=device, dtype=torch.float32) for k, (e, _) in AXES.items()}
    edges['disagreement'] = torch.tensor(DISAGREEMENT_EDGES, device=device, dtype=torch.float32)
    sums = {k: [] for k in list(edges) + ['shadowed', 'all', 'mola_shot_distance']}
    pixel_size = {}

    for b in loader:
        ctx, x, cond, dem, mola = (b[k].to(device) for k in ('ctx', 'x', 'cond', 'dem', 'mola'))
        pred = model(ctx, x, cond)[0]
        B = pred.shape[0]

        def stat(k):
            return b[k].float().to(device).view(-1, 1, 1, 1)

        pred_m = pred * stat('dem_std') + stat('dem_mean')
        dem_m = dem * stat('dem_std') + stat('dem_mean')
        err = (pred_m - dem_m).abs()
        keep = torch.tensor([p in keep_pairs for p in b['pair']], device=device).view(-1, 1, 1, 1)
        valid = (dem != 0) & (mola != 0) & keep

        # delivered surface of each crop, anchored to the shots in the crop
        pn = pred.float().cpu().numpy()[:, 0].astype(np.float64)
        vn = valid.cpu().numpy()[:, 0]
        shot_mask = b['shot_mask'].numpy()[:, 0]
        shot_m = b['shot_m'].numpy()[:, 0].astype(np.float64)
        surface = np.empty_like(pn)
        for i in range(B):
            mmean, mstd = float(b['mola_mean'][i]), float(b['mola_std'][i])
            tile = b['tile'][i]
            if tile not in pixel_size:
                with rasterio.open(dataset.root / tile) as src:
                    pixel_size[tile] = abs(src.transform.a)
            rr, cc = np.nonzero(shot_mask[i])
            v = vn[i] if vn[i].any() else np.ones_like(vn[i])
            surface[i], _ = anchor_window(pn[i] * mstd + mmean, v, mmean, mstd, rr.astype(np.float64),
                                          cc.astype(np.float64), shot_m[i][shot_mask[i]], pixel_size[tile],
                                          footprint_m)
        erra = (torch.from_numpy(surface).to(device, torch.float32).unsqueeze(1) - dem_m).abs()

        # the MOLA input (half-cell corrected) defines the disagreement; MOLA as
        # distributed is scored in double precision
        disagreement = (mola * stat('mola_std') + stat('mola_mean') - dem_m).abs()
        dem_d = b['dem'].double() * b['dem_std'].view(-1, 1, 1, 1) + b['dem_mean'].view(-1, 1, 1, 1)
        mola_dist = b['mola_dist'].double()
        err_mola = (mola_dist - dem_d).abs().numpy().ravel()

        # geometry of the predicted surface and the distance from the nearest shot
        inner = torch.zeros_like(valid)
        inner[:, :, BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
        inner = valid & inner
        nx, ny, nz, slope = surface_normals(pred_m)
        sx, sy, sz = (cond[:, j].view(-1, 1, 1, 1) for j in range(3))
        cos_i = (nx * sx + ny * sy + nz * sz).clamp(-1.0, 1.0)
        shadowed = cos_i <= 0
        aspect = (torch.rad2deg(torch.atan2(nx, ny)) - torch.rad2deg(torch.atan2(sx, sy)) + 180.0) % 360.0 - 180.0
        features = dict(shot_distance=-CONF_D0 * torch.log(x[:, -1:].clamp(min=1e-6)), slope=slope, aspect=aspect,
                        local_incidence=torch.rad2deg(torch.acos(cos_i)))
        lit = inner & ~shadowed

        model_errors = [e.cpu().numpy().ravel().astype(np.float64) for e in (err, erra)]
        sample = np.repeat(np.arange(B), err[0].numel())

        def add(name, idx, mask, nb, errors=model_errors):
            m = mask.cpu().numpy().ravel() if torch.is_tensor(mask) else mask
            flat = sample[m] * nb + (idx.cpu().numpy().ravel()[m] if idx is not None else 0)
            s = [np.bincount(flat, weights=e[m], minlength=B * nb) for e in errors]
            s.append(np.bincount(flat, minlength=B * nb).astype(np.float64))
            sums[name].append(np.stack(s, -1).reshape(B, nb, len(errors) + 1))

        add('disagreement', torch.bucketize(disagreement, edges['disagreement']), valid, len(DISAGREEMENT_EDGES) + 1)
        for k, (e, lit_only) in AXES.items():
            add(k, torch.bucketize(features[k], edges[k]), lit if lit_only else inner, len(e) + 1)
        add('shadowed', None, inner & shadowed, 1)
        add('all', None, valid, 1)
        add('mola_shot_distance', torch.bucketize(features['shot_distance'], edges['shot_distance']),
            inner.cpu().numpy().ravel() & np.isfinite(err_mola), len(AXES['shot_distance'][0]) + 1, [err_mola])

    out = {k: np.concatenate(v, 0) for k, v in sums.items()}
    out['keys'] = np.array([f"{r['pair_short']}|{r.get('ctx_product_id', '')}|{r['dem_tile']}" for r in dataset.rows])
    return out


def weighted_median(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[int(np.searchsorted(cw, 0.5 * cw[-1]))])


def bin_cells(sums, groups=None):
    """Cells of one Table 2 block from per-sample sums [samples, bins, errors +
    1]: for every bin with pixels, the weighted median of each error, the share
    of pixels in % and the number of pixels. groups merges bins (e.g. folding
    the aspect about the Sun's azimuth)."""
    if groups:
        sums = np.stack([sums[:, list(g)].sum(1) for g in groups], 1)
    total = sums[:, :, -1].sum()
    cells = []
    for j in range(sums.shape[1]):
        n = sums[:, j, -1]
        m = n > 0
        if not m.any():
            cells.append(None)
            continue
        cells.append([weighted_median(sums[m, j, q] / n[m], n[m]) for q in range(sums.shape[2] - 1)]
                     + [100.0 * n.sum() / total, float(n.sum())])
    return cells


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def median(v):
    return float(np.nanmedian(np.asarray(v, float)))


def table1_tiles(samples, mola_tiles, pairs):
    """Table 1 tile rows and the statistics quoted with them."""
    s = [r for r in samples if r['pair_short'] in pairs]
    mola = {r['dem_tile']: r for r in mola_tiles}

    def col(k):
        return np.array([float(r[k]) for r in s])

    def mola_col(k):
        return np.array([float(mola[r['dem_tile']][k]) for r in s])

    lab, ali, rel, mo, inc = col('label_mae'), col('aligned_mae'), col('relief_m'), mola_col('mola_mae'), \
        col('incidence_deg')
    out = dict(samples=len(s), tiles=len({r['dem_tile'] for r in s}), pairs=len({r['pair_short'] for r in s}),
               label_mae=median(lab), aligned_mae=median(ali), fine_r=median(col('fine_r')),
               fine_amp=median(col('fine_amp')), error_relief_pct=median(100 * ali / rel), mola_mae=median(mo),
               mola_fine_r=median(mola_col('fine_r')), mola_fine_amp=median(mola_col('fine_amp')),
               mola_error_relief_pct=median(100 * mo / mola_col('relief_m')), aligned_minus_label=median(ali - lab),
               relief_m=median(rel))

    # error by the solar incidence of the image (Table 2, solar-incidence block)
    out['by_incidence'] = {}
    for lo, hi in ((0, 40), (40, 50), (50, 60), (60, 70), (70, 99)):
        m = (inc >= lo) & (inc < hi)
        out['by_incidence'][f'{lo}-{hi}'] = dict(label_mae=median(lab[m]), aligned_mae=median(ali[m]),
                                                 share_pct=100 * float(m.mean()))

    # consistency between the views of a tile
    by_tile = {}
    for r in s:
        by_tile.setdefault(r['dem_tile'], []).append(r)
    multi = [v for v in by_tile.values() if len(v) >= 2]
    spread = [v for v in multi if np.ptp([float(r['incidence_deg']) for r in v]) > 5]

    def sd(views, k):
        return float(np.std([float(r[k]) for r in views], ddof=1))

    out['views'] = dict(
        tiles=len(multi), sd_label=median([sd(v, 'label_mae') for v in multi]),
        sd_label_frac=median([sd(v, 'label_mae') / np.mean([float(r['label_mae']) for r in v]) for v in multi]),
        sd_aligned=median([sd(v, 'aligned_mae') for v in multi]), tiles_over_5deg=len(spread),
        sd_label_over_5deg=median([sd(v, 'label_mae') for v in spread]),
        sd_aligned_over_5deg=median([sd(v, 'aligned_mae') for v in spread]))
    return out


TABLE2_BLOCKS = [
    ('disagreement', ['0-20', '20-50', '50-100', '>100'], None),
    ('shot_distance', ['<0.2 km', '0.2-0.5 km', '0.5-1 km', '1-2 km', '2-5 km', '>5 km'],
     [(0,), (1,), (2,), (3,), (4,), (5, 6)]),
    ('slope', ['<1', '1-2', '2-5', '5-10', '10-15', '15-20', '20-30', '>30'], None),
    ('aspect', ['0-45', '45-90', '90-135', '135-180'], [(3, 4), (2, 5), (1, 6), (0, 7)]),
    ('local_incidence', ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90'], None),
    ('shadowed', ['self-shadowed'], None),
    ('all', ['all'], None),
]


def table2_bins(bin_files):
    """Table 2 blocks from the per-band outputs of bin_errors."""
    parts = [np.load(f) for f in bin_files]
    out = {}
    for name, labels, groups in TABLE2_BLOCKS:
        cells = bin_cells(np.concatenate([p[name] for p in parts], 0), groups)
        out[name] = {lab: None if c is None else dict(zip(('label_mae', 'aligned_mae', 'share_pct', 'pixels'), c))
                     for lab, c in zip(labels, cells)}
    mola = bin_cells(np.concatenate([p['mola_shot_distance'] for p in parts], 0), TABLE2_BLOCKS[1][2])
    for lab, c in zip(TABLE2_BLOCKS[1][1], mola):
        out['shot_distance'][lab]['mola_mae'] = c[0]
    out['shadowed_share_of_all_pct'] = 100 * out['shadowed']['self-shadowed']['pixels'] / out['all']['all']['pixels']
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tiles', help='Table 1 measures of every test sample')
    t.add_argument('--model', required=True)
    t.add_argument('--root', required=True, help='test archive')
    t.add_argument('--manifest', required=True, help='comma-separated test manifests')
    t.add_argument('--pairs', required=True, help='test pairs to score')
    t.add_argument('--incidence-json', default=None, help='default: ROOT/ctx_overlapping_95pct_polar.json')
    t.add_argument('--out', required=True)
    t.add_argument('--footprint', type=float, default=FOOTPRINT_M,
                   help='shot footprint FWHM in metres (default %(default)s)')
    m = sub.add_parser('mola-tiles', help='MOLA as distributed on the same windows')
    m.add_argument('--root', required=True)
    m.add_argument('--samples', required=True, help='output of the tiles command')
    m.add_argument('--mola', required=True, help='Mars_MGS_MOLA_DEM_mosaic_global_463m.tif')
    m.add_argument('--out', required=True)
    m.add_argument('--jobs', type=int, default=8)
    b = sub.add_parser('bins', help='Table 2 per-sample sums for one test manifest')
    b.add_argument('--model', required=True)
    b.add_argument('--root', required=True)
    b.add_argument('--manifest', required=True, help='one of data/test_manifest_<band>.csv')
    b.add_argument('--pairs', required=True)
    b.add_argument('--mola', required=True, help='Mars_MGS_MOLA_DEM_mosaic_global_463m.tif')
    b.add_argument('--incidence-json', default=None)
    b.add_argument('--out', required=True, help='.npz')
    b.add_argument('--footprint', type=float, default=FOOTPRINT_M,
                   help='shot footprint FWHM in metres (default %(default)s)')
    s = sub.add_parser('tables', help='Table 1 tile rows and Table 2 from the outputs above')
    s.add_argument('--samples', required=True)
    s.add_argument('--mola-tiles', required=True)
    s.add_argument('--pairs', required=True)
    s.add_argument('--bins', nargs='*', default=[])
    s.add_argument('--out', default=None, help='JSON with every value')
    args = ap.parse_args()

    if args.cmd in ('tiles', 'bins'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(args.model, device)
        pairs = set(open(args.pairs).read().split())
        inc_json = args.incidence_json or str(Path(args.root) / 'ctx_overlapping_95pct_polar.json')
    if args.cmd == 'tiles':
        incidence = load_incidence(inc_json)
        rows = read_rows(args.manifest.split(','), pairs)
        out = []
        for k, row in enumerate(rows):
            r = score_sample(model, row, args.root, incidence, device, args.footprint)
            if r is not None:
                out.append(r)
            if (k + 1) % 500 == 0:
                print(f'{k + 1}/{len(rows)} samples', flush=True)
        write_csv(args.out, out)
        print(f'{len(out)} samples scored, {len(rows) - len(out)} skipped')
    elif args.cmd == 'mola-tiles':
        tiles = sorted({r['dem_tile'] for r in csv.DictReader(open(args.samples))})
        with Pool(args.jobs) as pool:
            out = [r for r in pool.map(score_mola_tile, [(args.root, t, args.mola) for t in tiles], chunksize=8) if r]
        write_csv(args.out, out)
        print(f'{len(out)} tiles')
    elif args.cmd == 'bins':
        dataset = TileDataset(args.root, args.manifest, split=None, incidence_json=inc_json)
        np.savez_compressed(args.out, **bin_errors(model, dataset, pairs, args.mola, device,
                                                   footprint_m=args.footprint))
    else:
        pairs = set(open(args.pairs).read().split())
        result = {'table1_tiles': table1_tiles(list(csv.DictReader(open(args.samples))),
                                               list(csv.DictReader(open(args.mola_tiles))), pairs)}
        if args.bins:
            result['table2'] = table2_bins(args.bins)
        text = json.dumps(result, indent=1)
        if args.out:
            open(args.out, 'w').write(text)
        print(text)


if __name__ == '__main__':
    main()

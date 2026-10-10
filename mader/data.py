"""Training and test samples from the tiled stereo-DEM archive.

Archive layout (one row of manifest.csv per image-DEM sample):

    dem/<tile>.tif              CTX stereo DEM, 18 m, areoid heights (m)
    ctx/<tile>__<image>.tif     registered CTX image on the DEM grid
    mola_bilinear/<tile>.tif    MOLA gridded DEM on the DEM grid
    d1_shot/<tile>.tif          distance to the nearest PEDR shot (m)
    shot/<tile>.tif             PEDR shot elevations at their pixels, NaN elsewhere

Manifest columns used: pair_short (stereo pair), split (train / val / test
label of the pair), dem_tile, ctx_tile, ctx_product_id, and optionally
sun_incidence / sun_azimuth.

Each sample is a random crop (384 px) of one tile. The stereo DEM and MOLA are
standardised with their own mean and standard deviation over the crop; the
image is stretched between its 1st and 99th percentiles. During training the
crop is flipped (the solar vector flipped with it), the image brightness,
contrast and noise are jittered, and with probability 0.3 the MOLA input is
removed within 6 pixels of every PEDR shot so that the network learns to
reconstruct terrain from the image where shots can check it.
"""
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from .model import relief_rho

NODATA = -9999.0
SENTINEL_FLOOR = -30000.0      # anything at or below is a no-data sentinel
STD_FLOOR_M = 5.0
DEFAULT_SUN = (45.0, 225.0)    # used only when an image has no solar geometry


def sun_vector(incidence_deg, azimuth_deg):
    """Unit vector towards the Sun, [east, north, up] (Eq. 1). The azimuth is
    measured clockwise from north."""
    i, a = np.radians(incidence_deg), np.radians(azimuth_deg)
    return np.array([np.sin(i) * np.sin(a), np.sin(i) * np.cos(a), np.cos(i)], dtype=np.float32)


def load_incidence(path):
    """product id -> (incidence, solar azimuth from north), from the dataset's
    image list (ctx_overlapping_95pct_polar.json)."""
    table = {}
    if not path:
        return table
    for entry in json.load(open(path)):
        for k, pid in enumerate(entry.get('original_images', []), start=1):
            inc = entry.get(f'original_incidence{k}')
            az = entry.get(f'original_solar_azimuth_from_north{k}')
            if pid and inc is not None and az is not None:
                table[pid] = (float(inc), float(az))
        for img in entry.get('additional_images', []):
            pid, inc, az = img.get('product_id', ''), img.get('incidence_angle'), img.get('solar_azimuth_from_north')
            if pid and inc is not None and az is not None:
                table[pid] = (float(inc), float(az))
    return table


def _sun_from_row(row):
    try:
        inc, az = float(row['sun_incidence']), float(row['sun_azimuth'])
    except (KeyError, TypeError, ValueError):
        return None
    return (inc, az) if math.isfinite(inc) and math.isfinite(az) else None


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _stats(a, valid, fallback=None):
    v = a[valid]
    if v.size == 0:
        return fallback if fallback is not None else (0.0, STD_FLOOR_M)
    return float(np.mean(v)), max(float(np.std(v)), STD_FLOOR_M)


class TileDataset(Dataset):
    """Samples of one split of the archive.

    root            archive directory
    manifest        manifest CSV (with a split column)
    split           'train', 'val' or 'test' (any label in the column);
                    None takes every row (a test manifest)
    incidence_json  image list with solar geometry; manifest columns are the fallback
    augment         True for training
    """

    def __init__(self, root, manifest, split, incidence_json=None, crop_size=384, augment=False,
                 conf_d0=2000.0, cutout_p=0.3, cutout_dilate_px=6):
        self.root = Path(root)
        self.crop_size = crop_size
        self.augment = augment
        self.conf_d0 = float(conf_d0)
        self.cutout_p = float(cutout_p)
        self.cutout_dilate_px = int(cutout_dilate_px)
        self.incidence = load_incidence(incidence_json)

        rows_by_pair = {}
        with open(manifest, newline='') as f:
            for row in csv.DictReader(f):
                if split is None or row['split'] == split:
                    rows_by_pair.setdefault(row['pair_short'], []).append(row)
        if not rows_by_pair:
            raise RuntimeError(f'no rows with split == {split!r} in {manifest}')
        # Pairs in sorted order, rows in manifest order within a pair. The order
        # fixes which crop each row receives in a seeded evaluation.
        self.rows = [r for p in sorted(rows_by_pair) for r in rows_by_pair[p]]
        print(f'[{split or "all"}] {len(self.rows)} samples from {len(rows_by_pair)} stereo pairs')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        stem = Path(row['dem_tile']).stem
        image_id = row.get('ctx_product_id', '')
        inc, az = self.incidence.get(image_id) or _sun_from_row(row) or DEFAULT_SUN
        sun = sun_vector(inc, az)

        ctx = _read(self.root / row['ctx_tile'])
        dem = _read(self.root / row['dem_tile'])
        mola = _read(self.root / 'mola_bilinear' / f'{stem}.tif')
        dist = _read(self.root / 'd1_shot' / f'{stem}.tif')
        shot_path = self.root / 'shot' / f'{stem}.tif'
        shot = _read(shot_path) if shot_path.exists() else np.full(dem.shape, np.nan, np.float32)

        # random crop, the same window for every layer
        H, W = dem.shape
        cs = min(self.crop_size, H, W)
        y0 = random.randint(0, max(0, H - cs))
        x0 = random.randint(0, max(0, W - cs))
        win = (slice(y0, y0 + cs), slice(x0, x0 + cs))
        ctx, dem, mola, shot = ctx[win], dem[win], mola[win], shot[win]
        conf = np.exp(-dist[win] / self.conf_d0).astype(np.float32)

        # standardisation, each layer with its own statistics over the crop
        mola_valid = (mola != NODATA) & (mola > SENTINEL_FLOOR) & np.isfinite(mola)
        mola_mean, mola_std = _stats(mola, mola_valid)
        dem_valid = (dem != NODATA) & (dem > SENTINEL_FLOOR) & np.isfinite(dem)
        dem_mean, dem_std = _stats(dem, dem_valid, fallback=(mola_mean, mola_std))

        finite = ctx[np.isfinite(ctx)]
        ctx_n = np.zeros_like(ctx)
        if finite.size > 0 and finite.max() > finite.min():
            p1, p99 = np.percentile(finite, [1, 99])
            if p99 - p1 > 1e-8:
                ctx_n = np.clip((ctx - p1) / (p99 - p1), 0.0, 1.0)
        mola_n = np.zeros_like(mola)
        mola_n[mola_valid] = (mola[mola_valid] - mola_mean) / mola_std
        both = dem_valid & mola_valid
        dem_n = np.zeros_like(dem)
        dem_n[both] = (dem[both] - dem_mean) / dem_std

        shot_mask = np.isfinite(shot) & (shot != NODATA)
        shot_m = np.where(shot_mask, shot, 0.0).astype(np.float32)

        if self.augment:
            ctx_n, dem_n, mola_n, shot_mask, shot_m, conf = _flip(
                sun, ctx_n, dem_n, mola_n, shot_mask, shot_m, conf)
            ctx_n = _jitter_image(ctx_n)

        cond = np.concatenate([sun, np.array([relief_rho(mola_std)], dtype=np.float32)])

        mola_in = mola_n
        if self.augment:
            mola_in = _cut_mola_near_shots(mola_n, shot_mask, self.cutout_p, self.cutout_dilate_px)

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {
            'ctx': t(ctx_n).float()[None],
            'x': t(np.stack([mola_in, conf]).astype(np.float32)),
            'cond': t(cond),
            'dem': t(dem_n).float()[None],
            'mola': t(mola_n).float()[None],
            'shot_m': t(shot_m).float()[None],
            'shot_mask': t(shot_mask)[None],
            'dem_mean': dem_mean, 'dem_std': dem_std,
            'mola_mean': mola_mean, 'mola_std': mola_std,
            'pair': row['pair_short'], 'tile': row['dem_tile'],
            'incidence': float(inc), 'conf_mean': float(conf.mean()), 'y0': y0, 'x0': x0,
        }


def _flip(sun, *layers):
    """Random east-west and north-south flips of every layer; the solar vector's
    east or north component changes sign with the image."""
    layers = list(layers)
    if random.random() < 0.5:
        layers = [np.fliplr(a).copy() for a in layers]
        sun[0] *= -1
    if random.random() < 0.5:
        layers = [np.flipud(a).copy() for a in layers]
        sun[1] *= -1
    return layers


def _jitter_image(img):
    """Brightness and contrast jitter (p = 0.5) and Gaussian noise (p = 0.4)."""
    if random.random() < 0.5:
        b = random.uniform(-0.12, 0.12)
        c = random.uniform(0.88, 1.12)
        img = np.clip(c * img + b, 0.0, 1.0).astype(np.float32)
    if random.random() < 0.4:
        sigma = random.uniform(0.005, 0.02)
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        img = np.clip(img + noise, 0.0, 1.0).astype(np.float32)
    return img


def _cut_mola_near_shots(mola_n, shot_mask, p, dilate_px):
    """With probability p, zero the standardised MOLA input within dilate_px of
    every PEDR shot of the crop. The draw uses a fresh, unseeded generator."""
    rng = np.random.default_rng()
    if rng.random() > p or not shot_mask.any():
        return mola_n
    out = mola_n.copy()
    out[binary_dilation(shot_mask, iterations=max(1, dilate_px))] = 0.0
    return out

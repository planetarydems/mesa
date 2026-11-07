"""Translation of registered images onto the PEDR shots (Section 2.5 step 2 of
the paper; Text S8).

The reference mosaics and the laser shots are different geodetic frames. Where
a registered CTX image overlaps a stereo pair whose orthoimages are registered
to the shots (Text S1), the image is matched to each orthoimage on a 24 m grid:
the logarithms of both are high-passed (Gaussian, sigma 4 px) and the
translation is found by normalised cross-correlation over +-14 px with
parabolic sub-pixel refinement. A match is kept if its correlation is at least
0.2 and its peak is not at the edge of the search, and an image is used if its
matches against the orthoimages agree within 60 m.

The field: the vector of a tile is the component-wise median over its images
(at least three). The field at any point is the mean of the tile vectors,
weighted by sqrt(number of images) and by a Gaussian of 20 deg FWHM in
great-circle distance, east and north averaged separately. It is evaluated at
the centre of each image and added to the origin of its geotransform; images
farther than 20 deg from any tile with a vector are not moved.

    python -m mesa.pedr_field measure --pairs overlaps.csv --out matches.csv
    python -m mesa.pedr_field fit --matches matches.csv --out field.json
    python -m mesa.pedr_field apply --field field.json --images 'aligned/*_aligned.vrt' --out aligned_pedr
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import gaussian_filter, map_coordinates

from .geo import ground_scale, parse_crs, xy_to_lonlat
from .register import write_vrt

GRID = 24.0
SEARCH = 14
HP_SIGMA = 4.0
MIN_PX = 20000
NCC_MIN = 0.2
MAX_SPREAD_M = 60.0
FWHM_DEG = 20.0
MAX_NEAREST_DEG = 20.0


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def _on_grid(path, crs, tr, w, h):
    with rasterio.open(path) as ds:
        out = np.full((h, w), np.nan, np.float32)
        reproject(rasterio.band(ds, 1), out, dst_transform=tr, dst_crs=crs, resampling=Resampling.average,
                  src_nodata=ds.nodata if ds.nodata is not None else 0, dst_nodata=np.nan)
    out = out.astype(float)
    out[out <= 0] = np.nan
    return out


def _highpass(a):
    m = np.isfinite(a)
    low = (gaussian_filter(np.nan_to_num(a, nan=0.0), HP_SIGMA)
           / np.maximum(gaussian_filter(m.astype(float), HP_SIGMA), 1e-6))
    o = a - low
    o[~m] = np.nan
    return o


def measure(image, ortho):
    """Displacement (east, north, metres on the ground) that moves the image onto
    the orthoimage, with the correlation, or None when there is too little overlap."""
    with rasterio.open(image) as ds:
        crs, fb = ds.crs, ds.bounds
    kind, R, lon0, lat_ts = proj = parse_crs(crs)
    with rasterio.open(ortho) as od:
        ob = transform_bounds(od.crs, crs, *od.bounds, densify_pts=21)
    left, bottom, right, top = max(fb.left, ob[0]), max(fb.bottom, ob[1]), min(fb.right, ob[2]), min(fb.top, ob[3])
    if right - left < 40 * GRID or top - bottom < 40 * GRID:
        return None
    w, h = int((right - left) / GRID), int((top - bottom) / GRID)
    tg = Affine(GRID, 0.0, left, 0.0, -GRID, top)
    ctx, ref = _on_grid(image, crs, tg, w, h), _on_grid(ortho, crs, tg, w, h)
    m = np.isfinite(ctx) & np.isfinite(ref)
    if m.sum() < MIN_PX:
        return None
    hc, hr = _highpass(np.log(ctx)), _highpass(np.log(ref))
    ys, xs = np.where(np.isfinite(hr) & np.isfinite(hc) & m)
    if ys.size < MIN_PX:
        return None
    sel = np.random.default_rng(0).choice(ys.size, size=min(80000, ys.size), replace=False)
    ys, xs = ys[sel], xs[sel]
    u = hr[ys, xs]

    def ncc(dy, dx):
        v = map_coordinates(hc, [ys + dy, xs + dx], order=1, mode='constant', cval=np.nan)
        k = np.isfinite(v)
        if k.sum() < MIN_PX // 4:
            return -1.0
        a, b = u[k] - u[k].mean(), v[k] - v[k].mean()
        d = math.sqrt(float((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 0 else -1.0

    grid = np.array([[ncc(dy, dx) for dx in range(-SEARCH, SEARCH + 1)] for dy in range(-SEARCH, SEARCH + 1)])
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    dy, dx = i - SEARCH, j - SEARCH
    railed = abs(dy) >= SEARCH - 1 or abs(dx) >= SEARCH - 1
    sdy = sdx = 0.0
    if 0 < i < 2 * SEARCH and 0 < j < 2 * SEARCH:          # parabolic sub-pixel peak
        a0, a1, a2 = grid[i - 1, j], grid[i, j], grid[i + 1, j]
        if a0 - 2 * a1 + a2 < 0:
            sdy = 0.5 * (a0 - a2) / (a0 - 2 * a1 + a2)
        b0, b1, b2 = grid[i, j - 1], grid[i, j], grid[i, j + 1]
        if b0 - 2 * b1 + b2 < 0:
            sdx = 0.5 * (b0 - b2) / (b0 - 2 * b1 + b2)
    lat = float(xy_to_lonlat(np.array([tg.c + tg.a * w / 2]), np.array([tg.f + tg.e * h / 2]), proj)[1][0])
    kx, ky = ground_scale(kind, lat, lat_ts)
    # sampling the image at (y + dy, x + dx) matches the orthoimage when the image content is
    # displaced by (+dx, +dy) px, so the image must move by (-dx, +dy) px east and north
    return dict(lat=round(lat, 4), ncc=round(float(grid[i, j]), 4), e_m=round(-(dx + sdx) * GRID * kx, 1),
                n_m=round((dy + sdy) * GRID * ky, 1), railed=int(railed))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def tile_centre(name):
    m = re.match(r'E([p+-])(\d{3})_N([p+-])(\d{2})$', name)
    sign = lambda s: -1 if s == '-' else 1
    return sign(m.group(1)) * int(m.group(2)) + 5.0, sign(m.group(3)) * int(m.group(4)) + 5.0


def tile_vectors(match_rows):
    """Per-tile median vectors from the per-match rows (tile, image, e_m, n_m, ncc, railed)."""
    per_image = {}
    for r in match_rows:
        if r.get('ncc') in ('', None) or float(r['ncc']) < NCC_MIN or int(r['railed']):
            continue
        per_image.setdefault((r['tile'], r['image']), []).append((float(r['e_m']), float(r['n_m'])))
    by_tile = {}
    for (tile, _), v in per_image.items():
        a = np.array(v)
        e, n = np.median(a[:, 0]), np.median(a[:, 1])
        if len(v) == 1 or np.max(np.hypot(a[:, 0] - e, a[:, 1] - n)) <= MAX_SPREAD_M:
            by_tile.setdefault(tile, []).append((e, n))
    tiles = {}
    for tile, v in by_tile.items():
        if len(v) >= 3:
            a = np.array(v)
            lon, lat = tile_centre(tile)
            med = np.median(a, axis=0)
            tiles[tile] = dict(lon=lon, lat=lat, n=len(v), e=float(med[0]), n_=float(med[1]))
    return tiles


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Field:
    """Gaussian-weighted mean of tile vectors on the sphere, east and north separately."""

    def __init__(self, tiles, fwhm_deg=FWHM_DEG, exclude=None):
        items = [v for k, v in tiles.items() if k != exclude]
        self.xyz = _unit(np.array([v['lon'] for v in items]), np.array([v['lat'] for v in items]))
        self.e = np.array([v['e'] for v in items])
        self.n = np.array([v['n_'] for v in items])
        self.w0 = np.sqrt(np.array([v['n'] for v in items], float))
        self.sigma = math.radians(fwhm_deg) / 2.3548

    def __call__(self, lon, lat):
        d = np.arccos(np.clip(_unit(np.atleast_1d(lon), np.atleast_1d(lat)) @ self.xyz.T, -1, 1))
        w = self.w0 * np.exp(-0.5 * (d / self.sigma) ** 2)
        s = w.sum(1)
        ok, sd = s > 1e-12, np.maximum(s, 1e-12)
        return np.where(ok, (w @ self.e) / sd, np.nan), np.where(ok, (w @ self.n) / sd, np.nan)

    def nearest_deg(self, lon, lat):
        q = _unit(np.atleast_1d(lon), np.atleast_1d(lat))
        return np.degrees(np.arccos(np.clip(q @ self.xyz.T, -1, 1)).min(1))


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def shift_image(src, dst, field):
    """Write dst (a VRT over src's pixels) with the field added to the origin.
    Returns the applied (east, north) in metres, or None if unsupported."""
    with rasterio.open(src) as s:
        t, w, h, crs = s.transform, s.width, s.height, s.crs
    kind, R, lon0, lat_ts = proj = parse_crs(crs)
    cx, cy = t.c + t.a * w / 2 + t.b * h / 2, t.f + t.d * w / 2 + t.e * h / 2
    lon, lat = xy_to_lonlat(np.array([cx]), np.array([cy]), proj)
    lon, lat = float(lon[0]), float(lat[0])
    e, n = field(lon, lat)
    if field.nearest_deg(lon, lat)[0] > MAX_NEAREST_DEG or not np.isfinite(e[0]):
        return None
    kx, ky = ground_scale(kind, lat, lat_ts)
    write_vrt(src, dst, Affine(t.a, t.b, t.c + float(e[0]) / kx, t.d, t.e, t.f + float(n[0]) / ky))
    return float(e[0]), float(n[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--pairs', required=True, help='CSV with columns tile, image (registered image), ortho')
    m.add_argument('--out', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--matches', required=True, nargs='+')
    f.add_argument('--out', required=True)
    a = sub.add_parser('apply')
    a.add_argument('--field', required=True)
    a.add_argument('--images', required=True, help="glob of registered images, e.g. 'aligned/*_aligned.vrt'")
    a.add_argument('--out', required=True, help='output directory')
    args = ap.parse_args()

    if args.cmd == 'measure':
        rows = []
        for r in csv.DictReader(open(args.pairs)):
            res = measure(r['image'], r['ortho'])
            if res is not None:
                rows.append(dict(tile=r['tile'], image=os.path.basename(r['image'])[:15], ortho=r['ortho'], **res))
        with open(args.out, 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=['tile', 'image', 'ortho', 'lat', 'ncc', 'e_m', 'n_m', 'railed'])
            w.writeheader()
            w.writerows(rows)
    elif args.cmd == 'fit':
        rows = [r for p in args.matches for r in csv.DictReader(open(p))]
        tiles = tile_vectors(rows)
        loo = [math.hypot(v['e'] - Field(tiles, exclude=k)(v['lon'], v['lat'])[0][0],
                          v['n_'] - Field(tiles, exclude=k)(v['lon'], v['lat'])[1][0]) for k, v in tiles.items()]
        json.dump(dict(fwhm_deg=FWHM_DEG, tiles=tiles, leave_one_out_median_m=float(np.median(loo))),
                  open(args.out, 'w'), indent=1)
        print(f'{len(tiles)} tile vectors; leave-one-tile-out residual median {np.median(loo):.0f} m')
    else:
        spec = json.load(open(args.field))
        field = Field(spec['tiles'], spec['fwhm_deg'])
        os.makedirs(args.out, exist_ok=True)
        for p in sorted(glob.glob(args.images)):
            stem = os.path.basename(p)[:15]
            if shift_image(p, os.path.join(args.out, f'{stem}_aligned.vrt'), field) is None:
                print(f'{stem}: no field support within {MAX_NEAREST_DEG} deg, not moved')


if __name__ == '__main__':
    main()

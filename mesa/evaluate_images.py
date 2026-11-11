"""Accuracy of whole CTX images against the test stereo DEMs (Table 1, whole-image
rows; Section 3.1 and Texts S7 and S10 of the paper).

At each of the 359 test pairs one CTX image of the product, usually not one of
the two images of the pair, was predicted whole by mesa.predict in one product
tile (pred_shot_anchor.tif: shot-anchored, before any levelling).
data/test_images.csv names the image and the tile, the pair's latitude band and
latitude (centre of the grid of its test tiles), its stereo DEM and the
co-registration of the image with the pair. The image of each pair is the one
selected for release 1 and kept for the paper (v3).

Stereo DEM. The paper scores each image against the full stereo DEM of its
pair (18 m, the pair's own projection; `<pair>.tif` in the directory given to
`score`): the tied DEM of the stereo pipeline without hole filling, or the
filled DEM masked to the pixels of the tied DEM, cropped to its valid
extent. These DEMs are part of the evaluation data of the paper. `labels`
rebuilds the stereo DEMs that release 1 scored against, the pair's test tiles
of the archive merged on their common grid (no-data where no tile has a
value), which cover only the test tiles of the pair.

Grid. The prediction and the MOLA gridded DEM as distributed (463 m, its own
georeferencing) are warped onto the stereo DEM's grid by bilinear interpolation
between the four nearest pixels (GDAL warp options XSCALE = YSCALE = 1, so that
the kernel does not widen where the scales of the two grids differ, as from an
equirectangular prediction to a sinusoidal stereo DEM). The three are cut to the
bounding box of the pixels where the prediction and the stereo DEM are both
valid, plus 20 px.

Co-registration. The transform was measured from the images alone, so it holds
for any model and is given here as data: a rotation and shift fitted by RANSAC
and least squares to SIFT matches between the CTX image and an orthoimage of the
stereo pair (the images of its test tiles where it has none), kept where 12 km
correlation windows or the pair's other orthoimage confirm it, otherwise the
consensus shift of the windows, and failing that the shift of the whole overlap
(Section 2.8, Text S7). The transforms of data/test_images.csv were fitted
over the overlap of each image with the full stereo DEM; where that fit was
not accepted and the stereo DEM is the one of release 1, the transform fitted
on the region of the test tiles is used (three pairs). Methods:
  rotation+shift  T maps pixel coordinates of a reference grid (spacing
  tile_region_rigid  ref_spacing_m, top-left corner ref_left, ref_top in the
                  stereo DEM's projection, counted from the crop ref_row0,
                  ref_col0); each stereo-DEM pixel takes the prediction at T
                  applied to its centre, by cubic-spline interpolation. No-data
                  is first filled from the nearest valid pixel (the spline
                  prefilter cannot take it), and pixels whose bilinear support
                  touches no-data are left out.
  shift           the consensus shift of the windows,
  original_shift  the shift of the whole overlap,
  tile_region_shift  the consensus shift on the region of the test tiles, all in
                  whole stereo-DEM pixels:
                  aligned[r, c] = pred[r + shift_rows, c + shift_cols]
  none            not co-registered (deployMAE only)

Measures, over the pixels where the surface and the stereo DEM are both valid:
  deployMAE       the prediction in place
  alignedMAE      the co-registered prediction
  labelMAE        the co-registered prediction rescaled to the stereo DEM's mean
                  and standard deviation over those pixels
  level           median of the surface minus the stereo DEM
  relief          2nd to 98th percentile of the stereo DEM
  error / relief  MAE as a percentage of relief
  <1 km r, amp.   evaluate.fine_structure (sigma 10.4 px)
MOLA is scored in place, over every valid stereo-DEM pixel of the grid.

Table 1 gives medians over the co-registered pairs, all latitudes and by band:
labelMAE, alignedMAE and the <1 km measures of the co-registered prediction,
deployMAE and error / relief of the prediction in place, and MOLA's MAE, <1 km
measures and error / relief. `tables` also gives the statistics of Section 3.1:
the change of the error with co-registration, the level in place, the pairs
poleward of 80 deg (|lat| >= 80) and the classes of in-place relief.

    python -m mesa.evaluate_images score --images data/test_images.csv --labels stereo_dems_full \\
        --preds preds --mola Mars_MGS_MOLA_DEM_mosaic_global_463m.tif --out images.csv
    python -m mesa.evaluate_images tables --scores images.csv --out images_tables.json

The predictions are read from PREDS/<tile>/<image>/pred_shot_anchor.tif.
"""
import argparse
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, map_coordinates

from .evaluate import fine_structure, median, read_rows, relief, write_csv

NODATA = -9999.0
MOLA_NODATA = -32768.0
PAD_PX = 20
MIN_OVERLAP_PX = 20000
MIN_PX = 1000
BANDS = ('eq', 'mid', 'polar')
RELIEF_CLASSES = ((0, 300), (300, 1000), (1000, 3000), (3000, math.inf))


# ---------------------------------------------------------------------------
# stereo DEM and grid
# ---------------------------------------------------------------------------

def build_label(root, tiles, out_path):
    """Merge the test tiles of one pair into its stereo DEM (float32, NaN for no-data)."""
    srcs = [rasterio.open(Path(root) / t) for t in sorted(tiles)]
    try:
        arr, transform = merge(srcs, nodata=NODATA, dtype='float32')
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    a = arr[0].astype(np.float32)
    a[(a <= -9000) | ~np.isfinite(a)] = np.nan
    profile = dict(driver='GTiff', height=a.shape[0], width=a.shape[1], count=1, dtype='float32', crs=crs,
                   transform=transform, nodata=np.nan, compress='deflate', tiled=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(a, 1)


def nan_nodata(a):
    """NaN for no-data and fill values, in place."""
    a[(a == NODATA) | (a < -1e5)] = np.nan
    return a


def warp(path, shape, transform, crs, src_nodata):
    """A raster resampled bilinearly onto a grid (float32, NaN for no-data)."""
    out = np.full(shape, NODATA, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src_nodata, dst_transform=transform, dst_crs=crs,
                  dst_nodata=NODATA, resampling=Resampling.bilinear, XSCALE=1, YSCALE=1)
    return nan_nodata(out)


def pair_grids(label_path, pred_path, mola_path):
    """Stereo DEM, prediction and MOLA on the stereo DEM's grid, cut to the
    pixels where the prediction and the stereo DEM are both valid plus 20 px
    (float64, NaN for no-data), and the transform of the cut grid."""
    with rasterio.open(label_path) as src:
        stereo = src.read(1).astype(np.float32)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    if nodata is not None:
        stereo[stereo == nodata] = np.nan
    nan_nodata(stereo)
    pred = warp(pred_path, stereo.shape, transform, crs, NODATA)
    mola = warp(mola_path, stereo.shape, transform, crs, MOLA_NODATA)
    both = np.isfinite(pred) & np.isfinite(stereo)
    if both.sum() < MIN_OVERLAP_PX:
        raise ValueError(f'{pred_path}: {int(both.sum())} px overlap the stereo DEM')
    rows, cols = np.nonzero(both)
    r0, r1 = max(0, rows.min() - PAD_PX), min(stereo.shape[0], rows.max() + 1 + PAD_PX)
    c0, c1 = max(0, cols.min() - PAD_PX), min(stereo.shape[1], cols.max() + 1 + PAD_PX)
    win = (slice(r0, r1), slice(c0, c1))
    t = transform
    cut = Affine(t.a, t.b, t.c + c0 * t.a, t.d, t.e, t.f + r0 * t.e)
    return stereo[win].astype(np.float64), pred[win].astype(np.float64), mola[win].astype(np.float64), cut


# ---------------------------------------------------------------------------
# co-registration
# ---------------------------------------------------------------------------

def rotate_shift(pred, transform, T, crop, spacing, left, top):
    """The prediction at T applied to each pixel centre (cubic spline), NaN
    where the bilinear support of that point touches no-data."""
    H, W = pred.shape
    r0, c0 = crop
    cols, rows = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    # pixel centres of the grid -> pixels of the reference grid, from its crop
    x = (transform.c + cols * transform.a - left) / spacing - c0 - 0.5
    y = (top - (transform.f + rows * transform.e)) / spacing - r0 - 0.5
    x2 = T[0, 0] * x + T[0, 1] * y + T[0, 2]
    y2 = T[1, 0] * x + T[1, 1] * y + T[1, 2]
    # back to map coordinates, then to (fractional) row and column of the grid
    X2 = left + (c0 + x2 + 0.5) * spacing
    Y2 = top - (r0 + y2 + 0.5) * spacing
    at = [(Y2 - transform.f) / transform.e - 0.5, (X2 - transform.c) / transform.a - 0.5]
    ok = np.isfinite(pred)
    filled = pred if ok.all() else pred[tuple(distance_transform_edt(~ok, return_distances=False, return_indices=True))]
    out = map_coordinates(filled, at, order=3, mode='nearest')
    support = map_coordinates(ok.astype(np.float64), at, order=1, mode='constant', cval=0.0)
    return np.where(support > 0.999, out, np.nan)


def shift(a, dr, dc):
    """out[r, c] = a[r + dr, c + dc], NaN where that falls outside a."""
    if dr == 0 and dc == 0:
        return a
    h, w = a.shape
    out = np.full_like(a, np.nan)
    y0, y1 = max(0, -dr), min(h, h - dr)
    x0, x1 = max(0, -dc), min(w, w - dc)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = a[y0 + dr:y1 + dr, x0 + dc:x1 + dc]
    return out


def coregister(pred, transform, row):
    """The prediction co-registered with the stereo DEM as row of
    data/test_images.csv says, or None."""
    if row['method'] in ('rotation+shift', 'tile_region_rigid'):
        T = np.array([[float(row[k]) for k in ('t00', 't01', 't02')], [float(row[k]) for k in ('t10', 't11', 't12')]])
        return rotate_shift(pred, transform, T, (int(row['ref_row0']), int(row['ref_col0'])),
                            float(row['ref_spacing_m']), float(row['ref_left']), float(row['ref_top']))
    if row['method'] in ('shift', 'original_shift', 'tile_region_shift'):
        return shift(pred, int(row['shift_rows']), int(row['shift_cols']))
    return None


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------

MEASURES = ('pixels', 'mae', 'rmse', 'level_m', 'relief_m', 'error_relief_pct', 'fine_r', 'fine_amp', 'fine_rmse')


def measures(surface, ref, rescaled=False):
    """Measures of a surface against the stereo DEM over the pixels where both
    are valid (None below 1000 px); with rescaled=True also labelMAE."""
    m = np.isfinite(surface) & np.isfinite(ref)
    if m.sum() < MIN_PX:
        return None
    s, t = surface[m], ref[m]
    d = s - t
    rel = relief(ref, m)
    out = dict(pixels=int(m.sum()), mae=float(np.abs(d).mean()), rmse=float(np.sqrt(np.mean(d ** 2))),
               level_m=float(np.median(d)), relief_m=rel, error_relief_pct=100 * float(np.abs(d).mean()) / rel)
    if rescaled:
        lab = (s - s.mean()) / s.std() * t.std() + t.mean() if s.std() > 0 else s
        out['label_mae'] = float(np.abs(lab - t).mean())
    fine = fine_structure(surface, ref, m)
    out.update(fine_r=fine['r'], fine_amp=fine['amp'], fine_rmse=fine['rmse'])
    return out


def score_grids(stereo, pred, mola, transform, row):
    """One row of the per-pair results (deploy_*, aligned_*, mola_*) from the
    grids of a pair."""
    out = dict(pair=row['pair'], image=row['image'], band=row['band'], lat=row['lat'], method=row['method'])
    aligned = coregister(pred, transform, row)
    for name, surface, keys in (('deploy', pred, MEASURES), ('aligned', aligned, MEASURES + ('label_mae',)),
                                ('mola', mola, MEASURES)):
        m = measures(surface, stereo, rescaled=name == 'aligned') if surface is not None else None
        out.update({f'{name}_{k}': (m[k] if m else '') for k in keys})
    return out


def score_pair(args):
    row, labels, preds, mola_path = args
    grids = pair_grids(Path(labels) / row['label'], Path(preds) / row['tile'] / row['image'] / 'pred_shot_anchor.tif',
                       mola_path)
    return score_grids(*grids, row)


# ---------------------------------------------------------------------------
# Table 1 and the statements of Section 3.1
# ---------------------------------------------------------------------------

def table1_images(scores):
    """Table 1 whole-image cells and the statistics quoted with them, from the
    rows written by `score` (strings or numbers)."""
    def num(v):
        return float(v) if v not in ('', None) else math.nan

    rows = [{k: (v if k in ('pair', 'image', 'band', 'method') else num(v)) for k, v in r.items()} for r in scores]
    s = [r for r in rows if np.isfinite(r['aligned_mae'])]

    def col(rs, k):
        return np.array([r[k] for r in rs], float)

    def cells(rs):
        return dict(pairs=len(rs), label_mae=median(col(rs, 'aligned_label_mae')),
                    aligned_mae=median(col(rs, 'aligned_mae')), deploy_mae=median(col(rs, 'deploy_mae')),
                    fine_r=median(col(rs, 'aligned_fine_r')), fine_amp=median(col(rs, 'aligned_fine_amp')),
                    error_relief_pct=median(col(rs, 'deploy_error_relief_pct')), mola_mae=median(col(rs, 'mola_mae')),
                    mola_fine_r=median(col(rs, 'mola_fine_r')), mola_fine_amp=median(col(rs, 'mola_fine_amp')),
                    mola_error_relief_pct=median(col(rs, 'mola_error_relief_pct')))

    out = {'all': cells(s)}
    for b in BANDS:
        out[b] = cells([r for r in s if r['band'] == b])

    dep, ali, lab = col(s, 'deploy_mae'), col(s, 'aligned_mae'), col(s, 'aligned_label_mae')
    text = dict(pairs_read=len(rows), pairs=len(s),
                alignment_change_m=median(dep - ali),            # deployMAE - alignedMAE
                lower_after_alignment=int((ali < dep).sum()), higher_after_alignment=int((ali > dep).sum()),
                aligned_minus_label_m=median(ali - lab), relief_m=median(col(s, 'deploy_relief_m')),
                level_m=median(np.abs(col(s, 'deploy_level_m'))), mola_level_m=median(np.abs(col(s, 'mola_level_m'))))
    p80 = [r for r in s if abs(r['lat']) >= 80]
    text['poleward_80'] = dict(pairs=len(p80), deploy_mae=median(col(p80, 'deploy_mae')),
                               mola_mae=median(col(p80, 'mola_mae')),
                               better_than_mola=int((col(p80, 'deploy_mae') < col(p80, 'mola_mae')).sum()),
                               fine_r=median(col(p80, 'aligned_fine_r')), fine_amp=median(col(p80, 'aligned_fine_amp')),
                               mola_fine_r=median(col(p80, 'mola_fine_r')),
                               mola_fine_amp=median(col(p80, 'mola_fine_amp')))
    text['by_relief'] = {}
    for lo, hi in RELIEF_CLASSES:
        c = [r for r in s if lo <= r['deploy_relief_m'] < hi]
        text['by_relief'][f'{lo}-{hi}'] = dict(pairs=len(c), deploy_mae=median(col(c, 'deploy_mae')),
                                               mola_mae=median(col(c, 'mola_mae')),
                                               error_relief_pct=median(col(c, 'deploy_error_relief_pct')),
                                               mola_error_relief_pct=median(col(c, 'mola_error_relief_pct')))
    return dict(table1_images=out, text=text)


def print_tables(result):
    names = dict(all='All latitudes', eq='Equatorial, <30 deg', mid='Mid-latitude, 30-60 deg', polar='Polar, >60 deg')
    print(f'{"Whole CTX images":30s} {"DEM":6s} {"labelMAE":>9s} {"alignedMAE":>11s} {"deployMAE":>10s} '
          f'{"<1 km r":>8s} {"<1 km amp":>10s} {"err/relief %":>13s}')
    for k, c in result['table1_images'].items():
        label = f'{names[k]} ({c["pairs"]})'
        print(f'{label:30s} {"MESA":6s} {c["label_mae"]:9.1f} {c["aligned_mae"]:11.1f} '
              f'{c["deploy_mae"]:10.1f} {c["fine_r"]:8.2f} {c["fine_amp"]:10.2f} {c["error_relief_pct"]:13.1f}')
        print(f'{"":30s} {"MOLA":6s} {"-":>9s} {"-":>11s} {c["mola_mae"]:10.1f} {c["mola_fine_r"]:8.2f} '
              f'{c["mola_fine_amp"]:10.2f} {c["mola_error_relief_pct"]:13.1f}')
    t = result['text']
    print(f'\nco-registration: deployMAE - alignedMAE median {t["alignment_change_m"]:.3f} m; lower at '
          f'{t["lower_after_alignment"]} pairs, higher at {t["higher_after_alignment"]}')
    print(f'alignedMAE - labelMAE median {t["aligned_minus_label_m"]:.2f} m; relief median {t["relief_m"]:.0f} m; '
          f'|level| in place median {t["level_m"]:.2f} m (MOLA {t["mola_level_m"]:.2f} m)')
    p = t['poleward_80']
    print(f'poleward of 80 deg: {p["pairs"]} pairs, deployMAE {p["deploy_mae"]:.2f} m vs MOLA {p["mola_mae"]:.2f} m '
          f'(better at {p["better_than_mola"]}); <1 km r {p["fine_r"]:.2f} (MOLA {p["mola_fine_r"]:.2f}), '
          f'amp {p["fine_amp"]:.2f} (MOLA {p["mola_fine_amp"]:.2f})')
    for k, c in t['by_relief'].items():
        print(f'relief {k} m: {c["pairs"]} pairs, deployMAE {c["deploy_mae"]:.1f} m (MOLA {c["mola_mae"]:.1f}), '
              f'error / relief {c["error_relief_pct"]:.1f}% (MOLA {c["mola_error_relief_pct"]:.1f}%)')


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    lb = sub.add_parser('labels', help='stereo DEM of every test pair from its test tiles (release 1)')
    lb.add_argument('--root', required=True, help='test archive')
    lb.add_argument('--manifest', required=True, help='comma-separated test manifests')
    lb.add_argument('--pairs', required=True)
    lb.add_argument('--out', required=True, help='directory for <pair>.tif')
    sc = sub.add_parser('score', help='measures of every test pair')
    sc.add_argument('--images', required=True, help='data/test_images.csv')
    sc.add_argument('--labels', required=True, help='directory of the stereo DEMs, <pair>.tif')
    sc.add_argument('--preds', required=True, help='PREDS/<tile>/<image>/pred_shot_anchor.tif')
    sc.add_argument('--mola', required=True, help='Mars_MGS_MOLA_DEM_mosaic_global_463m.tif')
    sc.add_argument('--out', required=True, help='CSV, one row per pair')
    sc.add_argument('--jobs', type=int, default=8)
    tb = sub.add_parser('tables', help='Table 1 whole-image cells and the statistics of Section 3.1')
    tb.add_argument('--scores', required=True, help='output of score')
    tb.add_argument('--out', default=None, help='JSON with every value')
    args = ap.parse_args()

    if args.cmd == 'labels':
        pairs = set(open(args.pairs).read().split())
        tiles = {}
        for r in read_rows(args.manifest.split(','), pairs):
            tiles.setdefault(r['pair_short'], set()).add(r['dem_tile'])
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for p in sorted(tiles):
            build_label(args.root, tiles[p], Path(args.out) / f'{p}.tif')
        print(f'{len(tiles)} stereo DEMs')
    elif args.cmd == 'score':
        rows = list(csv.DictReader(open(args.images)))
        with Pool(args.jobs) as pool:
            out = pool.map(score_pair, [(r, args.labels, args.preds, args.mola) for r in rows], chunksize=1)
        write_csv(args.out, out)
        print(f'{len(out)} pairs scored')
    else:
        result = table1_images(list(csv.DictReader(open(args.scores))))
        if args.out:
            open(args.out, 'w').write(json.dumps(result, indent=1))
        print_tables(result)


if __name__ == '__main__':
    main()

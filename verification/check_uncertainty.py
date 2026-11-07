"""Check the released uncertainty layer against the layer built with the
production scripts for the previous mosaic (epoch 98), whose calibration files,
look-up table and per-tile outputs are kept.

    python verification/check_uncertainty.py CALIB_DIR GRIDS LABELS PEDR BOXES_ROOT TAG LAYER_ROOT OUT N_PAIRS BOX

CALIB_DIR   calib/<pair>.npz, calib_infill/<pair>.npz and lut_e098.json
GRIDS       the co-registered test-pair grids the calibration used (<pair>/pred.tif, stereo.tif)
LABELS      stereo DEMs of the test pairs (<pair>.tif)
PEDR        PEDR database
BOXES_ROOT  product tiles (<box>/box_run/ with box_dem_<TAG>_deseam.tif, box_dem_<TAG>_filled.tif.prov.tif,
            box_seam_<TAG>_domid.tif, level_frames_<TAG>/FRAMES.txt and meta/<image>.env)
LAYER_ROOT  the production layer (<box>/<box>_expected_error_m.tif, _uncertainty_class.tif, _summary.json)

1. the look-up table pooled from the production calibration files;
2. the calibration of N_PAIRS test pairs (frame histograms and infill
   histograms), and of N_PAIRS further test pairs with infilled pixels; the
   infill of a pair is compared only if its product tiles are older than the
   calibration file, since some tiles were rebuilt afterwards;
3. the layer of one product tile.
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa import uncertainty as U  # noqa: E402
from mesa.pedr import PedrDatabase  # noqa: E402

calib, grids, labels, pedr_path, boxes, tag, layer_root, out, n_pairs, box = sys.argv[1:11]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
pedr = PedrDatabase(pedr_path)
ok = True

# 1. look-up table
lut = U.build_lut(sorted(glob.glob(f'{calib}/calib/*.npz')), sorted(glob.glob(f'{calib}/calib_infill/*.npz')))
ref = json.load(open(f'{calib}/lut_e098.json'))
worst = 0.0
for k in ('cell_median', 'cell_p90', 'cell_level', 'nolinc_median', 'nolinc_p90', 'global_median', 'infill_median',
          'infill_p90'):
    a = np.atleast_1d(np.array(lut[k], float))
    b = np.atleast_1d(np.array([np.nan if v is None else v for v in np.atleast_1d(ref[k])], float))
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    d = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
    worst = max(worst, d if same_nan else math.inf)
    print(f'LUT {k:14s} max |diff| {d:.2e}, same missing cells {same_nan}')
ok &= worst < 1e-9

# 2. calibration of some test pairs
files = sorted(glob.glob(f'{calib}/calib/*.npz'))
for f in files[::max(1, len(files) // int(n_pairs))][:int(n_pairs)]:
    z = np.load(f, allow_pickle=True)
    pair = str(z['pair'])
    mine = U.calibrate_pair(f'{grids}/{pair}/pred.tif', f'{grids}/{pair}/stereo.tif', float(z['inc']), float(z['az']),
                            pedr)
    same = (np.array_equal(mine['hist'], z['hist']) and np.array_equal(mine['sample'], z['sample'])
            and mine['n'] == int(z['n']))
    tiles = U.covering_tiles(f'{labels}/{pair}.tif', boxes, dem=f'box_run/box_dem_{tag}_deseam.tif',
                             prov=f'box_run/box_dem_{tag}_filled.tif.prov.tif')
    infill_npz = f'{calib}/calib_infill/{pair}.npz'
    zi = np.load(infill_npz, allow_pickle=True)
    if any(os.path.getmtime(f) > os.path.getmtime(infill_npz) for t in tiles for f in t):
        # the tile was rebuilt after the calibration: not the same input
        print(f'{pair}: frame histograms identical {same}; infill not compared (product tile rebuilt after '
              f'the calibration)', flush=True)
        ok &= same
        continue
    mi = U.calibrate_infill_pair(f'{labels}/{pair}.tif', tiles)
    same_i = np.array_equal(mi['hist'], zi['hist']) and np.array_equal(mi['hmodel'], zi['hmodel'])
    print(f'{pair}: frame histograms identical {same}, infill histograms identical {same_i} '
          f'({len(tiles)} tiles, {int(zi["n_infill"])} infilled px)', flush=True)
    ok &= same and same_i

# 2b. infill calibration of test pairs with infilled pixels whose product tiles are older than the calibration
checked = 0
for infill_npz in sorted(glob.glob(f'{calib}/calib_infill/*.npz')):
    zi = np.load(infill_npz, allow_pickle=True)
    if int(zi['n_infill']) == 0:
        continue
    pair = Path(infill_npz).stem
    tiles = U.covering_tiles(f'{labels}/{pair}.tif', boxes, dem=f'box_run/box_dem_{tag}_deseam.tif',
                             prov=f'box_run/box_dem_{tag}_filled.tif.prov.tif')
    if any(os.path.getmtime(f) > os.path.getmtime(infill_npz) for t in tiles for f in t):
        continue
    mi = U.calibrate_infill_pair(f'{labels}/{pair}.tif', tiles)
    same_i = np.array_equal(mi['hist'], zi['hist']) and np.array_equal(mi['hmodel'], zi['hmodel'])
    print(f'{pair}: {int(zi["n_infill"]):,} infilled px by distance class {zi["hist"].sum(1).tolist()}, '
          f'infill histograms identical {same_i}', flush=True)
    ok &= same_i
    checked += 1
    if checked == int(n_pairs):
        break

# 3. layer of one product tile; the sun of each image from its metadata, in the order of the dominant-image index
run = Path(boxes) / box / 'box_run'
frames = [line.split('\t')[0].strip() for line in open(run / f'level_frames_{tag}' / 'FRAMES.txt') if line.strip()]


def read_env(path):
    d = {}
    for line in open(path):
        if '=' in line and not line.startswith('#'):
            k, v = line.strip().split('=', 1)
            d[k] = v.strip().strip("'\"")
    return d


suns = []
for stem in frames:
    e = run / 'meta' / f'{stem}.env'
    suns.append((float(read_env(e)['INC']), float(read_env(e)['AZ'])) if e.exists() else None)
fractions = U.tile_layer(U.LUT(f'{calib}/lut_e098.json'), run / f'box_dem_{tag}_deseam.tif',
                         run / f'box_dem_{tag}_filled.tif.prov.tif', run / f'box_seam_{tag}_domid.tif', suns, pedr,
                         str(out / box))
for name in ('expected_error_m', 'uncertainty_class'):
    released, production = out / f'{box}_{name}.tif', Path(layer_root) / box / f'{box}_{name}.tif'
    with rasterio.open(released) as a, rasterio.open(production) as b:
        x, y = a.read(1), b.read(1)
        same_grid = a.transform == b.transform and a.shape == b.shape and a.crs == b.crs
    d = float(np.abs(x.astype(np.float64) - y).max())
    print(f'{box} {name}: same grid {same_grid}, max |diff| {d:.3e}, pixels differing {int((x != y).sum()):,} of '
          f'{x.size:,}, bitwise {np.array_equal(x, y)}')
    ok &= same_grid and np.array_equal(x, y)
prod = json.load(open(Path(layer_root) / box / f'{box}_summary.json'))['class_area_fraction']
print('class area fractions, released:', {k: round(v, 6) for k, v in fractions.items()})
print('class area fractions, production:', {k: round(v, 6) for k, v in prod.items()})
print(f'UNCERTAINTY CHECK {"PASS" if ok else "DIFF"}')

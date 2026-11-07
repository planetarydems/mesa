"""Check the stage-2 (mosaic) calibration of the uncertainty layer in this
package against the production run of October 2026 (Task 21 of the revision).

    python verification/check_uncertainty_mosaic.py PROD_NPZ_DIR PROD_LUT BOXES_ROOT TAG STAGE1_ROOT FOOT_ROOT \\
        LABELS LAYER_ROOT MOLA OUT PAIRS TILE

PROD_NPZ_DIR  production per-pair calibration files (<pair>.npz with h_al, h_in)
PROD_LUT      the frozen production table (lut_mosaic_v1.json)
BOXES_ROOT    product tiles (<tile>/box_run/box_dem_<TAG>_deseam_dlc2.tif, box_dem_<TAG>_filled.tif.prov.tif)
STAGE1_ROOT   stage-1 layer (<tile>/<tile>_expected_error_m.tif)
FOOT_ROOT     footprint layer (<tile>/<tile>_overlap_depth.tif)
LABELS        stereo DEMs of the test pairs (<pair>.tif)
LAYER_ROOT    the production stage-2 layer (<tile>/<tile>_median_error_m.tif, _p90_error_m.tif, _uncertainty_class.tif)
PAIRS         comma-separated test pairs to recalibrate; TILE one tile to remap

1. the stage-2 table pooled by build_mosaic_lut from the production files;
2. the calibration histograms of PAIRS (co-registered and in place);
3. the released rasters of TILE by remap_tile, pixel by pixel.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa import uncertainty as U  # noqa: E402

(npz_dir, prod_lut, boxes, tag, stage1, foot, labels, layer_root, mola, out, pairs, tile) = sys.argv[1:13]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
data = Path(__file__).resolve().parents[1] / 'data'
rows = {r['pair']: r for r in csv.DictReader(open(data / 'uncertainty_mosaic_coreg.csv'))}
lut1 = json.load(open(data / 'uncertainty_lut_single_image.json'))
report = {}
ok = True

# 1. table
lut = U.build_mosaic_lut(sorted(Path(npz_dir).glob('*.npz')), {p: r['site_id'] for p, r in rows.items()}, lut1['infill_median'])
ref = json.load(open(prod_lut))
d = {}
for k in ('pred_median', 'pred_p90', 'pred_level', 'infill_median', 'infill_p90', 'infill_level', 'pred_noscore_median', 'pred_noscore_p90'):
    d[k] = float(np.max(np.abs(np.atleast_1d(np.array(lut[k], float)) - np.atleast_1d(np.array(ref[k], float)))))
report['table_max_abs_diff'] = d
ok &= max(d.values()) < 1e-9
print('1. table', d)

# 2. calibration of some pairs
paths = lambda t: dict(dem=f'{boxes}/{t}/box_run/box_dem_{tag}_deseam_dlc2.tif', prov=f'{boxes}/{t}/box_run/box_dem_{tag}_filled.tif.prov.tif',
                       score=f'{stage1}/{t}/{t}_expected_error_m.tif', depth=f'{foot}/{t}/{t}_overlap_depth.tif')
cal = {}
for p in pairs.split(','):
    mine = U.calibrate_mosaic_pair(f'{labels}/{p}.tif', paths, rows[p], lut1['infill_median'], mola)
    prod = np.load(f'{npz_dir}/{p}.npz')
    r = dict(method=rows[p]['method'])
    for k in ('h_in', 'h_al'):
        a, b = mine[k].astype(np.int64), prod[k].astype(np.int64)
        r[k] = dict(n_release=int(a.sum()), n_production=int(b.sum()), identical=bool(np.array_equal(a, b)),
                    abs_count_diff=int(np.abs(a - b).sum()),
                    group_medians_max_abs_diff=float(np.nanmax(np.abs(np.array([U.hist_quantile(x, 0.5) for x in a]) - np.array([U.hist_quantile(x, 0.5) for x in b])))) if a.sum() else 0.0)
    cal[p] = r
    print('2.', p, json.dumps(r))
report['calibration'] = cal

# 3. remap of one tile
pr = paths(tile)
area = U.remap_tile(U.MosaicLUT(prod_lut), pr['dem'], pr['prov'], pr['score'], pr['depth'], str(out / tile))
cmp = {}
for name in ('median_error_m', 'p90_error_m', 'uncertainty_class'):
    with rasterio.open(out / f'{tile}_{name}.tif') as a, rasterio.open(f'{layer_root}/{tile}/{tile}_{name}.tif') as b:
        diff = 0
        for _, w in a.block_windows(1):
            x, y = a.read(1, window=w), b.read(1, window=w)
            diff += int((x != y).sum())
    cmp[name] = diff
    ok &= diff == 0
report['remap_pixels_differing'] = cmp
report['remap_class_area_m2'] = area
print('3. remap', tile, cmp)
report['ok'] = bool(ok)
json.dump(report, open(out / 'check_uncertainty_mosaic.json', 'w'), indent=1)
print('PASS' if ok else 'CHECK THE DIFFERENCES')

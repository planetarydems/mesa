"""Check the mosaic correction and the PEDR field against production.

    python verification/check_field.py FIELD_DIR CORRECTIONS_DIR BOXES_ROOT TILE [TILE ...]

FIELD_DIR        production pedr_field/ (ctx_vs_ortho*/<pair>.csv matches, field.json)
CORRECTIONS_DIR  production polar_prep/field/ (murray_robbins_fits.csv, polar_corrections.csv)
TILE             tiles whose aligned_pedr/ images are re-derived from aligned/
"""
import csv
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa.pedr_field import Field, shift_image, tile_vectors  # noqa: E402
from mesa.register import mosaic_corrections  # noqa: E402

field_dir, corr_dir, boxes = sys.argv[1:4]
tiles = sys.argv[4:]

# 1. Global CTX Mosaic corrections from the measured equatorial offsets
fits = list(csv.DictReader(open(os.path.join(corr_dir, 'murray_robbins_fits.csv'))))
prod = {r['safe']: (float(r['dx_m']), float(r['dy_m'])) for r in csv.DictReader(open(os.path.join(corr_dir, 'polar_corrections.csv')))}
mine = mosaic_corrections(fits, sorted(prod))
worst = max(max(abs(round(mine[t][0], 1) - prod[t][0]), abs(round(mine[t][1], 1) - prod[t][1])) for t in prod)
print(f'mosaic corrections: {len(prod)} tiles, largest difference after rounding to 0.1 m: {worst:.2f} m')

# 2. field fit from the production matches
rows = []
for p in sorted(glob.glob(os.path.join(field_dir, 'ctx_vs_ortho', '*.csv')) +
                glob.glob(os.path.join(field_dir, 'ctx_vs_ortho_polar', '*.csv'))):
    for r in csv.DictReader(open(p)):
        if r.get('error'):
            continue
        rows.append(dict(tile=r['box'], image=r['frame'], e_m=r['e_m'], n_m=r['n_m'], ncc=r['ncc'], railed=r['railed']))
tv = tile_vectors(rows)
pf = json.load(open(os.path.join(field_dir, 'field.json')))['boxes']
same_keys = set(tv) == set(pf)
d = max(max(abs(tv[k]['e'] - pf[k]['e']), abs(tv[k]['n_'] - pf[k]['n_']), abs(tv[k]['n'] - pf[k]['n_used']))
        for k in pf if k in tv)
print(f'field: {len(tv)} tile vectors (production {len(pf)}), same tiles {same_keys}, largest difference {d:.3e}')

# 3. apply to the images of some tiles, compare with the production aligned_pedr images
field = Field(tv)
with tempfile.TemporaryDirectory() as tmp:
    for t in tiles:
        rec = json.load(open(os.path.join(boxes, t, 'box_run', 'aligned_pedr', 'PEDR_FIELD.json')))['frames']
        worst, n = 0.0, 0
        for stem, v in sorted(rec.items()):
            if 'gt_new' not in v or not os.path.exists(v['src']):
                continue
            with rasterio.open(v['src']) as s:
                if max(abs(a - b) for a, b in zip(tuple(s.transform)[:6], v['gt_src'])) > 1e-6:
                    continue                                   # source relabelled since; not comparable
            dst = os.path.join(tmp, f'{stem}.vrt')
            shift_image(v['src'], dst, field)
            with rasterio.open(dst) as a, rasterio.open(os.path.join(boxes, t, 'box_run', 'aligned_pedr',
                                                                     f'{stem}_aligned.vrt')) as b:
                worst = max(worst, max(abs(x - y) for x, y in zip(tuple(a.transform)[:6], tuple(b.transform)[:6])))
                assert a.shape == b.shape and a.crs == b.crs
            n += 1
        print(f'{t}: {n} images shifted, largest geotransform difference from production {worst:.3e} m')

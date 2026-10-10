"""Check the released global-mosaic stages against the production mosaic.

    python verification/check_mosaic.py BOXES_ROOT PRODUCT_TAG PRODUCTION_DIR MOLA OUT_DIR TILE [TILE ...]

Each stage is fed with the production inputs of that stage: coarse from the
de-seamed tiles, level from the production coarse lattice, render from the
production coarse lattice and correction fields.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mader import mosaic  # noqa: E402

boxes, tag, prod, mola_path, out = sys.argv[1:6]
tiles = sys.argv[6:]
dem = f'box_run/box_dem_{tag}_deseam.tif'
prov = f'box_run/box_dem_{tag}_filled.tif.prov.tif'


def compare(a_path, b_path, label, band=1):
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        same_grid = a.transform == b.transform and a.shape == b.shape
        x, y = a.read(band).astype(np.float64), b.read(band).astype(np.float64)
    both = np.isfinite(x) & np.isfinite(y)
    nan_same = np.array_equal(np.isnan(x), np.isnan(y))
    d = np.abs(x[both] - y[both])
    print(f'{label}: same grid {same_grid}, same NaN mask {nan_same}, max |diff| {d.max() if d.size else 0:.3e}, '
          f'bitwise {np.array_equal(x, y, equal_nan=True)}', flush=True)


# coarse
lay = mosaic.Layout(boxes, os.path.join(out, 'stage_coarse'), dem, prov)
for t in tiles:
    t0 = time.time()
    mosaic.coarse(t, lay, threads=4)
    compare(os.path.join(lay.work, 'coarse', f'{t}.tif'), os.path.join(prod, 'coarse', f'{t}.tif'), f'coarse {t}')

# level (production coarse lattice as input)
lay = mosaic.Layout(boxes, os.path.join(out, 'stage_level'), dem, prov)
all_tiles = set(lay.tiles())
for t in tiles:
    mosaic.level(t, lay, all_tiles, coarse_dir=os.path.join(prod, 'coarse'))
    for band, name in ((1, 'correction'), (2, 'node differences'), (3, 'smoothed differences')):
        compare(os.path.join(lay.work, 'corr', f'{t}.tif'), os.path.join(prod, 'corr', f'{t}.tif'), f'level {t} {name}',
                band)

# render (production coarse lattice and corrections as inputs)
work = os.path.join(out, 'stage_render')
os.makedirs(work, exist_ok=True)
for d in ('coarse', 'corr'):
    if not os.path.exists(os.path.join(work, d)):
        os.symlink(os.path.join(prod, d), os.path.join(work, d))
lay = mosaic.Layout(boxes, work, dem, prov)
mola = mosaic.MolaBridge(mola_path)
for t in tiles:
    t0 = time.time()
    mosaic.render(t, lay, mola, all_tiles, threads=8)
    print(f'render {t}: {time.time() - t0:.0f} s', flush=True)
    compare(os.path.join(work, 'tiles', f'{t}.tif'), os.path.join(prod, 'tiles', f'{t}.tif'), f'render {t} DEM')
    compare(os.path.join(work, 'tiles', f'{t}_mask.tif'), os.path.join(prod, 'tiles', f'{t}_mask.tif'),
            f'render {t} mask')

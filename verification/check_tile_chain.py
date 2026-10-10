"""Check the released tile chain against a production tile, one stage at a
time, each stage fed with the production inputs of that stage.

    python verification/check_tile_chain.py STAGE BOX PRODUCT_TAG BOXES_ROOT OUT_DIR MOLA PEDR [INFILL_DIR]

STAGE    level   predicted DEMs -> levelled mosaic (+ offsets, levelled DEMs, weights)
         infill  levelled mosaic + CTX mosaic -> infill surface and provenance (needs INFILL_DIR
                 with coarse.pt and fine.pt)
         seams   levelled mosaic + infill surface -> joined tile; joined tile -> de-seamed tile
"""
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mader import infill, level, seams  # noqa: E402

stage, box, tag, boxes, out = sys.argv[1:6]
mola, pedr = sys.argv[6:8]
run = Path(boxes) / box / 'box_run'
out = Path(out) / box
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def compare(a_path, b_path, label):
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        grid = a.transform == b.transform and a.shape == b.shape and a.crs == b.crs
        x, y = a.read(1).astype(np.float64), b.read(1).astype(np.float64)
        nda, ndb = a.nodata, b.nodata
    va = np.isfinite(x) & (x != nda) if nda is not None else np.isfinite(x)
    vb = np.isfinite(y) & (y != ndb) if ndb is not None else np.isfinite(y)
    both = va & vb
    d = np.abs(x[both] - y[both])
    n_diff = int((d > 0).sum())
    print(f'{label}: same grid {grid}; valid px {int(va.sum()):,} / {int(vb.sum()):,} '
          f'(mask differs at {int((va != vb).sum()):,}); max |diff| {d.max() if d.size else 0:.3e}, '
          f'pixels differing {n_diff:,} ({100 * n_diff / max(both.sum(), 1):.4f}%), bitwise {np.array_equal(x, y)}',
          flush=True)
    return d.max() if d.size else 0.0


if stage == 'level':
    frames = sorted(glob.glob(str(run / f'preds_{tag}' / '*' / 'pred_shot_anchor.tif')))
    work = out / 'leveled'
    work.mkdir(exist_ok=True)
    by_name = level.frames_on_tile_grid(box, frames, work)
    for name, p in by_name.items():                       # warped copies of sinusoidal DEMs
        if p.startswith(str(work)):
            compare(p, run / f'preds_{tag}' / name / 'pred_shot_anchor_eqc.tif', f'warp {name}')
    screen = Path(boxes).parent / 'shape_error' / box / f'orbit_screen_{box}.txt'
    excl = [int(t) for t in screen.read_text().split()] if screen.exists() else None
    print(f'{len(by_name)} DEMs; orbit screen: {len(excl) if excl else "none"}', flush=True)
    level.level_tile(by_name, mola, pedr, out / 'leveled.tif', work, excl)
    print(f'levelled in {time.time() - t0:.0f} s', flush=True)
    prod = {r['frame'].replace('_pred_shot_anchor', ''): r for r in
            csv.DictReader(open(run / f'box_dem_{tag}_leveled_offsets.csv'))}
    mine = {r['frame']: r for r in csv.DictReader(open(out / 'leveled_offsets.csv'))}
    assert set(prod) == set(mine), (len(prod), len(mine))
    dd = max(abs(float(prod[k]['delta_m']) - float(mine[k]['delta_m'])) for k in prod)
    ties = sum(prod[k]['shot_tie_m'] != mine[k]['shot_tie_m'] or prod[k]['n_shots'] != mine[k]['n_shots']
               or prod[k]['n_overlaps'] != mine[k]['n_overlaps'] for k in prod)
    print(f'offsets: {len(prod)} DEMs, largest |delta difference| {dd:.3f} m (CSV keeps 3 decimals), '
          f'{ties} rows with a different tie, shot count or overlap count', flush=True)
    for name in sorted(by_name)[:3]:
        compare(work / f'{name}_leveled.tif', run / f'leveled_{tag}' / f'{name}_pred_shot_anchor_leveled.tif',
                f'levelled DEM {name}')
        compare(work / f'{name}_w.tif', run / f'leveled_{tag}' / f'{name}_pred_shot_anchor_w.tif', f'weight {name}')
    compare(out / 'leveled.tif', run / f'box_dem_{tag}_leveled.tif', 'LEVELLED MOSAIC')

elif stage == 'infill':
    d = Path(sys.argv[8])
    infill.fill_tile(run / f'box_dem_{tag}_leveled.tif', run / f'ctx_robbins_18m_{tag}.tif', mola,
                     d / 'coarse.pt', d / 'fine.pt', str(out / 'infill'))
    print(f'infill in {time.time() - t0:.0f} s', flush=True)
    compare(out / 'infill.prov.tif', run / f'box_dem_{tag}_filled.tif.prov.tif', 'PROVENANCE')
    compare(out / 'infill.model.tif', run / f'box_dem_{tag}_filled.tif.model.tif', 'INFILL SURFACE')

elif stage == 'seams':
    seams.seamline_merge(run / f'box_dem_{tag}_leveled.tif', run / f'box_dem_{tag}_filled.tif.model.tif',
                         out / 'final.tif')
    print(f'merge in {time.time() - t0:.0f} s', flush=True)
    compare(out / 'final.tif', run / f'box_dem_{tag}_FINAL.tif', 'JOINED TILE (FINAL)')
    levelled = sorted(glob.glob(str(run / f'leveled_{tag}' / '*_leveled.tif')))
    seams.deseam(run / f'box_dem_{tag}_FINAL.tif', run / f'box_dem_{tag}_filled.tif.prov.tif', levelled,
                 out / 'deseam.tif', str(out / 'deseam_seams'), jobs=int(os.environ.get('JOBS', 24)))
    print(f'de-seam in {time.time() - t0:.0f} s', flush=True)
    compare(out / 'deseam_seams_mask.tif', run / f'box_seam_{tag}_mask.tif', 'seam mask')
    compare(out / 'deseam_seams_domid.tif', run / f'box_seam_{tag}_domid.tif', 'dominant DEM index')
    gate_prod = {(int(r['row0']), int(r['col0'])): float(r['af_base'])
                 for r in csv.DictReader(open(run / f'box_deseam_base_{tag}.csv'))}
    gate_mine = {(int(r['row0']), int(r['col0'])): float(r['af_base'])
                 for r in csv.DictReader(open(out / 'deseam_seams_sites.csv')) if float(r['af_base']) > 0}
    print(f'sites with a seam: production {len(gate_prod)}, released {len(gate_mine)}, same keys '
          f'{set(gate_prod) == set(gate_mine)}, largest |af_base difference| '
          f'{max((abs(gate_prod[k] - gate_mine.get(k, 0)) for k in gate_prod), default=0):.2e}', flush=True)
    compare(out / 'deseam.tif', run / f'box_dem_{tag}_deseam.tif', 'DE-SEAMED TILE')
print(f'=== {stage} {box} done in {time.time() - t0:.0f} s')

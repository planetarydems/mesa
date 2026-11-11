"""Write data/test_images.csv for the paper (v3) from the co-registration
records of the manuscript's whole-image evaluation.

    python verification/make_test_images_v3.py OLD_CSV PAIR_FRAME RIGID_DIR MATCH_DIR STEP3_DIR OFFSETS LABELS GRIDS OUT_CSV

OLD_CSV     the release-1 data/test_images.csv (latitude of each pair)
PAIR_FRAME  pair_frame_r2.tsv: the image scored at each pair (columns r1_box, r1_stem)
RIGID_DIR   <pair>.json of the image-only co-registration (final_method, refs, tile_region)
MATCH_DIR   <pair>.json of the window matches (result.d18, the consensus shift)
STEP3_DIR   <pair>.json naming the registered image (frame_pedr) the transform was fitted on
OFFSETS     offsets_r2.csv: the image shift of the whole overlap (a_e_m, a_n_m)
LABELS      the full stereo DEMs, <pair>.tif
GRIDS       the manuscript's grids, <pair>/{pred,stereo}.tif (for the whole-overlap shift)
OUT_CSV     output

The transforms are those of the manuscript (v3, fixed release-1 image
selection): a rotation and shift where the windows confirmed it, the
consensus shift of the windows, the shift of the whole overlap, or, at three
pairs, the transform of the region of the test tiles. Every number is copied
at full precision; integer shifts are rounded as the manuscript's scorer
rounds them (Python's round).
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import transform_bounds

old_csv, pair_frame, rigid_dir, match_dir, step3_dir, offsets, labels, grids, out_csv = sys.argv[1:10]
old = {r['pair']: r for r in csv.DictReader(open(old_csv))}
sel = {r['pair']: r for r in csv.DictReader(open(pair_frame), delimiter='\t')}
off = {r['pair']: r for r in csv.DictReader(open(offsets))}


def jac(transform, crs, a):
    """d(east, north) / d(col, row) in metres at the median valid pixel of a (the scorer's jac)."""
    rr, cc = np.nonzero(np.isfinite(a))
    r0, c0 = float(np.median(rr)), float(np.median(cc))
    t = Transformer.from_crs(crs, CRS.from_proj4('+proj=longlat +R=3396190 +no_defs'), always_xy=True)

    def ll(c, r):
        x, y = transform * (c, r)
        return t.transform(x, y)
    lon0, lat0 = ll(c0, r0)
    k = math.cos(math.radians(lat0))

    def en(lon, lat):
        dl = ((lon - lon0 + 180.0) % 360.0) - 180.0
        return 3396190.0 * k * math.radians(dl), 3396190.0 * math.radians(lat - lat0)
    return np.array([en(*ll(c0 + 1, r0)), en(*ll(c0, r0 + 1))]).T


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        a[a < -1e5] = np.nan
        return a, s.transform, s.crs


rows = []
for pair in old:
    r = json.load(open(Path(rigid_dir) / f'{pair}.json'))
    m = r['final_method']
    row = dict(pair=pair, image=sel[pair]['r1_stem'], tile=sel[pair]['r1_box'], band=r.get('band') or old[pair]['band'],
               lat=old[pair]['lat'], label=f'{pair}.tif', method=m)
    row.update({k: '' for k in ('t00', 't01', 't02', 't10', 't11', 't12', 'ref_row0', 'ref_col0', 'ref_spacing_m',
                                'ref_left', 'ref_top', 'shift_rows', 'shift_cols')})
    T = None
    if m == 'rotation+shift':
        st3 = json.load(open(Path(step3_dir) / f'{pair}.json'))
        with rasterio.open(Path(labels) / f'{pair}.tif') as d:
            lcrs, lb = d.crs, d.bounds
        with rasterio.open(st3['frame_pedr']) as d:
            fb = transform_bounds(d.crs, lcrs, *d.bounds, densify_pts=41)
        v = r['refs'][r['rigid_gate']['ref']]
        T, crop, g = v['rigid']['T'], v['crop'], v['grid_m']
        left, top = max(lb.left, fb[0]), min(lb.top, fb[3])
    elif m == 'tile_region_rigid':
        t = r['tile_region']
        T, crop, g, left, top = t['T'], t['crop'], t['grid_m'], t['L'], t['Tp']
    elif m == 'shift':
        d = json.load(open(Path(match_dir) / f'{pair}.json'))['result']['d18']
        row.update(shift_rows=int(round(d[1])), shift_cols=int(round(d[0])))
    elif m == 'tile_region_shift':
        d = r['tile_region']['d18']
        row.update(shift_rows=int(round(d[1])), shift_cols=int(round(d[0])))
    elif m == 'original_shift':
        _, ptf, pcrs = read(Path(grids) / pair / 'pred.tif')
        st, _, _ = read(Path(grids) / pair / 'stereo.tif')
        o = off[pair]
        d = (-np.linalg.solve(jac(ptf, pcrs, st), np.array([float(o['a_e_m']), float(o['a_n_m'])]))).tolist()
        row.update(shift_rows=int(round(d[1])), shift_cols=int(round(d[0])))
    else:
        raise SystemExit(f'{pair}: method {m}')
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        row.update(t00=T[0, 0], t01=T[0, 1], t02=T[0, 2], t10=T[1, 0], t11=T[1, 1], t12=T[1, 2],
                   ref_row0=int(crop[0]), ref_col0=int(crop[1]), ref_spacing_m=float(g), ref_left=float(left),
                   ref_top=float(top))
    rows.append(row)

with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for row in rows:
        w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
from collections import Counter  # noqa: E402
print(f'{len(rows)} pairs -> {out_csv}: {dict(Counter(r["method"] for r in rows))}; '
      f'images changed from the old table at {sum(r["image"] != old[r["pair"]]["image"] for r in rows)} pairs')

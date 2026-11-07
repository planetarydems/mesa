"""Check the released evaluation of whole CTX images (Table 1 whole-image rows,
Section 3.1) against the manuscript's scoring of epoch 88.

    python verification/check_images.py ROOT MANIFEST_DIR PAIRS IMAGES LABELS_REF PREDS MOLA GRIDS_REF JSON_REF OUT [JOBS]

ROOT          the test archive (mars_ctx_dem_v4_pedr)
MANIFEST_DIR  manifest_v4s_test375_{eq,mid,polar}.csv
PAIRS         the 359 test pairs
IMAGES        data/test_images.csv
LABELS_REF    the stereo DEMs the manuscript was scored against, <pair>.tif
PREDS         the epoch-88 predictions, PREDS/<tile>/<image>/pred_shot_anchor.tif
MOLA          the distributed MOLA mosaic
GRIDS_REF     the manuscript's grids, GRIDS_REF/<pair>/{stereo,pred,mola}.tif
JSON_REF      the manuscript's per-pair scores, JSON_REF/<pair>.json (blocks
              deploy = in place, final = co-registered, mola; hp1k_* = <1 km)
OUT           output directory

1. stereo DEMs rebuilt from the test tiles vs LABELS_REF, bit for bit
2. pair_grids vs GRIDS_REF, pixel by pixel
3. every per-pair measure vs JSON_REF
4. Table 1 whole-image cells and the Section 3.1 statistics vs the same
   medians computed from JSON_REF, and vs the values printed in the manuscript
"""
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa import evaluate_images as ei  # noqa: E402
from mesa.evaluate import read_rows, write_csv  # noqa: E402

root, man_dir, pairs_file, images_csv, labels_ref, preds, mola_tif, grids_ref, json_ref, out = sys.argv[1:11]
jobs = int(sys.argv[11]) if len(sys.argv) > 11 else 16
out = Path(out)
labels = out / 'labels'
labels.mkdir(parents=True, exist_ok=True)
pairs = [p for p in open(pairs_file).read().split()]
rows = list(csv.DictReader(open(images_csv)))
ok = True


def difference(a, b):
    """|a - b|, relative where |b| > 1; 0 if both are NaN, inf if one is."""
    na, nb = math.isnan(a), math.isnan(b)
    if na or nb:
        return 0.0 if na and nb else math.inf
    return abs(a - b) / max(1.0, abs(b))


print(f'{len(rows)} rows in {images_csv}; same pairs in the same order as {pairs_file}: {[r["pair"] for r in rows] == pairs}')
ok &= [r['pair'] for r in rows] == pairs

# ---- 1. stereo DEMs ---------------------------------------------------------------------------------
manifests = [str(Path(man_dir) / f'manifest_v4s_test375_{b}.csv') for b in ('eq', 'mid', 'polar')]
tiles = {}
for r in read_rows(manifests, set(pairs)):
    tiles.setdefault(r['pair_short'], set()).add(r['dem_tile'])
n_same = 0
for p in pairs:
    f = labels / f'{p}.tif'
    if not f.exists():
        ei.build_label(root, tiles[p], f)
    with rasterio.open(f) as a, rasterio.open(Path(labels_ref) / f'{p}.tif') as b:
        same = (a.shape == b.shape and a.transform == b.transform and a.crs == b.crs
                and np.array_equal(a.read(1), b.read(1), equal_nan=True))
    n_same += same
print(f'stereo DEMs: {n_same} of {len(pairs)} identical to the reference ({sum(len(v) for v in tiles.values())} tiles)')
ok &= n_same == len(pairs)


# ---- 2. and 3. grids and per-pair measures ------------------------------------------------------------
def read_ref(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        t, nd = s.transform, s.nodata
    a[a == nd] = np.nan
    a[a < -1e5] = np.nan
    return a, t


def one(row):
    p = row['pair']
    stereo, pred, mola, transform = ei.pair_grids(labels / row['label'],
                                                  Path(preds) / row['tile'] / row['image'] / 'pred_shot_anchor.tif',
                                                  mola_tif)
    cmp = {}
    for name, a in (('stereo', stereo), ('pred', pred), ('mola', mola)):
        b, tb = read_ref(Path(grids_ref) / p / f'{name}.tif')
        if a.shape != b.shape or transform != tb:
            cmp[name] = None                 # different extent
            continue
        both = np.isfinite(a) & np.isfinite(b)
        cmp[name] = (int((np.isfinite(a) != np.isfinite(b)).sum()), float(np.abs(a[both] - b[both]).max()))
    s = ei.score_grids(stereo, pred, mola, transform, row)
    # MOLA on the prediction's in-place pixels only (diagnostic: MOLA is scored on every stereo-DEM pixel of the grid)
    m = np.isfinite(pred) & np.isfinite(stereo) & np.isfinite(mola)
    diag = dict(mola_mae_pred_pixels=float(np.abs(mola[m] - stereo[m]).mean()),
                mola_extra_pixels=int((np.isfinite(mola) & np.isfinite(stereo)).sum() - m.sum()))
    return s, cmp, diag


with Pool(jobs) as pool:
    res = pool.map(one, rows, chunksize=1)
scores = [r[0] for r in res]
write_csv(out / 'images.csv', scores)

for k in ('stereo', 'pred', 'mola'):
    c = [(r[1][k], r[0]['pair']) for r in res]
    extent = [p for v, p in c if v is None]
    nmask, pm = max((v[0], p) for v, p in c if v is not None)
    d, pd = max((v[1], p) for v, p in c if v is not None)
    print(f'grid {k:6s}: extent differs at {len(extent)} pairs {extent[:3]}; largest no-data mask difference {nmask} px '
          f'({pm}); largest |difference| {d:.3g} m ({pd})')
    ok &= not extent and nmask == 0 and d == 0.0

ref = {p: json.load(open(Path(json_ref) / f'{p}.json')) for p in pairs}
keys = dict(pixels='n', mae='mae', rmse='rmse', level_m='off', relief_m='relief', error_relief_pct='pct', fine_r='hp1k_r',
            fine_amp='hp1k_amp', fine_rmse='hp1k_rmse', label_mae='label')
n_method = sum(s['method'] != ref[s['pair']]['final_method'] for s in scores)
print(f'co-registration method differs from the reference at {n_method} pairs')
ok &= n_method == 0
for blk, rblk in (('deploy', 'deploy'), ('aligned', 'final'), ('mola', 'mola')):
    for k, rk in keys.items():
        if f'{blk}_{k}' not in scores[0]:
            continue
        d, at = 0.0, ''
        for s in scores:
            e = difference(s[f'{blk}_{k}'], ref[s['pair']][rblk][rk])
            if e > d:
                d, at = e, s['pair']
        print(f'  {blk:7s} {k:16s} vs {rblk}.{rk:9s} largest difference {d:.2e} (relative above 1) {at}')
        ok &= d < 1e-9

d = np.array([r[2]['mola_mae_pred_pixels'] for r in res]) - np.array([s['mola_mae'] for s in scores])
n_extra = sum(r[2]['mola_extra_pixels'] > 0 for r in res)
print(f'MOLA scored on stereo-DEM pixels outside the prediction at {n_extra} pairs; MOLA MAE on the prediction\'s pixels '
      f'only: median over pairs {ei.median([r[2]["mola_mae_pred_pixels"] for r in res]):.3f} m '
      f'(as scored {ei.median([s["mola_mae"] for s in scores]):.3f} m), largest per-pair change {np.abs(d).max():.3f} m')

# ---- 4. tables ---------------------------------------------------------------------------------------
result = ei.table1_images(scores)
json.dump(result, open(out / 'tables.json', 'w'), indent=1)
ei.print_tables(result)

lat = {r['pair']: float(r['lat']) for r in rows}
J = [dict(ref[p], lat=lat[p]) for p in pairs if ref[p].get('final')]
md = ei.median


def cells(js):
    return dict(pairs=len(js), label_mae=md([j['final']['label'] for j in js]), aligned_mae=md([j['final']['mae'] for j in js]),
                deploy_mae=md([j['deploy']['mae'] for j in js]), fine_r=md([j['final']['hp1k_r'] for j in js]),
                fine_amp=md([j['final']['hp1k_amp'] for j in js]), error_relief_pct=md([j['deploy']['pct'] for j in js]),
                mola_mae=md([j['mola']['mae'] for j in js]), mola_fine_r=md([j['mola']['hp1k_r'] for j in js]),
                mola_fine_amp=md([j['mola']['hp1k_amp'] for j in js]), mola_error_relief_pct=md([j['mola']['pct'] for j in js]))


expect = {'all': cells(J)}
for b in ei.BANDS:
    expect[b] = cells([j for j in J if j['band'] == b])
dep, ali = np.array([j['deploy']['mae'] for j in J]), np.array([j['final']['mae'] for j in J])
p80 = [j for j in J if abs(j['lat']) >= 80]
text = dict(alignment_change_m=md(dep - ali), lower_after_alignment=int((ali < dep).sum()),
            higher_after_alignment=int((ali > dep).sum()), aligned_minus_label_m=md([j['final']['mae'] - j['final']['label'] for j in J]),
            relief_m=md([j['deploy']['relief'] for j in J]), level_m=md([abs(j['deploy']['off']) for j in J]),
            mola_level_m=md([abs(j['mola']['off']) for j in J]),
            poleward_80=dict(pairs=len(p80), deploy_mae=md([j['deploy']['mae'] for j in p80]),
                             mola_mae=md([j['mola']['mae'] for j in p80]),
                             better_than_mola=sum(j['deploy']['mae'] < j['mola']['mae'] for j in p80),
                             fine_r=md([j['final']['hp1k_r'] for j in p80]), fine_amp=md([j['final']['hp1k_amp'] for j in p80]),
                             mola_fine_r=md([j['mola']['hp1k_r'] for j in p80]),
                             mola_fine_amp=md([j['mola']['hp1k_amp'] for j in p80])),
            by_relief={})
for lo, hi in ei.RELIEF_CLASSES:
    c = [j for j in J if lo <= j['deploy']['relief'] < hi]
    text['by_relief'][f'{lo}-{hi}'] = dict(pairs=len(c), deploy_mae=md([j['deploy']['mae'] for j in c]),
                                           mola_mae=md([j['mola']['mae'] for j in c]),
                                           error_relief_pct=md([j['deploy']['pct'] for j in c]),
                                           mola_error_relief_pct=md([j['mola']['pct'] for j in c]))


def flat(d, pre=''):
    o = {}
    for k, v in d.items():
        o.update(flat(v, f'{pre}{k}.') if isinstance(v, dict) else {pre + k: v})
    return o


mine, theirs = flat(dict(table1_images=result['table1_images'], text=result['text'])), \
    flat(dict(table1_images=expect, text=text))
worst = max(difference(mine[k], v) for k, v in theirs.items())
print(f'\ntables: {len(theirs)} values, largest difference from the medians of the reference scores {worst:.2e}')
ok &= worst < 1e-9

# values printed in the manuscript (Table 1; Section 3.1; Texts S7 and S10)
PAPER = {'all': (9.8, 10.6, 10.5, 0.75, 0.82, 1.9, 18.4, 0.33, 0.40, 3.2),
         'eq': (10.8, 11.9, 12.3, 0.77, 0.82, 2.0, 19.6, 0.31, 0.34, 3.3),
         'mid': (9.2, 10.1, 9.9, 0.75, 0.83, 1.5, 18.3, 0.40, 0.49, 2.6),
         'polar': (7.4, 8.2, 7.7, 0.46, 0.53, 3.7, 9.3, 0.39, 0.63, 3.6)}
COLS = ('label_mae', 'aligned_mae', 'deploy_mae', 'fine_r', 'fine_amp', 'error_relief_pct', 'mola_mae', 'mola_fine_r',
        'mola_fine_amp', 'mola_error_relief_pct')
DEC = (1, 1, 1, 2, 2, 1, 1, 2, 2, 1)
PAIRS = {'all': 359, 'eq': 223, 'mid': 109, 'polar': 27}
bad = []
for k, vals in PAPER.items():
    c = result['table1_images'][k]
    if c['pairs'] != PAIRS[k]:
        bad.append(f'{k} pairs')
    bad += [f'{k} {col}' for col, v, n in zip(COLS, vals, DEC) if round(c[col], n) != v]
t = result['text']
TEXT = [('alignment_change_m', 2, 0.01), ('lower_after_alignment', 0, 185), ('higher_after_alignment', 0, 174),
        ('aligned_minus_label_m', 1, 0.5), ('relief_m', 0, 574), ('level_m', 1, 2.5), ('mola_level_m', 1, 3.1),
        ('poleward_80.pairs', 0, 18), ('poleward_80.deploy_mae', 1, 7.8), ('poleward_80.mola_mae', 1, 8.9),
        ('poleward_80.better_than_mola', 0, 13), ('poleward_80.fine_r', 2, 0.44), ('poleward_80.mola_fine_r', 2, 0.41),
        ('poleward_80.fine_amp', 2, 0.53), ('poleward_80.mola_fine_amp', 2, 0.66),
        ('by_relief.0-300.pairs', 0, 101), ('by_relief.0-300.deploy_mae', 1, 8.2), ('by_relief.0-300.mola_mae', 1, 9.2),
        ('by_relief.1000-3000.deploy_mae', 1, 12.7), ('by_relief.1000-3000.mola_mae', 1, 31.8),
        ('by_relief.3000-inf.pairs', 0, 11), ('by_relief.3000-inf.deploy_mae', 1, 21.9),
        ('by_relief.3000-inf.mola_mae', 1, 55.6), ('by_relief.0-300.error_relief_pct', 1, 6.3),
        ('by_relief.0-300.mola_error_relief_pct', 1, 7.1), ('by_relief.1000-3000.error_relief_pct', 1, 0.8),
        ('by_relief.1000-3000.mola_error_relief_pct', 1, 1.9)]
ft = flat(t)
bad += [k for k, n, v in TEXT if round(ft[k], n) != v]
print(f'manuscript values: {sum(len(v) for v in PAPER.values()) + len(PAIRS) + len(TEXT)} checked, '
      f'{len(bad)} differ {bad if bad else ""}')
ok &= not bad
print(f'IMAGES CHECK {"PASS" if ok else "DIFF"}')

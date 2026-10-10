"""Check the released evaluation of the test tiles (Table 1 tile rows, Table 2)
against the manuscript's scoring of epoch 88.

    python verification/check_evaluate.py MODEL DATA_ROOT MANIFEST_DIR PAIRS REF_DIR MOLA_1K_CSV MOLA_AUDIT_CSV \
        MOLA_BINS_DIR MOLA_TIF OUT

MANIFEST_DIR    manifest_v4s_test375_{eq,mid,polar}.csv
PAIRS           the 359 test pairs
REF_DIR         the manuscript's epoch-88 workspace: tiles/e088_tile_1k_test359.csv
                (eval_tile_deploy_1k.py), bins/e088_359_<band>_med.npz
                (eval_illum_median_rel.py) and bins/table2_e088.json
MOLA_1K_CSV     tiles_moladist_1k.csv (fine-scale measures of MOLA per tile)
MOLA_AUDIT_CSV  tiles_moladist.csv (MOLA MAE and stereo relief per tile)
MOLA_BINS_DIR   mola_<band>.npz of mola_dist_bins_359.py (MOLA as distributed by
                distance from the nearest shot, Table 2)
MOLA_TIF        the distributed MOLA mosaic

The reference CSV rounds to 3 (MAE) and 4 (fine-scale) decimals; the released
values, rounded the same way, must equal it.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mader import evaluate as ev  # noqa: E402
from mader.data import TileDataset, load_incidence  # noqa: E402
from mader.model import load_model  # noqa: E402

model_path, root, man_dir, pairs_file, ref, mola_1k, mola_audit, mola_bins, mola_tif, out = sys.argv[1:11]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_path, device)
pairs = set(open(pairs_file).read().split())
manifests = [str(Path(man_dir) / f'manifest_v4s_test375_{b}.csv') for b in ('eq', 'mid', 'polar')]
ok = True

# ---- Table 1, tile rows: every sample -------------------------------------------------------------
samples_csv = out / 'tiles.csv'
if not samples_csv.exists():
    incidence = load_incidence(str(Path(root) / 'ctx_overlapping_95pct_polar.json'))
    rows = [ev.score_sample(model, r, root, incidence, device) for r in ev.read_rows(manifests, pairs)]
    ev.write_csv(samples_csv, [r for r in rows if r is not None])
mine = {(r['dem_tile'], r['ctx_product_id']): r for r in csv.DictReader(open(samples_csv))}
prod_csv = Path(ref) / 'tiles' / 'e088_tile_1k_test359.csv'
prod = {(r['dem_tile'], r['ctx_product_id']): r for r in csv.DictReader(open(prod_csv))}
print(f'tiles: released {len(mine)} samples, production {len(prod)}; same samples {set(mine) == set(prod)}')
ok &= set(mine) == set(prod)
how = {'affine': 'affine', 'offset': 'gain_gated', 'rejected': 'rejected', 'fit_failed': 'mola_offset',
       'few_shots': 'mola_offset'}
columns = (('label_mae', 'mae_label', 3), ('aligned_mae', 'mae_deploy', 3), ('fine_r', 'hp1k_corr_deploy', 4),
           ('fine_amp', 'hp1k_amp_deploy', 4), ('incidence_deg', 'inc_deg', 2))
for a, b, decimals in columns:
    same = sum(round(float(mine[k][a]), decimals) == float(prod[k][b]) for k in prod if k in mine)
    print(f'  {a:14s} rounded to {decimals} decimals equals {b} at {same} of {len(prod)} samples')
    ok &= same == len(prod)
n_how = sum(how[mine[k]['anchoring']] != prod[k]['deploy_path'] for k in prod if k in mine)
n_shots = sum(int(mine[k]['shots']) != int(prod[k]['n_shot_in_win']) for k in prod if k in mine)
print(f'  anchoring differs at {n_how} samples, shot count at {n_shots}')
ok &= n_how == 0 and n_shots == 0

# ---- MOLA on the same windows ----------------------------------------------------------------------
mola_csv = out / 'tiles_mola.csv'
if not mola_csv.exists():
    from multiprocessing import Pool
    tiles = sorted({k[0] for k in mine})
    with Pool(16) as pool:
        ev.write_csv(mola_csv, [r for r in pool.map(ev.score_mola_tile, [(root, t, mola_tif) for t in tiles], 8) if r])
mm = {r['dem_tile']: r for r in csv.DictReader(open(mola_csv))}
p1k = {r['dem_tile']: r for r in csv.DictReader(open(mola_1k))}
pau = {r['dem_tile']: r for r in csv.DictReader(open(mola_audit))}
for a, src, b in (('mola_mae', pau, 'mae_dist'), ('relief_m', pau, 'relief_m'), ('fine_r', p1k, 'hp1k_corr_dist'),
                  ('fine_amp', p1k, 'hp1k_amp_dist')):
    d = max(abs(float(mm[t][a]) - float(src[t][b])) for t in mm)
    print(f'  MOLA {a:9s} vs {b:15s} max |diff| {d:.2e} over {len(mm)} tiles')
    ok &= d < 1e-6
d = max(abs(float(mine[k]['relief_m']) - float(pau[k[0]]['relief_m'])) for k in mine)
print(f'  sample relief vs audit relief max |diff| {d:.2e}')

# ---- Table 2: per-sample sums of every bin ---------------------------------------------------------
pairs_of = {'disagreement': 'disagree', 'shot_distance': 'dist_to_shot_m', 'slope': 'slope_deg',
            'aspect': 'aspect_rel_sun_deg', 'local_incidence': 'local_incidence_deg', 'shadowed': 'shadowed',
            'all': 'all_full'}
bin_files = []
for band, man in zip(('eq', 'mid', 'polar'), manifests):
    f = out / f'bins_{band}.npz'
    if not f.exists():
        ds = TileDataset(root, man, split=None, incidence_json=str(Path(root) / 'ctx_overlapping_95pct_polar.json'))
        np.savez_compressed(f, **ev.bin_errors(model, ds, pairs, mola_tif, device))
    bin_files.append(f)
    a, b = np.load(f), np.load(Path(ref) / 'bins' / f'e088_359_{band}_med.npz')
    c = np.load(Path(mola_bins) / f'mola_{band}.npz')
    same_keys = list(a['keys']) == list(b['keys']) == list(c['keys'])
    worst_n = worst_s = 0.0
    compared = [(a[k], b[f'{pk}__ps']) for k, pk in pairs_of.items()] + [(a['mola_shot_distance'], c['ps'])]
    for x, y in compared:
        worst_n = max(worst_n, float(np.abs(x[..., -1] - y[..., -1]).max()))
        worst_s = max(worst_s, float((np.abs(x[..., :-1] - y[..., :-1]) / np.maximum(1.0, np.abs(y[..., :-1]))).max()))
    print(f'bins {band}: {len(a["keys"])} samples, same samples in the same order {same_keys}; largest difference in '
          f'pixel count {worst_n:.0f}, relative difference in error sums {worst_s:.1e}')
    ok &= same_keys and worst_n == 0 and worst_s < 1e-9

# ---- the tables -------------------------------------------------------------------------------------
t1 = ev.table1_tiles(list(mine.values()), list(mm.values()), pairs)
t2 = ev.table2_bins(bin_files)
json.dump(dict(table1_tiles=t1, table2=t2), open(out / 'tables.json', 'w'), indent=1)
print('Table 1, tiles:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t1.items()
                          if not isinstance(v, dict)})
print('  by solar incidence:', {k: [round(v['label_mae'], 2), round(v['aligned_mae'], 2), round(v['share_pct'], 1)]
                                for k, v in t1['by_incidence'].items()})
print('  views:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t1['views'].items()})
pj = json.load(open(Path(ref) / 'bins' / 'table2_e088.json'))
names = {'disagreement': 'disagree', 'shot_distance': 'dist', 'slope': 'slope', 'aspect': 'aspect',
         'local_incidence': 'local_inc', 'shadowed': 'shadowed', 'all': 'all_full'}
worst = 0.0
for k, pk in names.items():
    cells = [c for c in t2[k].values() if c is not None]
    for c, p in zip(cells, pj[pk]):
        worst = max(worst, abs(c['label_mae'] - p['label']), abs(c['aligned_mae'] - p['aligned']),
                    abs(c['share_pct'] - p['share']))
    print(f'Table 2 {k:15s} label ' + ' '.join(f'{c["label_mae"]:.1f}' for c in cells) + ' | aligned '
          + ' '.join(f'{c["aligned_mae"]:.1f}' for c in cells) + ' | share '
          + ' '.join(f'{c["share_pct"]:.2f}' for c in cells))
print('Table 2 MOLA by shot distance', ' '.join(f'{c["mola_mae"]:.3f}' for c in t2['shot_distance'].values()),
      '(mola_dist_bins_359: 12.892 15.037 18.251 21.940 30.643 65.462)')
print(f'Table 2 cells, largest difference from the manuscript\'s full-precision cells: {worst:.2e}')
ok &= worst < 1e-6
print(f'EVALUATE CHECK {"PASS" if ok else "DIFF"}')

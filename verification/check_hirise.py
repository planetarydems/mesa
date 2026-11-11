"""Check the released HiRISE evaluation (Tables 3 and 4, Section 3.2) against
the manuscript's scoring of epoch 88.

    python verification/check_hirise.py --ref REF_DIR --sample SAMPLE_DIR --placement PLACEMENT_DIR \
        --group GROUP_DIR --preds PRED_ROOT --out OUT [--jobs N]
    python verification/check_hirise.py --ref REF_DIR --inputs INPUT_DIR --sample SAMPLE_DIR --group GROUP_DIR \
        --out OUT --ref-prefix eval [--jobs N]          (the paper, v3)

REF_DIR        the manuscript's epoch-88 HiRISE workspace: imgreg/<DTM>.json and
               moladist/<DTM>.json (one per HiRISE DEM: every window, its image
               co-registration and its measures), e088_359_dtm.csv (per-DEM
               values) and e088_359.txt (the aggregate behind the tables)
SAMPLE_DIR     the HiRISE sample: hirise/sites_rb.csv (DEMs, their CTX images and
               test pairs), all/frames_all.csv (product tile of each image),
               sel_r1r2.csv (latitude, distance from training ground), labels/
               (the test stereo DEMs, mosaicked from their tiles), scores_f/
               (window geometry of the earlier scoring, for the ground shift)
PLACEMENT_DIR  ori_pick.csv (orthoimage of each DEM), modelfree_exclude.txt (the
               33 DEMs excluded by rule R4)
GROUP_DIR      hirise_dtm/, hirise_ori/, stereo_dems/, global_run/boxes/ and the
               MOLA mosaic
PRED_ROOT      the epoch-88 predicted DEMs, <tile>/<image>/

The input tables of mesa.evaluate_hirise are built from these files, every
HiRISE DEM of the sample is scored by the released code (records cached under
OUT/records), and each window, each DEM and each table cell is compared with
the manuscript's values. The per-DEM CSV rounds to 4 decimals.

For the paper (v3), --inputs gives the three input tables with the cluster's
paths (verification/make_hirise_inputs_v3.py), REF_DIR is the manuscript's v3
scoring with the windows of release 1 (imgreg/, moladist/, eval_dtm.csv,
eval.txt, text_numbers_eval.txt; --ref-prefix eval), and SAMPLE_DIR is only
read for scores_f/ (the window geometry behind the ground shift).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesa import evaluate_hirise as eh  # noqa: E402

ap = argparse.ArgumentParser()
for k in ('ref', 'sample', 'group', 'out'):
    ap.add_argument(f'--{k}', required=True)
ap.add_argument('--placement', default=None)
ap.add_argument('--preds', default=None)
ap.add_argument('--inputs', default=None, help='directory with prebuilt hirise_dems.csv, hirise_windows.csv, '
                                                'hirise_stereo_pairs.csv (absolute paths)')
ap.add_argument('--ref-prefix', default='e088_359', help='REF_DIR/<prefix>_dtm.csv and <prefix>.txt')
ap.add_argument('--jobs', type=int, default=24)
ap.add_argument('--only', default=None, help='comma-separated DEMs: score and compare these windows only')
args = ap.parse_args()
REF, SMP, G = Path(args.ref), Path(args.sample), Path(args.group)
PLC = Path(args.placement) if args.placement else None
OUT = Path(args.out)
(OUT / 'records').mkdir(parents=True, exist_ok=True)
MOLA = str(G / 'data' / 'Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff')
TEST_PAIRS = str(Path(__file__).resolve().parents[1] / 'data' / 'test_pairs.txt')
ok = True


def read_csv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---- input tables ----------------------------------------------------------------------------------------------------
only = set(args.only.split(',')) if args.only else None
if args.inputs:
    INP0 = Path(args.inputs)
    dems = [d for d in read_csv(INP0 / 'hirise_dems.csv') if not only or d['dem'] in only]
    keep = {d['dem'] for d in dems}
    windows = [w for w in read_csv(INP0 / 'hirise_windows.csv') if w['dem'] in keep]
    pairs = {r['pair']: r for r in read_csv(INP0 / 'hirise_stereo_pairs.csv')}
    excluded = set()
else:
    sites = read_csv(SMP / 'hirise' / 'sites_rb.csv')
    box = {(r['dtm'], r['product']): r['box'] for r in read_csv(SMP / 'all' / 'frames_all.csv') if r.get('box')}
    ortho = {r['dtm']: os.path.basename(r['url']) for r in read_csv(PLC / 'ori_pick.csv')}
    excluded = {l.split()[0] for l in open(PLC / 'modelfree_exclude.txt') if l.strip() and not l.startswith('#')}
    sel = {r['dtm']: r for r in read_csv(SMP / 'sel_r1r2.csv')}
    CTX_ID = re.compile(r'[A-Z][0-9A-Z][0-9]_[0-9]{6}_[0-9]{4}')
    stereo_dirs = {}
    for d in sorted(os.listdir(G / 'stereo_dems')):
        k = '+'.join(sorted(set(CTX_ID.findall(d))))
        if k.count('+') == 1:
            stereo_dirs.setdefault(k, d)
    dems, windows, pairs = [], [], {}
    for s in sites:
        if s['dtm'] in excluded or (only and s['dtm'] not in only):
            continue
        dems.append(dict(dem=s['dtm'], dem_path=s['dtm_path'], ortho_path=str(G / 'hirise_ori' / ortho[s['dtm']]),
                         lat=sel[s['dtm']]['lat'], dist_km=sel[s['dtm']]['dist_trainval_km'], pairs=s['label_pairs']))
        for stem in [x for x in s['stems'].split(';') if x]:
            b = box[(s['dtm'], stem)]
            windows.append(dict(dem=s['dtm'], image=stem, prediction=f'{args.preds}/{b}/{stem}',
                                registered_image=str(G / 'global_run' / 'boxes' / b / 'box_run' / 'aligned_pedr' /
                                                     f'{stem}_aligned.vrt')))
        for p in [x for x in s['label_pairs'].split(';') if x]:
            sd = stereo_dirs.get('+'.join(sorted(set(CTX_ID.findall(p)))))
            orthos = sorted(glob.glob(str(G / 'stereo_dems' / sd / 'mapprojected_images' / '*_map_tied.tif'))) if sd else []
            orthos = [o for o in orthos if os.path.basename(o)[:15] in p] or orthos
            pairs[p] = dict(pair=p, dem_path=str(SMP / 'labels' / f'{p}.tif'), orthoimages=';'.join(orthos[:2]))
INP = OUT / ('inputs_only' if only else 'inputs')
INP.mkdir(exist_ok=True)
write_csv(INP / 'hirise_dems.csv', dems)
write_csv(INP / 'hirise_windows.csv', windows)
write_csv(INP / 'stereo_pairs.csv', list(pairs.values()))
print(f'inputs: {len(dems)} HiRISE DEMs ({len(excluded)} excluded), {len(windows)} windows, {len(pairs)} stereo pairs',
      flush=True)

# ---- scoring (cached per DEM) ------------------------------------------------------------------------------------------
jobs = [j for j in eh.load_inputs(INP / 'hirise_dems.csv', INP / 'hirise_windows.csv', INP / 'stereo_pairs.csv', TEST_PAIRS)
        if not (OUT / 'records' / f'{j[0]["dem"]}.json').exists()]
jobs.sort(key=lambda j: -len(j[1]) * (1 + len(j[2])) * os.path.getsize(j[0]['dem_path']))     # longest first


def score(job):
    d, w, st = job
    rec = eh.score_dem(d, w, st, MOLA)
    tmp = OUT / 'records' / f'{d["dem"]}.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f, default=float)
    os.replace(tmp, OUT / 'records' / f'{d["dem"]}.json')
    return d['dem']


if jobs:
    with Pool(args.jobs) as pool:
        for k, dem in enumerate(pool.imap_unordered(score, jobs, chunksize=1)):
            if (k + 1) % 50 == 0:
                print(f'scored {k + 1}/{len(jobs)}', flush=True)
records = {d['dem']: json.load(open(OUT / 'records' / f'{d["dem"]}.json')) for d in dems}
test_pairs = set(open(TEST_PAIRS).read().split())

# ---- every window against the manuscript's records -----------------------------------------------------------------
worst = {}


def diff(name, a, b, rel=False):
    fa, fb = a is not None and math.isfinite(a), b is not None and math.isfinite(b)
    if not (fa and fb):
        d = 0.0 if fa == fb else float('inf')
    else:
        d = abs(a - b) / max(abs(b), 1e-12) if rel else abs(a - b)
    worst[name] = max(worst.get(name, 0.0), d)


def compare_measures(tag, mine, prod, full=True):
    p = prod['paper']
    diff(f'{tag} pixels', mine['px'], p.get('n', prod.get('n_px')))
    diff(f'{tag} deployMAE', mine['deploy_mae'], p['mae_unreg'])
    diff(f'{tag} offset', mine['offset'], p['offset_levelkept_m'])
    diff(f'{tag} relief', mine['relief'], p['ref_relief_p2p98_m'])
    if not full:
        return
    diff(f'{tag} alignedMAE', mine['aligned_mae'], p['mae_levelkept'])
    diff(f'{tag} labelMAE', mine['label_mae'], p['mae_label'])
    diff(f'{tag} slope error', mine['slope_rmse'], p['slope_rmse_deg'])
    if 'a_opt' in prod:
        diff(f'{tag} scale', mine['scale'], prod['a_opt'])
    for i, (bm, bp) in enumerate(zip(mine['bands'], p['bands'])):
        diff('bands pixels', bm['px'], bp['n'])
        for a, b in (('cov', 'cov'), ('var_z', 'vp'), ('var_h', 'vt'), ('sq_err', 'verr')):
            diff(f'bands {a} (relative)', bm[a], bp[b], rel=True)


counts = dict(windows=0, status_differs=0, gate_differs=0, method_differs=0, stereo_differs=0, surfaces_differ=0)
jac_missing = 0
for d in dems:
    dem = d['dem']
    J = json.load(open(REF / 'imgreg' / f'{dem}.json'))
    M = json.load(open(REF / 'moladist' / f'{dem}.json'))
    SF = {}
    sf = SMP / 'scores_f' / f'{dem}.json'
    if sf.exists():
        SF = json.load(open(sf)).get('frames', {})
    for rec in records[dem]:
        counts['windows'] += 1
        R, MR = J['frames'][rec['image']], M['frames'][rec['image']]
        if rec['status'] != R['status'] or rec['status'] != MR['status']:
            counts['status_differs'] += 1
            print(f'{dem} {rec["image"]}: status {rec["status"]} vs {R["status"]} / {MR["status"]}')
            continue
        if rec['status'] != 'ok':
            continue
        compare_measures('in place, model', rec['in_place']['model'], MR['inplace']['e098'], False)
        compare_measures('in place, MOLA', rec['in_place']['mola'], MR['inplace']['mola_dist'], False)
        compare_measures('in place, model (scorer)', rec['in_place']['model'], R['inplace']['e098'], False)
        ir, reg = R.get('image_reg') or {}, rec.get('registration') or {}
        prod_gate = ir.get('status') == 'ok' and bool(ir.get('img_gate'))
        if prod_gate != bool(reg.get('accepted')):
            counts['gate_differs'] += 1
            print(f'{dem} {rec["image"]}: co-registration {reg} vs {ir.get("status")} {ir.get("img_gate")}')
            continue
        if not prod_gate:
            continue
        counts['method_differs'] += reg['method'] != ir['method']
        diff('image shift x (m)', reg['dx'], ir['dx_m'])
        diff('image shift y (m)', reg['dy'], ir['dy_m'])
        diff('fine peak', reg['peak'], ir['peak'])
        jac = SF.get(rec['image']) or {}
        if jac.get('jac_col'):
            pc, pr = ir['dx_m'] / 18.0, -ir['dy_m'] / 18.0
            e = pc * jac['jac_col'][0] + pr * jac['jac_row'][0]
            n = pc * jac['jac_col'][1] + pr * jac['jac_row'][1]
            diff('ground shift east (m)', rec['shift_east_m'], e)
            diff('ground shift north (m)', rec['shift_north_m'], n)
        else:
            jac_missing += 1
        S = R.get('surfaces') or {}
        if ('e098' in S) != ('registered' in rec):
            counts['surfaces_differ'] += 1
            continue
        if 'registered' in rec:
            for mine, prod in (('model', 'e098'), ('model_no_shots', 'e098_noshot'), ('mola', 'mola_dist')):
                compare_measures(f'co-registered, {mine}', rec['registered'][mine], S[prod])
        LR = R.get('label_reg') or {}
        for p in [x for x in LR if x in test_pairs]:
            lr, q = LR[p], rec['stereo'].get(p)
            prod_ok = lr.get('status') == 'ok' and bool(lr.get('img_gate')) and f'label@{p}' in S
            if prod_ok != (q is not None and 'model' in q):
                counts['stereo_differs'] += 1
                print(f'{dem} {rec["image"]} {p}: stereo {q and q.get("registration")} vs {lr.get("status")} '
                      f'{lr.get("img_gate")} {f"label@{p}" in S}')
                continue
            if lr.get('status') == 'ok' and q is not None and q.get('registration'):
                diff('stereo image shift x (m)', q['registration']['dx'], lr['dx_m'])
                diff('stereo image shift y (m)', q['registration']['dy'], lr['dy_m'])
            if prod_ok:
                compare_measures('stereo, model', q['model'], S[f'e098@{p}'])
                compare_measures('stereo, stereo DEM', q['stereo'], S[f'label@{p}'])
                compare_measures('stereo, MOLA', q['mola'], S[f'mola_dist@{p}'])
print(f'\nwindows: {counts}; ground shift without the earlier window geometry: {jac_missing}')
ok &= all(v == 0 for k, v in counts.items() if k != 'windows')
print('largest |difference| per window measure (relative where marked):')
for k, v in worst.items():
    tol = 1e-9 if 'relative' in k else 1e-6
    flag = '' if v <= tol else '   <-- above 1e-9 relative / 1e-6 absolute'
    print(f'  {k:40s} {v:.3e}{flag}')
    ok &= v <= tol

if only:
    print(f'HIRISE CHECK (windows of {len(dems)} DEMs) {"PASS" if ok else "DIFF"}')
    sys.exit(0)

# ---- per-DEM values against the manuscript's per-DEM CSV ----------------------------------------------------------
rows = [r for d in dems for r in eh.dem_rows(d['dem'], records[d['dem']])]
write_csv(OUT / 'hirise_scores.csv', rows)
mine = {(r['set'], r['surface'], r['dem']): r for r in rows}
NAMES = {('inplace_all', 'e098'): ('in_place', 'model'), ('inplace_all', 'mola_dist'): ('in_place', 'mola'),
         ('imggated', 'e098'): ('registered', 'model'), ('imggated', 'e098_noshot'): ('registered', 'model_no_shots'),
         ('imggated', 'mola_dist'): ('registered', 'mola'), ('imggated', 'e098@'): ('stereo', 'model'),
         ('imggated', 'label@'): ('stereo', 'stereo'), ('imggated', 'mola_dist@'): ('stereo', 'mola')}
COLS = {'n': 'windows', 'mae_unreg': 'deploy_mae', 'mae_label': 'label_mae', 'mae_lk': 'aligned_mae', 'abs_off': 'abs_offset',
        'off': 'offset', 'slope_rmse_deg': 'slope_rmse', 'relief': 'relief', 'pct_unreg': 'deploy_pct'}
dtm_csv = REF / f'{args.ref_prefix}_dtm.csv'
if dtm_csv.exists():
    prod = [r for r in read_csv(dtm_csv) if (r['set'], r['cand']) in NAMES]
    keys_p = {(*NAMES[(r['set'], r['cand'])], r['dtm']) for r in prod}
    keys_m = {k for k in mine if k[:2] in NAMES.values()}
    print(f'\nper-DEM rows: released {len(keys_m)}, manuscript {len(keys_p)}; same DEMs and surfaces {keys_m == keys_p}')
    ok &= keys_m == keys_p
    for (s, k) in NAMES.values():
        sub = [r for r in prod if NAMES[(r['set'], r['cand'])] == (s, k)]
        line = []
        for pc, mc in COLS.items():
            ds = [abs(float(r[pc]) - mine[(s, k, r['dtm'])][mc]) for r in sub
                  if (s, k, r['dtm']) in mine and r[pc] not in ('', 'nan') and math.isfinite(mine[(s, k, r['dtm'])][mc])]
            if ds:
                line.append(f'{mc} {max(ds):.1e}')
                ok &= max(ds) <= 5.001e-5
        if s == 'registered' and k == 'model':
            ds = [abs(float(r['mean_abs_d_m']) - mine[(s, k, r['dtm'])]['shift_m']) for r in sub if r['mean_abs_d_m']]
            line.append(f'shift_m {max(ds):.1e} (CSV rounds to 0.1 m)')
            ok &= max(ds) <= 0.05001
        print(f'  {s:10s} {k:15s} {len(sub):3d} DEMs, max |diff|: ' + ', '.join(line))
else:
    print(f'\n{dtm_csv} not found: per-DEM comparison skipped')
    ok = False

# ---- tables and the numbers of Section 3.2 ----------------------------------------------------------------------------
t = eh.tables(dems, rows)
json.dump(t, open(OUT / 'hirise_tables.json', 'w'), indent=1)
print()
eh.print_tables(t)
T3, T4, X = t['table3'], t['table4'], t['text']


def close(value, printed):
    """value agrees with a number printed to its decimals (the per-DEM CSV behind
    some of them rounds to 4 decimals)."""
    dec = len(printed.split('.')[1]) if '.' in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10 ** -dec + 6e-5


agg = REF / f'{args.ref_prefix}.txt'
if agg.exists():
    txt = open(agg).read()

    def section(title):
        m = re.search(title, txt)
        out = {}
        for line in txt[m.end():].splitlines()[1:]:
            if line.startswith('#'):
                break
            if line.startswith('| ') and not line.startswith(('| candidate', '| wavelength', '|---')):
                c = [x.strip() for x in line.strip().strip('|').split('|')]
                out[c[0]] = c
        return out

    bad = []
    checks = [(section(r'## Gate-free, in place.*?\n\n### all: \d+ DTMs'), 'a',
               {'MADER': 'model', 'MOLA gridded DEM (distributed)': 'mola'}),
              (section(r'## Table 5\(a\).*?\n\n### all: \d+ DTMs'), 'b',
               {'MADER': 'model', 'MOLA gridded DEM (distributed, in place)': 'mola'}),
              (section(r'## Table 5\(b\).*?\n\n### all with a scored label: \d+ DTMs'), 'c',
               {'MADER': 'model', 'CTX stereo DEM (training label)': 'stereo',
                'MOLA gridded DEM (distributed, in place)': 'mola'})]
    n3 = 0
    for rowsx, part, names in checks:
        for nm, k in names.items():
            c = rowsx[nm]
            if int(c[1]) != T3[part]['dems']:
                bad.append(f'Table 3({part}) DEMs {T3[part]["dems"]} vs {c[1]}')
            for i, q in ((2, 'deploy_mae'), (3, 'label_mae'), (4, 'aligned_mae'), (6, 'abs_offset'), (13, 'deploy_pct')):
                if c[i] != '---' and q in T3[part][k]:
                    n3 += 1
                    if not close(T3[part][k][q], c[i].split()[0]):
                        bad.append(f'Table 3({part}) {k} {q} {T3[part][k][q]:.4f} vs {c[i]}')
    n4 = 0
    for title, part, ks, cols in ((r'### \(a\) image-registered HiRISE DTMs \(\d+ DTMs\)', 'a', ('model', 'mola'), (2, 5)),
                                  (r'### \(b\) identical pixels \(\d+ DTMs\)', 'b', ('model', 'stereo', 'mola'), (2, 5, 8))):
        m = re.search(title, txt)
        lines = [l for l in txt[m.end():].splitlines()[1:] if l.startswith('| ') and not l.startswith('| wavelength')]
        for i, line in enumerate(lines[:7]):
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            cells = [(T4[part]['model'][i]['hirise_rms'], c[1], 'HiRISE RMS')]
            cells += [(T4[part][k][i][q], c[j + o], f'{k} {q}') for k, j in zip(ks, cols)
                      for o, q in enumerate(('amp', 'r', 'error'))]
            for v, printed, what in cells:
                n4 += 1
                if not close(v, printed):
                    bad.append(f'Table 4({part}) {c[0]} {what} {v:.4f} vs {printed}')
    m = re.search(r'per-DTM mean \|d\| over accepted windows: median (\d+) m \[(\d+), (\d+)\], p90 (\d+) m, n = (\d+)', txt)
    sh = X['image_shift']
    mine_shift = (sh['median_m'], *sh['ci_m'], sh['p90_m'], T3['b']['dems'])
    if not all(close(a, b) for a, b in zip(mine_shift, m.groups())):
        bad.append(f'image shift {mine_shift} vs {m.groups()}')
    print(f'\nTable 3: {n3} cells, Table 4: {n4} cells and the image shift against the manuscript\'s aggregate: '
          f'{len(bad)} differ' + ''.join(f'\n  {b}' for b in bad))
    ok &= not bad
else:
    print(f'{agg} not found: table comparison skipped')
    ok = False

numbers = REF / ('text_numbers_e088.txt' if args.ref_prefix == 'e088_359' else f'text_numbers_{args.ref_prefix}.txt')
if numbers.exists():
    lines = open(numbers).read().splitlines()
    ip, rg, st = X['in_place'], X['registered'], X['stereo']
    expect = {
        'in place: n': [ip['dems']],
        'in place: deployMAE model vs MOLA': [ip['model'], ip['mola']],
        'in place: median relief': [ip['relief']],
        'in place: error / relief model vs MOLA (%)': [ip['model_pct'], ip['mola_pct']],
        'in place: distance 9-25 km (n, model vs MOLA)': list(X['distance_9_25_km'].values()),
        'in place: distance 25-inf km (n, model vs MOLA)': list(X['distance_over_25_km'].values()),
        'in place: eq (n, model vs MOLA)': list(X['equatorial'].values()),
        'in place: mid (n, model vs MOLA)': list(X['mid_latitude'].values()),
        'registered: n': [rg['dems']],
        'registered: deployMAE model vs MOLA': [rg['model'], rg['mola']],
        'registered: alignedMAE / labelMAE': [rg['aligned'], rg['label']],
        'registered: slope error (deg) model vs MOLA': [rg['model_slope'], rg['mola_slope']],
        'registered: |vertical offset| model vs MOLA': [rg['model_offset'], rg['mola_offset']],
        'not registered: n, model vs MOLA, polar': list(X['not_registered'].values()),
        '76: labelMAE model vs stereo': [st['model_label'], st['stereo_label']],
        '76: alignedMAE model vs stereo': [st['model_aligned'], st['stereo_aligned']],
        'three sources: geodetic (deploy - aligned) median': [X['change_with_alignment']['median']],
        'three sources: magnitude (aligned - label) median': [X['change_with_rescaling']],
        'three sources: magnitude without the shot fit': [X['change_with_rescaling_no_shots'], rg['dems']],
        'scale with the shots [CI]; MOLA statistics alone': [X['scale']['median'], *X['scale']['ci'], X['scale']['no_shots']],
        'image shift median [CI], p90, n (model-independent)': [sh['median_m'], *sh['ci_m'], sh['p90_m'], rg['dems']],
        'image offset per DTM: > 300 m count; p90 (mean_abs_d_m)': [sh['over_300_m'], rg['dems'], sh['p90_m']],
        'poleward of 60: n (>=80), model vs MOLA; co-registered': [X['poleward_60'][k] for k in
                                                                   ('dems', 'over_80', 'model', 'mola', 'registered')],
        'poleward of 80: model vs MOLA; co-registered': [X['poleward_80'][k] for k in ('model', 'mola', 'registered')],
        'Spearman(deployMAE, relief) in place': [X['spearman_error_relief']]}
    bad = []
    for key, mine_v in expect.items():
        line = next((l for l in lines if l.startswith(key)), key + ' (missing)')
        printed = re.findall(r'(?<![A-Za-z\d.])-?\d+(?:\.\d+)?', line[len(key):])     # not the 90 of 'p90'
        if len(printed) != len(mine_v) or not all(close(a, b) for a, b in zip(mine_v, printed)):
            bad.append(f'{key}: released {[round(v, 4) for v in mine_v]}, manuscript {line[len(key):].strip()}')
    print(f'Section 3.2 numbers: {len(expect)} statements against {numbers.name}: {len(bad)} differ'
          + ''.join(f'\n  {b}' for b in bad))
    ok &= not bad
else:
    print(f'{numbers} not found: Section 3.2 comparison skipped')

changes = REF.parent / 'changes' / 'metric_changes.json'
if changes.exists():
    S9 = json.load(open(changes))['table_s9']
    a, b = S9['e098']['change with alignment'], S9['e098']['change with rescaling']
    c = S9['e098_noshot']['change with rescaling']
    ca = X['change_with_alignment']
    tol = 1.01e-4      # a difference of two values rounded to 4 decimals
    same = (abs(ca['median'] - a['median']) < tol and all(abs(x - y) < tol for x, y in zip(ca['ci'], a['ci']))
            and abs(X['change_with_rescaling'] - b['median']) < tol and abs(X['change_with_rescaling_no_shots'] - c['median']) < tol)
    print(f'changes with alignment and rescaling (Table S7): released {ca["median"]:.4f} {[round(x, 4) for x in ca["ci"]]}, '
          f'{X["change_with_rescaling"]:.4f}, without the shots {X["change_with_rescaling_no_shots"]:.4f}; manuscript '
          f'{a["median"]:.4f} {[round(x, 4) for x in a["ci"]]}, {b["median"]:.4f}, {c["median"]:.4f} (per-DEM CSV, 4 decimals)')
    ok &= same

# the 76 DEMs with a stereo DEM: image shift of the stereo DEMs and of the predictions. The released code reports
# ground metres; the earlier computation took the shift in projected metres, over every co-registered stereo pair
counted = REF / 'numbers_e088_359.txt'
if counted.exists():
    m = re.search(r'image offsets to HiRISE at these DTMs: stereo DEM (\d+) m, frames (\d+) m \(n (\d+)\)', open(counted).read())
    proj = {}
    for d in dems:
        acc = {'stereo': [], 'model': []}
        for r in records[d['dem']]:
            for q in (r.get('stereo') or {}).values():
                if 'model' in q:
                    acc['stereo'].append(math.hypot(q['registration']['dx'], q['registration']['dy']))
                    acc['model'].append(math.hypot(r['registration']['dx'], r['registration']['dy']))
        if acc['model']:
            proj[d['dem']] = {k: float(np.mean(v)) for k, v in acc.items()}
    ps = [proj[x]['stereo'] for x in proj]
    pm = [proj[x]['model'] for x in proj]
    print(f'76 DEMs, image shift of the stereo DEMs and of the predictions: released (ground) '
          f'{X["stereo"]["stereo_shift_m"]:.1f} and {X["stereo"]["model_shift_m"]:.1f} m; in projected metres '
          f'{np.median(ps):.1f} and {np.median(pm):.1f} m over {len(proj)} DEMs; manuscript {m.group(1)} and {m.group(2)} m '
          f'(n {m.group(3)})')
print(f'HIRISE CHECK {"PASS" if ok else "DIFF"}')

"""Write the HiRISE input tables of the paper (v3): with paths relative to a
working directory as data/ ships them, and with the cluster's paths for the
check.

    python verification/make_hirise_inputs_v3.py FIXED_DIR SITES FRAMES R2_FRAMES ORI_PICK SEL EXCLUDE GROUP \
        LABELS STEREO_OVERRIDE TAG DATA_OUT WORK

FIXED_DIR        the manuscript's v3 HiRISE scoring with the windows of release 1
                 (imgreg/<DTM>.json; the co-registration outcome of release 1 applied)
SITES            sites_ff_r2.csv (HiRISE DEM path, its stereo pairs over the full stereo DEMs)
FRAMES           frames_r2.csv (product tile of each window)
R2_FRAMES        r2_frames.tsv (the registered image of the frames that release 2 re-placed)
ORI_PICK         ori_pick.csv (HiRISE orthoimage of each DEM)
SEL              sel_r1r2.csv (latitude, distance from the training and validation tiles)
EXCLUDE          modelfree_exclude.txt (the 33 DEMs excluded by rule R4)
GROUP            the group directory (hirise_dtm/, hirise_ori/, stereo_dems/, global_run/boxes/)
LABELS           the full stereo DEMs, <pair>.tif
STEREO_OVERRIDE  JSON {pair key: stereo directory} for the two pairs rebuilt outside stereo_dems/
TAG              prediction tag of the frames (global_run/boxes/<tile>/box_run/preds_<TAG>/)
DATA_OUT         directory for hirise_dems.csv, hirise_windows.csv, hirise_stereo_pairs.csv
                 (relative paths: hirise_dtm/, hirise_ortho/, preds/<tile>/<image>,
                 aligned_pedr/<tile>/, stereo_dems_full/, orthoimages/<stereo dir>/)
WORK             directory for the same three tables with the cluster's paths

The windows and their release-1 co-registration outcome (registered_r1) are
read from FIXED_DIR; the stereo DEM directory of a pair is found as the
manuscript's scorer finds it (first directory of stereo_dems/ in listing order
whose two CTX ids are those of the pair, unless overridden); its tied
orthoimages are those of the two images of the pair.
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

(fixed, sites_csv, frames_csv, r2_tsv, ori_csv, sel_csv, exclude, group, labels, override, tag, data_out,
 work) = sys.argv[1:14]
G, WORK, DATA = Path(group), Path(work), Path(data_out)
DATA.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
ID = re.compile(r'[A-Z][0-9A-Z][0-9]_[0-9]{6}_[0-9]{4}')


def key_of(s):
    return '+'.join(sorted(set(ID.findall(s))))


sites = {r['dtm']: r for r in csv.DictReader(open(sites_csv))}
box = {(r['dtm'], r['product']): r['box'] for r in csv.DictReader(open(frames_csv)) if r.get('box')}
r2ctx = {(r['box'], r['stem']): r['ctx'] for r in csv.DictReader(open(r2_tsv), delimiter='\t')
         if r['cls'] in ('fixed', 'new', 'rescued')}
ori = {r['dtm']: os.path.basename(r['url']) for r in csv.DictReader(open(ori_csv))}
sel = {r['dtm']: r for r in csv.DictReader(open(sel_csv))}
excluded = {l.split()[0] for l in open(exclude) if l.strip() and not l.startswith('#')}
ov = json.loads(override)
sdir = {}
for d in os.listdir(G / 'stereo_dems'):           # listing order, as the scorer
    k = key_of(d)
    if k.count('+') == 1:
        sdir.setdefault(k, d)

dems, windows, pairs = [], [], {}
adems, awindows, apairs = [], [], {}
for f in sorted(glob.glob(f'{fixed}/imgreg/*.json')):
    J = json.load(open(f))
    dem = J['dtm']
    if dem in excluded or dem not in sites:
        continue
    s = sites[dem]
    dems.append(dict(dem=dem, dem_path=f'hirise_dtm/{os.path.basename(s["dtm_path"])}', ortho_path=f'hirise_ortho/{ori[dem]}',
                     lat=sel[dem]['lat'], dist_km=sel[dem]['dist_trainval_km'], pairs=s['label_pairs']))
    adems.append(dict(dems[-1], dem_path=s['dtm_path'], ortho_path=str(G / 'hirise_ori' / ori[dem])))
    for stem, R in J['frames'].items():
        b = box[(dem, stem)]
        assert R.get('box') in (None, b), (dem, stem, R.get('box'), b)
        if (b, stem) in r2ctx:          # release 2 re-placed the frame: the image it was predicted from
            areg = r2ctx[(b, stem)]
            reg = f'aligned_pedr/{b}/{os.path.basename(areg)}'
        else:
            vr = sorted(glob.glob(str(G / 'global_run' / 'boxes' / b / 'box_run' / 'aligned_pedr' / f'{stem}*_aligned.vrt')))
            areg = vr[0] if vr else ''
            reg = f'aligned_pedr/{b}/{os.path.basename(areg)}' if vr else ''
        ir = R.get('image_reg') or {}
        windows.append(dict(dem=dem, image=stem, tile=b, prediction=f'preds/{b}/{stem}', registered_image=reg,
                            registered_r1=int(ir.get('status') == 'ok' and bool(ir.get('img_gate')))))
        awindows.append(dict(windows[-1], prediction=str(G / 'global_run' / 'boxes' / b / 'box_run' / f'preds_{tag}' / stem),
                             registered_image=areg))
    for p in [x for x in s['label_pairs'].split(';') if x]:
        if p in pairs:
            continue
        d = ov.get(key_of(p)) or sdir.get(key_of(p))
        orthos = sorted(glob.glob(str(G / 'stereo_dems' / d / 'mapprojected_images' / '*_map_tied.tif'))) if d else []
        orthos = [o for o in orthos if os.path.basename(o)[:15] in p] or orthos
        names = [f'orthoimages/{os.path.basename(os.path.dirname(os.path.dirname(o)))}/{os.path.basename(o)}'
                 for o in orthos[:2]]
        pairs[p] = dict(pair=p, dem_path=f'stereo_dems_full/{p}.tif', orthoimages=';'.join(names))
        apairs[p] = dict(pair=p, dem_path=str(Path(labels) / f'{p}.tif'), orthoimages=';'.join(orthos[:2]))


def write(name, rows, where=DATA):
    with open(where / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


write('hirise_dems.csv', dems)
write('hirise_windows.csv', windows)
write('hirise_stereo_pairs.csv', sorted(pairs.values(), key=lambda r: r['pair']))
write('hirise_dems.csv', adems, WORK)
write('hirise_windows.csv', awindows, WORK)
write('hirise_stereo_pairs.csv', sorted(apairs.values(), key=lambda r: r['pair']), WORK)
print(f'{len(dems)} HiRISE DEMs ({len(excluded)} excluded), {len(windows)} windows '
      f'({sum(w["registered_r1"] for w in windows)} co-registered in release 1, '
      f'{sum(1 for w in windows if (w["tile"], w["image"]) in r2ctx)} on a release-2 placement), '
      f'{len(pairs)} stereo pairs ({sum(1 for p in pairs.values() if not p["orthoimages"])} without orthoimage)')

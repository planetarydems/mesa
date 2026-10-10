"""Check the released registration against the production registration code.

    python verification/check_register.py BOX KIND BOXES_ROOT OUT_DIR N ORIG_DIR [QUADS_DIR | CORRECTIONS_CSV]

KIND robbins  equatorial tile: the 24 m crop of the controlled mosaic is rebuilt
              from the quadrangles with the production gdalwarp command
     murray   polar tile: the production corrected Global CTX Mosaic reference
              (murray_ref_24m.corrected.vrt) is used, and its correction is
              compared with the correction table

For up to N images of the tile, the original align_rasters.py (run as a
script, writing a VRT) and align_refine.refine (imported from ORIG_DIR) are run
on the same inputs and settings as the released register.coarse_fit and
register.refine, and the transforms are compared. The difference from the
archived production transform is printed too; it reflects the history of the
alignment campaigns (several references and settings were used over time), not
the code.
"""
import csv
import glob
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import rasterio
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mader.register import coarse_fit, refine  # noqa: E402

box, kind, boxes, out, n, orig_dir = sys.argv[1:7]
n = int(n)
run = Path(boxes) / box / 'box_run'
out = Path(out) / box
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('align_refine', os.path.join(orig_dir, 'align_refine.py'))
orig_refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig_refine)

if kind == 'robbins':
    m = re.search(r'E([p-])(\d+)_N([p-])(\d+)', box)
    lon = (-1 if m.group(1) == '-' else 1) * int(m.group(2))
    lat = (-1 if m.group(3) == '-' else 1) * int(m.group(4))
    cols, e = set(), lon
    while e <= lon + 10 + 1e-9:
        cols.add(int(((e - 180.0) % 360.0) // 45.0) % 8)
        e += 2.5
    base = 8 if lat + 5 >= 0 else 16
    quads = [os.path.join(sys.argv[7], f'MC{base + x:02d}_6mpp.16bit.tif') for x in sorted(cols)]
    R, B = 3396190.0, 0.3
    te = [f'{R * math.radians(v):.1f}' for v in (lon - B, lat - B, lon + 10 + B, lat + 10 + B)]
    ref = out / 'reference_24m.tif'
    if not ref.exists():
        subprocess.run(['gdalwarp', '-q', '-overwrite', '-t_srs',
                        '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs',
                        '-te', *te, '-tr', '24', '24', '-r', 'bilinear', '-ot', 'Int16', '-srcnodata', '-32768',
                        '-dstnodata', '-32768', '-multi', '-wo', 'NUM_THREADS=4', '-co', 'TILED=YES',
                        '-co', 'COMPRESS=NONE', '-co', 'BIGTIFF=IF_SAFER', '--config', 'GDAL_CACHEMAX', '2048',
                        *quads, str(ref)], check=True)
    nodata, offset, fallback = -32768.0, 32768.0, False
    orig_flags = ['--reproject-ref', '--ref-nodata', '-32768', '--ref-offset', '32768']
else:
    ref = run / 'murray_ref_24m.corrected.vrt'
    nodata, offset, fallback = 0.0, 0.0, True
    orig_flags = ['--reproject-ref', '--ref-nodata', '0', '--translation-fallback']
    corr = {r['safe']: r for r in csv.DictReader(open(sys.argv[7]))}[box]
    with rasterio.open(run / 'murray_ref_24m.tif') as a, rasterio.open(ref) as b:
        dx, dy = b.transform.c - a.transform.c, b.transform.f - a.transform.f
    print(f'reference correction: VRT origin moved by ({dx:+.2f}, {dy:+.2f}) m; table ({corr["dx_m"]}, {corr["dy_m"]}) m',
          flush=True)


def coeff_diff(a, b):
    return max(abs(x - y) for x, y in zip(tuple(a)[:6], tuple(b)[:6]))


records = [r for r in sorted(glob.glob(str(run / 'aligned' / '*.refine.json')))
           if len(Path(r).name.split('.')[0].split('_')) == 3][:n]
worst1 = worst2 = 0.0
for rec_path in records:
    rec = json.load(open(rec_path))
    stem = rec['stem']
    mapped = glob.glob(str(run / 'mapped' / f'{stem}*_mapped.tif'))
    if not mapped:
        print(f'{stem}: no mapped image')
        continue
    # stage 1: original script vs released function, same inputs and settings
    vrt = out / f'{stem}_orig.vrt'
    r = subprocess.run([sys.executable, os.path.join(orig_dir, 'align_rasters.py'), mapped[0], str(ref), *orig_flags,
                        '--apply', '--out', str(vrt)], capture_output=True, text=True)
    try:
        T1 = coarse_fit(mapped[0], ref, ref_nodata=nodata, ref_offset=offset, translation_fallback=fallback)
    except ValueError as e:
        T1 = None
        print(f'{stem}: released stage 1 rejected ({e}); original rc {r.returncode}', flush=True)
    if T1 is not None and r.returncode == 0:
        with rasterio.open(vrt) as v:
            d1 = coeff_diff(T1, v.transform)
        worst1 = max(worst1, d1)
        print(f'{stem}: stage 1 original vs released {d1:.3e}; released vs archived transform '
              f'{coeff_diff(T1, rec["transform_original"]):.2f} m', flush=True)
    # stage 2: original refine vs released refine, both from the archived stage-1 transform
    T0 = Affine(*rec['transform_original'])
    Ta, info = orig_refine.refine(mapped[0], str(ref), T0, ref_nodata=nodata, ref_offset=offset, verbose=False)
    Tb = refine(mapped[0], ref, T0, ref_nodata=nodata, ref_offset=offset)
    if (Ta is None) != (Tb is None):
        print(f'{stem}: stage 2 DIFFERENT decision: original {"rejected" if Ta is None else "applied"}, released '
              f'{"rejected" if Tb is None else "applied"}', flush=True)
        worst2 = float('inf')
    elif Ta is None:
        print(f'{stem}: stage 2 rejected by both ({info.get("why")})', flush=True)
    else:
        d2 = coeff_diff(Ta, Tb)
        worst2 = max(worst2, d2)
        print(f'{stem}: stage 2 applied by both ({info.get("kind")}), difference {d2:.3e}', flush=True)
print(f'REGISTER CHECK {box}: largest stage-1 difference {worst1:.3e}, stage-2 {worst2:.3e}')

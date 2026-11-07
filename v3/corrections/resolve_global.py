#!/usr/bin/env python3
"""Select a verified reviewed height layer on compute, with fail-closed hashes."""
import argparse,json,hashlib
from pathlib import Path
STAGE=Path('/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/site_corrections_global_2026-11-03')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def resolve(box,baseline,pointer=None):
    latest=json.load(open(pointer or STAGE/'LATEST.json'));mp=latest['manifest']
    if sha(mp)!=latest['sha256']:raise ValueError('manifest hash mismatch')
    m=json.load(open(mp))
    if m['status']!='applied_and_pixel_verified':raise ValueError('release not verified')
    r=m['boxes'].get(box)
    if r is None:return baseline
    if Path(baseline).resolve()!=Path(r['baseline']['path']).resolve():raise ValueError('baseline selection mismatch')
    if sha(baseline)!=r['baseline']['sha256']:raise ValueError('baseline hash mismatch')
    if sha(r['output'])!=r['vrt_sha256']:raise ValueError('VRT hash mismatch')
    for p in r['patches']:
        if sha(p['path'])!=p['sha256']:raise ValueError('correction patch hash mismatch')
    return r['output']
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('box');a.add_argument('baseline');a.add_argument('--pointer',help='Immutable copy of LATEST.json pinned once for this render');v=a.parse_args();print(resolve(v.box,v.baseline,v.pointer))

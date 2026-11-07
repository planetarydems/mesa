#!/usr/bin/env python3
import json,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from resolve_global import resolve,sha
with tempfile.TemporaryDirectory(prefix='seam-resolver-test-') as td:
    d=Path(td)
    for name in ('base','vrt','patch'):(d/name).write_text('original '+name)
    m=dict(status='applied_and_pixel_verified',boxes={'B':dict(baseline=dict(path=str(d/'base'),sha256=sha(d/'base')),output=str(d/'vrt'),vrt_sha256=sha(d/'vrt'),patches=[dict(path=str(d/'patch'),sha256=sha(d/'patch'))])})
    (d/'manifest').write_text(json.dumps(m));(d/'pointer').write_text(json.dumps(dict(manifest=str(d/'manifest'),sha256=sha(d/'manifest'))))
    assert resolve('B',str(d/'base'),d/'pointer')==str(d/'vrt')
    assert resolve('OTHER',str(d/'base'),d/'pointer')==str(d/'base')
    for name in ('base','vrt','patch','manifest'):
        p=d/name;saved=p.read_bytes();p.write_bytes(saved+b' altered')
        try:resolve('B',str(d/'base'),d/'pointer')
        except ValueError:pass
        else:raise AssertionError('failed to reject altered '+name)
        p.write_bytes(saved)
    try:resolve('B',str(d/'other'),d/'pointer')
    except ValueError:pass
    else:raise AssertionError('failed to reject wrong baseline')
print('RESOLVER_TEST_PASS: pinned manifest; changed baseline/patch/VRT/manifest and wrong baseline refused')

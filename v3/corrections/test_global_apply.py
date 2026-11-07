#!/usr/bin/env python3
import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from apply_global import *
from rasterio.transform import Affine
selftest()
md=dict(box='B',site=1,nearby_sites=[dict(site=1,weight=1),dict(site=2,weight=.7),dict(site=3,weight=0)])
assert neighbor_holds(md,{}, {})==['B_s0002']
assert neighbor_holds(md,{}, {'B_s0002':dict(decision='defer')})==['B_s0002']
assert neighbor_holds(md,dict(neighbor_clearances=[dict(site=2,decision='safe_in_this_footprint')]),{})==[]
assert neighbor_holds(md,{}, {'B_s0002':dict(decision='accept_local_morphology')})==[]
with tempfile.TemporaryDirectory(prefix='global-apply-test-') as td:
    td=Path(td);p=np.arange(400,dtype='float32').reshape(20,20);p[0,0]=-32768
    profile=dict(driver='GTiff',width=20,height=20,count=1,dtype='float32',crs='+proj=eqc +R=3396190',transform=Affine(18,0,0,0,-18,360),nodata=-32768.)
    with rasterio.open(td/'base.tif','w',**profile) as ds:ds.write(p,1)
    patches=[]
    for idx,(r,c) in enumerate([(3,4),(12,13)]):
        elev=p[r:r+4,c:c+4].copy()+7;weight=np.ones((4,4),dtype='float32');path=td/f'p{idx}.tif'
        pp=profile.copy();pp.update(count=3,width=4,height=4,transform=Affine(18,0,c*18,0,-18,360-r*18),nodata=float('nan'))
        with rasterio.open(path,'w',**pp) as ds:ds.write(elev,1);ds.write(weight,2);ds.write(weight*7,3)
        with rasterio.open(td/'base.tif') as b:patches.append(readpatch(path,b))
    assert intersection(patches[0],patches[1]) is None
    write_vrt(td/'derived.vrt',td/'base.tif',patches);qa=full_verify(td/'base.tif',td/'derived.vrt',patches)
    assert qa['pixels_checked']==400 and qa['pixels_changed']==32 and qa['validity_changes']==0
    try:full_verify(td/'base.tif',td/'derived.vrt',[patches[0],patches[0],patches[1]])
    except ValueError as e:assert 'overlap' in str(e)
    else:raise AssertionError('duplicate overlap not refused')
print('GLOBAL_APPLY_TEST_PASS: two-patch placement, exact outside pixels, nodata preservation, overlap refusal')

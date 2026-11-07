#!/usr/bin/env python3
"""Publish a reversible height-only correction stage, preserving all source files.

Output per box is an exact-grid VRT: baseline + finite composite patch. A manifest
is published only after a whole-box pixel comparison. Existing production stamps,
prediction captures, queues, dominance maps, and uncertainty estimates are untouched.
The correction weight/delta raster explicitly identifies derived heights; calibrated
uncertainty from the baseline must not be presented as verified on changed pixels.
"""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

TAG='ta_implicit_d1_spline100s1_e098v3'
G='/home/gridsan/groups/planetary_dtm/mars_dem/global_run'
SELECTED=[('Ep110_Np00',7,'tol05'),('Ep140_N-40',18,'tol05')]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def same(a,b):
    return a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)

def correction(base,trial,weight,protected):
    if base.shape!=trial.shape or base.shape!=weight.shape or base.shape!=protected.shape:raise ValueError('shape mismatch')
    if not np.all(np.isfinite(weight)) or (weight<0).any() or (weight>1).any():raise ValueError('invalid weight')
    if ((weight>0)&protected).any():raise ValueError('protected terrain intersects correction; revise footprint, do not clip')
    use=weight>0
    if (use&(~np.isfinite(base)|~np.isfinite(trial))).any():raise ValueError('missing height inside correction')
    out=base.copy();out[use]=base[use]+weight[use]*(trial[use]-base[use])
    if not same(base[~use],out[~use]):raise AssertionError('changed outside footprint')
    if not same(base[protected],out[protected]):raise AssertionError('changed protected terrain')
    return out,use

def add_source(band,path,src_size,dst_rect,nodata):
    s=SubElement(band,'ComplexSource');SubElement(s,'SourceFilename',relativeToVRT='0').text=str(path)
    SubElement(s,'SourceBand').text='1'
    SubElement(s,'SrcRect',xOff='0',yOff='0',xSize=str(src_size[0]),ySize=str(src_size[1]))
    SubElement(s,'DstRect',**{k:str(v) for k,v in zip(('xOff','yOff','xSize','ySize'),dst_rect)})
    if nodata is not None:SubElement(s,'NODATA').text=str(nodata)

def write_vrt(path,base,patch,r0,c0,h,w):
    with rasterio.open(base) as ds:
        root=Element('VRTDataset',rasterXSize=str(ds.width),rasterYSize=str(ds.height))
        SubElement(root,'SRS').text=ds.crs.to_wkt();SubElement(root,'GeoTransform').text=','.join(map(str,ds.transform.to_gdal()))
        band=SubElement(root,'VRTRasterBand',dataType='Float32',band='1')
        SubElement(band,'Description').text='Reviewed local seam correction; see MANIFEST.json and patch bands 2/3'
        if ds.nodata is not None:SubElement(band,'NoDataValue').text=str(ds.nodata)
        add_source(band,base,(ds.width,ds.height),(0,0,ds.width,ds.height),ds.nodata)
        add_source(band,patch,(w,h),(c0,r0,w,h),float('nan'))
        ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def verify_full(basepath,vrtpath,crop,r0,c0,use):
    counts={'pixels_checked':0,'pixels_changed':0,'pixels_outside_footprint':0,'outside_finite_bit_differences':0,'validity_changes':0}
    h,w=crop.shape
    with rasterio.open(basepath) as base,rasterio.open(vrtpath) as derived:
        assert (base.crs,base.transform,base.shape)==(derived.crs,derived.transform,derived.shape),'VRT grid mismatch'
        for r in range(0,base.height,1024):
            for c in range(0,base.width,2048):
                win=Window(c,r,min(2048,base.width-c),min(1024,base.height-r));before=base.read(1,window=win);after=derived.read(1,window=win)
                expected=before.copy();local=np.zeros(before.shape,bool)
                ya,yb=max(r,r0),min(r+before.shape[0],r0+h);xa,xb=max(c,c0),min(c+before.shape[1],c0+w)
                if ya<yb and xa<xb:
                    dst=np.s_[ya-r:yb-r,xa-c:xb-c];src=np.s_[ya-r0:yb-r0,xa-c0:xb-c0]
                    lu=use[src];target=expected[dst];target[lu]=crop[src][lu];local[dst]=lu
                if not same(expected,after):raise AssertionError(f'output mismatch at {r},{c}')
                outside=~local;finite=outside&np.isfinite(before)&np.isfinite(after)
                bitdiff=int((before[finite].view('uint32')!=after[finite].view('uint32')).sum())
                validity=int((base.read_masks(1,window=win)!=derived.read_masks(1,window=win)).sum())
                counts['pixels_checked']+=before.size;counts['pixels_changed']+=int((local&(before!=after)).sum());counts['pixels_outside_footprint']+=int(outside.sum())
                counts['outside_finite_bit_differences']+=bitdiff;counts['validity_changes']+=validity
                if bitdiff or validity:raise AssertionError('outside values or nodata validity changed')
            if r%4096==0:print('verified rows',r,'/',base.height,flush=True)
    return counts

def apply_one(box,sid,arm,sites,out):
    npz=Path(sites)/f'{box}_site{sid}.npz';d=np.load(npz);md=json.loads(str(d['metadata']))
    if md['protected_conflicts']:raise ValueError('known terrain protection conflict')
    basepath=md['sources']['production'];trialpath=md['arms'][arm]['source'];base,trial,weight,protected=[d[k] for k in ('production',arm,'weight','protected')]
    comp,use=correction(base,trial,weight,protected)
    if not same(comp,d[arm+'_composite']):raise ValueError('computed correction differs from reviewed candidate')
    r0,c0=md['r0'],md['c0'];h,w=base.shape;win=Window(c0,r0,w,h)
    sourcehash={}
    for label,path in [('production',basepath),('candidate',trialpath),('dominance',md['sources']['dominance'])]:
        sourcehash[label]={'path':path,'sha256':sha256(path),'stat':{'size':os.stat(path).st_size,'mtime_ns':os.stat(path).st_mtime_ns}}
    with rasterio.open(basepath) as ds,rasterio.open(trialpath) as ts:
        if (ds.crs,ds.transform,ds.shape)!=(ts.crs,ts.transform,ts.shape):raise ValueError('source grids differ')
        livebase=ds.read(1,window=win,masked=True).astype('float32').filled(np.nan);livetrial=ts.read(1,window=win,masked=True).astype('float32').filled(np.nan)
        if not same(livebase,base) or not same(livetrial,trial):raise ValueError('source changed after review')
        patch=out/f'{box}_site{sid}_correction.tif';profile=ds.profile.copy();profile.update(driver='GTiff',width=w,height=h,count=3,transform=ds.window_transform(win),dtype='float32',nodata=float('nan'),tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
        elev=np.full(base.shape,np.nan,dtype='float32');elev[use]=comp[use];delta=np.zeros(base.shape,dtype='float32');delta[use]=comp[use]-base[use]
        with rasterio.open(patch,'w',**profile) as dst:
            dst.write(elev,1);dst.write(weight,2);dst.write(delta,3)
            for band,title in [(1,'corrected elevation; NaN outside footprint'),(2,'correction weight; >0 flags derived height'),(3,'signed elevation change from production (m)')]:dst.set_band_description(band,title)
            dst.update_tags(box=box,site=str(sid),source_arm=arm,independent_elevation_validation='unavailable',baseline_uncertainty='not recalibrated for corrected pixels')
    vrt=out/f'{box}_reviewed.vrt';write_vrt(vrt,basepath,patch,r0,c0,h,w)
    qa=verify_full(basepath,vrt,comp,r0,c0,use)
    # Every named source must remain unchanged through verification, not just the crop.
    for item in sourcehash.values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('source changed during application')
    return dict(box=box,site=sid,arm=arm,sources=sourcehash,review_npz_sha256=sha256(npz),output=str(vrt),patch=str(patch),patch_sha256=sha256(patch),vrt_sha256=sha256(vrt),verification=qa,
                changed_gt1_km2=float((np.abs(delta)>1).sum()*.018**2),changed_gt20_km2=float((np.abs(delta)>20).sum()*.018**2),
                independent_reference_pixels=0,calibrated_uncertainty='unavailable on corrected pixels; retain separate correction flag',
                baseline_dominance_map='describes baseline contributors only; candidate source and weight are in this record and patch')

def selftest():
    p=np.arange(100,dtype='float32').reshape(10,10);t=p+5;w=np.zeros_like(p);w[3:7,3:7]=.5;protected=np.zeros(p.shape,bool);protected[0]=True
    out,use=correction(p,t,w,protected);assert same(out[~use],p[~use]);assert (out[use]==p[use]+2.5).all()
    for bad in ('overlap','nodata','weight','shape'):
        pp,tt,ww,pr=p.copy(),t.copy(),w.copy(),protected.copy()
        if bad=='overlap':pr[4,4]=True
        elif bad=='nodata':tt[4,4]=np.nan
        elif bad=='weight':ww[4,4]=1.1
        else:tt=tt[:-1]
        try:correction(pp,tt,ww,pr)
        except ValueError:pass
        else:raise AssertionError('failed refusal '+bad)
    # Real GDAL overlay, with nodata, round-trip equality and source placement.
    with tempfile.TemporaryDirectory(prefix='seam-stage-test-') as td:
        td=Path(td);profile=dict(driver='GTiff',width=10,height=10,count=1,dtype='float32',crs='+proj=eqc +R=3396190',transform=Affine(18,0,0,0,-18,180),nodata=-32768.)
        p[0,0]=-32768
        with rasterio.open(td/'base.tif','w',**profile) as ds:ds.write(p,1)
        crop=p[2:8,2:8].copy();use=np.zeros(crop.shape,bool);use[1:5,1:5]=True;crop[use]+=2.5
        patch=np.full(crop.shape,np.nan,dtype='float32');patch[use]=crop[use]
        profile.update(width=6,height=6,transform=Affine(18,0,36,0,-18,144),nodata=float('nan'))
        with rasterio.open(td/'patch.tif','w',**profile) as ds:ds.write(patch,1)
        write_vrt(td/'derived.vrt',td/'base.tif',td/'patch.tif',2,2,6,6)
        q=verify_full(td/'base.tif',td/'derived.vrt',crop,2,2,use);assert q['pixels_checked']==100 and q['pixels_changed']==16
    print('SELFTEST PASS: protected overlap, nodata, invalid weight, shape, and exact VRT round trip',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sites');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    out=Path(a.out).resolve()
    if not a.sites:raise ValueError('--sites required')
    if out.exists() and any(out.iterdir()):raise ValueError('destination must be empty; never overwrite an existing correction release')
    out.mkdir(parents=True,exist_ok=True);selftest()
    records=[]
    for box,sid,arm in SELECTED:
        print('APPLY',box,sid,arm,flush=True);records.append(apply_one(box,sid,arm,a.sites,out))
    manifest=dict(version=1,stage='reviewed local seam corrections',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        authorization='User 2026-11-03: apply the correction where appropriate; preserve real terrain.',
        status='applied_and_pixel_verified',scope='height-only per-box overlays; original production rasters retained',
        tolerance_policy='selected +0.5 m trials at these two sites; no blanket frame promotion',corrections=records,
        deferred=['Ep110_Np00:23','E-050_N-20:17'],rejected_ineffective=['E-050_N-20:20'])
    tmp=out/'MANIFEST.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,out/'MANIFEST.json')
    (out/'box_overrides.tsv').write_text('box\tdem\tcorrection_patch\n'+''.join(f"{r['box']}\t{r['output']}\t{r['patch']}\n" for r in records))
    print('APPLIED',str(out/'MANIFEST.json'),flush=True)

if __name__=='__main__':main()

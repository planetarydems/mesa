#!/usr/bin/env python3
"""Publish only independently reviewed, numerically checked, nonoverlapping patches.

Every release is immutable. Existing source rasters and prior patches are read-only.
Overlapping corrections require a separately reviewed union, never last-writer wins.
"""
import sys,os,json,csv,argparse,time,copy
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
science=Path(__file__).resolve().parents[2]/'seam_science_2026-11-03/code'
if science.exists():sys.path.append(str(science))
from apply_reviewed import sha256,same,correction,add_source,selftest
from release_lock import publication_lock,union_member_cases,union_review_concerns
from edge_certificate import validate_edge_certificate,validate_inventory_context
import numpy as np,rasterio
from rasterio.windows import Window
G=Path('/home/gridsan/groups/planetary_dtm/mars_dem/global_run')
D=G/'fix_2026-10/r2_delivery/failure_fixes/seam_remaining';C=D/'pilot/global_20261103'
STAGE=G/'v3_2026-10/site_corrections_global_2026-11-03'
OLD=G/'v3_2026-10/site_corrections_2026-11-03/MANIFEST.json'
T='ta_implicit_d1_spline100s1_e098v3'

def readpatch(path,base):
    with rasterio.open(path) as p:
        if p.crs!=base.crs or (p.transform.a,p.transform.b,p.transform.d,p.transform.e)!=(base.transform.a,base.transform.b,base.transform.d,base.transform.e):raise ValueError('patch grid differs')
        col,row=(~base.transform)*(p.transform.c,p.transform.f)
        if abs(row-round(row))+abs(col-round(col))>1e-6:raise ValueError('patch not aligned')
        return dict(path=str(path),r0=round(row),c0=round(col),h=p.height,w=p.width,elev=p.read(1),weight=p.read(2))

def intersection(a,b):
    ya,yb=max(a['r0'],b['r0']),min(a['r0']+a['h'],b['r0']+b['h']);xa,xb=max(a['c0'],b['c0']),min(a['c0']+a['w'],b['c0']+b['w'])
    if ya>=yb or xa>=xb:return None
    return (np.s_[ya-a['r0']:yb-a['r0'],xa-a['c0']:xb-a['c0']],np.s_[ya-b['r0']:yb-b['r0'],xa-b['c0']:xb-b['c0']])

def neighbor_holds(md,review,allreviews):
    clear={int(r['site']) for r in review.get('neighbor_clearances',[]) if r['decision']=='safe_in_this_footprint'};holds=[]
    for n in md['nearby_sites']:
        if n['site']==md['site'] or n['weight']<=0:continue
        case=md['box']+f"_s{n['site']:04d}"
        if n['site'] not in clear and allreviews.get(case,{}).get('decision')!='accept_local_morphology':holds.append(case)
    return holds

def full_verify(basepath,vrtpath,patches):
    counts=dict(pixels_checked=0,pixels_changed=0,pixels_outside_footprint=0,outside_finite_bit_differences=0,validity_changes=0)
    with rasterio.open(basepath) as base,rasterio.open(vrtpath) as derived:
        if (base.crs,base.transform,base.shape)!=(derived.crs,derived.transform,derived.shape):raise ValueError('output grid mismatch')
        for r in range(0,base.height,1024):
            for c in range(0,base.width,2048):
                h,w=min(1024,base.height-r),min(2048,base.width-c);win=Window(c,r,w,h);before=base.read(1,window=win);after=derived.read(1,window=win)
                expect=before.copy();mask=np.zeros((h,w),bool);block=dict(r0=r,c0=c,h=h,w=w)
                for p in patches:
                    cut=intersection(block,p)
                    if cut:
                        dst,src=cut;use=p['weight'][src]>0
                        if (mask[dst]&use).any():raise ValueError('overlap during full verification')
                        mask[dst]|=use;target=expect[dst];target[use]=p['elev'][src][use]
                if not same(expect,after):raise ValueError(f'pixel mismatch at {r},{c}')
                finite=(~mask)&np.isfinite(before)&np.isfinite(after)
                bd=int((before[finite].view('uint32')!=after[finite].view('uint32')).sum());vc=int((base.read_masks(1,window=win)!=derived.read_masks(1,window=win)).sum())
                if bd or vc:raise ValueError('outside values or nodata changed')
                counts['pixels_checked']+=before.size;counts['pixels_changed']+=int((mask&(before!=after)).sum());counts['pixels_outside_footprint']+=int((~mask).sum());counts['outside_finite_bit_differences']+=bd;counts['validity_changes']+=vc
        return counts

def write_vrt(path,base,patches):
    with rasterio.open(base) as ds:
        root=Element('VRTDataset',rasterXSize=str(ds.width),rasterYSize=str(ds.height));SubElement(root,'SRS').text=ds.crs.to_wkt();SubElement(root,'GeoTransform').text=','.join(map(str,ds.transform.to_gdal()))
        band=SubElement(root,'VRTRasterBand',dataType='Float32',band='1')
        if ds.nodata is not None:SubElement(band,'NoDataValue').text=str(ds.nodata)
        add_source(band,base,(ds.width,ds.height),(0,0,ds.width,ds.height),ds.nodata)
        for p in patches:add_source(band,p['path'],(p['w'],p['h']),(p['c0'],p['r0'],p['w'],p['h']),float('nan'))
        ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def main(proposal,release):
    selftest();out=STAGE/'releases'/release
    if out.exists():raise ValueError('release already exists; never overwrite')
    out.mkdir(parents=True);request=json.load(open(proposal));(out/'proposal.json').write_text(json.dumps(request,indent=2))
    seamcheck=json.load(open(str(proposal)+'.seam_check.json'))
    if seamcheck['manifest']['sha256']!=sha256(proposal):raise ValueError('seam diagnostic proposal mismatch')
    metric_holds=set(seamcheck['requires_metric_review']);(out/'seam_check.json').write_text(json.dumps(seamcheck,indent=2))
    latest=STAGE/'LATEST.json'
    if latest.exists():
        lp=json.load(open(latest));previous=Path(lp['manifest'])
        if sha256(previous)!=lp['sha256']:raise ValueError('prior manifest hash mismatch')
        prior_manifest=json.load(open(previous));boxes=copy.deepcopy(prior_manifest['boxes']);union_registry=copy.deepcopy(prior_manifest.get('union_registry',{}))
    else:
        previous=OLD;boxes={};union_registry={}
        for r in json.load(open(OLD))['corrections']:
            boxes[r['box']]=dict(baseline=r['sources']['production'],output=r['output'],vrt_sha256=r['vrt_sha256'],patches=[dict(case_id=r['box']+f"_s{r['site']:04d}",path=r['patch'],sha256=r['patch_sha256'],legacy_record=r)],verification=r['verification'])
    holds=[];applied=[];revoked=[];edge_inventory_contexts=[];bybox={};allreviews={r['case_id']:r for r in request['all_sol_reviews']};unionreviews={r['case_id']:r for r in request.get('all_union_reviews',[])}
    for review in request['cases']:
        b=review['case_id'].rsplit('_s',1)[0];bybox.setdefault(b,[]).append(review)
    for box in boxes:
        if any(g['box']==box for g in request['protected_sites']) or any(p.get('kind')=='same_candidate_union' or p['case_id'] in allreviews or any(c in allreviews for c in union_member_cases(p)) for p in boxes[box]['patches']):bybox.setdefault(box,[])
    for box,reviews in bybox.items():
        baseline=str(G/'boxes'/box/'box_run'/f'box_dem_{T}_deseam.tif');prior=boxes.get(box);patchrecords=copy.deepcopy(prior['patches']) if prior else [];newrecords=[]
        with rasterio.open(baseline) as base:
            patches=[];kept=[];boxrevoked=[]
            sites={int(r['site']):r for r in csv.DictReader(open(D/'campaign/list_c'/f'{box}.sites.csv'))}
            for r in patchrecords:
                if sha256(r['path'])!=r['sha256']:raise ValueError('prior patch changed')
                pp=readpatch(r['path'],base);yy,xx=np.ogrid[:pp['h'],:pp['w']];conflicts=[]
                for guard in request['protected_sites']:
                    if guard['box']!=box:continue
                    s=sites[guard['site']];m=(yy-(int(s['row'])-pp['r0']))**2+(xx-(int(s['col'])-pp['c0']))**2<=guard['radius_pixels']**2
                    if (m&(pp['weight']>0)).any():conflicts.append(guard)
                review=allreviews.get(r['case_id']);withdrawn=False;unresolved=[]
                if r.get('kind')=='same_candidate_union':
                    unresolved=union_review_concerns(r,allreviews,unionreviews)
                    withdrawn=bool(unresolved)
                elif review and 'review' in r and review['array_sha256']==r['review']['array_sha256']:
                    withdrawn=review['decision']!='accept_local_morphology'
                    md=json.load(open(C/'cards'/box/r['case_id']/'metadata.json'));unresolved=neighbor_holds(md,review,allreviews)
                if conflicts or withdrawn or unresolved:
                    if 'legacy_record' in r:
                        raise ValueError('inherited correction conflict requires selected-footprint review; refusing release: '+r['case_id'])
                    boxrevoked.append(dict(case_id=r['case_id'],reason='terrain conflict, withdrawn approval or unresolved weighted neighbour; restored baseline pending review',protected_sites=conflicts,withdrawn=withdrawn,unresolved_neighbors=unresolved))
                    if r.get('kind')=='same_candidate_union':
                        union_registry[r['case_id']].update(status='suspended',reason=boxrevoked[-1])
                else:patches.append(pp);kept.append(r)
            patchrecords=kept
            if prior and sha256(prior['output'])!=prior['vrt_sha256']:raise ValueError('prior VRT changed')
            sourcehash={}
            sites={int(r['site']):r for r in csv.DictReader(open(D/'campaign/list_c'/f'{box}.sites.csv'))}
            for review in reviews:
                case=review['case_id'];dest=C/'cards'/box/case
                try:
                    union_entries=[u for u in union_registry.values() if case in u['member_cases']]
                    if any(u['status']=='active' for u in union_entries):continue
                    if union_entries:raise ValueError('member of suspended/superseded union; explicit reviewed restoration required')
                    if any(r['case_id']==case for r in patchrecords):continue
                    md=json.load(open(dest/'metadata.json'));npz=dest/'arrays.npz';evidence=C/'numeric'/box/(case+'.json');numeric=json.load(open(evidence))
                    if Path(baseline.replace('_deseam.tif','_deseam_pcf80.tif')).exists():raise ValueError('polar baseline composition needs separately reviewed candidate')
                    if sha256(npz)!=review['array_sha256'] or md['array_sha256']!=review['array_sha256'] or numeric['array_sha256']!=review['array_sha256']:raise ValueError('review/evidence array hash mismatch')
                    if review['decision']!='accept_local_morphology' or not {'full.png','profiles.png'}.issubset(review['images_inspected']):raise ValueError('incomplete Sol review')
                    if case in metric_holds:raise ValueError('fixed original join diagnostic worsens; explicit contradiction review or revised candidate required')
                    unresolved=neighbor_holds(md,review,allreviews)
                    if unresolved:raise ValueError('weighted neighbours need clearance in this exact footprint:'+','.join(unresolved))
                    edge_verification=None
                    if md['box_edge']:
                        edge_verification=validate_edge_certificate(proposal,review,md,numeric,previous)
                        sourcehash.update(edge_verification['runtime_bindings'])
                        edge_inventory_contexts.append(edge_verification['inventory_context'])
                    elif numeric['automatic_holds']:
                        # user override (2026-11-06): only the user's own call, recorded with the exact holds it overrides
                        if not (review.get('user_override_numeric_hold') is True and str(review.get('reviewer','')).startswith('user (') and review.get('automatic_holds')==numeric['automatic_holds']):raise ValueError('numeric hold:'+str(numeric['automatic_holds']))
                        print('USER_OVERRIDE_NUMERIC_HOLD',case,numeric['automatic_holds'],flush=True)
                    for n,expected in review.get('review_image_sha256',{}).items():
                        if sha256(Path(review.get('review_image_directory',dest))/n)!=expected:raise ValueError('review image hash mismatch:'+n)
                    if numeric.get('schema_version')!=2:raise ValueError('numeric evidence must capture reference and shot sources')
                    for st in numeric['shot_inputs']+[r['source'] for r in numeric['reference_comparisons']]:
                        live=os.stat(st['path']) if os.path.exists(st['path']) else None
                        if ((live.st_size,live.st_mtime_ns) if live else (None,None))!=(st['size'],st['mtime_ns']):raise ValueError('numeric evidence source changed')
                    for st in md['source_stats'].values():
                        live=os.stat(st['path'])
                        if (live.st_size,live.st_mtime_ns)!=(st['size'],st['mtime_ns']):raise ValueError('review source changed')
                    z=np.load(npz);p,t,w=[z[k] for k in ('production','candidate','weight')];h,ww=p.shape;r0,c0=md['r0'],md['c0'];use=w>0
                    protected=np.zeros(p.shape,bool);yy,xx=np.ogrid[:h,:ww]
                    for guard in request['protected_sites']:
                        if guard['box']==box:
                            s=sites[guard['site']];protected|=(yy-(int(s['row'])-r0))**2+(xx-(int(s['col'])-c0))**2<=guard['radius_pixels']**2
                    comp,use=correction(p,t,w,protected)
                    if not same(comp,z['composite']):raise ValueError('reviewed composite mismatch')
                    candidate=dict(r0=r0,c0=c0,h=h,w=ww,weight=w)
                    for prev in patches:
                        cuts=intersection(candidate,prev)
                        if cuts and ((w[cuts[0]]>0)&(prev['weight'][cuts[1]]>0)).any():raise ValueError('overlaps existing correction; requires reviewed union')
                    if baseline not in sourcehash:
                        sourcehash[baseline]=sha256(baseline)
                        if prior and sourcehash[baseline]!=prior['baseline']['sha256']:raise RuntimeError('baseline changed since prior release')
                    with rasterio.open(md['candidate']) as trial:
                        if (base.crs,base.transform,base.shape)!=(trial.crs,trial.transform,trial.shape):raise ValueError('candidate grid mismatch')
                        win=Window(c0,r0,ww,h)
                        for ds,old in [(base,p),(trial,t)]:
                            live=ds.read(1,window=win,masked=True).astype('float32').filled(np.nan)
                            if not same(live,old):raise ValueError('source crop differs from review')
                    if md['candidate'] not in sourcehash:sourcehash[md['candidate']]=sha256(md['candidate'])
                    patch=out/(case+'.tif');profile=base.profile.copy();profile.update(driver='GTiff',width=ww,height=h,count=3,transform=base.window_transform(win),dtype='float32',nodata=float('nan'),tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=3)
                    elev=np.full(p.shape,np.nan,dtype='float32');elev[use]=comp[use];delta=np.zeros(p.shape,dtype='float32');delta[use]=comp[use]-p[use]
                    with rasterio.open(patch,'w',**profile) as dst:
                        dst.write(elev,1);dst.write(w,2);dst.write(delta,3)
                        for i,title in [(1,'corrected elevation; NaN outside correction'),(2,'weight; positive flags derived heights'),(3,'signed height change from production (m)')]:dst.set_band_description(i,title)
                        dst.update_tags(case_id=case,baseline_uncertainty='not recalibrated for corrected pixels',independent_reference_pixels=str(numeric['independent_changed_pixels']))
                    patches.append(dict(candidate,path=str(patch),elev=elev));r=dict(case_id=case,path=str(patch),sha256=sha256(patch),review=review,evidence=numeric,evidence_sha256=sha256(evidence),source_candidate=dict(path=md['candidate'],sha256=sourcehash[md['candidate']]),new_frames_covering_centre=md['new_frames_covering_centre'],display_new_frame=md['display_new_frame'],lon=float(md['site_row']['lon']),lat=float(md['site_row']['lat']),changed_gt1_km2=float((np.abs(delta)>1).sum()*.018**2),changed_gt20_km2=float((np.abs(delta)>20).sum()*.018**2));
                    if edge_verification is not None:r['edge_verification']=edge_verification
                    patchrecords.append(r);newrecords.append(r)
                except (ValueError,FileNotFoundError,KeyError) as e:
                    holds.append(dict(case_id=case,reason=str(e)));print('HOLD',case,str(e),flush=True)
        if not newrecords and not boxrevoked:continue
        if baseline not in sourcehash:
            sourcehash[baseline]=sha256(baseline)
            if prior and sourcehash[baseline]!=prior['baseline']['sha256']:raise RuntimeError('baseline changed since prior release')
        vrt=out/(box+'_reviewed.vrt');write_vrt(vrt,baseline,patches);qa=full_verify(baseline,vrt,patches)
        for path,expected in sourcehash.items():
            if sha256(path)!=expected:raise ValueError('source changed during application')
        st=os.stat(baseline)
        boxes[box]=dict(baseline=dict(path=baseline,sha256=sourcehash[baseline],stat=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)),output=str(vrt),vrt_sha256=sha256(vrt),patches=patchrecords,verification=qa)
        applied.extend(r['case_id'] for r in newrecords);revoked.extend(boxrevoked);print('APPLIED_BOX',box,len(newrecords),qa,flush=True)
    result=dict(version=3,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='applied_and_pixel_verified' if applied or revoked else 'no_new_applications',previous_manifest=str(previous),previous_manifest_sha256=sha256(previous),proposal_path=str(out/'proposal.json'),proposal_sha256=sha256(out/'proposal.json'),boxes=boxes,union_registry=union_registry,applied_this_release=applied,revoked_this_release=revoked,holds=holds,
                policy='+0.5 m pre-screen; Sol morphology and exact-footprint numeric review; protected terrain retained; overlapping patches require separate union review',uncertainty='Baseline calibrated uncertainty is unverified on corrected pixels; patch band 2 flags them.')
    for context in edge_inventory_contexts:validate_inventory_context(context)
    manifest=out/'MANIFEST.json';manifest.write_text(json.dumps(result,indent=2))
    if applied or revoked:
        tmp=STAGE/'LATEST.json.tmp';tmp.write_text(json.dumps(dict(manifest=str(manifest),sha256=sha256(manifest)),indent=2));os.replace(tmp,latest)
    print('RELEASE',manifest,'APPLIED',len(applied),'HELD',len(holds),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('proposal');a.add_argument('release');v=a.parse_args()
    with publication_lock(STAGE):main(v.proposal,v.release)

#!/usr/bin/env python3
"""Narrow, fail-closed edge evidence certificate. Never publishes a surface.

Detached proposal sidecar avoids a proposal/certificate hash cycle. All scientific
sources and evidence artifacts are rehashed by the application on Slurm.
"""
import argparse,hashlib,json,math,os
from pathlib import Path
CASE='Ep010_N-10_s0008'
ARRAY='91edd3061597fa6dcece2f60e7c5380614c61733e7b50395e22a63b04c8d97dc'
SELECTED='f8d1a706009a119219beeb5f0b2f1fe2f89575c9cf176d3a23d509085f1936f2'
EDGE='box_edge_requires_adjacent_box_QA'
CELLS=('Ep010_N-10','Ep010_N-20','Ep020_N-10','Ep020_N-20')
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def require(ok,why):
 if not ok:raise ValueError('edge certificate: '+why)
def load_bound(record):
 require(set(('path','sha256'))<=set(record),'missing file binding')
 p=Path(record['path']);require(p.is_absolute(),'nonabsolute artifact path')
 require(sha256(p)==record['sha256'],'file changed: '+str(p))
 return json.loads(p.read_text())
def validate_inventory_context(context):
 """Header-only all-box recheck; callable again immediately before publication."""
 import re
 import rasterio
 from rasterio.warp import transform_bounds
 inv=load_bound(context['inventory']);bounds=context['context_bounds'];target=context['common_crs']
 paths=[Path(v['path']) for v in inv['headers']];roots={p.parents[2] for p in paths}
 require(len(roots)==1,'mixed inventory roots');root=next(iter(roots));tag='box_dem_ta_implicit_d1_spline100s1_e098v3_deseam.tif'
 pattern=re.compile(r'^E([p-])(\d{3})_N([p-])(\d{2})$')
 current=sorted(p.name for p in root.iterdir() if pattern.match(p.name) and (p/'box_run'/tag).is_file())
 require(current==inv['all_boxes'],'renderer source inventory changed')
 possible=set(context['possible_contributors']);require({v['box'] for v in inv['headers'] if v['renderer_possible']}==possible,'possible cohort changed')
 for old in inv['headers']:
  with rasterio.open(old['path']) as ds:bb=transform_bounds(ds.crs,target,*ds.bounds,densify_pts=101)
  require(tuple(bb)==tuple(old['bounds_common_crs']),'live physical bounds changed: '+old['box'])
  reaches=bb[0]<bounds[2] and bb[2]>bounds[0] and bb[1]<bounds[3] and bb[3]>bounds[1]
  require(bool(reaches)==old['bounds_reach_context'],'live physical reach changed: '+old['box'])
  require(not reaches or old['box'] in possible,'new unaccounted physical contributor: '+old['box'])
 return {'headers_checked':len(paths),'possible_contributors':len(possible)}

def check_proof(result,reports,old):
 require(result.get('validation_pass') is True,'validation did not pass')
 require(result.get('case_id')==CASE and result.get('array_sha256')==ARRAY,'result case/array')
 counts={'pad2_vs_pad4':139,'pad2_vs_pad8':139,'allcontributors_pad2_vs003':117}
 groups={k:{} for k in counts}
 for c in result.get('checks',[]):
  group=c.get('comparison');require(group in groups,'unexpected comparison')
  name=c.get('array');require(name and name not in groups[group],'duplicate/unnamed check')
  value=c.get('maxabs');require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0,'nonfinite comparison')
  require(c.get('pass') is True and value<=(.002 if name.endswith('_dem') else 0),'failed/nonexact comparison: '+name)
  groups[group][name]=c
 for k,n in counts.items():require(len(groups[k])==n,'incomplete comparison '+k)
 require(set(groups['pad2_vs_pad4'])==set(groups['pad2_vs_pad8']),'padding keys differ')
 for cell in CELLS:
  for mode in ('raw','currentS2'):
   for phase in ('before','after'):
    for label in ('dem','wsum','cset','fallback','hardseam_band','short_bridged','seam_restored'):
     key=f'{cell}_{mode}_{phase}_{label}'
     require(all(key in v for v in groups.values()),'missing central proof '+key)
 for n,r in reports.items():
  require(r['padding_km']==n and r['array_sha256']==ARRAY and r['selected_pin']['sha256']==SELECTED,'padding identity')
  require(r['rule']=={'feather_km':4,'dataedge_taper_m':1000,'resampling':'bilinear','nominal_cells':list(CELLS)},'render rule changed')
  require(not r['missing_noncovering_S2'],'missing S2 binding')
  require(len(r['orders'])==4 and all(len(v)==9 and v[0]==k and len(set(v))==9 for k,v in r['orders'].items()),'incomplete contributor order')
  require(r['orders']==reports[2]['orders'],'contributor order changed')
  possible=set(sum(r['orders'].values(),[]));require(len(possible)==16 and set(r['actual_valid_counts_scene'])==possible,'incomplete coverage enumeration')
  require({k for k,v in r['actual_valid_counts_scene'].items() if v>0}==set(CELLS),'unexpected actual contributor')
  require(set(r['composition'])=={c+'_'+m for c in CELLS for m in ('raw','currentS2')},'incomplete composition modes')
  for v in r['composition'].values():
   require(all(v[k]==0 for k in ('hardseam_band_pixels','remaining_gap_pixels','fallback_pixels')),'gap/seam/fallback appeared')
   edge=v['physical_edge_strips']['south_1000'];require(edge['enter_weight_or_fallback']==0 and edge['primary_renderer_weight']['max']==0,'south edge enters composition')
   require(edge['composition_delta']['min']==edge['composition_delta']['max']==0 and edge['before_after_remaining_gaps']==[0,0],'south composition not excluded')
 require(old['array_sha256']==ARRAY and old['selected_pin']['sha256']==SELECTED,'003 identity')

def validate_edge_certificate(proposal,review,md,numeric,current_manifest):
 require(review['case_id']==CASE and review['array_sha256']==ARRAY,'unsupported case/array')
 require(md['box_edge'] is True and numeric['automatic_holds']==[EDGE],'nonedge hold or no edge flag')
 side=Path(str(proposal)+'.edge_certificates.json');bundle=json.loads(side.read_text())
 require(bundle.get('schema_version')==1 and bundle.get('proposal_sha256')==sha256(proposal),'proposal sidecar mismatch')
 entries=bundle.get('certificates',[]);require(len(entries)==1 and entries[0]['case_id']==CASE and entries[0]['array_sha256']==ARRAY,'sidecar scope')
 cert=load_bound(entries[0]);require(cert.get('schema_version')==1 and cert.get('case_id')==CASE and cert.get('array_sha256')==ARRAY,'certificate identity')
 require(sha256(current_manifest)==SELECTED==cert.get('selected_manifest_sha256'),'selected baseline changed')
 require(cert.get('completed_job_id')=='6156348' and cert.get('publication_scope')=='exact_native_patch_under_pinned_R2_rule','certificate scope/job')
 files=cert['files'];needed={'result','report003','pad2','pad4','pad8','inventory2','inventory4','inventory8','sol_review','metadata','numeric','precheck'}
 require(set(files)==needed,'certificate file roles')
 docs={k:load_bound(v) for k,v in files.items()}
 require(docs['metadata']==md and docs['numeric']==numeric,'live scientific evidence differs')
 row=docs['sol_review'];row=row[0] if isinstance(row,list) and len(row)==1 else row;require(row['case_id']==CASE and row['array_sha256']==ARRAY and row['decision']=='accept_local_morphology','Sol acceptance missing')
 require(review['decision']=='accept_local_morphology' and set(row['images_inspected'])=={'triage.png','full.png','profiles.png'},'incomplete acceptance views')
 require(review.get('review_file_sha256')==files['sol_review']['sha256'],'proposal points to another Sol review')
 require(row['image_sha256']==md['image_sha256'],'Sol image identity')
 neighbours={v['site'] for v in md['nearby_sites'] if v['site']!=md['site'] and v['weight']>0}
 for rr in (row,review):
  clear={v['site'] for v in rr.get('neighbor_clearances',[]) if v.get('decision')=='safe_in_this_footprint' and v.get('reason')}
  require(neighbours<=clear,'missing primary-footprint neighbour clearance')
 reports={n:docs['pad'+str(n)] for n in (2,4,8)};check_proof(docs['result'],reports,docs['report003'])
 require(docs['result']['reference003_report_sha256']==files['report003']['sha256'],'003 result binding')
 sourcepins={}
 runtime={str(side):sha256(side),entries[0]['path']:entries[0]['sha256'],str(current_manifest):SELECTED}
 runtime.update({v['path']:v['sha256'] for v in files.values()})
 for n,r in reports.items():
  require(docs['result']['padding_report_sha256'][str(n)]==files['pad'+str(n)]['sha256'],'padding result binding')
  inv=docs['inventory'+str(n)];require(r['inventory_sha256']==files['inventory'+str(n)]['sha256'] and inv['orders']==r['orders'],'inventory binding/order')
  require(not [v for v in inv['headers'] if v['bounds_reach_context'] and not v['renderer_possible']],'unaccounted physical coverage')
  require(set(inv['all_boxes'])=={v['box'] for v in inv['headers']},'incomplete grid inventory')

  arrays_path=Path(files['pad'+str(n)]['path']).parent/'central_arrays.npz'
  require(r['central_arrays_sha256']==sha256(arrays_path),'padding scientific arrays changed');runtime[str(arrays_path)]=r['central_arrays_sha256']
 for r in [docs['report003']]+list(reports.values()):
  require(len(r['sources_before_after_identical'])==(24 if r is docs['report003'] else 48),'missing source bindings')
  for path,record in r['sources_before_after_identical'].items():
   require(record['path']==path,'source path identity')
   if path in sourcepins:require(sourcepins[path]==record,'source changed across jobs')
   sourcepins[path]=record
 for path,record in sourcepins.items():
  if Path(path).name=='review_numeric.json':require(json.loads(Path(path).read_text())==numeric,'scientific numeric evidence changed')
  if Path(path).name=='review_metadata.json':require(json.loads(Path(path).read_text())==md,'scientific metadata changed')
  st=os.stat(path);require(st.st_size==record['size'] and st.st_mtime_ns==record['mtime_ns'] and sha256(path)==record['sha256'],'live source changed: '+path)
 for f,h in docs['report003']['artifacts'].items():
  path=Path(files['report003']['path']).parent/f;require(sha256(path)==h,'003 artifact changed: '+f);runtime[str(path)]=h
 runtime.update({path:v['sha256'] for path,v in sourcepins.items()})
 pm=docs['report003']['grid']['transform'][0];deg=3396190*math.pi/180
 cx=float(md['site_row']['lon'])*deg;cy=float(md['site_row']['lat'])*deg;half=18000.
 col0=math.floor((cx-half+180*deg)/pm);row0=math.floor((90*deg-(cy+half))/pm);width=math.ceil(2*half/pm)
 left=-180*deg+col0*pm;top=90*deg-row0*pm
 inventory_context={'inventory':files['inventory8'],'common_crs':docs['report003']['grid']['crs'],'context_bounds':[left,top-width*pm,left+width*pm,top],'possible_contributors':sorted(set(sum(reports[8]['orders'].values(),[])))}
 validate_inventory_context(inventory_context)
 pre=docs['precheck'];require(not pre['requires_metric_review'] and len(pre['sites_data'])==1,'precheck failed')
 site=pre['sites_data'][0];require(site['case_id']==CASE and site['array_sha256']==ARRAY,'precheck identity')
 require(all(v['after']['abs_p50']<v['before']['abs_p50'] for v in site['distances_px'].values()) and set(site['distances_px'])=={'4','8','16','32'},'precheck distances do not improve')
 return {'certificate_path':entries[0]['path'],'certificate_sha256':entries[0]['sha256'],'sidecar_sha256':sha256(side),'completed_job_id':'6156348','selected_manifest_sha256':SELECTED,'publication_scope':cert['publication_scope'],'runtime_bindings':runtime,'inventory_context':inventory_context}

def generate(campaign,proposal,review_path):
 """Assemble already-hashed evidence bindings only; runtime validator rehashes scientific files."""
 c=Path(campaign);o=c/'edge_corner_004';files={}
 paths={'result':o/'result.json','report003':c/'edge_corner_003/output/report.json','sol_review':Path(review_path),'metadata':c/'cards/Ep010_N-10'/CASE/'metadata.json','numeric':c/'numeric/Ep010_N-10'/(CASE+'.json'),'precheck':c/'edge_corner_precheck_003/result.json'}
 for n in (2,4,8):paths['pad'+str(n)]=o/f'output_pad{n}/report.json';paths['inventory'+str(n)]=o/f'output_pad{n}/inventory.json'
 for key,path in paths.items():files[key]={'path':str(path.resolve()),'sha256':sha256(path)}
 cert={'schema_version':1,'case_id':CASE,'array_sha256':ARRAY,'selected_manifest_sha256':SELECTED,'completed_job_id':'6156348','publication_scope':'exact_native_patch_under_pinned_R2_rule','files':files,'not_final_v3_certification':True}
 proposal_hash=sha256(proposal)
 out=o/('certificate_'+Path(proposal).stem+'.json');side=Path(str(proposal)+'.edge_certificates.json')
 require(not out.exists() and not side.exists(),'never overwrite certificate/sidecar')
 out.write_text(json.dumps(cert,indent=2)+'\n');side.write_text(json.dumps({'schema_version':1,'proposal_sha256':proposal_hash,'certificates':[{'case_id':CASE,'array_sha256':ARRAY,'path':str(out),'sha256':sha256(out)}]},indent=2)+'\n')
 return str(side)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('campaign');a.add_argument('proposal');a.add_argument('review');v=a.parse_args();print(generate(v.campaign,v.proposal,v.review))

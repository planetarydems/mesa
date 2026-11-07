#!/usr/bin/env python3
"""bigC_diag.py BOX [THR] -- where does the staged pcf80 correction exceed THR m (default 40)? Clusters of |C| > THR: centre lat/lon,
size, max |C|, provenance (1 = infill), local v3 relief, and PEDR shots within 1 km: MAE (shot - surface) before / after. Read-only."""
import json, os, sys
import numpy as np, rasterio
from scipy.ndimage import label, find_objects, map_coordinates, maximum_filter, minimum_filter
from pyproj import Transformer, CRS
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"; PC = f"{G}/global_run/v3_2026-10/polar_combfix80"
P = f"{G}/global_run/fix_2026-10/r2_delivery/pedr_census"; B = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
LL = CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
src = f"{G}/global_run/boxes/{B}/box_run/box_dem_{V3}_deseam.tif"; out = f"{PC}/boxes/{B}/box_dem_{V3}_deseam_pcf80.tif"
with rasterio.open(src) as S, rasterio.open(out) as O: Z = S.read(1).astype(np.float64); F = O.read(1).astype(np.float64); tr = S.transform; crs = S.crs
with rasterio.open(f"{G}/global_run/boxes/{B}/box_run/box_dem_{V3}_filled.tif.prov.tif") as Pv: prov = Pv.read(1)
v = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4); C = np.where(v, Z - F, 0.0)
lab, n = label(np.abs(C) > THR); objs = find_objects(lab)
inv = Transformer.from_crs(crs, LL, always_xy=True); fwd = Transformer.from_crs(LL, crs, always_xy=True)
sh = np.load(f"{P}/shots/{B}.shots.npz"); sx, sy = fwd.transform(sh["lon"].astype(float), sh["lat"].astype(float))
sc = (np.asarray(sx) - tr.c) / tr.a - 0.5; sr = (np.asarray(sy) - tr.f) / tr.e - 0.5
rows = []
for i, o in enumerate(objs):
    m = lab[o] == i + 1; npx = int(m.sum())
    if npx < 20: continue
    rr, cc = np.nonzero(m); r0, c0 = rr.mean() + o[0].start, cc.mean() + o[1].start
    lon, lat = inv.transform(tr.c + (c0 + 0.5) * tr.a, tr.f + (r0 + 0.5) * tr.e)
    w = (slice(max(0, int(r0) - 60), int(r0) + 60), slice(max(0, int(c0) - 60), int(c0) + 60))
    relief = float(np.nanmax(np.where(v[w], Z[w], np.nan)) - np.nanmin(np.where(v[w], Z[w], np.nan)))
    near = (np.abs(sr - r0) < 56) & (np.abs(sc - c0) < 56) & (sr > 0) & (sc > 0) & (sr < Z.shape[0] - 1) & (sc < Z.shape[1] - 1)
    if near.sum() >= 3:
        zb = map_coordinates(Z, [sr[near], sc[near]], order=1); za = map_coordinates(F, [sr[near], sc[near]], order=1); e = sh["elev"][near]
        g = np.isfinite(zb) & np.isfinite(za); d0 = np.median((e - zb)[g]); shots = (int(g.sum()), round(float(np.mean(np.abs(e - zb - d0)[g])), 1), round(float(np.mean(np.abs(e - za - d0)[g])), 1))
    else: shots = (int(near.sum()), None, None)
    rows.append(dict(lat=round(float(lat), 3), lon=round(float(lon), 3), px=npx, maxC=round(float(np.abs(C[o][m]).max()), 1), infill_frac=round(float((prov[o][m] == 1).mean()), 2),
                     relief_2km=round(relief, 0), shots_n_mae_before_after=shots))
rows.sort(key=lambda r: -r["maxC"])
res = dict(box=B, thr=THR, clusters=len(rows), px_total=int(sum(r["px"] for r in rows)), top=rows[:12],
           infill_share_of_bigC=round(float((prov[np.abs(C) > THR] == 1).mean()), 3) if (np.abs(C) > THR).any() else None)
os.makedirs(f"{PC}/diag", exist_ok=True); json.dump(res, open(f"{PC}/diag/bigC_{B}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "top"})); [print(json.dumps(r)) for r in rows[:6]]

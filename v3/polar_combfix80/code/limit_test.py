#!/usr/bin/env python3
"""limit_test.py OUT BOX -- SCRATCH test (nothing installed is touched) of per-pixel limits on the polar comb correction.
Coordinator 10-30: 6 boxes refused by the 80 m cap; do limited corrections fix this without losing the comb reduction on normal ground?
Rule from the user (10-30): corrections must not harm real terrain.
Inputs: _deseam surface Z, staged notch correction C1 (boxes/<BOX>/*_pcf80_C.tif), pre-notch correction C0 (prenotch/<BOX>/ or scratch run).
N = C1 - C0 is the notch term. limit(C, L) = C x local gain, gain = min(1, L / envelope), envelope = max |C| over 21 px (one comb period)
smoothed with a 6 px Gaussian: scales a whole bar group down smoothly instead of clipping, so it adds no kinks or harmonics.
Variants: pre (C0) | notch (C1) | limN8 (C0 + limit(N, 8)) | lim20 (limit(C1, 20)) | lim20N8 (limit(C0, 20) + limit(N, 8)) |
gateN (C0 + w N; w 1 below 10 deg slope, 0 above 20 deg) | gate20 (limit(C0, 20) + limit(w N, 8)).
Scores: |C| stats in the PRODUCT area (own 10x10 + 2 km, where the mosaic weight is > 0) and in the margin; comb per 512 px window
(box_eval.py definition, own cell, >= 80 deg, split frames / infill) ; PEDR shots; top clusters + site windows saved for figures."""
import json, math, os, re, sys, time
os.environ.setdefault("PROJ_NETWORK", "OFF")
import numpy as np, rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects, map_coordinates, median_filter
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"; R = 3396190.0
PC = f"{G}/global_run/v3_2026-10/polar_combfix80"; W = f"{G}/global_run/v3_2026-10/polar_mola_correction"
P = f"{G}/global_run/fix_2026-10/r2_delivery/pedr_census"; MOLA = f"{G}/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff"
sys.path.insert(0, f"{W}/code")
import combfix as CF
from site_eval import comb_amp, spec, peak
OUT, BOX = sys.argv[1], sys.argv[2]; os.makedirs(f"{OUT}/{BOX}", exist_ok=True); t0 = time.time()
mm = re.match(r"E([p-])(\d{3})_N([p-])(\d{2})", BOX); LO = (1 if mm.group(1) == "p" else -1) * int(mm.group(2)); LA = (1 if mm.group(3) == "p" else -1) * int(mm.group(4)); LC = LO + 5.0
DEGKM = math.pi * R / 180.0 / 1000.0; wrap = lambda a: (a + 180.0) % 360.0 - 180.0
src = f"{G}/global_run/boxes/{BOX}/box_run/box_dem_{V3}_deseam.tif"
with rasterio.open(src) as S: Z = S.read(1); tr = S.transform; crs = S.crs; nd = S.nodata
H, Wd = Z.shape; valid = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4)
if nd is not None and np.isfinite(nd): valid &= Z != nd
lon0 = float(crs.to_dict().get("lon_0", 0.0)); latr = np.degrees((tr.f + (np.arange(H) + 0.5) * tr.e) / R); xc = tr.c + (np.arange(Wd) + 0.5) * tr.a
with rasterio.open(f"{G}/global_run/boxes/{BOX}/box_run/box_dem_{V3}_filled.tif.prov.tif") as Pv: PROV = Pv.read(1)
def rd(p):
    with rasterio.open(p) as S_: return S_.read(1)
C1 = rd(f"{PC}/boxes/{BOX}/box_dem_{V3}_deseam_pcf80_C.tif")
f0 = f"{PC}/prenotch/{BOX}/box_dem_{V3}_deseam_pcf80_C.tif"; f0b = f"{OUT}/c0/{BOX}/box_dem_{V3}_deseam_pn_C.tif"
C0 = rd(f0b if os.path.exists(f0b) else f0)
# product area: own cell + 2 km (feather_weight > 0)
D = np.zeros((H, Wd), bool); LON = None
dy = np.minimum(latr - LA, (LA + 10) - latr) * DEGKM
for r in range(H):
    cl = math.cos(math.radians(latr[r]))
    rel = wrap(lon0 + np.degrees(xc / (R * max(cl, 1e-6))) - LC); dx = (5.0 - np.abs(rel)) * DEGKM * cl
    D[r] = np.minimum(dx, dy[r]) > -2.0
D &= valid
def limit(C, L):
    env = gaussian_filter(maximum_filter(np.abs(C), size=21), 6.0); return C * np.minimum(1.0, L / np.maximum(env, 1e-6)).astype(np.float32)
Zf = np.where(valid, Z, np.nanmedian(Z[valid])).astype(np.float32); gy, gx = np.gradient(gaussian_filter(Zf, 3.0), 18.0); SL = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32); del gy, gx, Zf
w = np.clip((20.0 - SL) / 10.0, 0, 1).astype(np.float32); N = C1 - C0; lN = limit(N, 8.0); lC0 = limit(C0, 20.0)
V = dict(pre=C0, notch=C1, limN8=C0 + lN, lim20=limit(C1, 20.0), lim20N8=lC0 + lN, gateN=C0 + w * N, gate20=lC0 + limit(w * N, 8.0))
print(BOX, "variants built", round(time.time() - t0), flush=True)
res = dict(box=BOX, shape=[H, Wd], valid_px=int(valid.sum()), product_px=int(D.sum()), notch_term_p99=round(float(np.percentile(np.abs(N[D]), 99)), 2) if D.any() else None, stats={})
for k, C in V.items():
    a = np.abs(C[D]); b = np.abs(C[valid & ~D])
    res["stats"][k] = dict(prod_max=round(float(a.max()), 2) if a.size else 0, prod_p999=round(float(np.percentile(a, 99.9)), 2) if a.size else 0, prod_rms=round(float(np.sqrt(np.mean(a.astype(np.float64) ** 2))), 3) if a.size else 0,
                           prod_gt20=int((a > 20).sum()), prod_gt40=int((a > 40).sum()), prod_gt80=int((a > 80).sum()), margin_max=round(float(b.max()), 2) if b.size else 0, margin_gt80=int((b > 80).sum()))
# windows (box_eval definition)
with rasterio.open(MOLA) as M:
    mt, MH, MW = M.transform, M.height, M.width
    rlo = max(0, int((math.radians(min(90.0, latr.max() + 0.5)) * R - mt.f) / mt.e) - 2); rhi = min(MH, int((math.radians(max(-90.0, latr.min() - 0.5)) * R - mt.f) / mt.e) + 3)
    A = M.read(1, window=Window(0, rlo, MW, rhi - rlo)).astype(np.float64)
def mola(lon, lat):
    col = np.mod((np.radians(lon) * R - mt.c) / mt.a, MW); row = (np.radians(lat) * R - mt.f) / mt.e - rlo
    c0 = np.floor(col).astype(np.int64); r0 = np.clip(np.floor(row).astype(np.int64), 0, A.shape[0] - 2); fc, fr = col - c0, row - r0; c0 %= MW; c1 = (c0 + 1) % MW
    return (A[r0, c0] * (1 - fc) + A[r0, c1] * fc) * (1 - fr) + (A[r0 + 1, c0] * (1 - fc) + A[r0 + 1, c1] * fc) * fr
def lonlat(r0, c0, h, w_):
    X, LAT = np.meshgrid(xc[c0:c0 + w_], latr[r0:r0 + h]); return wrap(lon0 + np.degrees(X / (R * np.maximum(np.cos(np.radians(LAT)), 1e-6)))), LAT
PAR = dict(CF.DEF)
def band(z, m, lat, lon):
    zf = CF.fill(z, m); zf = zf - gaussian_filter(zf, 64); Bf, _ = CF.sector_filters(z.shape, CF.meridian_phi(lat, lon), PAR); return CF.apply(zf, Bf)
NW = 512; win = []
for r0 in range(0, H - NW + 1, NW):
    if np.abs(latr[r0:r0 + NW]).max() < 80: continue
    for c0 in range(0, Wd - NW + 1, NW):
        v = valid[r0:r0 + NW, c0:c0 + NW]
        if v.mean() < 0.9: continue
        lon, lat = lonlat(r0, c0, NW, NW); lom = float(np.median(lon)); lam = float(lat[NW // 2, NW // 2])
        if not (LO <= lom < LO + 10 and LA <= lam < LA + 10): continue
        z = Z[r0:r0 + NW, c0:c0 + NW].astype(np.float64); k0 = peak(*spec(z)[:3]); c = slice(32, -32); bm = band(mola(lon, lat), np.ones_like(v), lat, lon)
        bz = band(z, v, lat, lon)
        row = dict(r0=r0, c0=c0, lat=round(lam, 3), lon=round(lom, 3), infill=round(float((PROV[r0:r0 + NW, c0:c0 + NW] == 1).mean()), 3), relief=round(float(np.ptp(z[v])), 1), slope_p95=round(float(np.percentile(SL[r0:r0 + NW, c0:c0 + NW][v], 95)), 1),
                   band_v3=round(float(np.std(bz[c, c])), 3), comb_v3=round(comb_amp(z, k0), 3), band_mola=round(float(np.std(bm[c, c])), 3), corr_v3=round(float(np.corrcoef(bz[c, c].ravel(), bm[c, c].ravel())[0, 1]), 3))
        for k, C in V.items():
            f = z - C[r0:r0 + NW, c0:c0 + NW]; bf = band(f, v, lat, lon)
            row[f"band_{k}"] = round(float(np.std(bf[c, c])), 3); row[f"comb_{k}"] = round(comb_amp(f, k0), 3); row[f"corr_{k}"] = round(float(np.corrcoef(bf[c, c].ravel(), bm[c, c].ravel())[0, 1]), 3)
            row[f"maxC_{k}"] = round(float(np.abs(C[r0:r0 + NW, c0:c0 + NW][v]).max()), 2)
        win.append(row)
res["n_windows"] = len(win); json.dump(win, open(f"{OUT}/{BOX}/windows.json", "w"))
print(BOX, "windows", len(win), round(time.time() - t0), flush=True)
# shots in the own cell
sf = f"{P}/shots/{BOX}.shots.npz"; res["shots"] = {}
if os.path.exists(sf):
    Sd = np.load(sf); la, lo, e, o = Sd["lat"].astype(float), Sd["lon"].astype(float), Sd["elev"].astype(float), Sd["orbit"]
    x = R * np.radians(wrap(lo - lon0)) * np.cos(np.radians(la)); y = R * np.radians(la); cc = (x - tr.c) / tr.a - 0.5; rr = (y - tr.f) / tr.e - 0.5
    ok = (cc >= 0) & (cc <= Wd - 1) & (rr >= 0) & (rr <= H - 1) & (np.abs(la) >= 80); la, e, o, cc, rr = la[ok], e[ok], o[ok], cc[ok], rr[ok]
    Zn = np.where(valid, Z, np.nan).astype(np.float32); zr = map_coordinates(Zn, [rr, cc], order=1, mode="nearest"); sl = map_coordinates(SL, [rr, cc], order=1, mode="nearest"); g = np.isfinite(zr)
    med = np.nanmedian((zr - e)[g]); res["shots"]["n"] = int(g.sum())
    for k, C in V.items():
        zf = zr - map_coordinates(C, [rr, cc], order=1, mode="nearest"); ar = np.abs(zr - e - med); af = np.abs(zf - e - med); sh = {}
        for lab, m in (("all", g), ("steep20", g & (sl > 20)), ("flat5", g & (sl < 5))):
            if m.sum() >= 30: sh[lab] = [int(m.sum()), round(float(ar[m].mean()), 3), round(float(af[m].mean()), 3), round(float(np.mean(af[m] > ar[m] + 2)), 4), round(float(np.mean(af[m] < ar[m] - 2)), 4)]
        res["shots"][k] = sh
# top clusters of the notch correction: in the product area and in the margin
yy = lambda r: latr[int(r)]
def clusters(mask, thr, top):
    lab, n = label(np.abs(C1) * mask > thr); ob = find_objects(lab); rows = []
    for i, o_ in enumerate(ob):
        if o_ is None: continue
        m_ = lab[o_] == i + 1; npx = int(m_.sum())
        if npx < 10: continue
        vals = np.abs(C1[o_]) * m_; ir, ic = np.unravel_index(np.argmax(vals), vals.shape); r_, c_ = ir + o_[0].start, ic + o_[1].start
        lon_, lat_ = lonlat(r_, c_, 1, 1); rows.append(dict(r=int(r_), c=int(c_), lat=round(float(lat_[0, 0]), 3), lon=round(float(lon_[0, 0]), 3), px=npx, maxC1=round(float(vals.max()), 1), C0=round(float(C0[r_, c_]), 1), slope=round(float(SL[r_, c_]), 1),
                                 infill=round(float((PROV[o_][m_] == 1).mean()), 2)))
    rows.sort(key=lambda q: -q["maxC1"]); return rows[:top]
res["clusters_product"] = clusters(D, 30.0, 6); res["clusters_margin"] = clusters(valid & ~D, 30.0, 3)
os.makedirs(f"{OUT}/clusters", exist_ok=True)
for kind, rows in (("prod", res["clusters_product"][:4]), ("marg", res["clusters_margin"][:2])):
    for i, q in enumerate(rows):
        r0, c0 = max(0, min(H - 512, q["r"] - 256)), max(0, min(Wd - 512, q["c"] - 256)); sl_ = (slice(r0, r0 + 512), slice(c0, c0 + 512)); lon, lat = lonlat(r0, c0, 512, 512)
        np.savez_compressed(f"{OUT}/clusters/{BOX}_{kind}{i}.npz", z=Z[sl_], prov=PROV[sl_], mola=mola(lon, lat).astype(np.float32), lat=lat.astype(np.float32), lon=lon.astype(np.float32), D=D[sl_], peak=np.array([q["r"] - r0, q["c"] - c0]), info=json.dumps(q),
                            **{f"C_{k}": v[sl_] for k, v in V.items() if k in ("pre", "notch", "lim20N8", "gate20", "limN8")})
# site windows (HiRISE etc.) for the local referee
sites = json.load(open(f"{OUT}/sites.json")).get(BOX, {}); os.makedirs(f"{OUT}/sites", exist_ok=True)
for s, (left, top, n) in sites.items():
    c0 = int(round((left - tr.c) / tr.a)); r0 = int(round((top - tr.f) / tr.e)); sl_ = (slice(r0, r0 + n), slice(c0, c0 + n))
    if r0 < 0 or c0 < 0 or r0 + n > H or c0 + n > Wd: print("site off raster", s); continue
    np.savez_compressed(f"{OUT}/sites/{s}.npz", z=Z[sl_], **{f"C_{k}": v[sl_] for k, v in V.items()})
json.dump(res, open(f"{OUT}/{BOX}/limit.json", "w"), indent=1); print(BOX, "done", round(time.time() - t0), json.dumps(res["stats"]["notch"]), flush=True)

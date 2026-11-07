#!/usr/bin/env python3
"""box_correct.py OUT_ROOT BOX [PARAMS_JSON] -- box-level removal of the MOLA-input comb from a v3 box surface (sandbox; products only read).

Input : global_run/boxes/<BOX>/box_run/box_dem_<V3>_deseam.tif (the v3 box surface).
MOLA  : the 463 m mosaic sampled bilinearly at every 18 m pixel centre (MolaSampler convention), i.e. what the network saw up to the
        frame grid's sub-pixel offset (absorbed by the quadrature term of combfix).
Method: combfix.correct() tile by tile (1024 px tiles, 192 px tapered overlap), in parallel over tiles.
Output: OUT_ROOT/<BOX>/box_dem_<V3>_deseam_combfix.tif (v3 - C), .._combfix_C.tif (C, float32), combfix.json (stats incl. correction
        RMS by 0.5 deg latitude), and a 1/8-resolution npz of |k|, R2 and the weight for figures."""
import json, math, os, sys, time
os.environ.setdefault("PROJ_NETWORK", "OFF")
import numpy as np, rasterio
from rasterio.windows import Window
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combfix as CF

G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"
MOLA = f"{G}/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff"; R = 3396190.0
OUT, BOX = sys.argv[1], sys.argv[2]; PAR = dict(CF.DEF, NOTCH_FEATHER=8.0, **(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}))
SRC = os.environ.get("CF_SRC", f"{G}/global_run/boxes/{BOX}/box_run/box_dem_{V3}_deseam.tif")
TAG = os.environ.get("CF_TAG", "combfix")
os.makedirs(f"{OUT}/{BOX}", exist_ok=True)
S_ = {}


def lonlat(r0, c0, h, w):
    tr, lon0 = S_["tr"], S_["lon0"]
    y = tr.f + (np.arange(r0, r0 + h) + 0.5) * tr.e; x = tr.c + (np.arange(c0, c0 + w) + 0.5) * tr.a
    lat = np.degrees(y / R); X, LAT = np.meshgrid(x, lat)
    lon = lon0 + np.degrees(X / (R * np.cos(np.radians(LAT))))
    return ((lon + 180) % 360 - 180).astype(np.float64), LAT.astype(np.float64)


def mola_tile(lon, lat):
    A, mt, rlo, MW = S_["A"], S_["mt"], S_["rlo"], S_["MW"]
    col = np.mod((np.radians(lon) * R - mt.c) / mt.a, MW); row = (np.radians(lat) * R - mt.f) / mt.e - rlo
    c0 = np.floor(col).astype(np.int64); r0 = np.clip(np.floor(row).astype(np.int64), 0, A.shape[0] - 2); fc, fr = col - c0, row - r0
    c0 %= MW; c1 = (c0 + 1) % MW
    return (A[r0, c0] * (1 - fc) + A[r0, c1] * fc) * (1 - fr) + (A[r0 + 1, c0] * (1 - fc) + A[r0 + 1, c1] * fc) * fr


def work(t):
    r0, c0, h, w = t; HZ, WZ = S_["Z"].shape; P_ = int(PAR["PAD"])
    # tile + a halo of real neighbouring data (P_ px); mirrored only where the halo leaves the box
    a0, a1, b0, b1 = max(0, r0 - P_), min(HZ, r0 + h + P_), max(0, c0 - P_), min(WZ, c0 + w + P_)
    Z = S_["Z"][a0:a1, b0:b1].astype(np.float64); v = S_["valid"][a0:a1, b0:b1]
    nm = (S_["PROV"][a0:a1, b0:b1] == 1).astype(np.float64) if "PROV" in S_ else None      # infill pixels (notch add-on)
    lon, lat = lonlat(a0, b0, a1 - a0, b1 - b0)
    if not v[r0 - a0:r0 - a0 + h, c0 - b0:c0 - b0 + w].any() or np.abs(lat).max() < PAR["LAT0"]: return t, None
    M = mola_tile(lon, lat); Zn = np.where(v, Z, np.nan)
    pads = ((P_ - (r0 - a0), P_ - (a1 - r0 - h)), (P_ - (c0 - b0), P_ - (b1 - c0 - w)))
    if any(x for pr in pads for x in pr):
        Zn = np.pad(CF.fill(Zn, v), pads, mode="reflect"); M = np.pad(M, pads, mode="reflect"); v = np.pad(v, pads, mode="reflect")
        lat = np.pad(lat, pads, mode="edge"); lon = np.pad(lon, pads, mode="edge"); Zn = np.where(v, Zn, np.nan)
        if nm is not None: nm = np.pad(nm, pads, mode="reflect")
    if nm is not None: nm = gaussian_filter(nm, PAR["NOTCH_FEATHER"])                    # feathered infill mask (no step at the frame edge)
    p = dict(PAR, TILE=max(h, w) + 2 * P_ + 1, PAD=0)                   # one FFT tile per worker tile (halo already added)
    C, parts = CF.correct(Zn, M, lat, lon, valid=v, p=p, return_parts=True, notch_mask=nm)
    cr = (slice(P_, P_ + h), slice(P_, P_ + w)); C = C[cr]; parts = {k: x[cr] for k, x in parts.items()}
    return t, (C.astype(np.float32), np.hypot(parts["a"], parts["b"]).astype(np.float32)[::8, ::8], parts["R2"].astype(np.float32)[::8, ::8],
               parts["w"].astype(np.float32)[::8, ::8])


def main():
    t0 = time.time()
    with rasterio.open(SRC) as S:
        prof = S.profile; H, W = S.shape; tr = S.transform; crs = S.crs; nd = S.nodata; Z = S.read(1)
    lon0 = float(crs.to_dict().get("lon_0", 0.0))
    valid = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4)
    if nd is not None and np.isfinite(nd): valid &= Z != nd
    ymax, ymin = tr.f, tr.f + H * tr.e; lats = np.degrees(np.array([ymax, ymin]) / R)
    with rasterio.open(MOLA) as M:
        mt, MH, MW, mnd = M.transform, M.height, M.width, M.nodata
        rlo = max(0, int((math.radians(min(90, lats.max() + 0.5)) * R - mt.f) / mt.e) - 2); rhi = min(MH, int((math.radians(max(-90, lats.min() - 0.5)) * R - mt.f) / mt.e) + 3)
        A = M.read(1, window=Window(0, rlo, MW, rhi - rlo)).astype(np.float32)
    if mnd is not None: A[A == mnd] = np.nan
    S_.update(Z=Z, valid=valid, tr=tr, lon0=lon0, A=A, mt=mt, rlo=rlo, MW=MW)
    if PAR["NOTCH_FRAC"] > 0:                                                              # infill notch add-on: provenance 1 = infill
        pf = os.environ.get("CF_PROV", f"{G}/global_run/boxes/{BOX}/box_run/box_dem_{V3}_filled.tif.prov.tif")
        with rasterio.open(pf) as PS: prov = PS.read(1)
        if prov.shape != Z.shape: raise RuntimeError(f"provenance {prov.shape} != surface {Z.shape}")
        S_["PROV"] = prov
    T, OV = 1024, PAR["OV"]; step = T - 2 * OV
    # worker tiles = the FFT tiles of combfix (so taper/overlap logic is identical); blended here with the same cosine taper
    rows = list(range(0, max(1, H - 2 * OV), step)) if H > T else [0]; cols = list(range(0, max(1, W - 2 * OV), step)) if W > T else [0]
    tiles = []
    for r0 in rows:
        for c0 in cols:
            r1, c1 = min(H, r0 + T), min(W, c0 + T); tiles.append((max(0, r1 - T), max(0, c1 - T), r1 - max(0, r1 - T), c1 - max(0, c1 - T)))
    Cacc = np.zeros((H, W), np.float32); Wacc = np.zeros((H, W), np.float32)
    K8 = np.zeros(((H + 7) // 8, (W + 7) // 8), np.float32); R8 = np.zeros_like(K8); W8 = np.zeros_like(K8); N8 = np.zeros_like(K8)
    nproc = int(os.environ.get("NPROC", "8"))
    with Pool(nproc) as pool:
        for (r0, c0, h, w), res in pool.imap_unordered(work, tiles, chunksize=1):
            if res is None: continue
            C, k8, r8, w8 = res
            tw = np.ones((h, w), np.float32)
            for ax, (lo, hi, n_) in enumerate(((r0, r0 + h, H), (c0, c0 + w, W))):
                L = hi - lo; tt = np.ones(L, np.float32)
                if lo > 0: tt[:OV] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(OV) + 0.5) / OV)
                if hi < n_: tt[-OV:] = np.minimum(tt[-OV:], 0.5 + 0.5 * np.cos(np.pi * (np.arange(OV) + 0.5) / OV))
                tw = tw * (tt[:, None] if ax == 0 else tt[None, :])
            Cacc[r0:r0 + h, c0:c0 + w] += tw * C; Wacc[r0:r0 + h, c0:c0 + w] += tw
            tw8 = tw[::8, ::8]; sl8 = (slice(r0 // 8, r0 // 8 + k8.shape[0]), slice(c0 // 8, c0 // 8 + k8.shape[1]))
            K8[sl8] += tw8 * k8; R8[sl8] += tw8 * r8; W8[sl8] += tw8 * w8; N8[sl8] += tw8
    C = np.where(Wacc > 0, Cacc / np.maximum(Wacc, 1e-9), 0).astype(np.float32); C[~valid] = 0
    if PAR["LIM_T"] > 0: C = CF.limit(C, PAR["LIM_T"]).astype(np.float32); C[~valid] = 0     # rule v3: no correction beyond the size HiRISE has tested
    Zo = np.where(valid, Z - C, Z).astype(np.float32)
    p = dict(prof); p.update(driver="GTiff", dtype="float32", compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512, bigtiff="IF_SAFER")
    for arr, nm in ((Zo, f"box_dem_{V3}_deseam_{TAG}.tif"), (C, f"box_dem_{V3}_deseam_{TAG}_C.tif")):
        tmp = f"{OUT}/{BOX}/.tmp_{nm}"
        with rasterio.open(tmp, "w", **p) as w_: w_.write(arr, 1)
        os.replace(tmp, f"{OUT}/{BOX}/{nm}")
    n8 = np.maximum(N8, 1e-6); np.savez_compressed(f"{OUT}/{BOX}/{TAG}_diag8.npz", k=K8 / n8, R2=R8 / n8, w=W8 / n8)
    # stats by latitude (0.5 deg bins) on valid pixels
    yy = tr.f + (np.arange(H) + 0.5) * tr.e; latr = np.degrees(yy / R)
    byl = {}
    for b0 in np.arange(np.floor(np.abs(latr).min() * 2) / 2, np.ceil(np.abs(latr).max() * 2) / 2, 0.5):
        rs = np.flatnonzero((np.abs(latr) >= b0) & (np.abs(latr) < b0 + 0.5))
        if len(rs) == 0: continue
        cc = C[rs][valid[rs]]
        if cc.size: byl[f"{b0:.1f}"] = dict(rms=round(float(np.sqrt(np.mean(cc.astype(np.float64) ** 2))), 3), p99=round(float(np.percentile(np.abs(cc), 99)), 2), mean=round(float(cc.mean()), 3))
    cv = C[valid].astype(np.float64)
    res = dict(box=BOX, src=SRC, params=PAR, tiles=len(tiles), seconds=round(time.time() - t0), correction_rms_m=round(float(np.sqrt(np.mean(cv ** 2))), 3),
               correction_p99_m=round(float(np.percentile(np.abs(cv), 99)), 2), correction_max_m=round(float(np.abs(cv).max()), 2),
               frac_w_gt_half=round(float((W8 / n8 > 0.5)[N8 > 0].mean()), 4), by_lat=byl)
    json.dump(res, open(f"{OUT}/{BOX}/{TAG}.json", "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "by_lat"}), flush=True)


if __name__ == "__main__":
    main()

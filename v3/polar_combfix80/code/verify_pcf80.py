#!/usr/bin/env python3
"""verify_pcf80.py BOX -- checks of the polar comb filter output for one box before it may be installed (read-only on the product).
Fails (exit 1) unless: same grid / CRS / nodata; identical valid mask; zero change where |lat| < 80 deg; |correction| <= CMAX m over the whole raster,
CMAX = 1.2 x LIM_T from code/params.json (the limiter's smooth envelope overshoots L by 9 % on a bar and at most 18 % on an isolated spike), so this checks that the limiter worked.
Writes $PC/boxes/<BOX>/verify.json: source fingerprint (size, mtime, md5), output md5, correction RMS / p99 / max overall and per 1 deg."""
import hashlib, json, math, os, sys
import numpy as np, rasterio
G = "/home/gridsan/groups/planetary_dtm/mars_dem"; V3 = "ta_implicit_d1_spline100s1_e098v3"; R = 3396190.0
PC = os.environ.get("PC", f"{G}/global_run/v3_2026-10/polar_combfix80"); BOX = sys.argv[1]
_P = json.load(open(os.environ.get("PARAMS_JSON", f"{PC}/code/params.json"))); CMAX = float(os.environ.get("CMAX", 1.2 * _P["LIM_T"]))
src = f"{G}/global_run/boxes/{BOX}/box_run/box_dem_{V3}_deseam.tif"; out = f"{PC}/boxes/{BOX}/box_dem_{V3}_deseam_pcf80.tif"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


with rasterio.open(src) as S, rasterio.open(out) as O:
    same_grid = S.shape == O.shape and S.transform == O.transform and S.crs == O.crs and S.nodata == O.nodata
    Z = S.read(1); F = O.read(1); tr = S.transform; nd = S.nodata
vz = np.isfinite(Z) & (Z > -1e4) & (Z < 3e4); vf = np.isfinite(F) & (F > -1e4) & (F < 3e4)
if nd is not None and np.isfinite(nd): vz &= Z != nd; vf &= F != nd
lat = np.degrees((tr.f + (np.arange(Z.shape[0]) + 0.5) * tr.e) / R)
C = np.where(vz, Z.astype(np.float64) - F, 0.0)
south = np.abs(lat) < 80.0
eq_change = float(np.abs(C[south]).max()) if south.any() else 0.0
nonvalid_same = bool(np.array_equal(np.where(~vz, Z, 0), np.where(~vz, F, 0)) if nd is None or not np.isfinite(nd) else np.array_equal(Z[~vz], F[~vz]))
cv = C[vz]; byl = {}
for b0 in range(int(np.floor(np.abs(lat).min())), int(np.ceil(np.abs(lat).max()))):
    rs = (np.abs(lat) >= b0) & (np.abs(lat) < b0 + 1); c = C[rs][vz[rs]]
    if c.size: byl[str(b0)] = dict(rms=round(float(np.sqrt(np.mean(c ** 2))), 3), p99=round(float(np.percentile(np.abs(c), 99)), 2), max=round(float(np.abs(c).max()), 2))
st = os.stat(src)
res = dict(box=BOX, source=src, source_size=st.st_size, source_mtime=st.st_mtime, source_md5=md5(src), output=out, output_md5=md5(out),
           same_grid=bool(same_grid), same_valid_mask=bool(np.array_equal(vz, vf)), nonvalid_unchanged=nonvalid_same,
           max_change_below_80=round(eq_change, 6), correction_rms_m=round(float(np.sqrt(np.mean(cv ** 2))), 3) if cv.size else 0.0,
           correction_p99_m=round(float(np.percentile(np.abs(cv), 99)), 2) if cv.size else 0.0, correction_max_m=round(float(np.abs(cv).max()), 2) if cv.size else 0.0,
           frac_gt_20m=round(float(np.mean(np.abs(cv) > 20)), 6) if cv.size else 0.0, by_lat=byl)
res["ok"] = bool(res["same_grid"] and res["same_valid_mask"] and res["nonvalid_unchanged"] and res["max_change_below_80"] == 0.0 and res["correction_max_m"] <= CMAX)
json.dump(res, open(f"{PC}/boxes/{BOX}/verify.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("by_lat", "source", "output")}), flush=True); sys.exit(0 if res["ok"] else 1)

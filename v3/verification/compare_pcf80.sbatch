#!/bin/bash
#SBATCH -p debug-cpu -N 1 -c 32 --mem 150G -t 00:45:00 -J mesa_v3_pcf80
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/release_check/v3/logs/%x_%j.out
# compare_pcf80.sbatch POLAR_BOX : replay the polar comb correction of one box twice (worker counts 44 and 28) from the
# released polar_combfix80 code on the production _deseam.tif (read only) into the scratch root, and compare the outputs
# with the installed _deseam_pcf80.tif pixel by pixel and header by header. Writes only under $R.
G=/home/gridsan/groups/planetary_dtm/mars_dem; R=$G/release_check/v3; C=$R/code; V3=ta_implicit_d1_spline100s1_e098v3
POLAR=${1:?box}; PY=$HOME/miniforge/envs/isis/bin/python3; RES=$R/results/$POLAR; mkdir -p "$RES"
S=$G/global_run/boxes/$POLAR/box_run/box_dem_${V3}_deseam.tif; INST=$G/global_run/boxes/$POLAR/box_run/box_dem_${V3}_deseam_pcf80.tif
unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for NP in 44 28; do
  PCR=$R/pcf80_np$NP; rm -rf "$PCR"; mkdir -p "$PCR/boxes" "$PCR/logs"
  t0=$(date +%s)
  ( export PC=$PCR; CF_SRC=$S CF_TAG=pcf80 NPROC=$NP $PY "$C/polar_combfix80/code/box_correct.py" "$PCR/boxes" "$POLAR" "$(cat "$C/polar_combfix80/code/params.json")" > "$PCR/logs/box_$POLAR.txt" 2>&1 )
  echo "np $NP rc=$? $(( $(date +%s) - t0 )) s md5 $(md5sum "$PCR/boxes/$POLAR/box_dem_${V3}_deseam_pcf80.tif" | cut -c1-32)"
done
echo "installed md5 $(md5sum "$INST" | cut -c1-32)"
$PY - "$INST" "$R/pcf80_np44/boxes/$POLAR/box_dem_${V3}_deseam_pcf80.tif" "$R/pcf80_np28/boxes/$POLAR/box_dem_${V3}_deseam_pcf80.tif" "$S" <<'PY' | tee "$RES/pcf80_compare.txt"
import sys, numpy as np, rasterio
inst, a, b, src = sys.argv[1:5]
def info(p):
    with rasterio.open(p) as d:
        return dict(driver=d.driver, dtype=d.dtypes[0], shape=d.shape, crs=str(d.crs), transform=tuple(d.transform)[:6], nodata=d.nodata,
                    profile={k: v for k, v in d.profile.items() if k not in ("crs", "transform")}, tags=d.tags(), band_tags=d.tags(1), desc=d.descriptions)
I, A, B = info(inst), info(a), info(b)
for k in I:
    print(f"header {k}: installed == np44: {I[k] == A[k]}; np44 == np28: {A[k] == B[k]}")
    if I[k] != A[k]: print("   installed:", I[k]); print("   np44     :", A[k])
with rasterio.open(inst) as d: zi = d.read(1)
with rasterio.open(a) as d: za = d.read(1)
with rasterio.open(b) as d: zb = d.read(1)
with rasterio.open(src) as d: zs = d.read(1); tr = d.transform
def cmp(name, x, y):
    eq = (x == y) | (np.isnan(x) & np.isnan(y)); n = (~eq).sum()
    print(f"pixels {name}: differing {n} of {x.size} ({100.0 * n / x.size:.4f} %)", end="")
    if n:
        dd = np.abs(x - y)[~eq]; rows = np.where((~eq).any(axis=1))[0]
        print(f"; max |diff| {np.nanmax(dd):.4g} m, median {np.nanmedian(dd):.4g} m, p99 {np.nanpercentile(dd, 99):.4g} m, rows {rows.min()}-{rows.max()}", end="")
    print()
cmp("installed vs np44", zi, za); cmp("installed vs np28", zi, zb); cmp("np44 vs np28", za, zb)
cmp("source vs installed", zs, zi); cmp("source vs np44", zs, za)
# latitude of the differing rows (box grid: EQC-like, y in metres from the equator on a 3396190 m sphere)
R_ = 3396190.0
eq = (zi == za) | (np.isnan(zi) & np.isnan(za))
if (~eq).any():
    rows = np.where((~eq).any(axis=1))[0]
    lat = lambda r: np.degrees((tr.f + (r + 0.5) * tr.e) / R_)
    print(f"differing rows span latitude {lat(rows.max()):.3f} to {lat(rows.min()):.3f} deg")
PY
echo "done $(date)"

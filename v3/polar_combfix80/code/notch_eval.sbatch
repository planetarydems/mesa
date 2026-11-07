#!/bin/bash
#SBATCH -J pcf80ne -p debug-cpu -N1 -c 48 --mem=170G -t 01:30:00
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/polar_combfix80/logs/%x_%j.out
# Coordinator 10-29: infill comb before/after on the boxes the sweep re-staged with the infill notch (user: "sure redo it ."). Scored with the
# polar session's box_eval.py (read-only use of $W/code), windows/shots >= 80 deg, split CTX frames / infill. Tags in $E/<BOX>/ follow its
# summ_notch.py: pcf80 = BEFORE (v3 - the pre-notch correction saved in prenotch/<BOX>/), cfn80 = AFTER (the installed pcf80 file now).
G=/home/gridsan/groups/planetary_dtm/mars_dem; PC=$G/global_run/v3_2026-10/polar_combfix80; W=$G/global_run/v3_2026-10/polar_mola_correction
V3=ta_implicit_d1_spline100s1_e098v3; E=$PC/notch_eval; mkdir -p $E; cd $PC
unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 WIN_LAT_MIN=80; PY=~/miniforge/envs/isis/bin/python3
$PY code/pcf80_state.py > $E/state.txt
for B in $(ls prenotch); do
  grep -q "^$B current" $E/state.txt || { echo "$B not current after the sweep, skipped"; continue; }
  mkdir -p $E/$B; S=$G/global_run/boxes/$B/box_run/box_dem_${V3}_deseam.tif
  $PY - $S prenotch/$B/box_dem_${V3}_deseam_pcf80_C.tif $E/$B/box_dem_${V3}_deseam_pcf80.tif <<'PYEOF'
import sys, numpy as np, rasterio
s, c, o = sys.argv[1:4]
with rasterio.open(s) as S, rasterio.open(c) as C:
    p = S.profile; Z = S.read(1); K = C.read(1)
p.update(driver="GTiff", dtype="float32", compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512, bigtiff="IF_SAFER")
with rasterio.open(o, "w", **p) as w: w.write((Z - K).astype(np.float32), 1)
PYEOF
  ln -sfn $G/global_run/boxes/$B/box_run/box_dem_${V3}_deseam_pcf80.tif $E/$B/box_dem_${V3}_deseam_cfn80.tif
  echo "{\"params\": {\"LAT0\": 80.0, \"LAT1\": 81.0}}" > $E/$B/pcf80.json; echo "{\"params\": $(cat code/params.json)}" > $E/$B/cfn80.json
done
n=0
for B in $(ls -d $E/E* | xargs -n1 basename); do
  for T in pcf80 cfn80; do ( OMP_NUM_THREADS=4 $PY $W/code/box_eval.py $E $B $T > $E/$B/eval_$T.txt 2>&1 || echo "$B $T eval FAILED" ) & n=$((n+1)); [ $((n % 10)) -eq 0 ] && wait; done
done; wait
cp $W/code/summ_notch.py code/summ_notch_pmc.py 2>/dev/null; $PY code/summ_notch_pmc.py $E > $E/summary.md 2>&1; head -c 3500 $E/summary.md; echo; echo "$(date +%T) done"

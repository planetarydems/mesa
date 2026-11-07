#!/usr/bin/env bash
# rejoin_box_v3.sh <BOX> : for a box the v3 fleet FINISHED with the old join (state/<BOX>/.done; 2026-10-26). Computes the v3 join
#   (code/seamline_merge_v3.py, flags 32/8/30, from the fill's leveled + model surfaces) and the infill-vs-MOLA rule
#   (code/mola_infill_fix.py) OUT OF PLACE, then swaps FINAL and the v3 surface (box_dem_<TAG>_deseam.tif + _mask/_mif.json) in
#   back to back and re-stamps both (fresh_tag_v3.py), so a fleet worker scan sees the finished box stale for seconds only
#   (mif_box_tag_v3.sh does the same work in place, for boxes still claimed by the fleet). Writes FINAL.merge.json (join md5).
#   Products are untouched until the swap; a failure before it leaves the box as it was (temp dir removed).
#   CHECK=1: code/join_check.py on the old FINAL and on the new one (--ref old) before the swap; JSON in $V3D/rejoin/.
# Exit 0 done or already joined; 2 not a finished box / guard; 4 join failed; 3 infill rule failed; 5 stamp refused after the swap.
# Env: TAG (required), BOXES_ROOT, V3D, MIF_T_POLAR (30), MIF_T_OTHER (150), CHECK.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS=$BASE/global_run/predict_v4s3d_e098/downstream
source "$DS/tag_guard.sh" || exit 2
NAME=${1:?box}; TAG=${TAG:?TAG}; V3D=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}
tag_check_writable "$TAG" || exit 2
OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$NAME/box_run
FT="python3 $V3D/code/fresh_tag_v3.py"; PY=${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3}
FINAL=$OUT/box_dem_${TAG}_FINAL.tif; PROV=$OUT/box_dem_${TAG}_filled.tif.prov.tif; DES=$OUT/box_dem_${TAG}_deseam.tif
LEVELED=$OUT/box_dem_${TAG}_leveled.tif; MODEL=$OUT/box_dem_${TAG}_filled.tif.model.tif
LOG=$OUT/build_final_${TAG}.log
JOIN=$V3D/code/seamline_merge_v3.py; JMD5=$(md5sum "$JOIN" | cut -c1-32); MARK=$FINAL.merge.json
echo "[$(date +%T)] $NAME rejoin tag=$TAG join $JMD5" | tee -a "$LOG"
if grep -q "\"$JMD5\"" "$MARK" 2>/dev/null && ! [ "$FINAL" -nt "$MARK" ]; then echo "[rejoin] $NAME already joined with $JMD5, skip" | tee -a "$LOG"; exit 0; fi
$FT check "$OUT" "$TAG" FINAL prov deseam > /dev/null 2>&1 || { echo "[rejoin] $NAME: FINAL/prov/deseam not all fresh -> not a finished box" | tee -a "$LOG"; exit 2; }
PREF=$($FT ids "$OUT" "$TAG" FINAL) || exit 2
TMP=$OUT/.rejoin_tmp_$TAG; rm -rf "$TMP"; mkdir -p "$TMP/mif"
trap 'rm -rf "$TMP"' EXIT
python3 "$JOIN" --dem "$LEVELED" --model "$MODEL" --out "$TMP/FINAL.tif" --buffer 32 --feather 8 --min-hole-px 30 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -s "$TMP/FINAL.tif" ] || { echo "[rejoin] join FAILED rc=$rc (products unchanged)" | tee -a "$LOG"; exit 4; }
if [ -n "${CHECK:-}" ]; then
  mkdir -p "$V3D/rejoin"
  python3 "$V3D/code/join_check.py" "$LEVELED" "$MODEL" "$FINAL" --label "$NAME old join" --json "$V3D/rejoin/check_${NAME}_old.json" 2>&1 | tee -a "$LOG"
  python3 "$V3D/code/join_check.py" "$LEVELED" "$MODEL" "$TMP/FINAL.tif" --ref "$FINAL" --label "$NAME v3 join" --json "$V3D/rejoin/check_${NAME}_v3.json" 2>&1 | tee -a "$LOG"
fi
( unset PROJ_DATA PROJ_LIB; "$PY" "$V3D/code/mola_infill_fix.py" "$NAME" "$TMP/FINAL.tif" "$TMP/mif" --prov "$PROV" --out "$TMP/mif/deseam.tif" \
    --t-polar "${MIF_T_POLAR:-30}" --t-other "${MIF_T_OTHER:-150}" ) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -s "$TMP/mif/deseam.tif" ] || { echo "[rejoin] infill rule FAILED rc=$rc (products unchanged)" | tee -a "$LOG"; exit 3; }
# ---- swap (seconds) ----
$FT check "$OUT" "$TAG" FINAL prov deseam > /dev/null 2>&1 && [ "$($FT ids "$OUT" "$TAG" FINAL)" = "$PREF" ] \
  || { echo "[rejoin] $NAME changed while the re-join ran -> not swapped (products unchanged)" | tee -a "$LOG"; exit 2; }
mv -f "$TMP/FINAL.tif" "$FINAL"
$FT stamp --expect "$PREF" "$OUT" "$TAG" FINAL 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[rejoin] FINAL NOT stamped after the swap" | tee -a "$LOG"; exit 5; }
PRED=$($FT ids "$OUT" "$TAG" deseam) || exit 5
mv -f "$TMP/mif/deseam_mask.tif" "${DES%.tif}_mask.tif"; mv -f "$TMP/mif/deseam.json" "${DES%.tif}_mif.json"; mv -f "$TMP/mif/deseam.tif" "$DES"
$FT stamp --expect "$PRED" "$OUT" "$TAG" deseam 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[rejoin] deseam NOT stamped after the swap" | tee -a "$LOG"; exit 5; }
printf '{"merge": "%s", "md5": "%s", "flags": "--buffer 32 --feather 8 --min-hole-px 30", "job": "%s", "date": "%s", "via": "rejoin_box_v3.sh"}\n' \
    "$JOIN" "$JMD5" "${SLURM_JOB_ID:-local}" "$(date '+%F %T')" > "$MARK"
echo "[$(date +%T)] $NAME rejoin done -> $FINAL, $DES" | tee -a "$LOG"

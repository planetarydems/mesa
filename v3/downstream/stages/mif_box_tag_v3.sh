#!/usr/bin/env bash
# mif_box_tag_v3.sh <BOX> : v3 chain stage after the fill (2026-10-24). No de-seam splice. The v3 surface of the box is
#   box_dem_<TAG>_deseam.tif = FINAL with the infill-vs-MOLA rule (code/mola_infill_fix.py: inpainted cells (2.3 km) whose
#   median departure from MOLA exceeds 30 m poleward of 60 deg, 150 m elsewhere, replaced by edge-matched MOLA),
# stamped by code/fresh_tag_v3.py (node deseam: inputs FINAL + prov). The name "deseam" is kept so the lattice filter and the
# global render read it unchanged. Companions: box_dem_<TAG>_deseam_mask.tif (1 = replaced) and .json (counts).
# 2026-10-26: FIRST re-join FINAL with code/seamline_merge_v3.py (order-independent join, edge-matched small gaps) unless
#   FINAL.merge.json names that script's md5: the fill stage of the running fleet still calls src/inpainting/seamline_merge.py
#   (fixed path inside the running workers); the re-join reuses the fill's leveled + model surfaces (same flags 32/8/30), rewrites
#   FINAL atomically, re-stamps it (fresh_tag_v3.py stamp FINAL) and writes the marker; the deseam/mif output then rebuilds.
# 2026-10-27: the infill rule is v2.1 (code/mola_infill_fix.py: smooth weight, robust edge matching; no 2.3 km blocks) and is
#   followed by the seam relaxation at abutting frames (code/seam_relax_v3.py --ramp 32: hard hand-overs between frames get a
#   +/- half-offset correction blended with the leveller's weights, reach <= ~55 px). Both run in a temp dir; the v3 surface is
#   swapped in once and stamped. Marker <deseam>.mif.json names both scripts' md5 and the ramp: a fresh surface whose marker
#   differs is rebuilt (finished boxes pick the change up on a re-run of this stage).
# 2026-10-30 (user: "Deploy these fixes now", "yes install the infill fix at the same time"): after the rule, the seam stage (code/seam_stage_v3.py: R1's Poisson + min-cut with the rim fix, PEDR datum term and guard;
#   seam lines from the dominant-frame map only; nothing changes beyond 37 px of the hand-over band), then the relaxation of hard hand-overs as before (it reuses the stage's dominant-frame cache and clears hard joints in chunks
#   the guard rejected). Order chosen by test: rule -> seam stage -> relaxation. The stage also leaves box_seam_<TAG>_domid.tif + _mask.tif (node mask, stamped), which the uncertainty layer and seam_qc read, and the
#   per-line report box_deseam_lines_<TAG>.tsv. The marker names the rule, relaxation and seam scripts' md5s and the band settings.
# Env: TAG (required), BOXES_ROOT (default global_run/boxes), V3D (default global_run/v3_2026-10/downstream), MIF_T_POLAR (30), MIF_T_OTHER (150),
#   RELAX_RAMP (32), RELAX_NP (default 8 inside xeon-g6-volta jobs, whose 2 worker lanes share the node with a GPU fill; else 16).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS=$BASE/global_run/predict_v4s3d_e098/downstream
source "$DS/tag_guard.sh" || exit 2
NAME=${1:?box}; TAG=${TAG:?TAG}; V3D=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}
tag_check_writable "$TAG" || exit 2
OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$NAME/box_run
FT="python3 $V3D/code/fresh_tag_v3.py"
FINAL=$OUT/box_dem_${TAG}_FINAL.tif; PROV=$OUT/box_dem_${TAG}_filled.tif.prov.tif; DES=$OUT/box_dem_${TAG}_deseam.tif
LOG=$OUT/build_final_${TAG}.log
echo "[$(date +%T)] $NAME mif tag=$TAG" | tee -a "$LOG"
$FT check "$OUT" "$TAG" FINAL prov > /dev/null 2>&1 || { echo "[mif] FINAL/prov not fresh -> run the fill first" | tee -a "$LOG"; exit 2; }
# ---- v3 join (2026-10-26) ----
JOIN=$V3D/code/seamline_merge_v3.py; JMD5=$(md5sum "$JOIN" | cut -c1-32); MARK=$FINAL.merge.json
LEVELED=$OUT/box_dem_${TAG}_leveled.tif; MODEL=$OUT/box_dem_${TAG}_filled.tif.model.tif
if ! grep -q "\"$JMD5\"" "$MARK" 2>/dev/null || [ "$FINAL" -nt "$MARK" ]; then
  echo "[join-v3] $NAME: re-joining FINAL with $JOIN ($JMD5)" | tee -a "$LOG"
  PRE=$($FT ids "$OUT" "$TAG" FINAL) || exit 2
  python3 "$JOIN" --dem "$LEVELED" --model "$MODEL" --out "${FINAL%.tif}.v3join.partial.tif" \
      --buffer 32 --feather 8 --min-hole-px 30 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && [ -s "${FINAL%.tif}.v3join.partial.tif" ] || { echo "[join-v3] FAILED rc=$rc" | tee -a "$LOG"; exit 4; }
  mv -f "${FINAL%.tif}.v3join.partial.tif" "$FINAL"
  $FT stamp --expect "$PRE" "$OUT" "$TAG" FINAL 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[join-v3] FINAL NOT stamped (leveled/model changed meanwhile?)" | tee -a "$LOG"; exit 4; }
  printf '{"merge": "%s", "md5": "%s", "flags": "--buffer 32 --feather 8 --min-hole-px 30", "job": "%s", "date": "%s"}\n' \
      "$JOIN" "$JMD5" "${SLURM_JOB_ID:-local}" "$(date '+%F %T')" > "$MARK"
fi
MIFC=$V3D/code/mola_infill_fix.py; RLX=$V3D/code/seam_relax_v3.py; RAMP=${RELAX_RAMP:-32}
RNP=${RELAX_NP:-$([ "${SLURM_JOB_PARTITION:-}" = xeon-g6-volta ] && echo 8 || echo 16)}   # 2026-10-28: 8 on the shared volta node
SST=$V3D/code/seam_stage_v3.py; SB=${SEAM_BAND:-32}; SFT=${SEAM_FEATHER:-4}; SLAM=${SEAM_LAM:-1e-2}; SHALO=64
SIG="mif $(md5sum "$MIFC" | cut -c1-32) relax $(md5sum "$RLX" | cut -c1-32) ramp $RAMP seam $(md5sum "$SST" | cut -c1-32) band $SB feather $SFT lam $SLAM"; DMARK=$DES.mif.json
if $FT check "$OUT" "$TAG" deseam > /dev/null 2>&1 && grep -qF "\"$SIG\"" "$DMARK" 2>/dev/null; then
  echo "[mif] $DES fresh (v3-joined FINAL, $SIG), skip" | tee -a "$LOG"; exit 0; fi
PRE=$($FT ids "$OUT" "$TAG" deseam) || exit 2
TMP=$OUT/.mif_tmp_$TAG; rm -rf "$TMP"; mkdir -p "$TMP"
( unset PROJ_DATA PROJ_LIB; ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} "$V3D/code/mola_infill_fix.py" "$NAME" "$FINAL" "$TMP" --prov "$PROV" --out "$TMP/deseam.tif" \
    --t-polar "${MIF_T_POLAR:-30}" --t-other "${MIF_T_OTHER:-150}" ) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -s "$TMP/deseam.tif" ] || { echo "[mif] FAILED rc=$rc" | tee -a "$LOG"; rm -rf "$TMP"; exit 3; }
( unset PROJ_DATA PROJ_LIB; export BOXES_ROOT=${BOXES_ROOT:-$BASE/global_run/boxes} V3D GDAL_CACHEMAX=256 GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} -W ignore "$SST" "$NAME" "$TMP/deseam.tif" "$TMP/seamed.tif" \
    --tag "$TAG" --band-px "$SB" --feather "$SFT" --lam "$SLAM" --halo "$SHALO" --cache "$TMP" --np "$RNP" --seam-prefix "$TMP/seam" ) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -s "$TMP/seamed.tif" ] && [ -s "$TMP/seam_domid.tif" ] && [ -s "$TMP/seam_mask.tif" ] || { echo "[seam] FAILED rc=$rc" | tee -a "$LOG"; rm -rf "$TMP"; exit 3; }
( unset PROJ_DATA PROJ_LIB; export BOXES_ROOT=${BOXES_ROOT:-$BASE/global_run/boxes}; ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} "$RLX" "$NAME" "$TMP/seamed.tif" "$TMP/relaxed.tif" \
    --tag "$TAG" --ramp "$RAMP" --cache "$TMP" --np "$RNP" ) 2>&1 | grep -v "^\[blend" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -s "$TMP/relaxed.tif" ] || { echo "[relax] FAILED rc=$rc" | tee -a "$LOG"; rm -rf "$TMP"; exit 3; }
MV() { mv -f "$1" "$2" || { echo "[mif] move FAILED $1 -> $2, surface and marker left as they were" | tee -a "$LOG"; exit 3; }; }   # a failed swap must never stamp the old surface under the new marker
MV "$TMP/deseam_mask.tif" "${DES%.tif}_mask.tif"; MV "$TMP/deseam.json" "${DES%.tif}_mif.json"; MV "$TMP/relaxed.tif.json" "${DES%.tif}_relax.json"
MV "$TMP/seamed.tif.json" "${DES%.tif}_seam.json"; MV "$TMP/seamed.tif.lines.tsv" "$OUT/box_deseam_lines_${TAG}.tsv"
MV "$TMP/seam_domid.tif" "$OUT/box_seam_${TAG}_domid.tif"; MV "$TMP/seam_mask.tif" "$OUT/box_seam_${TAG}_mask.tif"; [ -s "$TMP/seam_offset.tif" ] && MV "$TMP/seam_offset.tif" "$OUT/box_seam_${TAG}_offset.tif"
MV "$TMP/relaxed.tif" "$DES"
rm -rf "$TMP"
$FT stamp "$OUT" "$TAG" mask 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[seam] box_seam mask NOT stamped" | tee -a "$LOG"; exit 3; }
$FT stamp --expect "$PRE" "$OUT" "$TAG" deseam 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[mif] NOT stamped (FINAL/prov changed while it ran?)" | tee -a "$LOG"; exit 3; }
printf '{"sig": "%s", "pass1": "%s", "mif": "%s", "relax": "%s", "ramp": %s, "seam": "%s", "job": "%s", "date": "%s"}\n' "$SIG" "$SIG" "$MIFC" "$RLX" "$RAMP" "$SST" "${SLURM_JOB_ID:-local}" "$(date '+%F %T')" > "$DMARK"   # pass1: a later additive pass records itself as pass2 and is swept on its own key, so adding it does not make every box stale
echo "[$(date +%T)] $NAME mif done -> $DES" | tee -a "$LOG"

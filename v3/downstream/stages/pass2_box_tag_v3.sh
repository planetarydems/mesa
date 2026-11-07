#!/usr/bin/env bash
# pass2_box_tag_v3.sh <BOX> : DRAFT (2026-10-30, seam-remaining; not installed). Additive second pass of the v3 seam stage, run AFTER the combined mif stage (rule v2.2 -> seam stage -> relaxation) on a finished box,
# without rebuilding it. It reads box_dem_<TAG>_deseam.tif, solves only the hand-over lines whose census offset is still >= P2_MINOFF (20 m) with code/seam_pass2.py (lam 1e-3, band 64, feather 8, 32 px lip trim,
# no rim taper (option kept), no PEDR in the solve, the stage's guard), and swaps the result in once, stamped. Marker: the first pass's own <deseam>.mif.json (which records "pass1"); this pass ADDS "pass2" (its signature),
#   "pass2_on" (the pass1 it ran on), "pass2_skipped", "pass2_lines", "pass2_km" and leaves "sig"/"pass1" as they are (r1-seam-install's scheme, INSTALL.md). A box is skipped when pass2 == the current signature AND
#   pass2_on == pass1 AND the surface is fresh; a box whose first pass left no line >= P2_MINOFF gets "pass2_skipped": true and no new surface.
# Order for a new box: mif_box_tag_v3.sh then this (call it at the end of the mif stage, or as the worker's next stage); for finished boxes: pass2_sweep.sbatch. QA re-runs after it (qa_box_tag_v3.sh), as after the mif stage.
# Writes only in the box's own directory (same-directory renames: cross-directory renames hung on /home on 2026-10-30). box_seam_<TAG>_domid/mask/offset are untouched (the geometry does not change).
# Skips (rc 2, nothing solved, one line in the log) when the first-pass marker is missing, has no pass1 key (older mif stage), or P1_SIG is set and differs from its pass1.
# Env: TAG (required), BOXES_ROOT, V3D, P1_SIG (optional: the installed first-pass signature to require), P2_BAND (64), P2_FEATHER (8), P2_LAM (1e-3), P2_TRIM (32), P2_TAPER (0), P2_AUTH (0), P2_MINOFF (20), P2_NP (16; 8 on volta), V3_PY.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
BASE=/home/gridsan/groups/planetary_dtm/mars_dem; DS=$BASE/global_run/predict_v4s3d_e098/downstream; source "$DS/tag_guard.sh" || exit 2
NAME=${1:?box}; TAG=${TAG:?TAG}; V3D=${V3D:-$BASE/global_run/v3_2026-10/downstream}; tag_check_writable "$TAG" || exit 2
OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$NAME/box_run; FT="python3 $V3D/code/fresh_tag_v3.py"; DES=$OUT/box_dem_${TAG}_deseam.tif; LOG=$OUT/build_final_${TAG}.log; DMARK=$DES.mif.json
P2=$V3D/code/seam_pass2.py; SB=${P2_BAND:-64}; SFT=${P2_FEATHER:-8}; SLAM=${P2_LAM:-1e-3}; STR=${P2_TRIM:-32}; STP=${P2_TAPER:-0}; SAU=${P2_AUTH:-0}; SMO=${P2_MINOFF:-20}
RNP=${P2_NP:-$([ "${SLURM_JOB_PARTITION:-}" = xeon-g6-volta ] && echo 8 || echo 16)}
echo "[$(date +%T)] $NAME pass2 tag=$TAG" | tee -a "$LOG"
$FT check "$OUT" "$TAG" deseam > /dev/null 2>&1 || { echo "[pass2] deseam not fresh -> run the mif stage first" | tee -a "$LOG"; exit 2; }
[ -s "$DMARK" ] || { echo "[pass2] $NAME: no first-pass marker ($DMARK) -> skip, nothing solved" | tee -a "$LOG"; exit 2; }
ON=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); p=d.get("pass1"); print(p if p else "")' "$DMARK" 2>/dev/null)
[ -n "$ON" ] || { echo "[pass2] $NAME: first-pass marker has no pass1 key (an older mif stage wrote it: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sig",""))' "$DMARK" 2>/dev/null | cut -c1-80)) -> skip, nothing solved" | tee -a "$LOG"; exit 2; }
if [ -n "${P1_SIG:-}" ] && [ "$ON" != "$P1_SIG" ]; then echo "[pass2] $NAME: first pass on this box is not the installed one (pass1 '$ON' != '$P1_SIG') -> skip, nothing solved" | tee -a "$LOG"; exit 2; fi
SIG="pass2 $(md5sum "$P2" | cut -c1-32) band $SB feather $SFT lam $SLAM trim $STR taper $STP auth $SAU minoff $SMO"
if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("pass2")==sys.argv[2] and d.get("pass2_on")==sys.argv[3] else 1)' "$DMARK" "$SIG" "$ON" 2>/dev/null; then echo "[pass2] $DES current ($SIG on $ON), skip" | tee -a "$LOG"; exit 0; fi
PRE=$($FT ids "$OUT" "$TAG" deseam) || exit 2
TMPO=${DES%.tif}.pass2.partial.tif; rm -f "$TMPO" "$TMPO".json "$TMPO".lines.tsv "$TMPO".pairs.npy; TMPC=$OUT/.pass2_cache_$TAG; mkdir -p "$TMPC"
( unset PROJ_DATA PROJ_LIB; export BOXES_ROOT=${BOXES_ROOT:-$BASE/global_run/boxes} V3D GDAL_CACHEMAX=256 GDAL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  ${V3_PY:-/home/gridsan/wmitchell/miniforge/envs/isis/bin/python3} -W ignore "$P2" "$NAME" "$DES" "$TMPO" --tag "$TAG" --band-px "$SB" --feather "$SFT" --lam "$SLAM" --trim "$STR" --taper "$STP" --authority "$SAU" --min-off "$SMO" --halo 96 --cache "$TMPC" --np "$RNP" ) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; rm -rf "$TMPC"
[ "$rc" -eq 0 ] && [ -s "$TMPO" ] && [ -s "$TMPO.json" ] || { echo "[pass2] FAILED rc=$rc, surface and markers left as they were" | tee -a "$LOG"; rm -f "$TMPO" "$TMPO".json "$TMPO".lines.tsv "$TMPO".pairs.npy; exit 3; }
NSOLVED=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["solved"])' "$TMPO.json"); KMACC=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["km_accepted"])' "$TMPO.json")
MARK() { python3 - "$DMARK" "$SIG" "$ON" "$1" "$NSOLVED" "$KMACC" "${SLURM_JOB_ID:-local}" <<'PY'
import json, os, sys
p, sig, on, sk, n, km, job = sys.argv[1:8]; d = json.load(open(p)); d.update(pass2=sig, pass2_on=on, pass2_skipped=(sk == "true"), pass2_lines=int(n), pass2_km=float(km), pass2_job=job, pass2_date=__import__("time").strftime("%F %T"))
json.dump(d, open(p + ".partial", "w")); os.replace(p + ".partial", p)
PY
}
if [ "$NSOLVED" = 0 ]; then rm -f "$TMPO" "$TMPO".lines.tsv "$TMPO".pairs.npy; mv -f "$TMPO.json" "${DES%.tif}_pass2.json"; MARK true; echo "[$(date +%T)] $NAME pass2: no line >= $SMO m, surface unchanged" | tee -a "$LOG"; exit 0; fi
mv -f "$TMPO.json" "${DES%.tif}_pass2.json" && mv -f "$TMPO.lines.tsv" "$OUT/box_pass2_lines_${TAG}.tsv" && rm -f "$TMPO.pairs.npy" && mv -f "$TMPO" "$DES" || { echo "[pass2] swap FAILED, old surface kept" | tee -a "$LOG"; exit 3; }
$FT stamp --expect "$PRE" "$OUT" "$TAG" deseam 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[pass2] NOT stamped (FINAL/prov changed while it ran?)" | tee -a "$LOG"; exit 3; }
MARK false; echo "[$(date +%T)] $NAME pass2 done: $NSOLVED lines, $KMACC km accepted -> $DES" | tee -a "$LOG"

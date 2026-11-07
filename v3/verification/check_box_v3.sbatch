#!/bin/bash
#SBATCH -p xeon-p8 -N 1 -c 48 --mem 180G -t 04:00:00 -J mesa_v3_check
#SBATCH -o /home/gridsan/groups/planetary_dtm/mars_dem/release_check/v3/logs/%x_%j.out
# check_box_v3.sbatch BOX [POLAR_BOX]
#
# Equivalence check of the released v3 per-box chain (release/mader/v3) against one production box.
# The chain is replayed from the released scripts on a COPY of the box's inputs under the scratch root
# $R = $G/release_check/v3 (never a production path). Production files are only read: the box's level,
# fill and join products (copied), the MOLA mosaic, the PEDR shot tables, the orbit screens, the R1
# reference tile of the QA, and the de-seamed surface whose md5 is the reference.
#
#   1. copy leveled, ctx, filled (+ model, prov), FINAL and leveled_<TAG>/ into $R/boxes/BOX/box_run,
#      restore the provenance stamps there (fresh_tag_v3.py stamp; the copies have new inodes)
#   2. join replay: seamline_merge_v3.py on the copied leveled + model, md5 against the production FINAL
#   3. mif stage (infill rule v2.2 -> seam stage -> relaxation), pass 2, QA, all from the released copy
#      ($V3D), with the worker counts of the production sweeps (14)
#   4. md5 of the replayed _deseam.tif (and of the seam domid/mask/offset rasters) against production;
#      the stage markers are compared after removing paths, dates and job ids
#   5. POLAR_BOX (optional): the polar comb correction replayed with the released polar_combfix80 code
#      on the production _deseam.tif of that box (read only), md5 against the installed _deseam_pcf80.tif
# Results: $R/results/BOX/RESULT.txt (one line per comparison), logs in $R/logs. The copied rasters are
# removed at the end; the markers, md5 lists and logs stay.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check/v3
C=$R/code                                        # rsync of release/mader/v3
export TAG=ta_implicit_d1_spline100s1_e098v3
export V3D=$C/downstream
export BOXES_ROOT=$R/boxes
export QA_DIR=$R/qa
export V3_PY=$HOME/miniforge/envs/isis/bin/python3
BOX=${1:?box}; POLAR=${2:-}
[ "$BOXES_ROOT" != "$G/global_run/boxes" ] || { echo "refusing to run on the production boxes"; exit 9; }
P=$G/global_run/boxes/$BOX/box_run; O=$BOXES_ROOT/$BOX/box_run; RES=$R/results/$BOX
mkdir -p "$O" "$RES" "$QA_DIR" "$R/logs"
LOGP="$RES/RESULT.txt"; : > "$LOGP"
say() { echo "[$(date +%T)] $*"; }
res() { echo "$*" | tee -a "$LOGP"; }
m5() { md5sum "$1" | cut -c1-32; }
say "host $(hostname) job ${SLURM_JOB_ID:-local} box $BOX polar ${POLAR:-none} code $C"
say "released code: $(cd "$C" && md5sum downstream/code/seamline_merge_v3.py downstream/code/mola_infill_fix.py downstream/code/seam_stage_v3.py downstream/code/seam_relax_v3.py downstream/code/seam_pass2.py | cut -c1-32 | tr '\n' ' ')"

# ---- 1. stage-in -------------------------------------------------------------------------------------------
B=box_dem_${TAG}
for f in ${B}_leveled.tif ${B}_filled.tif ${B}_filled.tif.model.tif ${B}_filled.tif.prov.tif ${B}_FINAL.tif \
         ${B}_FINAL.tif.merge.json ${B}_leveled.frames.txt ${B}_leveled_offsets.csv ${B}_leveled.provenance.json \
         ctx_robbins_18m_${TAG}.tif; do
  cp -p "$P/$f" "$O/$f" || { res "FAIL stage-in: $f"; exit 2; }
done
cp -rp "$P/leveled_$TAG" "$O/" || { res "FAIL stage-in: leveled_$TAG"; exit 2; }
say "stage-in done: $(du -sh "$O" | cut -f1)"
for f in ${B}_FINAL.tif.merge.json ${B}_deseam.tif.mif.json ${B}_deseam_mif.json ${B}_deseam_seam.json ${B}_deseam_relax.json ${B}_deseam_pass2.json; do
  cp -p "$P/$f" "$RES/production_$f" 2>/dev/null || true
done
cp -p "$G/global_run/v3_2026-10/downstream/qa/$BOX.txt" "$RES/production_qa.txt" 2>/dev/null || true
PROD_DES=$(m5 "$P/${B}_deseam.tif"); PROD_FINAL=$(m5 "$P/${B}_FINAL.tif")
PROD_DOMID=$(m5 "$P/box_seam_${TAG}_domid.tif"); PROD_MASK=$(m5 "$P/box_seam_${TAG}_mask.tif"); PROD_OFFSET=$(m5 "$P/box_seam_${TAG}_offset.tif")
res "production ${B}_deseam.tif md5 $PROD_DES ($(stat -c %s "$P/${B}_deseam.tif") bytes, mtime $(stat -c %y "$P/${B}_deseam.tif" | cut -c1-19))"
res "production ${B}_FINAL.tif md5 $PROD_FINAL"
[ "$(m5 "$O/${B}_FINAL.tif")" = "$PROD_FINAL" ] || { res "FAIL copy of FINAL differs from production"; exit 2; }
# the stamps name inode + size + mtime of each input; the copies have new inodes
python3 "$V3D/code/fresh_tag_v3.py" stamp "$O" "$TAG" frames prov filled model FINAL || { res "FAIL re-stamp"; exit 2; }
python3 "$V3D/code/fresh_tag_v3.py" check "$O" "$TAG" FINAL prov frames || { res "FAIL freshness after re-stamp"; exit 2; }
res "stage-in OK: copies stamped fresh"

# ---- 2. join replay ----------------------------------------------------------------------------------------
t0=$(date +%s)
python3 "$V3D/code/seamline_merge_v3.py" --dem "$O/${B}_leveled.tif" --model "$O/${B}_filled.tif.model.tif" \
    --out "$RES/FINAL_replay.tif" --buffer 32 --feather 8 --min-hole-px 30 > "$R/logs/join_${BOX}_${SLURM_JOB_ID:-local}.log" 2>&1
rc=$?; J=$(m5 "$RES/FINAL_replay.tif" 2>/dev/null || echo none)
if [ "$rc" -eq 0 ] && [ "$J" = "$PROD_FINAL" ]; then res "join OK: replayed FINAL md5 $J = production ($(( $(date +%s) - t0 )) s)"
else res "join FAIL: rc=$rc replayed md5 $J production $PROD_FINAL"; fi
rm -f "$RES/FINAL_replay.tif"

# ---- 3. mif stage, pass 2, QA -------------------------------------------------------------------------------
t0=$(date +%s)
RELAX_NP=14 bash "$V3D/stages/mif_box_tag_v3.sh" "$BOX" > "$R/logs/mif_${BOX}_${SLURM_JOB_ID:-local}.log" 2>&1; rc=$?
res "mif stage rc=$rc ($(( $(date +%s) - t0 )) s)"
[ "$rc" -eq 0 ] || { tail -5 "$R/logs/mif_${BOX}_${SLURM_JOB_ID:-local}.log"; }
t0=$(date +%s)
P2_NP=14 bash "$V3D/stages/pass2_box_tag_v3.sh" "$BOX" > "$R/logs/pass2_${BOX}_${SLURM_JOB_ID:-local}.log" 2>&1; rc=$?
res "pass 2 rc=$rc ($(( $(date +%s) - t0 )) s)"
t0=$(date +%s)
NP=10 bash "$V3D/stages/qa_box_tag_v3.sh" "$BOX" > "$R/logs/qa_${BOX}_${SLURM_JOB_ID:-local}.log" 2>&1; rc=$?
res "qa rc=$rc ($(( $(date +%s) - t0 )) s)"

# ---- 4. comparison -----------------------------------------------------------------------------------------
D=$(m5 "$O/${B}_deseam.tif" 2>/dev/null || echo none)
if [ "$D" = "$PROD_DES" ]; then res "RESULT deseam OK: replayed ${B}_deseam.tif md5 $D = production"
else res "RESULT deseam FAIL: replayed md5 $D production $PROD_DES"; fi
for k in domid mask offset; do
  f="$O/box_seam_${TAG}_$k.tif"; v=$(m5 "$f" 2>/dev/null || echo none); eval "p=\$PROD_$(echo $k | tr a-z A-Z)"
  [ "$v" = "$p" ] && res "seam $k OK: md5 $v = production" || res "seam $k DIFF: $v vs $p"
done
for f in ${B}_FINAL.tif.merge.json ${B}_deseam.tif.mif.json ${B}_deseam_mif.json ${B}_deseam_seam.json ${B}_deseam_relax.json ${B}_deseam_pass2.json; do
  cp -p "$O/$f" "$RES/replay_$f" 2>/dev/null || true
  python3 - "$RES/production_$f" "$RES/replay_$f" "$f" <<'PY' | tee -a "$LOGP"
import json, re, sys
def clean(d):
    if isinstance(d, dict): return {k: clean(v) for k, v in d.items() if k not in ("job", "date", "pass2_job", "pass2_date", "surface", "cache", "seam_prefix", "mif", "relax", "seam", "merge", "secs")}
    if isinstance(d, list): return [clean(x) for x in d]
    if isinstance(d, str): return re.sub(r"/home/gridsan/\S*", "<path>", d)
    return d
try:
    a, b = clean(json.load(open(sys.argv[1]))), clean(json.load(open(sys.argv[2])))
except Exception as e:
    print(f"marker {sys.argv[3]}: not compared ({e})"); sys.exit(0)
if a == b: print(f"marker {sys.argv[3]} OK: same values (paths, dates, job ids and timings removed)")
else:
    diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    print(f"marker {sys.argv[3]} DIFF: {diff}")
PY
done
grep -h "VERDICT" "$RES/production_qa.txt" 2>/dev/null | sed 's/^/production QA: /' | tee -a "$LOGP"
grep -h "VERDICT" "$QA_DIR/$BOX.txt" 2>/dev/null | sed 's/^/replayed   QA: /' | tee -a "$LOGP"
md5sum "$O"/*.tif "$O"/*.json > "$RES/replay_md5.txt" 2>/dev/null

# ---- 5. polar comb correction ------------------------------------------------------------------------------
if [ -n "$POLAR" ]; then
  PCR=$R/pcf80; mkdir -p "$PCR/boxes" "$PCR/logs"; PRES=$R/results/$POLAR; mkdir -p "$PRES"
  PS=$G/global_run/boxes/$POLAR/box_run/${B}_deseam.tif; PM=$G/global_run/boxes/$POLAR/box_run/${B}_deseam_pcf80.json
  cp -p "$PM" "$PRES/production_${B}_deseam_pcf80.json" 2>/dev/null || true
  PROD_PCF=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_md5"])' "$PM")
  PROD_SRC=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_md5"])' "$PM")
  CUR_SRC=$(m5 "$PS"); CUR_PCF=$(m5 "$G/global_run/boxes/$POLAR/box_run/${B}_deseam_pcf80.tif")
  echo "polar $POLAR: production _deseam.tif md5 $CUR_SRC (marker source_md5 $PROD_SRC), installed _deseam_pcf80.tif md5 $CUR_PCF (marker output_md5 $PROD_PCF)" | tee -a "$PRES/RESULT.txt"
  t0=$(date +%s)
  ( unset PROJ_DATA PROJ_LIB; export PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PC=$PCR
    CF_SRC=$PS CF_TAG=pcf80 NPROC=44 $V3_PY "$C/polar_combfix80/code/box_correct.py" "$PCR/boxes" "$POLAR" "$(cat "$C/polar_combfix80/code/params.json")" \
      > "$PCR/logs/box_$POLAR.txt" 2>&1 && PARAMS_JSON=$C/polar_combfix80/code/params.json $V3_PY "$C/polar_combfix80/code/verify_pcf80.py" "$POLAR" > "$PCR/logs/verify_$POLAR.txt" 2>&1 )
  rc=$?; F="$PCR/boxes/$POLAR/${B}_deseam_pcf80.tif"; V=$(m5 "$F" 2>/dev/null || echo none)
  if [ "$rc" -eq 0 ] && [ "$V" = "$CUR_PCF" ] && [ "$V" = "$PROD_PCF" ]; then echo "RESULT pcf80 OK: replayed ${B}_deseam_pcf80.tif md5 $V = installed = marker ($(( $(date +%s) - t0 )) s)" | tee -a "$PRES/RESULT.txt"
  else echo "RESULT pcf80 FAIL: rc=$rc replayed md5 $V installed $CUR_PCF marker $PROD_PCF" | tee -a "$PRES/RESULT.txt"; tail -3 "$PCR/logs/box_$POLAR.txt" "$PCR/logs/verify_$POLAR.txt"; fi
  cp -p "$PCR/boxes/$POLAR/verify.json" "$PRES/replay_verify.json" 2>/dev/null || true
  cp -p "$PCR/boxes/$POLAR/combfix.json" "$PRES/replay_combfix.json" 2>/dev/null || true
  rm -f "$PCR/boxes/$POLAR"/*.tif
fi

# ---- clean-up of the copies --------------------------------------------------------------------------------
rm -rf "$O/leveled_$TAG" "$O"/.mif_tmp_* "$O"/.pass2_cache_*; rm -f "$O"/*.tif
say "done; results in $RES"
cat "$LOGP"

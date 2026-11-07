#!/usr/bin/env bash
# v3 COPY (2026-10-24) of fix_2026-10/rebuild/stages/build_box_final_tag_r2.sh. Changes (level_only only; the fill phase is the
# release-2 one, but the v3 fleet runs the fill through predict_v4s3d_e098/downstream/ctx_fill_*_tag.sbatch as release 2 did):
#   * solver = <V3D>/code/level_frames_box_v3.py --demote-ramp ${DEMOTE_RAMP:-512} (feathered demotion, raster border counted
#     as a data edge, unweighted *_w.tif); weights = ${V3_WEIGHTS:-<V3D>/weights}/<BOX>.csv (the release-2 factors with the
#     non-translational frames set to 0: excluded wherever other coverage exists);
#   * input guard = <V3D>/code/v3_input_guard.py; the queue is QUEUE_TSV (required).
# V3D = env V3D, default global_run/v3_2026-10/downstream. 2026-10-26: the fill phase joins with $V3D/code/seamline_merge_v3.py.
# Previous header: fix_2026-10 RELEASE-2 COPY (2026-10-16): identical to predict_v4s3d_e098/downstream/build_box_final_tag.sh except the level
# solve runs fix_2026-10/blend/code/level_frames_box_v2.py with --frame-weights fix_2026-10/blend/weights/<BOX>.csv
# (blend-weight factors of hazy frames, CTX image QC 2026-10-16) when that file exists, and the provenance records it.
# build_box_final_tag.sh <level_only|fill> <BOX> : TAG-AWARE COPY of global_run/build_box_final.sh (2026-09-16)
# for the ta_implicit_d1_v4s3d_e098 re-prediction. Stage-0 mosaic (level -> feather -> fill -> splice) producing
# box_dem_<TAG>_FINAL.tif from preds_<TAG>. The original is untouched; diffs (see README.md):
#   * TAG=${TAG:-ta_implicit_d1_v4s3d_e098}; the production tags are refused (tag_guard.sh).
#   * NO output-side PEDR shift: the preds are already in the PEDR frame (aligned_pedr inputs). The auto-switch to
#     preds_shifted_$TAG VRTs and the SHIFTED marker logic are removed; the script REFUSES if any
#     preds_shifted_<TAG>* or leveled_<TAG>.SHIFTED* exists (both phases).
#   * Frames come from the uniform-CRS list level_frames_<TAG>/*/pred_shot_anchor.tif written by
#     warp_preds_to_eqc_tag.py (symlinks to preds_<TAG>/<stem>/pred_shot_anchor.tif or its _eqc copy).
#   * phase `level` (per-frame ctx_on_grid paste = the wrong float ctx) is removed: only level_only and fill.
#   * Tagged names: ctx_robbins_18m_<TAG>.tif, build_final_<TAG>.log, lock .level_<TAG>.lock.
#   * Atomic publishes: the leveled mosaic (+ _offsets.csv) is written into .level_tmp_<TAG>/ and renamed into
#     place; the fill writes into .fill_tmp_<TAG>/ and renames (model.tif last, it is the skip gate); FINAL via
#     .partial rename. A killed job can no longer leave a half-written file that a later stage trusts.
#   * An existing leveled mosaic is reused only if it was solved from the same FRAMES.txt (else exit 6).
#   * level provenance JSON (frames, orbit screen sha1, PEDR db, MOLA, taper).
#   * FRESHNESS (review finding D1, 2026-09-17; fresh_tag.py): the level publish writes leveled_<TAG>/SOLVE.json stamped
#     with the new mosaic; the fill first moves every product NOT built from the current leveled mosaic + ctx to
#     <file>.stale.<epoch> (quarantine-stale), reuses the model / FINAL only when their provenance stamps match, and
#     stamps what it builds (refusing when an input changed while it ran). Exit 6 lists every tagged downstream product.
#   * Python exit codes are checked (PIPESTATUS), not only output existence.
# Unchanged: level_frames_box.py flags (--mola ORIGINAL 463 m, --pedr-db Stored_Data_for_Codes/MOLA_SQL_Table/mola,
# --taper-px ${TAPER_PX:-512}, --exclude-orbits shape_error/<BOX>/orbit_screen_<BOX>.txt when present), fill flags.
#
# module load BEFORE set -u (profile/modulefiles reference unset vars).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail

BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS=$BASE/global_run/predict_v4s3d_e098/downstream
source "$DS/tag_guard.sh" || exit 2
cd "$BASE"
PHASE=${1:?usage: build_box_final_tag.sh <level_only|fill> <BOX>}
NAME=${2:?usage: build_box_final_tag.sh <level_only|fill> <BOX>}
TAG=${TAG:-$DEFAULT_TAG}
tag_check_writable "$TAG" || exit 2
JOBS=${JOBS:-$(nproc)}

MOLA=$BASE/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff
PEDR=$BASE/Stored_Data_for_Codes/MOLA_SQL_Table/mola
COARSE=${COARSE_CKPT:-$(python3 "$DS/fresh_tag.py" ckpt)}   # ONE definition (fresh_tag.py); stamped as a root of the fill
export COARSE_CKPT=$COARSE
FINE=${FINE_CKPT:-$(python3 "$DS/fresh_tag.py" fine-ckpt)}   # 18 m stage (2026-09-24); "none" = coarse-only fill
export FINE_CKPT=$FINE
BOXES_ROOT=${BOXES_ROOT:-$BASE/global_run/boxes}      # test hook; production = global_run/boxes
OUT=$BOXES_ROOT/$NAME/box_run
PRED=$OUT/preds_$TAG
FLIST=$OUT/level_frames_$TAG
LEVELED=$OUT/box_dem_${TAG}_leveled.tif
CTX=$OUT/ctx_robbins_18m_${TAG}.tif
FILLED=$OUT/box_dem_${TAG}_filled.tif
FINAL=$OUT/box_dem_${TAG}_FINAL.tif
LOG=$OUT/build_final_${TAG}.log
[ -d "$PRED" ] || { echo "no preds dir: $PRED" >&2; exit 2; }
tag_refuse_shift "$OUT" "$TAG" || exit 4
nfr=$(ls -d "$PRED"/*/ 2>/dev/null | wc -l)
echo "[$(date +%T)] $NAME build_final_tag phase=$PHASE tag=$TAG frames=$nfr jobs=$JOBS" | tee -a "$LOG"

V3D=${V3D:-$BASE/global_run/v3_2026-10/downstream}; export V3D
R2_INPUT_GUARD=$V3D/code/v3_input_guard.py
R2_QUEUE=${QUEUE_TSV:?QUEUE_TSV (the v3 frame queue) is required}
case "$PHASE" in
level_only)
  # ---- (1) joint level + feather-blend mosaic (CPU) ----
  # level_frames_box.py takes NO ctx argument; only the fill consumes the ctx.
  [ -s "$FLIST/FRAMES.txt" ] || { echo "[level] no $FLIST/FRAMES.txt -- run warp_preds_to_eqc_tag.py $NAME first" | tee -a "$LOG"; exit 2; }
  if [ -s "$LEVELED" ]; then
    if cmp -s "$FLIST/FRAMES.txt" "${LEVELED%.tif}.frames.txt" &&
       python3 "$R2_INPUT_GUARD" "$NAME" --tag "$TAG" --queue "$R2_QUEUE" --out "$OUT"; then
      echo "[level] $LEVELED exists (verified current frames, contexts, weights and solver), skip solve" | tee -a "$LOG"
      python3 "$DS/fresh_tag.py" check "$OUT" "$TAG" frames > /dev/null 2>&1 \
        || { echo "[level] REFUSE: missing/stale SOLVE stamp; preserve and review old r2 outputs before re-level" | tee -a "$LOG"; exit 6; }
    else
      {
        echo "[level] $LEVELED exists but frame list or input/weight/solver identity differs -> refusing to reuse it."
        echo "[level] To re-level: move BOTH $LEVELED and ${LEVELED%.tif}.frames.txt aside, then resubmit"
        echo "[level] (_offsets.csv, .provenance.json and leveled_$TAG/ are rewritten by the new solve)."
        echo "[level] Every tagged downstream product below was built from the OLD solve. After the re-level the fill"
        echo "[level] (ctx_fill_box_tag.sh / build_box_final_tag.sh fill) moves each stale one to <file>.stale.<epoch> and"
        echo "[level] rebuilds it, deseam_box_tag.sh redoes every stale stage, and export_aoi_tag.py refuses stale products:"
        python3 "$DS/fresh_tag.py" list "$OUT" "$TAG" 2>&1 | sed 's/^/[level]   /'
        echo "[level] To move them all aside by hand now: python3 $DS/fresh_tag.py quarantine $OUT $TAG model filled prov"
      } | tee -a "$LOG"
      exit 6
    fi
  else
    # TAPER_PX 512: see global_run/build_box_final.sh (the 64-px cap stops spanning frames dominating seams).
    FRAMES_GLOB="$FLIST/*/pred_shot_anchor.tif"
    nlist=$(grep -c . "$FLIST/FRAMES.txt")
    nglob=$(ls $FRAMES_GLOB 2>/dev/null | wc -l)
    [ "$nlist" -eq "$nglob" ] && [ "$nlist" -gt 0 ] || { echo "[level] frame list $nlist rows != $nglob links -> refusing" | tee -a "$LOG"; exit 2; }
    echo "[level] frames: $nglob from $FRAMES_GLOB (unshifted, PEDR-frame predictions)" | tee -a "$LOG"
    LEVEL_EXTRA=""
    SCRTXT=$BASE/global_run/shape_error/$NAME/orbit_screen_$NAME.txt
    if [ -s "$SCRTXT" ]; then
      LEVEL_EXTRA="--exclude-orbits $SCRTXT"
      echo "[level] screening orbits from $SCRTXT" | tee -a "$LOG"
    fi
    LOCK=$OUT/.level_${TAG}.lock
    tag_lock "$LOCK" || { echo "[level] lock held: $LOCK" | tee -a "$LOG"; exit 7; }
    trap 'tag_unlock "$LOCK"' EXIT
    # stale-frame wipe of THIS tag's work dir only (the blend is immune, the de-seam globs are not)
    rm -f "$OUT/leveled_$TAG"/*_leveled.tif "$OUT/leveled_$TAG"/*_w.tif "$OUT/leveled_$TAG"/SOLVE.json "$OUT/leveled_$TAG"/SOLVE.json.inputs.json
    TMPD=$OUT/.level_tmp_$TAG
    rm -rf "$TMPD"; mkdir -p "$TMPD"
    R2_LEVEL=$V3D/code/level_frames_box_v3.py
    R2_WCSV=${V3_WEIGHTS:-$V3D/weights}/$NAME.csv
    R2_WARG=""; [ -s "$R2_WCSV" ] && R2_WARG="--frame-weights $R2_WCSV"
    R2_COVERAGE_REF=none
    R2_COVERAGE_ARGS=()
    if [ -s "$OUT/box_dem_ta_implicit_d1_implrel100s1_e088_leveled.tif" ]; then
      R2_COVERAGE_REF=$OUT/box_dem_ta_implicit_d1_implrel100s1_e088_leveled.tif
      R2_COVERAGE_ARGS=(--coverage-reference "$R2_COVERAGE_REF")
    fi
    echo "[level] v3 level script $R2_LEVEL (demote ramp ${DEMOTE_RAMP:-512} px) weights: ${R2_WARG:-none}; coverage grid: $R2_COVERAGE_REF" | tee -a "$LOG"
    R2_INPUT_SNAPSHOT=$TMPD/input_identity.json
    python3 "$R2_INPUT_GUARD" "$NAME" --tag "$TAG" --queue "$R2_QUEUE" --out "$OUT" --snapshot "$R2_INPUT_SNAPSHOT" \
      || { echo "[level] REFUSE: prediction inputs do not match current queue" | tee -a "$LOG"; exit 6; }
    python3 "$R2_LEVEL" $R2_WARG "${R2_COVERAGE_ARGS[@]}" --demote-ramp "${DEMOTE_RAMP:-512}" \
      --frames-glob "$FRAMES_GLOB" $LEVEL_EXTRA \
      --mola "$MOLA" --pedr-db "$PEDR" --taper-px "${TAPER_PX:-512}" \
      --out "$TMPD/$(basename "$LEVELED")" --work-dir "$OUT/leveled_$TAG" --jobs "$JOBS" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -s "$TMPD/$(basename "$LEVELED")" ]; then
      echo "[level] FAILED (rc=$rc): no $TMPD/$(basename "$LEVELED")" | tee -a "$LOG"; exit 3
    fi
    python3 "$R2_INPUT_GUARD" "$NAME" --tag "$TAG" --queue "$R2_QUEUE" --out "$OUT" --expect-snapshot "$R2_INPUT_SNAPSHOT" \
      || { echo "[level] REFUSE publication: inputs changed during solve; old published raster retained" | tee -a "$LOG"; exit 6; }
    OFFS=$(basename "${LEVELED%.tif}")_offsets.csv
    [ -s "$TMPD/$OFFS" ] && mv -f "$TMPD/$OFFS" "$OUT/$OFFS"
    cp -f "$FLIST/FRAMES.txt" "${LEVELED%.tif}.frames.txt"
    scr_sha=none; scr_n=0
    if [ -n "$LEVEL_EXTRA" ]; then scr_sha=$(sha1sum "$SCRTXT" | cut -d' ' -f1); scr_n=$(wc -w < "$SCRTXT"); fi
    cat > "${LEVELED%.tif}.provenance.json" <<EOF
{"tag": "$TAG", "box": "$NAME", "date": "$(date '+%F %T')", "job": "${SLURM_JOB_ID:-local}", "host": "$(hostname)",
 "frames": $nglob, "frame_list": "$FLIST/FRAMES.txt", "frame_list_sha1": "$(sha1sum "$FLIST/FRAMES.txt" | cut -d' ' -f1)",
 "orbit_screen": "$( [ -n "$LEVEL_EXTRA" ] && echo "$SCRTXT" || echo none)", "orbit_screen_sha1": "$scr_sha", "orbit_screen_n": $scr_n,
 "pedr_db": "$PEDR", "mola": "$MOLA", "taper_px": ${TAPER_PX:-512}, "output_shift": "none (PEDR-frame predictions)",
 "script": "$(readlink -f "$0")", "script_md5": "$(md5sum "$(readlink -f "$0")" | cut -d' ' -f1)",
 "level_script": "$R2_LEVEL", "level_script_md5": "$(md5sum "$R2_LEVEL" | cut -d' ' -f1)",
 "coverage_reference": "$R2_COVERAGE_REF", "coverage_reference_identity": $(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["coverage_reference"]))' "$R2_INPUT_SNAPSHOT"),
 "frame_weights": "$( [ -n "$R2_WARG" ] && echo "$R2_WCSV" || echo none)", "frame_weights_sha1": "$( [ -n "$R2_WARG" ] && sha1sum "$R2_WCSV" | cut -d' ' -f1 || echo none)"}
EOF
    mv -f "$R2_INPUT_SNAPSHOT" "${LEVELED%.tif}.input_identity.json"
    mv -f "$TMPD/$(basename "$LEVELED")" "$LEVELED"          # atomic publish (rename in the same tree)
    rmdir "$TMPD" 2>/dev/null || true
    # D1: tie the per-frame leveled/weight tifs (read by the de-seam) to THIS mosaic
    printf '{"leveled": "%s", "frames": %s, "frame_list_sha1": "%s", "job": "%s", "date": "%s"}\n' "$(basename "$LEVELED")" "$nglob" \
      "$(sha1sum "$FLIST/FRAMES.txt" | cut -d' ' -f1)" "${SLURM_JOB_ID:-local}" "$(date '+%F %T')" > "$OUT/leveled_$TAG/SOLVE.json"
    python3 "$DS/fresh_tag.py" stamp "$OUT" "$TAG" frames 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[level] FAILED to stamp leveled_$TAG/SOLVE.json" | tee -a "$LOG"; exit 3; }
    stale=$(python3 "$DS/fresh_tag.py" list --stale-only "$OUT" "$TAG" 2>&1)
    if [ -n "$stale" ]; then
      echo "[level] NOTE: products from the previous solve are now stale (the fill moves them aside and rebuilds):" | tee -a "$LOG"
      echo "$stale" | sed 's/^/[level]   /' | tee -a "$LOG"
    fi
  fi
  # level_only stops here ON PURPOSE and writes NO ctx: the fill ctx comes from the Murray-Lab mosaic
  # (build_box_ctx_from_murray_tag.sh, run by ctx_fill_box_tag.sh), never from the per-frame ctx_on_grid paste.
  echo "[$(date +%T)] $NAME level_only done (no ctx written) -> $LEVELED" | tee -a "$LOG"
  exit 0
  ;;

fill)
  [ -s "$LEVELED" ] || { echo "[fill] need $LEVELED (run level_only first)" >&2; exit 2; }
  [ -s "$CTX" ]     || { echo "[fill] need $CTX (run build_box_ctx_from_murray_tag.sh first)" >&2; exit 2; }
  # D1: nothing built from a previous level solve / ctx / fill is ever reused: move every non-fresh product aside
  python3 "$DS/fresh_tag.py" quarantine-stale "$OUT" "$TAG" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[fill] quarantine of stale products FAILED" | tee -a "$LOG"; exit 5; }
  # ---- (3) coarse-inpaint hole fill (GPU, or CPU when CUDA_VISIBLE_DEVICES="") ----
  if python3 "$DS/fresh_tag.py" check "$OUT" "$TAG" model filled prov > /dev/null 2>&1; then
    echo "[fill] ${FILLED}.model.tif exists, skip (fresh: stamped from the current leveled mosaic + ctx)" | tee -a "$LOG"
  else
    PRE=$(python3 "$DS/fresh_tag.py" ids "$OUT" "$TAG" model) || exit 2
    TMPF=$OUT/.fill_tmp_$TAG
    rm -rf "$TMPF"; mkdir -p "$TMPF"
    TF=$TMPF/$(basename "$FILLED")
    python3 src/inpainting/fill_holes_coarse.py \
      --dem "$LEVELED" --ctx "$CTX" --mola "$MOLA" --coarse-ckpt "$COARSE" \
      --out "$TF" --stride 1536 --batch 8 --emit-model 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -s "$TF.model.tif" ] || [ ! -s "$TF.prov.tif" ] || [ ! -s "$TF" ]; then
      echo "[fill] FAILED (rc=$rc): incomplete outputs in $TMPF" | tee -a "$LOG"; exit 3
    fi
    # ---- (3b) 18 m detail pass (docs/HOLE_INPAINTING_GLOBAL.md s8): coarse + fine nets on every tile touching a hole
    #      or the merge buffer; the coarse surface is kept elsewhere. Replaces the model surface the splice reads.
    if [ "$FINE" != "none" ]; then
      python3 src/inpainting/fill_holes_fine.py \
        --dem "$LEVELED" --ctx "$CTX" --mola "$MOLA" --coarse-ckpt "$COARSE" --fine-ckpt "$FINE" \
        --coarse-model "$TF.model.tif" --out "$TMPF/fine" --batch 16 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      if [ "$rc" -ne 0 ] || [ ! -s "$TMPF/fine.model.tif" ]; then
        echo "[fill] FINE pass FAILED (rc=$rc): no $TMPF/fine.model.tif" | tee -a "$LOG"; exit 3
      fi
      mv -f "$TMPF/fine.model.tif" "$TF.model.tif"
    fi
    mv -f "$TF.prov.tif" "${FILLED}.prov.tif"
    mv -f "$TF" "$FILLED"
    mv -f "$TF.model.tif" "${FILLED}.model.tif"               # last of the three
    rmdir "$TMPF" 2>/dev/null || true
    python3 "$DS/fresh_tag.py" stamp --expect "$PRE" "$OUT" "$TAG" prov filled model 2>&1 | tee -a "$LOG"   # the skip gate
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[fill] NOT stamped (leveled/ctx changed while the fill ran?) -> stale" | tee -a "$LOG"; exit 3; }
  fi
  # ---- (4) min-cut splice of the fill into the leveled base -> FINAL ----
  if python3 "$DS/fresh_tag.py" check "$OUT" "$TAG" FINAL > /dev/null 2>&1; then
    echo "[splice] $FINAL exists, skip (fresh: stamped from the current leveled mosaic + model)" | tee -a "$LOG"
  else
    PRE=$(python3 "$DS/fresh_tag.py" ids "$OUT" "$TAG" FINAL) || exit 2
    python3 "$V3D/code/seamline_merge_v3.py" \
      --dem "$LEVELED" --model "${FILLED}.model.tif" --out "${FINAL%.tif}.partial.tif" \
      --buffer 32 --feather 8 --min-hole-px 30 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -s "${FINAL%.tif}.partial.tif" ]; then
      echo "[splice] FAILED (rc=$rc): no ${FINAL%.tif}.partial.tif" | tee -a "$LOG"; exit 4
    fi
    mv -f "${FINAL%.tif}.partial.tif" "$FINAL"
    python3 "$DS/fresh_tag.py" stamp --expect "$PRE" "$OUT" "$TAG" FINAL 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[splice] FINAL NOT stamped (leveled/model changed while the splice ran?) -> stale" | tee -a "$LOG"; exit 4; }
    printf '{"merge": "%s", "md5": "%s", "flags": "--buffer 32 --feather 8 --min-hole-px 30", "job": "%s", "date": "%s", "via": "build_box_final_tag_v3.sh fill"}\n' \
        "$V3D/code/seamline_merge_v3.py" "$(md5sum "$V3D/code/seamline_merge_v3.py" | cut -c1-32)" "${SLURM_JOB_ID:-local}" "$(date '+%F %T')" > "$FINAL.merge.json"
  fi
  echo "[$(date +%T)] $NAME fill phase done -> $FINAL" | tee -a "$LOG"
  ;;

*) echo "phase must be level_only|fill (plain 'level' writes the wrong ctx and is not offered)" >&2; exit 1 ;;
esac

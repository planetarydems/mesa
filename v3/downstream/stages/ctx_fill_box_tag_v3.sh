#!/bin/bash
# ctx_fill_box_tag_v3.sh : v3 COPY (2026-10-28) of predict_v4s3d_e098/downstream/ctx_fill_box_tag.sh (unchanged there: R1/R2
# provenance). Only change: the fill phase is $V3D/stages/build_box_final_tag_v3.sh fill (identical to DS build_box_final_tag.sh
# fill except the join: $V3D/code/seamline_merge_v3.py + FINAL.merge.json), so the mif stage's in-place re-join is skipped.
# ctx_fill_box_tag.sh <BOX> [BOX ...] | --list <file> : TAG-AWARE COPY of the body of global_run/ctx_fill_box.sbatch
# (== polar_prep/campaign_schedule/ctx_fill_cpu.sbatch minus its CPU settings), 2026-09-16.
# Run through ctx_fill_gpu_tag.sbatch (volta) or ctx_fill_cpu_tag.sbatch (xeon-p8, CUDA hidden); never directly on
# the login node.  Finishes Stage-0 for already-LEVELLED boxes of TAG:
#   (a) ctx_robbins_18m_<TAG>.tif from the Murray-Lab mosaic (build_box_ctx_from_murray_tag.sh) -- NOT mosaic_ctx.py
#   (b) fill = coarse inpaint + min-cut seamline splice -> box_dem_<TAG>_FINAL.tif (build_box_final_tag.sh fill)
# Diffs vs the original (README.md):
#   * TAG=${TAG:-ta_implicit_d1_v4s3d_e098}; production tags refused; output-side-shift refusal per box.
#   * Tagged ctx (ctx_robbins_18m_<TAG>.tif) and tagged lock (.fill_<TAG>.lock, owner file + dead-job takeover):
#     the stale-grid branch can therefore only ever rename THIS tag's ctx, never the old tag's ctx_robbins_18m.tif.
#   * MURRAY_DIR chosen per box (murray_lab eq / murray_lab_18m polar-band) unless set explicitly.
#   * A ctx under the 90% coverage gate is moved aside to ctx_robbins_18m_<TAG>.lowcov.tif (the original left it in
#     place, and the next run's grid check then said "exists, skip" forever).
#   * EXIT STATUS: non-zero when any requested box ends without a FINAL (the original always exited 0, so an
#     afterok de-seam would have started on nothing).  WALL_H (default 12) sets the walltime budget.
#   * FRESHNESS (review finding D1, 2026-09-17; fresh_tag.py): a box is skipped only when its FINAL and fill mask are
#     FRESH (provenance stamps match the current leveled mosaic, ctx and model). Otherwise, under the fill lock, every
#     product not built from the current leveled mosaic (filled/model/prov, FINAL, de-seam mask/sites/gate/deseam/report)
#     is moved to <file>.stale.<epoch> before the rebuild. The ctx coverage gate now also re-runs on a reused ctx.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS=$BASE/global_run/predict_v4s3d_e098/downstream
V3D=${V3D:-$BASE/global_run/v3_2026-10/downstream}; export V3D
source "$DS/tag_guard.sh" || exit 2
cd "$BASE"
TAG=${TAG:-$DEFAULT_TAG}
tag_check_writable "$TAG" || exit 2
export TAG
export PYTHONWARNINGS=ignore
BOXES_ROOT=${BOXES_ROOT:-$BASE/global_run/boxes}

if [ "${1:-}" = "--list" ]; then
  LIST="$2"
  [ -s "$LIST" ] || LIST="$BASE/$2"
  [ -s "$LIST" ] || LIST="$BASE/global_run/$2"
  [ -s "$LIST" ] || { echo "FATAL: box list not found: $2"; exit 2; }
  mapfile -t BOXES < "$LIST"
  echo "[list] $LIST -> ${#BOXES[@]} boxes"
else
  BOXES=("$@")
fi
[ "${#BOXES[@]}" -gt 0 ] || { echo "FATAL: no boxes given"; exit 2; }
SAFETY=$((40*60))
END=$(( $(date +%s) + ${WALL_H:-12}*3600 - SAFETY ))
nofinal=0

for NAME in "${BOXES[@]}"; do
  [ -z "$NAME" ] && continue
  if (( $(date +%s) >= END )); then
    echo "[budget] walltime low -> stopping cleanly; rerun to continue"; nofinal=$((nofinal+1)); break
  fi
  OUT=$BOXES_ROOT/$NAME/box_run
  LEVELED=$OUT/box_dem_${TAG}_leveled.tif
  CTX=$OUT/ctx_robbins_18m_${TAG}.tif
  FINAL=$OUT/box_dem_${TAG}_FINAL.tif
  echo "=================== $NAME (tag $TAG) ==================="
  tag_refuse_shift "$OUT" "$TAG" || { nofinal=$((nofinal+1)); continue; }
  [ -s "$LEVELED" ] || { echo "[skip] $NAME: not levelled yet"; nofinal=$((nofinal+1)); continue; }
  # CTX_ONLY=1 (2026-09-21, level+ctx fleet while the inpainter is retrained): build/validate the ctx, touch nothing else
  if [ "${CTX_ONLY:-0}" != 1 ]; then
  # D1: skip ONLY a FINAL (and fill mask) built from the current leveled mosaic; an existing FINAL is not enough
  if why=$(python3 "$DS/fresh_tag.py" check "$OUT" "$TAG" FINAL prov 2>&1); then
    echo "[skip] $NAME: FINAL exists and is fresh (stamped from the current leveled mosaic)"; continue
  fi
  [ -s "$FINAL" ] && { echo "[stale] $NAME: FINAL exists but was NOT built from the current leveled mosaic/fill:"; echo "$why" | sed 's/^/[stale]   /'; }
  fi
  # per-box claim so parallel fill instances (or overlapping sweeps) never double-fill one box
  LOCK=$OUT/.fill_${TAG}.lock
  tag_lock "$LOCK" || { echo "[skip] $NAME: fill lock held"; nofinal=$((nofinal+1)); continue; }
  # D1: move every stale product (filled/model/prov, FINAL, de-seam family) to <file>.stale.<epoch> before rebuilding
  if [ "${CTX_ONLY:-0}" != 1 ]; then
  python3 "$DS/fresh_tag.py" quarantine-stale "$OUT" "$TAG" \
    || { echo "[stale] $NAME: quarantine FAILED"; tag_unlock "$LOCK"; nofinal=$((nofinal+1)); continue; }
  fi
  export MURRAY_DIR_BOX=${MURRAY_DIR:-$(murray_dir_for "$NAME")}

  # The ctx must exist AND sit on the SAME grid as the leveled tif (transform + size), not merely exist.
  ctx_state=absent
  if [ -s "$CTX" ]; then
    ctx_state=$(python3 - "$LEVELED" "$CTX" <<'PY'
import sys
try:
    import rasterio
    d = rasterio.open(sys.argv[1]); c = rasterio.open(sys.argv[2])
    tm = tuple(round(x, 3) for x in d.transform[:6]) == tuple(round(x, 3) for x in c.transform[:6])
    cov = c.width >= d.width and c.height >= d.height
    print("ok" if (tm and cov) else "stale")
except Exception:
    print("stale")
PY
)
  fi
  if [ "$ctx_state" = ok ]; then
    echo "[ctx] $NAME: exists on current grid, skip"
  else
    if [ "$ctx_state" = stale ]; then
      echo "[ctx] $NAME: STALE grid (leveled changed since ctx build) -> rebuilding"
      mv -f "$CTX" "${CTX%.tif}.stale_grid.tif"            # tagged file only
    fi
    MURRAY_DIR=$MURRAY_DIR_BOX BOXES_ROOT=$BOXES_ROOT bash "$DS/build_box_ctx_from_murray_tag.sh" "$NAME" \
      || { echo "[ctx] $NAME FAILED"; tag_unlock "$LOCK"; nofinal=$((nofinal+1)); continue; }
    [ -s "$CTX" ] || { echo "[ctx] $NAME: no output"; tag_unlock "$LOCK"; nofinal=$((nofinal+1)); continue; }
  fi
  # Coverage gate: refuse to fill against a ctx covering <90% of the DEM's valid pixels. D1: also for a REUSED ctx
  # (same grid after a re-level does not mean the same valid footprint).
  cov=$(python3 - "$CTX" "$LEVELED" <<'PY'
import sys
try:
    import rasterio
    c = rasterio.open(sys.argv[1]); d = rasterio.open(sys.argv[2])
    cv = c.read(1, out_shape=(1000, 1000)); dv = d.read(1, out_shape=(1000, 1000))
    dm = dv != d.nodata
    print(round(100.0 * ((cv != c.nodata) & dm).sum() / max(dm.sum(), 1), 1))
except Exception:
    print(0)
PY
)
  echo "[ctx] $NAME: ctx@dem=${cov}%"
  if ! awk "BEGIN{exit !($cov >= 90)}"; then
    echo "[ctx] $NAME: coverage ${cov}% < 90% (missing Murray tiles? MURRAY_DIR=$MURRAY_DIR_BOX) -- SKIP fill; ctx moved to ${CTX%.tif}.lowcov.tif"
    mv -f "$CTX" "${CTX%.tif}.lowcov.tif"
    tag_unlock "$LOCK"; nofinal=$((nofinal+1)); continue
  fi

  if [ "${CTX_ONLY:-0}" = 1 ]; then
    echo "[ctx-only] $NAME: ctx on the current grid, coverage ${cov}% -> done (no fill)"; tag_unlock "$LOCK"; continue
  fi
  # Stale-fill guard (D1): build_box_final_tag.sh fill re-runs quarantine-stale after the ctx step (a rebuilt ctx makes
  # the model stale) and reuses the model / FINAL only when their provenance stamps match (fresh_tag.py).

  JOBS=${SLURM_CPUS_PER_TASK:-20} BOXES_ROOT=$BOXES_ROOT bash "$V3D/stages/build_box_final_tag_v3.sh" fill "$NAME" \
    || echo "[fill] $NAME FAILED (rc=$?)"
  if python3 "$DS/fresh_tag.py" check "$OUT" "$TAG" FINAL prov; then echo "[done] $NAME -> $FINAL"; else nofinal=$((nofinal+1)); fi
  tag_unlock "$LOCK"
done
echo "=== ctx_fill_tag batch complete: boxes without FINAL = $nofinal ==="
[ "$nofinal" -eq 0 ]

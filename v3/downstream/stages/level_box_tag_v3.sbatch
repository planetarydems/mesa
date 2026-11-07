#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH --job-name=v4level
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/predict_v4s3d_e098/downstream/logs/level_%x_%j.out
# level_box_tag_v3.sbatch <BOX> : v3 COPY (2026-10-24) of the release-2 level sbatch, calls build_box_final_tag_v3.sh (same dir). Was: RELEASE-2 COPY; TAG-AWARE COPY of global_run/level_box.sbatch (2026-09-16). One box of TAG:
#   1. preflight_tag.py --stage quick   (every queue frame .done, counts, guards, no preds_shifted_<TAG>*/SHIFTED)
#   2. warp_preds_to_eqc_tag.py         (sinu frames of equatorial boxes -> _eqc copies; uniform level_frames_<TAG>/)
#   3. preflight_tag.py --stage full    (one CRS / nodata / 18 m / north-up over the exact frame list)
#   4. build_box_final_tag.sh level_only
# Diffs vs the original: tagged level lock held across steps 2-4, explicit --mem=180G (level ~120 GB peak on 48 cores; p8 nodes have 192 GB), phase fixed to
# level_only (plain `level` pastes the wrong ctx), steps 1-3, logs under downstream/logs, and the EXIT STATUS is the
# chain's (the original always exited 0, so afterok could not stop the fill after a failed level).
# Env: TAG (default ta_implicit_d1_v4s3d_e098), ALLOW_FAILED (default 0), TAPER_PX (default 512),
#      PREFLIGHT_EXTRA (e.g. --allow-non-northup, tests only).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
set -uo pipefail
BASE=/home/gridsan/groups/planetary_dtm/mars_dem
DS=$BASE/global_run/predict_v4s3d_e098/downstream
source "$DS/tag_guard.sh" || exit 2
NAME=${1:?box}
export TAG=${TAG:-$DEFAULT_TAG}
tag_check_writable "$TAG" || exit 2
JOBS=${SLURM_CPUS_PER_TASK:-48}
AF=${ALLOW_FAILED:-0}
echo "=== level_box_tag box=$NAME tag=$TAG node=$(hostname) cpus=$JOBS job=${SLURM_JOB_ID:-} start=$(date) ==="
# One lock around warp + frame list + level: a second level job for the same box must not swap level_frames_<TAG>/
# under a running solve. build_box_final_tag.sh re-enters the same lock (same owner token).
export TAG_LOCK_OWNER=${SLURM_JOB_ID:-local$$}
OUT=${BOXES_ROOT:-$BASE/global_run/boxes}/$NAME/box_run
LOCK=$OUT/.level_${TAG}.lock
[ -d "$OUT" ] || { echo "=== level_box_tag box=$NAME: no $OUT ==="; exit 2; }
tag_lock "$LOCK" || { echo "=== level_box_tag box=$NAME: level lock held ($LOCK) ==="; exit 7; }
trap 'tag_unlock "$LOCK"' EXIT

python3 "$DS/preflight_tag.py" --tag "$TAG" --stage quick --allow-failed "$AF" ${PREFLIGHT_EXTRA:-} "$NAME" \
  || { echo "=== level_box_tag box=$NAME: quick preflight FAILED -> not levelling ==="; exit 10; }
( unset PROJ_DATA PROJ_LIB; python3 "$DS/warp_preds_to_eqc_tag.py" "$NAME" --tag "$TAG" --jobs 8 --allow-failed "$AF" ) \
  || { echo "=== level_box_tag box=$NAME: warp / frame list FAILED ==="; exit 11; }
python3 "$DS/preflight_tag.py" --tag "$TAG" --stage full --allow-failed "$AF" ${PREFLIGHT_EXTRA:-} "$NAME" \
  || { echo "=== level_box_tag box=$NAME: full preflight FAILED (mixed CRS / stale list / shift dir) ==="; exit 12; }
V3D=${V3D:-$BASE/global_run/v3_2026-10/downstream}; export V3D
JOBS=$JOBS bash "$V3D/stages/build_box_final_tag_v3.sh" level_only "$NAME"
rc=$?
echo "=== level_box_tag box=$NAME rc=$rc finished=$(date) ==="
exit $rc

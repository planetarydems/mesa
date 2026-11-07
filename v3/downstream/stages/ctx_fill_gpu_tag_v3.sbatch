#!/bin/bash
#SBATCH --job-name=v3ctxfill_gpu
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=8:00:00
#SBATCH --output=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/ctxfill_gpu_%x_%j.out
# ctx_fill_gpu_tag_v3.sbatch <BOX> : v3 COPY (2026-10-28) of predict_v4s3d_e098/downstream/ctx_fill_gpu_tag.sbatch
# (unchanged there: R1/R2 provenance). Only change: runs $V3D/stages/ctx_fill_box_tag_v3.sh, whose fill phase is
# $V3D/stages/build_box_final_tag_v3.sh fill (join = $V3D/code/seamline_merge_v3.py + FINAL.merge.json marker).
# Original: ctx_fill_gpu_tag.sbatch <BOX> [BOX ...] : GPU (volta) wrapper of ctx_fill_box_tag.sh -- header of
# global_run/ctx_fill_box.sbatch plus an explicit --mem (fill ~108 GB RSS; without --mem the job D-state throttles).
# Env: TAG (default ta_implicit_d1_v4s3d_e098), MURRAY_DIR (default per box). Exit status = ctx_fill_box_tag.sh's.
set -uo pipefail
V3D=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}; export V3D
echo "=== ctx_fill_gpu_tag boxes=$* tag=${TAG:-default} node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) ==="
WALL_H=${WALL_H:-8} bash "$V3D/stages/ctx_fill_box_tag_v3.sh" "$@"
rc=$?
echo "=== ctx_fill_gpu_tag rc=$rc finished=$(date) ==="
exit $rc

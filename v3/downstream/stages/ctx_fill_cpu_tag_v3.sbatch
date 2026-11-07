#!/bin/bash
#SBATCH --job-name=v3ctxfill_cpu
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream/logs/ctxfill_cpu_%x_%j.out
# ctx_fill_cpu_tag_v3.sbatch <BOX> : v3 COPY (2026-10-28) of predict_v4s3d_e098/downstream/ctx_fill_cpu_tag.sbatch
# (unchanged there: R1/R2 provenance). Only change: runs $V3D/stages/ctx_fill_box_tag_v3.sh, whose fill phase is
# $V3D/stages/build_box_final_tag_v3.sh fill (join = $V3D/code/seamline_merge_v3.py + FINAL.merge.json marker).
# Original: ctx_fill_cpu_tag.sbatch <BOX> [BOX ...] : CPU (xeon-p8) wrapper of ctx_fill_box_tag.sh -- the settings of
# polar_prep/campaign_schedule/ctx_fill_cpu.sbatch (CUDA hidden -> fill_holes_coarse.py CPU fallback, same weights;
# CPU-vs-GPU parity rms 1e-4 m; ~42 min/box) plus an explicit --mem (p8 nodes have 192 GB; fill ~108 GB RSS).
# MURRAY_DIR is NOT defaulted to murray_lab_18m here (the original did, which is wrong for equatorial boxes):
# ctx_fill_box_tag.sh picks murray_lab / murray_lab_18m per box unless MURRAY_DIR is set.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48} MKL_NUM_THREADS=${MKL_NUM_THREADS:-48}
V3D=${V3D:-/home/gridsan/groups/planetary_dtm/mars_dem/global_run/v3_2026-10/downstream}; export V3D
echo "=== ctx_fill_cpu_tag boxes=$* tag=${TAG:-default} node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) ==="
WALL_H=${WALL_H:-12} bash "$V3D/stages/ctx_fill_box_tag_v3.sh" "$@"
rc=$?
echo "=== ctx_fill_cpu_tag rc=$rc finished=$(date) ==="
exit $rc

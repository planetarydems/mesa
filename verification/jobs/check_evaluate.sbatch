#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH -t 04:00:00
#SBATCH -J mesa_evaluate
#SBATCH -o %x_%j.out
# Table 1 tile rows and Table 2 of the released evaluation against the manuscript's epoch-88 scoring (V100, as the manuscript runs).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=4
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mader; O=$R/out
V=$G/data/mars_ctx_dem_v4_pedr; A=$G/experiments/v4s3d_e098_paper/test375/aligned
W=$G/experiments/implrel_e088_paper; R95=$G/experiments/v4rel_e095_paper/out_2026-09-29
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff
cd $C
echo "=== $(hostname) $(date)"; nvidia-smi -L
FOOTPRINT_M=150 python3 verification/check_evaluate.py $O/mader_e088.pt $V $A $C/data/test_pairs.txt $W $R95/tiles/tiles_moladist_1k.csv \
  $R/ref/tiles_moladist.csv $A/out359mola $MOLA $O/evaluate
echo "=== done $(date)"

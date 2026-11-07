#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH -t 04:00:00
#SBATCH -J mesa_check_gpu
#SBATCH -o %x_%j.out
# Equivalence checks of the released code against production, GPU node (SuperCloud):
# network, data loader, training steps, the frame that production predicted on a V100, and the tile evaluation.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mader; O=$R/out
M=$G/models/v3_models/ta_implicit_d1_implrel100_s1_fp32_e088; ORIG=$M/code/ablation
V=$G/data/mars_ctx_dem_v4_pedr
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff; PEDR=$G/Stored_Data_for_Codes/MOLA_SQL_Table/mola
TAG=ta_implicit_d1_implrel100s1_e088
mkdir -p $O; cd $C
echo "=== $(hostname) $(date)"; nvidia-smi -L
python3 verification/check_model.py $ORIG $M $O/mader_e088.pt $V $V/manifest_v4s_test_eq.csv
echo "--- data $(date +%T)"
python3 verification/check_data.py $ORIG $V $V/manifest_v4sT_ge10.csv 64
echo "--- training steps $(date +%T)"
OMP_NUM_THREADS=20 python3 verification/check_train_steps.py $ORIG $V $V/manifest_v4sT_ge10.csv 4 2
echo "--- frame (production: xeon-g6-volta, 5 threads) $(date +%T)"
OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 python3 verification/check_predict.py $O/mader_e088.pt $MOLA $PEDR \
  $O/predict_gpu $TAG $G/global_run/boxes \
  E-020_N-30 B01_010038_1501 $G/global_run/boxes/E-020_N-30/box_run/aligned_pedr/B01_010038_1501_aligned.vrt 72.78 305.91
echo "--- tile evaluation (Tables 1 and 2) $(date +%T)"
A=$G/experiments/v4s3d_e098_paper/test375/aligned; W=$G/experiments/implrel_e088_paper
python3 verification/check_evaluate.py $O/mader_e088.pt $V $A $C/data/test_pairs.txt $W \
  $G/experiments/v4rel_e095_paper/out_2026-09-29/tiles/tiles_moladist_1k.csv $R/ref/tiles_moladist.csv $A/out359mola $MOLA \
  $O/evaluate
echo "=== done $(date)"

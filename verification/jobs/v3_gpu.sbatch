#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH -t 03:00:00
#SBATCH -J mesa_v3_gpu
#SBATCH -o %x_%j.out
# The model of the paper (epoch 98, B-spline decoder): export of the released weights, network against the production
# network (CPU and GPU), 4 training steps (CPU), and the v3 frame that production predicted on a V100 (5 threads).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
mkdir -p $O $WT; cd $C
echo "=== $(hostname) $(date)"; nvidia-smi -L
python3 verification/check_model.py $ORIG $M $WT/mesa_e098.pt $V $V/manifest_v4s_test_eq.csv
echo "--- training steps $(date +%T)"
MODEL_CONFIG=mesa_e098 OMP_NUM_THREADS=20 python3 verification/check_train_steps.py $ORIG $V $V/manifest_v4sT_ge10.csv 4 2
echo "--- frame (production: xeon-g6-volta, 5 threads) $(date +%T)"
FOOTPRINT_M=150 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 python3 verification/check_predict.py \
  $WT/mesa_e098.pt $MOLA $PEDR $O/predict_gpu $TAG $B \
  Ep040_N-50 P18_007966_1361 $B/Ep040_N-50/box_run/aligned_pedr/P18_007966_1361_aligned.vrt 78.49 314.27000000000004
echo "=== done $(date)"

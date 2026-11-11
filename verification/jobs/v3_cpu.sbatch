#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 03:00:00
#SBATCH -J mesa_v3_cpu
#SBATCH -o %x_%j.out
# Two v3 frames that production predicted on xeon-p8 nodes (CPU, 5 threads per frame), one of them reaching 87.5 S
# (the PEDR query over all longitudes), re-predicted with the released code and the released weights.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" FOOTPRINT_M=150 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
mkdir -p $O/predict_cpu; cd $C
echo "=== $(hostname) $(date)"
python3 verification/check_predict.py $WT/mesa_e098.pt $MOLA $PEDR $O/predict_cpu $TAG $B \
  E-020_Np20 P18_007968_2023 $B/E-020_Np20/box_run/aligned_pedr/P18_007968_2023_aligned.vrt 41.56 276.76 \
  > $O/predict_cpu/eq.log 2>&1 &
python3 verification/check_predict.py $WT/mesa_e098.pt $MOLA $PEDR $O/predict_cpu $TAG $B \
  E-060_N-90 J06_046962_0925 $B/E-060_N-90/box_run/aligned_pedr/J06_046962_0925_aligned.vrt 81.77 298.33 \
  > $O/predict_cpu/polar.log 2>&1 &
wait
cat $O/predict_cpu/eq.log $O/predict_cpu/polar.log
echo "=== done $(date)"

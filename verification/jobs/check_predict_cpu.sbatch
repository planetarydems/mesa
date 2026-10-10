#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH -t 02:00:00
#SBATCH -J mader_check_cpu
#SBATCH -o %x_%j.out
# Frames that production predicted on xeon-p8 nodes (CPU, 8 threads per frame), re-predicted with the released code.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mader; O=$R/out
M=$G/models/v3_models/ta_implicit_d1_implrel100_s1_fp32_e088; ORIG=$M/code/ablation
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff; PEDR=$G/Stored_Data_for_Codes/MOLA_SQL_Table/mola
TAG=ta_implicit_d1_implrel100s1_e088; B=$G/global_run/boxes
mkdir -p $O/cpu; cd $C
echo "=== $(hostname) $(date)"
python3 verification/check_model.py $ORIG $M $O/cpu/mader_e088.pt
python3 verification/check_predict.py $O/cpu/mader_e088.pt $MOLA $PEDR $O/predict_cpu $TAG $B \
  Ep090_N-70 B09_013265_1194 $B/Ep090_N-70/box_run/aligned_pedr/B09_013265_1194_aligned.vrt 47.85 295.95000000000005 \
  Ep050_Np30 B01_009956_2169 $B/Ep050_Np30/box_run/aligned_pedr/B01_009956_2169_aligned.vrt 45.45 262.89
echo "=== done $(date)"

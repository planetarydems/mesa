#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -J mader_uncert
#SBATCH -o %x_%j.out
# Uncertainty layer of the released code against the production layer of the previous mosaic (epoch 98),
# in the environment the layer was made in (rasterio 1.4.3 with GDAL 3.6.2: the bilinear warp of the infill
# calibration differs at the centimetre level under GDAL 3.9).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF
PY=$HOME/.conda/envs/planetary_dtm/bin/python
G=/home/gridsan/groups/planetary_dtm/mars_dem; R=$G/release_check
U=$R/ref/uncertainty_e098_calibration
cd $R/mader
echo "=== $(hostname) $(date)"
$PY verification/check_uncertainty.py $U $R/ref/test_grids_e098 \
  $G/experiments/v4s3d_e098_paper/labels $G/Stored_Data_for_Codes/MOLA_SQL_Table/mola $G/global_run/boxes \
  ta_implicit_d1_v4s3d_e098 $G/global_run/uncertainty_e098 $R/out/uncertainty 8 E-020_N-30
echo "=== done $(date)"

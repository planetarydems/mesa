#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 06:00:00
#SBATCH -J mesa_mosaic
#SBATCH -o %x_%j.out
# Global-mosaic stages (coarse, level, render) of the released code vs the production e088 mosaic.
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 GDAL_CACHEMAX=1024 OMP_NUM_THREADS=8
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check
PY=$HOME/miniforge/envs/isis/bin/python3        # the environment the production mosaic ran in
cd $R/mader
echo "=== $(hostname) $(date)"
$PY verification/check_mosaic.py $G/global_run/boxes ta_implicit_d1_implrel100s1_e088 \
  $G/global_run/global_cog_implrel100s1_e088 $G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff $R/out/mosaic \
  E-030_N-10 Ep090_N-70 E-180_N-10 E-010_Np80
echo "=== done $(date)"

#!/bin/bash
#SBATCH -p debug-cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 01:00:00
#SBATCH -J mader_images
#SBATCH -o %x_%j.out
# Whole-image rows of Table 1 and the statistics of Section 3.1: the released evaluation against the manuscript's epoch-88 scoring.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mader; O=$R/out/images
V=$G/data/mars_ctx_dem_v4_pedr; P=$G/experiments/v4s3d_e098_paper; A=$P/test375/aligned
W=$G/experiments/implrel_e088_paper
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff
cd $C
echo "=== $(hostname) $(date)"
python3 -c "import rasterio, scipy, numpy; print('rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
python3 verification/check_images.py $V $A $C/data/test_pairs.txt $C/data/test_images.csv $P/labels $W/preds $MOLA \
  $W/grids_noshift $W/frames_json $O 16
# the command line of the module on the scores just written
python3 -m mader.evaluate_images tables --scores $O/images.csv --out $O/tables_cli.json > $O/tables_cli.txt && \
  cmp -s $O/tables.json $O/tables_cli.json && echo "tables command: same JSON as the check" || echo "tables command: JSON DIFFERS"
echo "=== done $(date)"

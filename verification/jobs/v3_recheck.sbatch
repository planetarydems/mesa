#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 02:00:00
#SBATCH -J mesa_v3_recheck
#SBATCH -o %x_%j.out
# Table 1, whole-image rows, with the final check script and the data/test_images.csv of the release (v3_images
# built that table from the manuscript's records; the two must be the same file).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
D=$O/images_final; mkdir -p $D; cd $C
echo "=== $(hostname) $(date)"
python3 -c "import rasterio, scipy, numpy; print('rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
cmp $C/data/test_images.csv $O/images/test_images_v3.csv && echo "data/test_images.csv = the table built from the manuscript's records"
V3_TABLE=$O/ref/table1_v3_complete.json python3 verification/check_images.py - - $C/data/test_pairs.txt $C/data/test_images.csv \
  $WF/labels_full $X/ctx/preds $MOLA $X/ctx/fixed_r1/grids_noshift_full $X/ctx/fixed_r1/frames_json $D 16
python3 -m mesa.evaluate_images score --images $C/data/test_images.csv --labels $WF/labels_full --preds $X/ctx/preds \
  --mola $MOLA --out $D/images_cli.csv --jobs 16 && python3 -m mesa.evaluate_images tables --scores $D/images_cli.csv \
  --out $D/tables_cli.json > $D/tables_cli.txt && cmp -s $D/tables.json $D/tables_cli.json && \
  echo "score + tables commands: same JSON as the check" || echo "score + tables commands: JSON DIFFERS"
echo "=== done $(date)"

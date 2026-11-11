#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 02:00:00
#SBATCH -J mesa_v3_images
#SBATCH -o %x_%j.out
# Table 1, whole-image rows, for the model of the paper: data/test_images.csv rebuilt from the manuscript's v3
# co-registration records, then the released scoring of the 359 v3 product frames against the full stereo DEMs,
# compared grid by grid and measure by measure with the manuscript's v3 scoring.
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
D=$O/images; mkdir -p $D; cd $C
echo "=== $(hostname) $(date)"
python3 -c "import rasterio, scipy, numpy; print('rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
[ -s $D/test_images_v3.csv ] || python3 verification/make_test_images_v3.py $O/ref/test_images_release1.csv \
  $X/inputs/pair_frame_r2.tsv $T11/rigid_json_359 $T11/match_json_359 $X/ctx/fixed_r1/step3 $X/inputs/offsets_r2.csv \
  $WF/labels_full $X/ctx/fixed_r1/grids_noshift_full $D/test_images_v3.csv
V3_TABLE=$O/ref/table1_v3_complete.json python3 verification/check_images.py - - $C/data/test_pairs.txt $D/test_images_v3.csv \
  $WF/labels_full $X/ctx/preds $MOLA $X/ctx/fixed_r1/grids_noshift_full $X/ctx/fixed_r1/frames_json $D 16
python3 -m mesa.evaluate_images tables --scores $D/images.csv --out $D/tables_cli.json > $D/tables_cli.txt && \
  cmp -s $D/tables.json $D/tables_cli.json && echo "tables command: same JSON as the check" || echo "tables command: JSON DIFFERS"
echo "=== done $(date)"

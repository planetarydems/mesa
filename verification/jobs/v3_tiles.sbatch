#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 04:00:00
#SBATCH -J mesa_v3_tiles
#SBATCH -o %x_%j.out
# Table 1, tile rows, for the model of the paper: every one of the 6,396 test samples scored by the released code
# (CPU, 12 processes x 4 threads as the manuscript's replay) against the manuscript's v3 sample table, with the
# production anchoring footprint (150 m); then the same with the released default (168 m).
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
A=$G/experiments/v4s3d_e098_paper/test375/aligned
MAN=$C/data/test_manifest_eq.csv,$C/data/test_manifest_mid.csv,$C/data/test_manifest_polar.csv
cd $C
echo "=== $(hostname) $(date)"
for FP in 150 168; do
  D=$O/tiles$FP; mkdir -p $D
  [ -s $D/tiles.csv ] || python3 verification/run_tiles_chunked.py $WT/mesa_e098.pt $V $MAN $C/data/test_pairs.txt $D/tiles.csv 12 4 $FP
  echo "--- footprint $FP m: check against the manuscript's v3 samples $(date +%T)"
  [ $FP = 168 ] && [ -s $O/tiles150/tiles_mola.csv ] && cp -p $O/tiles150/tiles_mola.csv $D/
  FOOTPRINT_M=$FP REF_TILES=$X/tiles/v3_tile_1k_test359.csv SKIP_BINS=1 V3_TABLE=$O/ref/table1_v3_complete.json \
    python3 verification/check_evaluate.py $WT/mesa_e098.pt $V $A $C/data/test_pairs.txt $X \
    $G/experiments/v4rel_e095_paper/out_2026-09-29/tiles/tiles_moladist_1k.csv $R/ref/tiles_moladist.csv - $MOLA $D
  python3 -m mesa.evaluate tables --samples $D/tiles.csv --mola-tiles $D/tiles_mola.csv --pairs $C/data/test_pairs.txt \
    --out $D/tables_cli.json > /dev/null && echo "tables command written: $D/tables_cli.json"
done
echo "=== done $(date)"

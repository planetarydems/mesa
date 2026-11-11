#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 03:00:00
#SBATCH -J mesa_v3_tilesenv
#SBATCH -o %x_%j.out
# Table 1, tile rows, in the Python environment of the manuscript's v3 tile replay (the conda environment
# planetary_dtm: PyTorch 2.5.1 on the CPU, NumPy 1.26, SciPy 1.16), 12 processes x 4 threads, footprint 150 m,
# against the manuscript's v3 sample table. v3_tiles ran the same in the PyTorch 2.0 module environment.
source /etc/profile 2>/dev/null
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
source /home/gridsan/groups/planetary_dtm/mars_dem/release_check/mesa_v3/verification/jobs/v3_env.sh
PY=$HOME/.conda/envs/planetary_dtm/bin/python
A=$G/experiments/v4s3d_e098_paper/test375/aligned
MAN=$C/data/test_manifest_eq.csv,$C/data/test_manifest_mid.csv,$C/data/test_manifest_polar.csv
D=$O/tiles150_env; mkdir -p $D; cd $C
echo "=== $(hostname) $(date)"
$PY -c "import torch, numpy, scipy, rasterio; print('torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'rasterio', rasterio.__version__, 'GDAL', rasterio.__gdal_version__)"
[ -s $D/tiles.csv ] || $PY verification/run_tiles_chunked.py $WT/mesa_e098.pt $V $MAN $C/data/test_pairs.txt $D/tiles.csv 12 4 150
[ -s $O/tiles150/tiles_mola.csv ] && cp -p $O/tiles150/tiles_mola.csv $D/
FOOTPRINT_M=150 REF_TILES=$X/tiles/v3_tile_1k_test359.csv SKIP_BINS=1 V3_TABLE=$O/ref/table1_v3_complete.json \
  $PY verification/check_evaluate.py $WT/mesa_e098.pt $V $A $C/data/test_pairs.txt $X \
  $G/experiments/v4rel_e095_paper/out_2026-09-29/tiles/tiles_moladist_1k.csv $R/ref/tiles_moladist.csv - $MOLA $D
$PY -m mesa.evaluate tables --samples $D/tiles.csv --mola-tiles $D/tiles_mola.csv --pairs $C/data/test_pairs.txt \
  --out $D/tables_cli.json > /dev/null && echo "tables command written: $D/tables_cli.json"
echo "=== done $(date)"

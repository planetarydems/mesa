#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH -t 06:00:00
#SBATCH -J mesa_hirise
#SBATCH -o %x_%j.out
# HiRISE evaluation (Tables 3 and 4, Section 3.2) of the released code vs the manuscript's epoch-88 scoring, in the Python
# environment that scoring ran in (GDAL 3.6.2, NumPy 1.26, SciPy 1.16, PyTorch 2.5). ONLY=DEM[,DEM] checks a few windows.
source /etc/profile 2>/dev/null
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PROJ_NETWORK=OFF OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
G=/home/gridsan/groups/planetary_dtm/mars_dem; R=$G/release_check
W=$G/experiments/implrel_e088_paper; S=$R/ref
PY=$HOME/.conda/envs/planetary_dtm/bin/python
cd $R/mader
echo "=== $(hostname) $(date)"
$PY verification/check_hirise.py --ref $W/hirise --sample $S/hirise_sample --placement $S/hirise_placement \
  --group $G --preds $W/preds --out ${OUT:-$R/out/hirise} --jobs ${JOBS:-32} ${ONLY:+--only $ONLY}
echo "=== done $(date)"

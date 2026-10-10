#!/bin/bash
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o %x_%j.out
# Tile-chain equivalence: sbatch -p xeon-p8 --cpus-per-task=48 --mem=180G -J mader_level check_chain.sbatch level BOX ...
#                         sbatch -p xeon-g6-volta --gres=gpu:volta:1 --cpus-per-task=20 --mem=180G -J mader_infill check_chain.sbatch infill BOX ...
source /etc/profile 2>/dev/null; module load anaconda/2023a-pytorch 2>/dev/null
unset PROJ_DATA PROJ_LIB
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 GDAL_CACHEMAX=256
G=/home/gridsan/groups/planetary_dtm/mars_dem
R=$G/release_check; C=$R/mader; O=$R/out/chain
MOLA=$G/data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff; PEDR=$G/Stored_Data_for_Codes/MOLA_SQL_Table/mola
TAG=ta_implicit_d1_implrel100s1_e088
STAGE=$1; shift
mkdir -p $O $R/out/infill_nets
ln -sfn $G/src/inpainting/runs/coarse_global_v2/best.pt $R/out/infill_nets/coarse.pt
ln -sfn $G/src/inpainting/runs/fine_global_v1/best.pt $R/out/infill_nets/fine.pt
cd $C
echo "=== $(hostname) $(date) stage $STAGE boxes $*"
for BOX in "$@"; do
  python3 verification/check_tile_chain.py $STAGE $BOX $TAG $G/global_run/boxes $O $MOLA $PEDR $R/out/infill_nets
done
echo "=== done $(date)"
